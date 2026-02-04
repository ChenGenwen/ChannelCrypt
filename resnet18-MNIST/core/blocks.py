"""
块管理模块
负责:
1. 解析接口层配置,建立块索引
2. 提取指定层的指定块(conv权重 + BN参数)
3. 对块应用Q变换
4. 对层应用块置换
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

# ============================================================
# BlockManager 类
# ============================================================
class BlockManager:
    """
    块管理器
    维护模型中所有需要处理的接口层的块索引信息
    """
    
    def __init__(self, model: nn.Module, interface_config: Dict):
        """
        Args:
            model: PyTorch模型
            interface_config: 从models/interfaces.py获取的配置,格式:
                {
                    'layers': [(conv_name, bn_name, block_size), ...],
                    'arch': 'resnet18',
                    'strategy': 'layer3_layer4'
                }
        """
        self.model = model
        self.interface_config = interface_config
        
        # 存储每层的块信息
        # 格式: {layer_name: {'conv': module, 'bn': module, 'c': int, 'g': int, ...}}
        self.layer_info = {}
        
        # 构建块索引
        self._build_block_index()
        
        print(f"[BlockManager] Initialized with {len(self.layer_info)} layers")
    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """
        根据名称获取模型中的模块
        
        Args:
            name: 模块名称 (e.g., 'layer3.0.conv2')
        
        Returns:
            对应的nn.Module
        
        Raises:
            KeyError: 如果模块不存在
        """
        module_dict = dict(self.model.named_modules())
        if name not in module_dict:
            raise KeyError(f"Module '{name}' not found in model")
        return module_dict[name]
    
    def _build_block_index(self):
        """
        遍历接口配置,为每层建立块索引
        """
        for conv_name, bn_name, block_size in self.interface_config['layers']:
            # 获取模块
            conv = self._get_module_by_name(conv_name)
            bn = self._get_module_by_name(bn_name)
            
            # 检查conv类型
            if not isinstance(conv, nn.Conv2d):
                raise TypeError(f"{conv_name} is not nn.Conv2d, got {type(conv)}")
            
            # 检查bn类型
            if not isinstance(bn, nn.BatchNorm2d):
                raise TypeError(f"{bn_name} is not nn.BatchNorm2d, got {type(bn)}")
            
            # 获取通道数
            num_channels = conv.weight.shape[0]
            
            # 检查能否整除
            if num_channels % block_size != 0:
                raise ValueError(
                    f"Layer {conv_name}: num_channels={num_channels} not divisible by "
                    f"block_size={block_size}"
                )
            
            # 计算块数
            num_blocks = num_channels // block_size
            
            # 存储信息(用conv_name作为主键)
            self.layer_info[conv_name] = {
                'conv': conv,
                'bn': bn,
                'bn_name': bn_name,
                'c': block_size,
                'g': num_blocks,
                'num_channels': num_channels,
            }
            
            print(f"  ✓ {conv_name}: {num_channels} channels, "
                  f"block_size={block_size}, num_blocks={num_blocks}")
    
    def get_layer_names(self) -> List[str]:
        """返回所有接口层的名称列表"""
        return list(self.layer_info.keys())
    
    def get_layer_info(self, layer_name: str) -> Dict:
        """
        获取某层的块信息
        
        Returns:
            {'conv': module, 'bn': module, 'c': int, 'g': int, ...}
        """
        if layer_name not in self.layer_info:
            raise KeyError(f"Layer '{layer_name}' not in interface config")
        return self.layer_info[layer_name]
    
    def get_block_weight(self, layer_name: str, block_idx: int) -> torch.Tensor:
        """
        获取某层某块的卷积权重
        
        Args:
            layer_name: 层名称
            block_idx: 块索引 (0-based)
        
        Returns:
            权重张量 [c, Cin, k, k]
        """
        info = self.get_layer_info(layer_name)
        c = info['c']
        g = info['g']
        
        if not 0 <= block_idx < g:
            raise IndexError(f"Block index {block_idx} out of range [0, {g})")
        
        # 计算切片范围
        start = block_idx * c
        end = (block_idx + 1) * c
        
        # 返回权重块
        return info['conv'].weight[start:end]  # [c, Cin, k, k]
    
    def get_block_bn(self, layer_name: str, block_idx: int) -> Dict[str, torch.Tensor]:
        """
        获取某层某块的BN参数
        
        Returns:
            {
                'weight': gamma [c],    # scale参数
                'bias': beta [c],       # shift参数
                'running_mean': [c],    # 运行均值(推理用)
                'running_var': [c]      # 运行方差(推理用)
            }
        """
        info = self.get_layer_info(layer_name)
        c = info['c']
        g = info['g']
        bn = info['bn']
        
        if not 0 <= block_idx < g:
            raise IndexError(f"Block index {block_idx} out of range [0, {g})")
        
        # 计算切片范围
        start = block_idx * c
        end = (block_idx + 1) * c
        
        return {
            'weight': bn.weight[start:end],           # gamma
            'bias': bn.bias[start:end],               # beta
            'running_mean': bn.running_mean[start:end],
            'running_var': bn.running_var[start:end],
        }
    
    def set_block_weight(self, layer_name: str, block_idx: int, 
                        new_weight: torch.Tensor):
        """
        设置某层某块的卷积权重(用于Lock/Assemble)
        
        Args:
            layer_name: 层名称
            block_idx: 块索引
            new_weight: 新权重 [c, Cin, k, k]
        """
        info = self.get_layer_info(layer_name)
        c = info['c']
        start = block_idx * c
        end = (block_idx + 1) * c
        
        # 检查形状
        expected_shape = info['conv'].weight[start:end].shape
        if new_weight.shape != expected_shape:
            raise ValueError(f"Shape mismatch: expected {expected_shape}, got {new_weight.shape}")
        
        # 写回(注意要用.data避免影响梯度图)
        info['conv'].weight.data[start:end] = new_weight
    
    def set_block_bn(self, layer_name: str, block_idx: int, 
                    new_bn_params: Dict[str, torch.Tensor]):
        """
        设置某层某块的BN参数
        
        Args:
            new_bn_params: {'weight': [c], 'bias': [c], 'running_mean': [c], 'running_var': [c]}
        """
        info = self.get_layer_info(layer_name)
        c = info['c']
        bn = info['bn']
        start = block_idx * c
        end = (block_idx + 1) * c
        
        # 写回所有BN参数
        bn.weight.data[start:end] = new_bn_params['weight']
        bn.bias.data[start:end] = new_bn_params['bias']
        bn.running_mean.data[start:end] = new_bn_params['running_mean']
        bn.running_var.data[start:end] = new_bn_params['running_var']
    
    def apply_Q_to_block(self, layer_name: str, block_idx: int, 
                        Q: torch.Tensor):
        """
        对某块应用Q变换: W_new = Q @ Flat(W_old)
        
        这是Lock/Assemble的核心操作
        
        Args:
            layer_name: 层名称
            block_idx: 块索引
            Q: 正交矩阵 [c, c]
        """
        info = self.get_layer_info(layer_name)
        c = info['c']
        
        # 1. 获取当前权重块
        W_blk = self.get_block_weight(layer_name, block_idx)  # [c, Cin, k, k]
        
        # 2. 展平
        Cin, k, _ = W_blk.shape[1:]
        W_flat = W_blk.reshape(c, -1)  # [c, d], d = Cin*k*k
        
        # 3. 应用Q变换
        W_transformed = Q @ W_flat  # [c, d]
        
        # 4. 恢复形状
        W_new = W_transformed.reshape(c, Cin, k, k)
        
        # 5. 写回
        self.set_block_weight(layer_name, block_idx, W_new)
    
    def permute_blocks(self, layer_name: str, pi: torch.Tensor):
        """
        对某层应用块级置换
        同时置换conv权重和BN参数
        
        Args:
            layer_name: 层名称
            pi: 置换数组 [g], pi[i]表示第i个位置应该放置原来第pi[i]个块
        """
        info = self.get_layer_info(layer_name)
        g = info['g']
        c = info['c']
        conv = info['conv']
        bn = info['bn']
        
        # 检查pi长度
        if len(pi) != g:
            raise ValueError(f"Permutation length {len(pi)} != num_blocks {g}")
        
        # 1. 备份原始权重和BN参数
        conv_weight_old = conv.weight.data.clone()  # [Cout, Cin, k, k]
        bn_weight_old = bn.weight.data.clone()
        bn_bias_old = bn.bias.data.clone()
        bn_mean_old = bn.running_mean.data.clone()
        bn_var_old = bn.running_var.data.clone()
        
        # 2. 按照置换重新排列
        for i in range(g):
            source_idx = pi[i]  # 第i个位置要放置的源块索引
            
            # 计算切片范围
            target_start = i * c
            target_end = (i + 1) * c
            source_start = source_idx * c
            source_end = (source_idx + 1) * c
            
            # 置换conv权重
            conv.weight.data[target_start:target_end] = conv_weight_old[source_start:source_end]
            
            # 置换BN参数
            bn.weight.data[target_start:target_end] = bn_weight_old[source_start:source_end]
            bn.bias.data[target_start:target_end] = bn_bias_old[source_start:source_end]
            bn.running_mean.data[target_start:target_end] = bn_mean_old[source_start:source_end]
            bn.running_var.data[target_start:target_end] = bn_var_old[source_start:source_end]


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    import torchvision.models as models
    from models.interfaces import get_interface_config
    
    # 创建模型
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)  # 修改为CIFAR-10的10类
    
    # 获取接口配置
    config = get_interface_config('resnet18', 'layer3_layer4')
    
    # 初始化BlockManager
    bm = BlockManager(model, config)
    
    # 测试获取块
    print("\n=== Test Block Access ===")
    layer_name = 'layer3.0.conv2'
    W_blk = bm.get_block_weight(layer_name, 0)
    print(f"Block weight shape: {W_blk.shape}")
    
    bn_params = bm.get_block_bn(layer_name, 0)
    print(f"BN gamma shape: {bn_params['weight'].shape}")
    
    # 测试Q变换
    print("\n=== Test Q Transform ===")
    c = bm.get_layer_info(layer_name)['c']
    Q = torch.eye(c)  # 用单位矩阵测试
    W_before = bm.get_block_weight(layer_name, 0).clone()
    bm.apply_Q_to_block(layer_name, 0, Q)
    W_after = bm.get_block_weight(layer_name, 0)
    print(f"Identity transform preserves weights: {torch.allclose(W_before, W_after)}")