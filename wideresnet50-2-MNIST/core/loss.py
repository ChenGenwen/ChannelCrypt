"""
同质化损失模块
负责:
1. 计算层内同质化损失 L_conf
2. 对选中块(S内)和未选中块(S外)应用不同权重
"""

import torch
import torch.nn as nn
from typing import Dict
from core.features import compute_layer_phi_list, normalize_phi

# ============================================================
# ConfuseLoss 类
# ============================================================
class ConfuseLoss(nn.Module):
    """
    同质化损失
    
    拉近每层内所有块的特征,使得:
    1. 选中的块(S内)特征高度相似 (权重 α_hi)
    2. 未选中块(S外)也趋于同质化 (权重 α_lo)
    3. 攻击者难以通过特征统计区分哪些块被处理
    
    损失公式:
        L_conf = Σ_t Σ_i α_{t,i} ||φ(t,i) - φ_bar_t||²
    
    其中:
        - t: 层索引
        - i: 块索引
        - φ(t,i): 第t层第i块的特征向量
        - φ_bar_t: 第t层所有块的平均特征
        - α_{t,i} = α_hi (if i∈S_t) else α_lo
    """
    
    def __init__(self, 
                 block_manager,
                 key_manager,
                 alpha_hi: float = 2.0,
                 alpha_lo: float = 1.0,
                 rho: float = 0.5,
                 include_phi4: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            block_manager: BlockManager实例
            key_manager: KeyManager实例
            alpha_hi: 选中块的拉近权重 (通常 > α_lo)
            alpha_lo: 未选中块的拉近权重
            rho: 选块比例
            include_phi4: 是否在损失中包含φ4谱峰值占比特征
            device: 计算设备
        """
        super().__init__()
        
        self.block_manager = block_manager
        self.key_manager = key_manager
        self.alpha_hi = alpha_hi
        self.alpha_lo = alpha_lo
        self.rho = rho
        self.include_phi4 = include_phi4
        self.device = device
        
        # 预计算每层的选块集合S (因为由密钥决定,训练中不变)
        self.selection_cache = {}
        for layer_name in self.block_manager.get_layer_names():
            info = self.block_manager.get_layer_info(layer_name)
            g = info['g']
            S = self.key_manager.generate_selection(layer_name, g, rho)
            self.selection_cache[layer_name] = set(S)  # 用set加速查找
        
        print(f"[ConfuseLoss] Initialized with α_hi={alpha_hi}, α_lo={alpha_lo}, ρ={rho}")
    
    def forward(self) -> torch.Tensor:
        """
        计算总的同质化损失
        
        Returns:
            标量损失张量
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 遍历所有接口层
        for layer_name in self.block_manager.get_layer_names():
            info = self.block_manager.get_layer_info(layer_name)
            g = info['g']
            S = self.selection_cache[layer_name]
            
            # 计算该层所有块的原始特征
            phi_matrix = compute_layer_phi_list(
                self.block_manager, 
                self.key_manager, 
                layer_name, 
                device=self.device,
                include_phi4=self.include_phi4
            )  # [g, phi_dim]
            
            # 按列z-score归一化（仅用于损失计算）
            # 归一化后每列均值为0, ||φ - φ_bar||² 简化为 ||φ||²
            phi_matrix = normalize_phi(phi_matrix)
            squared_dist = (phi_matrix ** 2).sum(dim=1)  # [g]
            
            # 应用加权
            for i in range(g):
                alpha = self.alpha_hi if i in S else self.alpha_lo
                total_loss += alpha * squared_dist[i]
        
        return total_loss
    
    def compute_layer_loss(self, layer_name: str) -> Dict:
        """
        计算单层的详细损失信息 (用于分析)
        
        Returns:
            {
                'total': 总损失,
                'selected_avg': 选中块的平均损失,
                'unselected_avg': 未选中块的平均损失,
                'num_selected': 选中块数量,
                'num_unselected': 未选中块数量
            }
        """
        info = self.block_manager.get_layer_info(layer_name)
        g = info['g']
        S = self.selection_cache[layer_name]
        
        # 计算原始特征并归一化（与 forward 一致）
        phi_matrix = compute_layer_phi_list(
            self.block_manager,
            self.key_manager,
            layer_name,
            device=self.device,
            include_phi4=self.include_phi4
        )
        phi_matrix = normalize_phi(phi_matrix)
        
        # 归一化后均值为0, 直接平方求和
        squared_dist = (phi_matrix ** 2).sum(dim=1)  # [g]
        
        # 分组统计
        selected_losses = []
        unselected_losses = []
        total_loss = 0.0
        
        for i in range(g):
            loss_i = squared_dist[i].item()
            if i in S:
                selected_losses.append(loss_i)
                total_loss += self.alpha_hi * loss_i
            else:
                unselected_losses.append(loss_i)
                total_loss += self.alpha_lo * loss_i
        
        return {
            'total': total_loss,
            'selected_avg': sum(selected_losses) / len(selected_losses) if selected_losses else 0,
            'unselected_avg': sum(unselected_losses) / len(unselected_losses) if unselected_losses else 0,
            'num_selected': len(selected_losses),
            'num_unselected': len(unselected_losses),
        }


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    import torchvision.models as models
    from models.interfaces import get_interface_config
    from core.blocks import BlockManager
    from core.crypto import KeyManager
    
    print("=== Test ConfuseLoss ===")
    
    # 创建模型
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)
    
    # 初始化管理器
    interface_config = get_interface_config('resnet18', 'layer4_only')
    block_manager = BlockManager(model, interface_config)
    key_manager = KeyManager('4a7d1ed414474e4033ac29ccb8653d9b')
    
    # 初始化损失
    confuse_loss = ConfuseLoss(
        block_manager=block_manager,
        key_manager=key_manager,
        alpha_hi=2.0,
        alpha_lo=1.0,
        rho=0.5,
        device='cpu'
    )
    
    # 计算损失
    print("\n1. Computing total loss...")
    loss = confuse_loss()
    print(f"   Total loss: {loss.item():.4f}")
    
    # 分层统计
    print("\n2. Per-layer analysis:")
    for layer_name in block_manager.get_layer_names():
        layer_loss = confuse_loss.compute_layer_loss(layer_name)
        print(f"   {layer_name}:")
        print(f"      Total: {layer_loss['total']:.4f}")
        print(f"      Selected avg: {layer_loss['selected_avg']:.4f} "
              f"(n={layer_loss['num_selected']})")
        print(f"      Unselected avg: {layer_loss['unselected_avg']:.4f} "
              f"(n={layer_loss['num_unselected']})")