"""
接口层配置
定义 WideResNet-50-2 中需要处理的卷积层+BN层对
只处理 layer4 的 3 个 conv2 层（1024通道，块大小64）
"""

import torch.nn as nn


# WideResNet-50-2 接口层配置（固定配置，不使用自动发现）
INTERFACES = {
    'wideresnet50_2': {
        'layers': [
            # layer4: 每个 conv2 输出 1024 通道, 块大小 64, 共 16 块
            ('layer4.0.conv2', 'layer4.0.bn2', 64),
            ('layer4.1.conv2', 'layer4.1.bn2', 64),
            ('layer4.2.conv2', 'layer4.2.bn2', 64),
        ],
        'arch': 'wideresnet50_2',
    }
}


def get_interface_config(arch='wideresnet50_2', model=None):
    """
    获取接口配置
    
    Args:
        arch: 模型架构名称
        model: 模型实例（此参数保留是为了兼容性，实际不使用）
    
    Returns:
        dict: {
            'layers': [(conv_name, bn_name, block_size), ...],
            'arch': 'wideresnet50_2',
        }
    """
    if arch not in INTERFACES:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # 直接返回预定义的配置
    return INTERFACES[arch].copy()


def validate_interface_config(model, config):
    """
    验证接口配置是否与模型匹配
    """
    print(f"\nValidating interface configuration...")
    print(f"Found {len(config['layers'])} interface layers:")
    
    for conv_name, bn_name, block_size in config['layers']:
        # 显示配置的层
        print(f"  - {conv_name} + {bn_name}, block_size={block_size}")
    
    # 获取所有模块
    modules_dict = dict(model.named_modules())
    
    for conv_name, bn_name, block_size in config['layers']:
        # 1. 检查 conv 层
        if conv_name not in modules_dict:
            raise ValueError(f"Conv layer '{conv_name}' not found in model")
        conv = modules_dict[conv_name]
        
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"'{conv_name}' is not nn.Conv2d, got {type(conv)}")
        
        # 2. 检查 bn 层
        if bn_name not in modules_dict:
            raise ValueError(f"BN layer '{bn_name}' not found in model")
        bn = modules_dict[bn_name]
        
        if not isinstance(bn, nn.BatchNorm2d):
            raise ValueError(f"'{bn_name}' is not nn.BatchNorm2d, got {type(bn)}")
        
        # 3. 检查通道数
        num_channels = conv.weight.shape[0]
        if num_channels % block_size != 0:
            raise ValueError(
                f"Layer {conv_name}: num_channels={num_channels} not divisible by "
                f"block_size={block_size}"
            )
        
        # 4. 验证匹配
        if bn.num_features != num_channels:
            raise ValueError(
                f"Channel mismatch: {conv_name} has {num_channels} channels "
                f"but {bn_name} has {bn.num_features} features"
            )
        
        num_blocks = num_channels // block_size
        print(f"  ✓ {conv_name}: {num_channels} channels, "
              f"block_size={block_size}, num_blocks={num_blocks}")
    
    print("Interface configuration validated successfully!\n")