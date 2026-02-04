"""
接口层配置
定义 ResNet18 (ImageNet) 中需要处理的卷积层+BN层对
只处理 layer3 和 layer4（靠近输出的关键层）
"""

# ============================================================
# ResNet18 (ImageNet) 接口配置（仅layer3和layer4）
# ============================================================
INTERFACES = {
    'resnet18': {
        'layers': [
            # layer3: 256通道, 块大小16, 共16块
            ('layer3.0.conv2', 'layer3.0.bn2', 16),
            ('layer3.1.conv2', 'layer3.1.bn2', 16),
            
            # layer4: 512通道, 块大小32, 共16块
            ('layer4.0.conv2', 'layer4.0.bn2', 32),
            ('layer4.1.conv2', 'layer4.1.bn2', 32),
        ],
        'arch': 'resnet18',
    }
}


def get_interface_config(arch='resnet18'):
    """
    获取接口配置
    
    Args:
        arch: 模型架构名称（目前只支持'resnet18'）
    
    Returns:
        dict: {
            'layers': [(conv_name, bn_name, block_size), ...],
            'arch': 'resnet18',
        }
    
    Raises:
        ValueError: 如果架构不支持
    """
    if arch not in INTERFACES:
        raise ValueError(f"Unsupported architecture: {arch}. Only 'resnet18' is supported.")
    
    return INTERFACES[arch]


def validate_interface_config(model, config):
    """
    验证接口配置是否与模型匹配
    检查:
    1. 层是否存在
    2. 通道数是否能被块大小整除
    
    Args:
        model: PyTorch模型
        config: get_interface_config返回的配置
    
    Raises:
        ValueError: 如果配置不合法
    """
    import torch.nn as nn
    
    print("\nValidating interface configuration...")
    
    for conv_name, bn_name, block_size in config['layers']:
        # 获取所有模块
        modules_dict = dict(model.named_modules())
        
        # 1. 检查conv层是否存在
        if conv_name not in modules_dict:
            raise ValueError(f"Conv layer '{conv_name}' not found in model")
        conv = modules_dict[conv_name]
        
        # 检查是否是Conv2d
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"'{conv_name}' is not nn.Conv2d, got {type(conv)}")
        
        # 2. 检查bn层是否存在
        if bn_name not in modules_dict:
            raise ValueError(f"BN layer '{bn_name}' not found in model")
        bn = modules_dict[bn_name]
        
        # 检查是否是BatchNorm2d
        if not isinstance(bn, nn.BatchNorm2d):
            raise ValueError(f"'{bn_name}' is not nn.BatchNorm2d, got {type(bn)}")
        
        # 3. 检查通道数是否能被块大小整除
        num_channels = conv.weight.shape[0]
        if num_channels % block_size != 0:
            raise ValueError(
                f"Layer {conv_name}: num_channels={num_channels} not divisible by "
                f"block_size={block_size}. Remainder: {num_channels % block_size}"
            )
        
        # 计算块数
        num_blocks = num_channels // block_size
        
        print(f"  ✓ {conv_name}: {num_channels} channels, "
              f"block_size={block_size}, num_blocks={num_blocks}")
    
    print("Interface configuration validated successfully!\n")


# ============================================================
# 使用示例和测试
# ============================================================
if __name__ == '__main__':
    import torch
    import torchvision.models as models
    
    print("="*60)
    print("Testing Interface Configuration (ImageNet ResNet18)")
    print("="*60)
    
    # 1. 获取配置
    config = get_interface_config('resnet18')
    print(f"\nArchitecture: {config['arch']}")
    print(f"Number of layers to process: {len(config['layers'])}")
    print("\nLayers:")
    for conv_name, bn_name, block_size in config['layers']:
        print(f"  - {conv_name} + {bn_name}, block_size={block_size}")
    
    # 2. 创建标准 ResNet18 模型并验证
    print("\n" + "-"*60)
    print("Creating standard ResNet18 model (for ImageNet)...")
    model = models.resnet18(weights=None)
    
    # 3. 验证配置
    try:
        validate_interface_config(model, config)
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    print("="*60)