"""
接口层配置
定义VGG16-BN中需要处理的卷积层+BN层对
使用自动发现机制，适配修改后的模型结构
"""

import torch.nn as nn

def auto_discover_vgg16bn_interfaces(model, min_channels=512, block_size=32):
    """
    自动发现VGG16-BN中的Conv+BN对
    
    Args:
        model: VGG16-BN模型
        min_channels: 最小通道数（只处理>=这个通道数的层）
        block_size: 块大小
    
    Returns:
        list of (conv_name, bn_name, block_size)
    """
    layers = []
    features_dict = dict(model.features.named_children())
    indices = sorted([int(k) for k in features_dict.keys()])
    
    for idx in indices:
        layer = features_dict[str(idx)]
        
        # 找到Conv2d层
        if isinstance(layer, nn.Conv2d):
            out_channels = layer.out_channels
            
            # 只处理大通道数的层
            if out_channels >= min_channels:
                # 检查下一层是否是BN
                next_idx = idx + 1
                if str(next_idx) in features_dict:
                    next_layer = features_dict[str(next_idx)]
                    if isinstance(next_layer, nn.BatchNorm2d):
                        # 检查通道数是否能被block_size整除
                        if out_channels % block_size == 0:
                            conv_name = f'features.{idx}'
                            bn_name = f'features.{next_idx}'
                            layers.append((conv_name, bn_name, block_size))
    
    return layers


# ============================================================
# VGG16-BN 接口配置（动态生成）
# ============================================================
def get_interface_config(arch='vgg16_bn', model=None):
    """
    获取接口配置
    
    Args:
        arch: 模型架构名称
        model: 可选，提供模型实例用于自动发现层
    
    Returns:
        dict: {
            'layers': [(conv_name, bn_name, block_size), ...],
            'arch': 'vgg16_bn',
        }
    """
    if arch != 'vgg16_bn':
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # 如果提供了模型，自动发现
    if model is not None:
        layers = auto_discover_vgg16bn_interfaces(model, min_channels=512, block_size=32)
        return {
            'layers': layers,
            'arch': arch,
        }
    
    # 否则返回空配置（需要后续提供模型）
    return {
        'layers': [],  # 占位，需要后续用模型填充
        'arch': arch,
    }


def validate_interface_config(model, config):
    """
    验证接口配置是否与模型匹配
    如果config['layers']为空，自动发现并填充
    """
    import torch.nn as nn
    
    # 如果layers为空，自动发现
    if not config['layers']:
        print("Auto-discovering interface layers...")
        config['layers'] = auto_discover_vgg16bn_interfaces(model, min_channels=512, block_size=32)
        if not config['layers']:
            raise ValueError("No valid interface layers found!")
    
    print("\nValidating interface configuration...")
    
    for conv_name, bn_name, block_size in config['layers']:
        # 获取所有模块
        modules_dict = dict(model.named_modules())
        
        # 1. 检查conv层
        if conv_name not in modules_dict:
            raise ValueError(f"Conv layer '{conv_name}' not found in model")
        conv = modules_dict[conv_name]
        
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"'{conv_name}' is not nn.Conv2d, got {type(conv)}")
        
        # 2. 检查bn层
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
        print(f"  ✓ {conv_name} + {bn_name}: {num_channels} channels, "
              f"block_size={block_size}, num_blocks={num_blocks}")
    
    print("Interface configuration validated successfully!\n")


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models import create_model
    
    print("="*60)
    print("Testing Auto-Discovery for VGG16-BN")
    print("="*60)
    
    # 创建适配后的模型
    model = create_model('vgg16_bn', num_classes=10, pretrained=False)
    
    # 打印结构
    print("\nAdapted VGG16-BN Structure:")
    print("-"*60)
    for idx, (name, layer) in enumerate(model.features.named_children()):
        layer_type = layer.__class__.__name__
        if isinstance(layer, nn.Conv2d):
            print(f"features.{name}: {layer_type} (out={layer.out_channels})")
        elif isinstance(layer, nn.BatchNorm2d):
            print(f"features.{name}: {layer_type} (features={layer.num_features})")
        elif isinstance(layer, nn.MaxPool2d):
            print(f"features.{name}: {layer_type}")
    
    # 测试自动发现
    print("\n" + "="*60)
    print("Auto-Discovered Interface Layers")
    print("="*60)
    config = get_interface_config('vgg16_bn', model=model)
    
    print(f"\nArchitecture: {config['arch']}")
    print(f"Number of layers found: {len(config['layers'])}")
    print("\nLayers:")
    for conv_name, bn_name, block_size in config['layers']:
        print(f"  - {conv_name} + {bn_name}, block_size={block_size}")
    
    # 验证
    print("\n" + "="*60)
    try:
        validate_interface_config(model, config)
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    print("="*60)