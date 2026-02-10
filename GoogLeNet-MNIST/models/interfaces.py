"""
接口层配置 - GoogLeNet专用（修复版）
定义GoogLeNet中需要处理的卷积层+BN层对
处理 inception5a 和 inception5b 的所有4个分支输出层（共8层）

Inception5a输出通道：256(branch1) + 320(branch2) + 128(branch3) + 128(branch4)
Inception5b输出通道：384(branch1) + 384(branch2) + 128(branch3) + 128(branch4)

统一使用 block_size=16
"""

# ============================================================
# GoogLeNet 接口配置
# ============================================================
INTERFACES = {
    'googlenet': {
        'layers': [
            # ========================================
            # Inception5a (4个分支)
            # ========================================
            # branch1: 1×1 conv → 256通道, 16块
            ('inception5a.branch1.conv', 'inception5a.branch1.bn', 16),
            
            # branch2: 1×1→3×3, 取3×3层 → 320通道, 20块
            # ✅ 修复：必须是 .conv 和 .bn（访问BasicConv2d内部的层）
            ('inception5a.branch2.1.conv', 'inception5a.branch2.1.bn', 16),
            
            # branch3: 1×1→5×5, 取5×5层 → 128通道, 8块
            # ✅ 修复：必须是 .conv 和 .bn
            ('inception5a.branch3.1.conv', 'inception5a.branch3.1.bn', 16),
            
            # branch4: pool→1×1 → 128通道, 8块
            # ✅ 修复：必须是 .conv 和 .bn
            ('inception5a.branch4.1.conv', 'inception5a.branch4.1.bn', 16),
            
            # ========================================
            # Inception5b (4个分支)
            # ========================================
            # branch1: 1×1 conv → 384通道, 24块
            ('inception5b.branch1.conv', 'inception5b.branch1.bn', 16),
            
            # branch2: 1×1→3×3, 取3×3层 → 384通道, 24块
            # ✅ 修复：必须是 .conv 和 .bn
            ('inception5b.branch2.1.conv', 'inception5b.branch2.1.bn', 16),
            
            # branch3: 1×1→5×5, 取5×5层 → 128通道, 8块
            # ✅ 修复：必须是 .conv 和 .bn
            ('inception5b.branch3.1.conv', 'inception5b.branch3.1.bn', 16),
            
            # branch4: pool→1×1 → 128通道, 8块
            # ✅ 修复：必须是 .conv 和 .bn
            ('inception5b.branch4.1.conv', 'inception5b.branch4.1.bn', 16),
        ],
        'arch': 'googlenet',
    }
}


def get_interface_config(arch='googlenet'):
    """
    获取接口配置
    
    Args:
        arch: 模型架构名称（目前只支持'googlenet'）
    
    Returns:
        dict: {
            'layers': [(conv_name, bn_name, block_size), ...],
            'arch': 'googlenet',
        }
    
    Raises:
        ValueError: 如果架构不支持
    """
    if arch not in INTERFACES:
        raise ValueError(f"Unsupported architecture: {arch}. Only 'googlenet' is supported.")
    
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
    print("Testing GoogLeNet Interface Configuration (FIXED VERSION)")
    print("="*60)
    
    # 1. 获取配置
    config = get_interface_config('googlenet')
    print(f"\nArchitecture: {config['arch']}")
    print(f"Number of layers to process: {len(config['layers'])}")
    print("\nLayers:")
    for conv_name, bn_name, block_size in config['layers']:
        print(f"  - {conv_name} + {bn_name}, block_size={block_size}")
    
    # 2. 创建GoogLeNet模型并验证
    print("\n" + "-"*60)
    print("Creating GoogLeNet model...")
    
    # 注意：这里使用标准GoogLeNet，实际训练时会在models/__init__.py中适配CIFAR-100
    model = models.googlenet(pretrained=False, aux_logits=False)
    
    # 3. 验证配置
    try:
        validate_interface_config(model, config)
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 统计总块数
    total_blocks = 0
    for conv_name, bn_name, block_size in config['layers']:
        modules_dict = dict(model.named_modules())
        if conv_name in modules_dict:
            conv = modules_dict[conv_name]
            num_channels = conv.weight.shape[0]
            num_blocks = num_channels // block_size
            total_blocks += num_blocks
    
    print("\n" + "="*60)
    print(f"Total blocks across all layers: {total_blocks}")
    print("="*60)