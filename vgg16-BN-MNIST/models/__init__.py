"""
模型模块
提供模型创建和加载功能（MNIST专用）
"""

import torch
import torch.nn as nn
import torchvision.models as models

from .interfaces import get_interface_config, validate_interface_config

__all__ = [
    'get_interface_config',
    'validate_interface_config',
    'create_model',
    'load_model',
]


def _adapt_vgg_for_mnist(model):
    """
    将标准VGG16-BN适配为MNIST（28×28灰度图）
    
    修改点：
    1. 第一层卷积：3通道 → 1通道（灰度图）
    2. 减少池化层：5次 → 3次（适配小图像）
    3. 调整全连接层输入
    """
    # 1. 修改第一个卷积层：3通道 → 1通道
    first_conv = model.features[0]
    model.features[0] = nn.Conv2d(
        in_channels=1,  # MNIST是灰度图
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None
    )
    
    # 2. 重建features，只保留前3个maxpool
    features = []
    pool_count = 0
    max_pools = 3  # 28 → 14 → 7 → 3
    
    for layer in model.features:
        if isinstance(layer, nn.MaxPool2d):
            if pool_count < max_pools:
                features.append(layer)
                pool_count += 1
            # 跳过后面的池化层
        else:
            features.append(layer)
    
    model.features = nn.Sequential(*features)
    
    # 3. 调整avgpool和分类器
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, model.classifier[6].out_features)
    )
    
    return model


def create_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    创建模型（MNIST专用）
    
    Args:
        arch: 模型架构（只支持 'vgg16_bn'）
        num_classes: 分类数（MNIST固定为10）
        pretrained: 是否使用预训练权重（保持False）
    
    Returns:
        PyTorch模型（已适配MNIST）
    """
    if arch == 'vgg16_bn':
        # 创建标准VGG16-BN
        model = models.vgg16_bn(weights=None)
        
        # 适配MNIST
        model = _adapt_vgg_for_mnist(model)
        
        # 修改最后的分类层
        model.classifier[6] = nn.Linear(4096, num_classes)
        
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    return model


def load_model(checkpoint_path: str, arch: str, num_classes: int, 
               device: str = 'cpu') -> nn.Module:
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        arch: 模型架构
        num_classes: 分类数
        device: 设备
    
    Returns:
        加载好权重的模型
    """
    # 创建模型结构（已包含MNIST适配）
    model = create_model(arch, num_classes, pretrained=False)
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'test_acc' in checkpoint:
            print(f"  Test accuracy: {checkpoint['test_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model