"""
模型模块
提供模型创建和加载功能
支持: WideResNet-50-2
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


def create_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    创建模型（适配 CIFAR-100）
    
    Args:
        arch: 模型架构 ('wideresnet50_2')
        num_classes: 分类数
        pretrained: 是否使用预训练权重
    
    Returns:
        PyTorch模型（已适配32×32输入）
    """
    if arch == 'wideresnet50_2':
        # 创建 Wide ResNet-50-2
        if pretrained:
            model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            model = models.wide_resnet50_2(weights=None)
        
        # 适配 CIFAR-100 (32×32 输入)
        # 1. 修改 conv1: 7×7 stride=2 → 3×3 stride=1
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 2. 移除 maxpool (32×32太小，不需要早期下采样)
        model.maxpool = nn.Identity()
        
        # 3. 修改最后的全连接层以适配类别数
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
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
    # 创建模型结构
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