"""
模型模块
提供模型创建和加载功能
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
    创建模型
    
    Args:
        arch: 模型架构 (目前只支持 'resnet18')
        num_classes: 分类数 (ImageNet: 1000)
        pretrained: 是否使用预训练权重
    
    Returns:
        标准 PyTorch ResNet18 模型（224×224 输入）
    """
    if arch == 'resnet18':
        # 创建标准ResNet18（保持原始结构，适用于 224×224 输入）
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
        
        # 如果类别数不是1000，修改最后的全连接层
        if num_classes != 1000:
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
    # 创建模型结构（标准 ResNet18）
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