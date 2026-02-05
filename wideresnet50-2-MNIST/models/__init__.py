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


def _adapt_resnet_for_mnist(model):
    """
    将标准ResNet适配为MNIST（28×28灰度输入）
    """
    # 1. 修改conv1: 灰度输入
    model.conv1 = nn.Conv2d(
        in_channels=1,      # 改: 3 → 1（灰度图）
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # 2. 去掉maxpool
    model.maxpool = nn.Identity()
    
    return model


def create_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    创建模型
    """
    if arch == 'wideresnet50_2':
        model = models.wide_resnet50_2(weights=None)
        
        # 适配MNIST的28×28灰度输入
        model = _adapt_resnet_for_mnist(model)  # 改: 函数名
        
        # 修改全连接层
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