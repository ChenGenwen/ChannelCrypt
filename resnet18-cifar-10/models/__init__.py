"""
模型模块
提供模型创建和加载功能 - 支持MNIST和CIFAR数据集
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


def _adapt_resnet_for_small_images(model, input_channels=1):
    """
    将标准ResNet适配为小尺寸图像（28×28 MNIST 或 32×32 CIFAR）
    
    标准ResNet开头设计是给224×224用的:
        conv1: 7×7, stride=2  → 224→112
        maxpool: 3×3, stride=2 → 112→56
    
    对于28×28或32×32的输入会导致信息损失太大，严重影响准确率
    
    修改后:
        conv1: 3×3, stride=1  → 保持分辨率
        去掉maxpool
    
    Args:
        model: ResNet模型
        input_channels: 输入通道数 (1=灰度MNIST, 3=RGB CIFAR)
    """
    # 1. 替换conv1: 7×7 stride=2 → 3×3 stride=1
    model.conv1 = nn.Conv2d(
        in_channels=input_channels,  # 支持1通道(MNIST)或3通道(CIFAR)
        out_channels=64,
        kernel_size=3,    # 从7改为3
        stride=1,         # 从2改为1（不下采样）
        padding=1,        # 对应padding也改为1
        bias=False
    )
    
    # 2. 去掉maxpool（用Identity替换）
    model.maxpool = nn.Identity()
    
    return model


def create_model(arch: str, num_classes: int, pretrained: bool = False,
                 input_channels: int = 1) -> nn.Module:
    """
    创建模型
    
    Args:
        arch: 模型架构 (目前只支持 'resnet18')
        num_classes: 分类数
        pretrained: 是否使用预训练权重（对小数据集无意义，保持False）
        input_channels: 输入通道数（1=MNIST灰度图，3=CIFAR彩色图）
    
    Returns:
        PyTorch模型（已适配小尺寸图像输入）
    """
    if arch == 'resnet18':
        # 创建标准ResNet18
        model = models.resnet18(weights=None)
        
        # 适配小尺寸图像（MNIST 28×28 或 CIFAR 32×32）
        model = _adapt_resnet_for_small_images(model, input_channels)
        
        # 修改最后的全连接层
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
    # 创建模型结构（已包含小尺寸图像适配）
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