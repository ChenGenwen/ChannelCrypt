"""
模型模块 - GoogLeNet专用
提供GoogLeNet模型创建和CIFAR-10适配功能
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


def _adapt_googlenet_for_cifar(model):
    """
    将标准GoogLeNet适配为CIFAR-10（32×32输入）
    
    标准GoogLeNet开头设计是给224×224用的:
        conv1: 7×7, stride=2, padding=3  → 224→112
        maxpool1: 3×3, stride=2         → 112→56
        
    对于32×32的输入会导致:
        conv1: 7×7, stride=2  → 32→16
        maxpool1: 3×3, stride=2 → 16→8
    信息损失太大，严重影响准确率
    
    修改后:
        conv1: 3×3, stride=1  → 32→32（保持分辨率）
        maxpool1: Identity    → 不下采样
        maxpool2: 3×3, stride=2 → 32→16（第一次下采样推迟到这里）
    """
    # 1. 替换conv1: 7×7 stride=2 → 3×3 stride=1
    # 保持输出通道数64不变
    model.conv1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
    )
    
    # 2. 去掉maxpool1（用Identity替换，完全不下采样）
    model.maxpool1 = nn.Identity()
    
    # 3. maxpool2保持不变（3×3 stride=2），这里才开始第一次下采样
    # 这样特征图尺寸变化: 32→32→16→8→4（更合理）
    
    return model


def create_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    创建模型
    
    Args:
        arch: 模型架构 (目前只支持 'googlenet')
        num_classes: 分类数
        pretrained: 是否使用预训练权重（对CIFAR无意义，保持False）
    
    Returns:
        PyTorch模型（已适配32×32输入）
    """
    if arch == 'googlenet':
        # 创建标准GoogLeNet (aux_logits=False 移除辅助分类器，简化训练)
        model = models.googlenet(
            pretrained=pretrained,
            aux_logits=False,  # CIFAR-10训练不需要辅助分类器
            init_weights=not pretrained
        )
        
        # 适配CIFAR-10的32×32输入
        model = _adapt_googlenet_for_cifar(model)
        
        # 修改最后的全连接层
        in_features = model.fc.in_features  # 通常是1024
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
    # 创建模型结构（已包含CIFAR适配）
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