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


def _adapt_vgg_for_cifar(model):
    """
    将标准VGG适配为CIFAR-10/100（32×32输入）
    
    标准VGG设计是给224×224用的，对于32×32需要调整：
    1. 减少下采样次数（保留3次maxpool而不是5次）
    2. 调整全连接层输入维度
    """
    # VGG16的features部分已经包含卷积+池化
    # 我们需要移除部分maxpool层来适配小图像
    
    # 重建features部分，只保留前3个maxpool
    features = []
    pool_count = 0
    max_pools = 3  # 只保留3次池化：32->16->8->4
    
    for layer in model.features:
        if isinstance(layer, nn.MaxPool2d):
            if pool_count < max_pools:
                features.append(layer)
                pool_count += 1
            # 超过max_pools的池化层跳过
        else:
            features.append(layer)
    
    model.features = nn.Sequential(*features)
    
    # 调整avgpool和分类器
    # 经过3次池化后特征图大小为 4×4
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    # 保持分类器结构，但调整输入
    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, model.classifier[6].out_features)  # 保持原输出类别数
    )
    
    return model


def create_model(arch: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """
    创建模型
    
    Args:
        arch: 模型架构 ('vgg16_bn')
        num_classes: 分类数
        pretrained: 是否使用预训练权重（对CIFAR保持False）
    
    Returns:
        PyTorch模型（已适配32×32输入）
    """
    if arch == 'vgg16_bn':
        # 创建标准VGG16-BN
        model = models.vgg16_bn(weights=None)
        
        # 适配CIFAR的32×32输入
        model = _adapt_vgg_for_cifar(model)
        
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