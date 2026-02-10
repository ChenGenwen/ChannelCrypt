"""
训练Base模型
不添加任何混淆机制的标准训练
每隔TEST_INTERVAL轮在测试集上验证，自动保存最佳模型
training_history.csv实时写入
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_model
from utils.logger import Logger

# ============================================================
# 全局参数
# ============================================================
TEST_INTERVAL = 5       # 每隔5轮测试一次


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_data_loaders(config: dict):
    """
    创建数据加载器
    
    Returns:
        train_loader, test_loader
    """
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    
    # 训练集变换(带数据增强)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # 测试集变换(不增强)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    data_dir = config['dataset']['data_dir']
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    """
    训练一个epoch
    
    Returns:
        avg_loss, avg_acc
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        train_loader,
        desc=f"Train [{epoch}/{num_epochs}]",
        leave=False,
        dynamic_ncols=True
    )

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total += targets.size(0)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}"
        })

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


def test_epoch(model, test_loader, criterion, device, epoch, num_epochs):
    """
    测试一个epoch
    
    Returns:
        avg_loss, avg_acc
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        test_loader,
        desc=f"Test  [{epoch}/{num_epochs}]",
        leave=False,
        dynamic_ncols=True
    )

    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total += targets.size(0)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })

    avg_loss = total_loss / len(test_loader)
    avg_acc = 100.0 * correct / total
    return avg_loss, avg_acc


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path =  r'D:\Model IP Protection\locked\GoogLeNet-cifar-100\config\config.yaml'
    config = load_config(config_path)
    
    print("="*60)
    print("Training Base Model (ResNet18 on CIFAR-100)")
    print("="*60)
    
    # ============================================================
    # 2. 设置设备和随机种子
    # ============================================================
    device = torch.device('cuda' if config['device']['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    seed = config['device']['seed']
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
    
    # ============================================================
    # 3. 创建数据加载器
    # ============================================================
    print("\nLoading CIFAR-100 dataset...")
    train_loader, test_loader = get_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ============================================================
    # 4. 创建模型
    # ============================================================
    print("\nCreating model...")
    model = create_model(
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)
    print(f"Model: {config['model']['arch']}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # ============================================================
    # 5. 定义损失函数和优化器
    # ============================================================
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr_base'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['lr_schedule_base']['milestones'],
        gamma=config['training']['lr_schedule_base']['gamma']
    )
    
    # ============================================================
    # 6. 创建日志记录器
    # ============================================================
    logger = Logger(
        log_dir=config['paths']['log_dir'],
        checkpoint_dir=config['paths']['checkpoint_dir'],
        file_suffix='(base)'
    )
    logger.set_config(config)

    # ============================================================
    # 7. 训练循环
    # ============================================================
    num_epochs = config['training']['epochs_base']
    
    logger.log(f"\nStarting training for {num_epochs} epochs...")
    logger.log(f"Test interval: every {TEST_INTERVAL} epochs")
    logger.log(f"Initial learning rate: {config['training']['lr_base']}")
    logger.log(f"LR milestones: {config['training']['lr_schedule_base']['milestones']}\n")
    
    for epoch in range(1, num_epochs + 1):
        # ---- 训练阶段（每轮都执行）----
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, num_epochs
        )
        
        # ---- 学习率更新（每轮都执行）----
        scheduler.step()
        
        # ---- 测试+记录+保存（每TEST_INTERVAL轮或最后一轮）----
        if epoch % TEST_INTERVAL == 0 or epoch == num_epochs:
            test_loss, test_acc = test_epoch(
                model, test_loader, criterion, device,
                epoch, num_epochs
            )
            
            # 记录到内存
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }
            logger.update(epoch, metrics)
            
            # 实时写入CSV（每次测试后立刻追加一行）
            logger.append_history_csv(epoch, metrics)
            
            # 打印本轮结果
            logger.log(
                f"Epoch {epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}%"
            )
            
            # 保存最佳模型
            logger.save_best_model(
                model, epoch, test_acc,
                config['paths']['base_model']
            )
    
    # ============================================================
    # 8. 保存汇总和总结
    # ============================================================
    logger.save_best_results_csv(model_type='base')
    logger.summary()
        
    print("\n" + "="*60)
    print("Base model training completed!")
    print(f"Best model saved to: {config['paths']['checkpoint_dir']}/{config['paths']['base_model']}")
    print(f"Best test accuracy: {logger.best_acc:.2f}% at epoch {logger.best_epoch}")
    print("="*60)


if __name__ == '__main__':
    main()