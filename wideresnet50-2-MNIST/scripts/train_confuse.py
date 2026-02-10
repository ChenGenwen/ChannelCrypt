"""
训练Confuse模型
在base模型基础上添加同质化损失进行微调
每隔METRICS_INTERVAL轮评估混淆指标，自动保存最佳模型
training_history.csv和best_results.csv实时写入

修改说明：
- 优化器从 SGD 改为 Adam
- 移除了 momentum 参数
- 评估混淆指标时包含φ4特征
- 专门适配 WideResNet-50-2 + MNIST
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

from models import load_model, get_interface_config, validate_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager
from core.loss import ConfuseLoss
from core.features import compute_layer_phi_list, normalize_phi
from utils.logger import Logger
from utils.metrics import ConfuseMetrics


# ============================================================
# 全局参数
# ============================================================
METRICS_INTERVAL = 5    # 每隔5轮评估混淆指标


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_data_loaders(config: dict):
    """
    创建数据加载器
    """
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    
    # 训练集变换（MNIST简化增强）
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # 测试集变换
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    data_dir = config['dataset']['data_dir']
    
    trainset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.MNIST(
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


def evaluate_confuse_metrics(block_manager, key_manager, rho, device='cpu',
                            include_phi4=True):
    """
    评估所有接口层的混淆效果指标
    
    Returns:
        dict: {
            layer_name: {
                'raw': {
                    'homogeneity': float,
                    'overall_similarity': float,
                    'selected_similarity': float,
                    'unselected_similarity': float,
                    'cross_similarity': float,
                },
                'normalized': {同上}
            }
        }
    """
    results = {}
    
    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        g = info['g']
        
        # 生成选块集合
        S = key_manager.generate_selection(layer_name, g, rho)
        
        # 计算原始特征矩阵 [g, phi_dim]
        phi_matrix_raw = compute_layer_phi_list(
            block_manager, key_manager, layer_name, device=device,
            include_phi4=include_phi4
        )
        
        # 归一化特征矩阵
        phi_matrix_norm = normalize_phi(phi_matrix_raw)
        
        # 转换为列表格式（ConfuseMetrics 需要）
        phi_list_raw = [phi_matrix_raw[i] for i in range(g)]
        phi_list_norm = [phi_matrix_norm[i] for i in range(g)]
        
        # 计算原始特征的指标
        sim_raw = ConfuseMetrics.compute_intra_similarity(phi_list_raw, S)
        homo_raw = ConfuseMetrics.compute_layer_homogeneity(phi_list_raw)
        
        # 计算归一化特征的指标
        sim_norm = ConfuseMetrics.compute_intra_similarity(phi_list_norm, S)
        homo_norm = ConfuseMetrics.compute_layer_homogeneity(phi_list_norm)
        
        results[layer_name] = {
            'raw': {
                'homogeneity': homo_raw,
                'overall_similarity': sim_raw['overall_similarity'],
                'selected_similarity': sim_raw['selected_similarity'],
                'unselected_similarity': sim_raw['unselected_similarity'],
                'cross_similarity': sim_raw['cross_similarity'],
            },
            'normalized': {
                'homogeneity': homo_norm,
                'overall_similarity': sim_norm['overall_similarity'],
                'selected_similarity': sim_norm['selected_similarity'],
                'unselected_similarity': sim_norm['unselected_similarity'],
                'cross_similarity': sim_norm['cross_similarity'],
            }
        }
    
    return results


def train_epoch(model, train_loader, criterion, confuse_loss, optimizer, 
               lambda_conf, device, epoch, num_epochs):
    """
    训练一个epoch
    
    Returns:
        avg_task_loss, avg_confuse_loss, avg_total_loss, avg_acc
    """
    model.train()
    
    total_task_loss = 0.0
    total_confuse_loss = 0.0
    total_loss_sum = 0.0
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
        
        # 前向传播
        outputs = model(inputs)
        
        # 任务损失
        task_loss = criterion(outputs, targets)
        
        # 同质化损失（在Q空间计算）
        conf_loss = confuse_loss()
        
        # 总损失
        total_loss = task_loss + lambda_conf * conf_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 统计
        total += targets.size(0)
        total_task_loss += task_loss.item()
        total_confuse_loss += conf_loss.item()
        total_loss_sum += total_loss.item()
        
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            "task": f"{task_loss.item():.4f}",
            "conf": f"{conf_loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}"
        })
    
    avg_task_loss = total_task_loss / len(train_loader)
    avg_confuse_loss = total_confuse_loss / len(train_loader)
    avg_total_loss = total_loss_sum / len(train_loader)
    avg_acc = 100.0 * correct / total
    
    return avg_task_loss, avg_confuse_loss, avg_total_loss, avg_acc


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
    config_path = r'D:\Model IP Protection\locked\wideresnet50-2-MNIST\config\config.yaml'
    config = load_config(config_path)
    
    print("="*60)
    print("Training Confuse Model (WideResNet-50-2 + MNIST)")
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
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ============================================================
    # 4. 加载base模型
    # ============================================================
    base_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['base_model']
    )
    
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(
            f"Base model not found: {base_model_path}\n"
            f"Please run train_base.py first!"
        )
    
    print(f"\nLoading base model from: {base_model_path}")
    model = load_model(
        base_model_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    model = model.to(device)
    print("Base model loaded successfully!")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    
    # ============================================================
    # 5. 初始化混淆机制
    # ============================================================
    print("\nInitializing confuse mechanism...")
    
    # 获取接口配置
    interface_config = get_interface_config(config['model']['arch'], model=model)
    
    # 验证接口配置
    validate_interface_config(model, interface_config)
    
    # 初始化BlockManager
    block_manager = BlockManager(model, interface_config)
    
    # 初始化KeyManager
    key_manager = KeyManager(config['confuse']['key'])
    
    # 初始化ConfuseLoss
    confuse_loss = ConfuseLoss(
        block_manager=block_manager,
        key_manager=key_manager,
        alpha_hi=config['confuse']['alpha_hi'],
        alpha_lo=config['confuse']['alpha_lo'],
        rho=config['confuse']['rho'],
        include_phi4=True,  # 包含φ4特征
        device=device
    )
    
    # ============================================================
    # 6. 定义损失函数和优化器
    # ============================================================
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr_confuse'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config['training']['lr_schedule_confuse']['milestones'],
        gamma=config['training']['lr_schedule_confuse']['gamma']
    )
    
    # ============================================================
    # 7. 创建日志记录器
    # ============================================================
    logger = Logger(
        log_dir=config['paths']['log_dir'],
        checkpoint_dir=config['paths']['checkpoint_dir'],
        file_suffix='(confuse)'
    )
    logger.set_config(config)
    
    # ============================================================
    # 8. 评估base模型的混淆指标（作为基准）
    # ============================================================
    print("\n" + "="*60)
    print("Evaluating base model metrics (before confuse training)...")
    print("="*60)
    
    with torch.no_grad():
        base_metrics = evaluate_confuse_metrics(
            block_manager, key_manager,
            rho=config['confuse']['rho'],
            device=device,
            include_phi4=True
        )
    
    # 存储base指标
    logger.store_base_metrics(base_metrics)
    
    # 打印base指标
    for layer_name in sorted(base_metrics.keys()):
        print(f"\n{layer_name}:")
        print(f"  Raw metrics:")
        for k, v in base_metrics[layer_name]['raw'].items():
            print(f"    {k}: {v:.4f}")
        print(f"  Normalized metrics:")
        for k, v in base_metrics[layer_name]['normalized'].items():
            print(f"    {k}: {v:.4f}")
    
    # ============================================================
    # 9. 训练循环
    # ============================================================
    num_epochs = config['training']['epochs_confuse']
    lambda_conf = config['confuse']['lambda']
    
    print("\n" + "="*60)
    print(f"Starting confuse training for {num_epochs} epochs...")
    print("="*60)
    print(f"Optimizer: Adam")
    print(f"Lambda (confuse weight): {lambda_conf}")
    print(f"Alpha_hi: {config['confuse']['alpha_hi']}")
    print(f"Alpha_lo: {config['confuse']['alpha_lo']}")
    print(f"Rho (selection ratio): {config['confuse']['rho']}")
    print(f"Metrics interval: every {METRICS_INTERVAL} epochs")
    print(f"Initial learning rate: {config['training']['lr_confuse']}")
    print(f"LR milestones: {config['training']['lr_schedule_confuse']['milestones']}\n")
    
    for epoch in range(1, num_epochs + 1):
        # ---- 训练阶段 ----
        task_loss, conf_loss, total_loss, train_acc = train_epoch(
            model, train_loader, criterion, confuse_loss, optimizer,
            lambda_conf, device, epoch, num_epochs
        )
        
        # ---- 学习率更新 ----
        scheduler.step()
        
        # ---- 测试 ----
        test_loss, test_acc = test_epoch(
            model, test_loader, criterion, device, epoch, num_epochs
        )
        
        # ---- 记录基本指标 ----
        metrics = {
            'train_task_loss': task_loss,
            'train_confuse_loss': conf_loss,
            'train_total_loss': total_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
        }
        
        # ---- 评估混淆指标（每METRICS_INTERVAL轮） ----
        if epoch % METRICS_INTERVAL == 0 or epoch == num_epochs:
            print(f"\n[Epoch {epoch}] Evaluating confuse metrics...")
            with torch.no_grad():
                current_metrics = evaluate_confuse_metrics(
                    block_manager, key_manager,
                    rho=config['confuse']['rho'],
                    device=device,
                    include_phi4=True
                )
            
            # 存储当前epoch的指标快照
            logger.store_metrics_snapshot(epoch, current_metrics)
            
            # 打印混淆指标
            for layer_name in sorted(current_metrics.keys()):
                print(f"\n  {layer_name}:")
                print(f"    Normalized metrics:")
                for k, v in current_metrics[layer_name]['normalized'].items():
                    print(f"      {k}: {v:.4f}")
        
        # ---- 更新日志 ----
        logger.update(epoch, metrics)
        logger.append_history_csv(epoch, metrics)
        
        # ---- 打印本轮结果 ----
        logger.log(
            f"Epoch {epoch}: "
            f"task_loss={task_loss:.4f}, conf_loss={conf_loss:.4f}, "
            f"train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%"
        )
        
        # ---- 保存最佳模型 ----
        logger.save_best_model(
            model, epoch, test_acc,
            config['paths']['confuse_model']
        )
    
    # ============================================================
    # 10. 保存汇总和总结
    # ============================================================
    logger.save_best_results_csv(model_type='confuse')
    logger.summary()
    
    print("\n" + "="*60)
    print("Confuse model training completed!")
    print(f"Best model saved to: {config['paths']['checkpoint_dir']}/{config['paths']['confuse_model']}")
    print(f"Best test accuracy: {logger.best_acc:.2f}% at epoch {logger.best_epoch}")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run scripts/lock.py to lock the model")
    print("  2. Run scripts/assemble.py to test recovery")
    print("  3. Run scripts/test.py to verify lock/assemble mechanism")


if __name__ == '__main__':
    main()