"""
训练Confuse模型
在base模型基础上添加同质化损失进行微调
每隔TEST_INTERVAL轮在测试集上验证，自动保存最佳模型（纯 test_acc）
每隔METRICS_INTERVAL轮评估混淆指标并输出对比表
training_history(confuse).csv 实时写入
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

from models import load_model, get_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager
from core.loss import ConfuseLoss
from utils.logger import Logger
from utils.metrics import ConfuseMetrics

# ============================================================
# 全局参数
# ============================================================
TEST_INTERVAL = 5       # 每隔5轮测试一次
METRICS_INTERVAL = 5  # 每隔5轮评估混淆指标一次
ADD_PHI4 = False         # 损失和评估中是否包含φ4谱峰值占比特征


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_data_loaders(config: dict):
    """
    创建MNIST数据加载器
    
    Returns:
        train_loader, test_loader
    """
    mean = config['dataset']['mean']
    std = config['dataset']['std']
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
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
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, confuse_loss, optimizer,
                device, lambda_conf, epoch, num_epochs):
    """
    训练一个epoch(带同质化损失, tqdm进度条)

    Returns:
        avg_task_loss, avg_confuse_loss, avg_total_loss, avg_acc
    """
    model.train()

    total_task_loss = 0.0
    total_confuse_loss = 0.0
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

        # 任务损失
        outputs = model(inputs)
        loss_task = criterion(outputs, targets)

        # 同质化损失
        loss_conf = confuse_loss()

        # 总损失
        loss = loss_task + lambda_conf * loss_conf

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_task_loss += loss_task.item()
        total_confuse_loss += loss_conf.item()
        total_loss += loss.item()

        total += targets.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({
            "task": f"{loss_task.item():.4f}",
            "conf": f"{loss_conf.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.5f}"
        })

    avg_task_loss = total_task_loss / len(train_loader)
    avg_confuse_loss = total_confuse_loss / len(train_loader)
    avg_total_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * correct / total

    return avg_task_loss, avg_confuse_loss, avg_total_loss, avg_acc


def test_epoch(model, test_loader, criterion, device, epoch, num_epochs):
    """
    测试一个epoch (tqdm进度条)

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

            total_loss += loss.item()
            total += targets.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })

    avg_loss = total_loss / len(test_loader)
    avg_acc = 100.0 * correct / total

    return avg_loss, avg_acc


def evaluate_confuse_metrics(block_manager, key_manager, device, rho):
    """
    评估混淆效果指标
    对每层分别用原始特征和归一化特征各算一组指标

    Returns:
        dict: {layer_name: {'raw': {...}, 'normalized': {...}}}
    """
    from core.features import compute_layer_phi_list, normalize_phi

    metrics_summary = {}

    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        g = info['g']

        # 拿到原始特征矩阵
        phi_raw = compute_layer_phi_list(
            block_manager, key_manager, layer_name, device,
            include_phi4=ADD_PHI4
        ).detach()  # [g, phi_dim], 评估不需要梯度

        # 归一化版本
        phi_norm = normalize_phi(phi_raw)  # [g, phi_dim]

        # 生成选块集合
        S = key_manager.generate_selection(layer_name, g, rho)

        # 拆成 list 送入指标函数
        raw_list  = [phi_raw[i]  for i in range(g)]
        norm_list = [phi_norm[i] for i in range(g)]

        # 原始特征的指标
        raw_sim  = ConfuseMetrics.compute_intra_similarity(raw_list, S)
        raw_homo = ConfuseMetrics.compute_layer_homogeneity(raw_list)

        # 归一化特征的指标
        norm_sim  = ConfuseMetrics.compute_intra_similarity(norm_list, S)
        norm_homo = ConfuseMetrics.compute_layer_homogeneity(norm_list)

        metrics_summary[layer_name] = {
            'raw': {
                'homogeneity': raw_homo,
                **raw_sim,
            },
            'normalized': {
                'homogeneity': norm_homo,
                **norm_sim,
            },
        }

    return metrics_summary


def print_metrics_table(base_metrics, current_metrics):
    """
    按层输出指标对比表（5 列）：
        Metric | Base(origin) | Current(origin) | Base(normalized) | Current(normalized)

    每行指标均匀填充 base 和 current 的 raw / normalized 值。

    Args:
        base_metrics: base 模型基准，与 evaluate_confuse_metrics 同结构
                      {layer: {'raw': {...}, 'normalized': {...}}}
        current_metrics: 当前 epoch 的评估结果，同结构
    """
    metric_order = [
        'homogeneity',
        'overall_similarity',
        'selected_similarity',
        'unselected_similarity',
        'cross_similarity',
    ]
    # 列宽
    col_metric = 28
    col_val    = 18

    header = (
        f"{'Metric':<{col_metric}}"
        f"{'Base(origin)':>{col_val}}"
        f"{'Current(origin)':>{col_val}}"
        f"{'Base(normalized)':>{col_val}}"
        f"{'Current(normalized)':>{col_val}}"
    )
    sep = '-' * len(header)

    layer_names = sorted(current_metrics.keys())

    for layer_name in layer_names:
        print(f"\n  {layer_name}:")
        print(f"    {sep}")
        print(f"    {header}")
        print(f"    {sep}")

        base_layer   = base_metrics.get(layer_name, {})
        base_raw     = base_layer.get('raw', {})
        base_norm    = base_layer.get('normalized', {})
        cur_raw      = current_metrics[layer_name].get('raw', {})
        cur_norm     = current_metrics[layer_name].get('normalized', {})

        for m in metric_order:
            v_base_origin = base_raw.get(m, float('nan'))
            v_cur_origin  = cur_raw.get(m, float('nan'))
            v_base_norm   = base_norm.get(m, float('nan'))
            v_cur_norm    = cur_norm.get(m, float('nan'))

            print(
                f"    {m:<{col_metric}}"
                f"{v_base_origin:>{col_val}.4f}"
                f"{v_cur_origin:>{col_val}.4f}"
                f"{v_base_norm:>{col_val}.4f}"
                f"{v_cur_norm:>{col_val}.4f}"
            )

        print(f"    {sep}")


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = r'D:\Model IP Protection\locked\vgg16-BN-MNIST\config\config.yaml'
    config = load_config(config_path)

    print("=" * 60)
    print("Training Confuse Model (with Homogenization Loss)")
    print("=" * 60)

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
    # 3. 加载base模型
    # ============================================================
    base_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['base_model']
    )

    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"Base model not found: {base_model_path}")

    print(f"\nLoading base model from: {base_model_path}")
    model = load_model(
        base_model_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    print("Base model loaded successfully!")

    # ============================================================
    # 4. 初始化混淆组件
    # ============================================================
    print("\nInitializing confuse components...")

    # 获取接口配置
    interface_config = get_interface_config(config['model']['arch'], model=model)

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
        include_phi4=ADD_PHI4,
        device=device
    )

    # ============================================================
    # 5. 创建数据加载器
    # ============================================================
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(config)

    # ============================================================
    # 6. 定义损失函数和优化器
    # ============================================================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['lr_confuse'],
        momentum=config['training']['momentum'],
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
    # 8. 采集 base 模型的混淆指标基准
    # ============================================================
    print("\nEvaluating base model metrics as baseline...")
    with torch.no_grad():
        base_metrics_summary = evaluate_confuse_metrics(
            block_manager, key_manager, device, config['confuse']['rho']
        )
    logger.store_base_metrics(base_metrics_summary)

    print("\n" + "=" * 60)
    print("Base Model Metrics (Baseline)")
    print("=" * 60)
    # 用 base 自身作为 base 列，current 也是 base（起始状态）
    print_metrics_table(base_metrics_summary, base_metrics_summary)
    print("=" * 60)

    # ============================================================
    # 9. 训练循环
    # ============================================================
    num_epochs = config['training']['epochs_confuse']
    lambda_conf = config['confuse']['lambda']

    logger.log(f"\nStarting confuse training for {num_epochs} epochs...")
    logger.log(f"Test interval: every {TEST_INTERVAL} epochs")
    logger.log(f"Metrics interval: every {METRICS_INTERVAL} epochs")
    logger.log(f"Lambda (confuse weight): {lambda_conf}")
    logger.log(f"Alpha_hi: {config['confuse']['alpha_hi']}, Alpha_lo: {config['confuse']['alpha_lo']}")
    logger.log(f"Rho (selection ratio): {config['confuse']['rho']}\n")

    for epoch in range(1, num_epochs + 1):
        # ---- 训练阶段（每轮都执行）----
        task_loss, conf_loss, total_loss, train_acc = train_epoch(
            model, train_loader, criterion, confuse_loss, optimizer,
            device, lambda_conf, epoch, num_epochs
        )

        # ---- 学习率更新（每轮都执行）----
        scheduler.step()

        # ---- 测试+记录+保存（每TEST_INTERVAL轮或最后一轮）----
        if epoch % TEST_INTERVAL == 0 or epoch == num_epochs:
            test_loss, test_acc = test_epoch(
                model, test_loader, criterion, device,
                epoch, num_epochs
            )

            # 写入 CSV 的字段：只保留 train_loss / train_acc / test_loss / test_acc
            metrics = {
                'train_loss': total_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            }
            logger.update(epoch, metrics)

            # 实时写入CSV
            logger.append_history_csv(epoch, metrics)

            # 打印本轮结果（控制台仍显示 task_loss 和 conf_loss）
            logger.log(
                f"Epoch {epoch}: "
                f"task_loss={task_loss:.4f}, conf_loss={conf_loss:.4f}, "
                f"train_acc={train_acc:.2f}%, test_acc={test_acc:.2f}%"
            )

            # 保存最佳模型（纯 test_acc）
            logger.save_best_model(
                model, epoch, test_acc,
                config['paths']['confuse_model']
            )

        # ---- 混淆指标评估（每METRICS_INTERVAL轮）----
        if epoch % METRICS_INTERVAL == 0:
            print(f"\n{'=' * 60}")
            print(f"Confuse Metrics at Epoch {epoch}")
            print(f"{'=' * 60}")

            with torch.no_grad():
                metrics_summary = evaluate_confuse_metrics(
                    block_manager, key_manager, device, config['confuse']['rho']
                )

            # 保存快照（用于最后写 best_results CSV）
            logger.store_metrics_snapshot(epoch, metrics_summary)

            # 控制台输出对比表
            print_metrics_table(base_metrics_summary, metrics_summary)

            print(f"{'=' * 60}\n")

    # ============================================================
    # 10. 保存汇总和总结
    # ============================================================
    # 如果 best_epoch 不在任何快照中，补采一次最终状态
    if logger.best_epoch not in logger.metrics_snapshots:
        with torch.no_grad():
            final_metrics = evaluate_confuse_metrics(
                block_manager, key_manager, device, config['confuse']['rho']
            )
        logger.store_metrics_snapshot(logger.best_epoch, final_metrics)

    logger.save_best_results_csv(model_type='confuse')
    logger.summary()

    print("\n" + "=" * 60)
    print("Confuse model training completed!")
    print(f"Best model saved to: {config['paths']['checkpoint_dir']}/{config['paths']['confuse_model']}")
    print(f"Best test accuracy: {logger.best_acc:.2f}% at epoch {logger.best_epoch}")
    print("=" * 60)


if __name__ == '__main__':
    main()