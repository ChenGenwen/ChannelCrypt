"""
训练Confuse模型（Metrics 记录模式，仅 origin）
在base模型基础上添加同质化损失进行微调
每轮评估混淆指标（仅原始特征），打印对比表，保存 JSON（不测试精度）
"""

import os
import sys
import json
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
from utils.metrics import ConfuseMetrics

# ============================================================
# 全局参数
# ============================================================
METRICS_INTERVAL = 1  # 每轮评估混淆指标一次
ADD_PHI4 = False       # 损失和评估中是否包含φ4谱峰值占比特征


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

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_dir = config['dataset']['data_dir']

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
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
    评估混淆效果指标（仅原始特征，不做归一化）

    Returns:
        dict: {layer_name: {'homogeneity': ..., 'overall_similarity': ...}}
    """
    from core.features import compute_layer_phi_list

    metrics_summary = {}

    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        g = info['g']

        phi_raw = compute_layer_phi_list(
            block_manager, key_manager, layer_name, device,
            include_phi4=ADD_PHI4
        ).detach()

        S = key_manager.generate_selection(layer_name, g, rho)
        raw_list = [phi_raw[i] for i in range(g)]

        raw_sim  = ConfuseMetrics.compute_intra_similarity(raw_list, S)
        raw_homo = ConfuseMetrics.compute_layer_homogeneity(raw_list)

        metrics_summary[layer_name] = {
            'homogeneity': raw_homo,
            **raw_sim,
        }

    return metrics_summary


def print_metrics_table(base_metrics, current_metrics):
    """
    按层输出指标对比表（3 列）：
        Metric | Base | Current

    Args:
        base_metrics: base 模型基准  {layer: {'homogeneity': ..., ...}}
        current_metrics: 当前 epoch   {layer: {'homogeneity': ..., ...}}
    """
    metric_order = [
        'homogeneity',
        'overall_similarity',
        'selected_similarity',
        'unselected_similarity',
        'cross_similarity',
    ]
    col_metric = 28
    col_val    = 18

    header = (
        f"{'Metric':<{col_metric}}"
        f"{'Base':>{col_val}}"
        f"{'Current':>{col_val}}"
    )
    sep = '-' * len(header)

    layer_names = sorted(current_metrics.keys())

    for layer_name in layer_names:
        print(f"\n  {layer_name}:")
        print(f"    {sep}")
        print(f"    {header}")
        print(f"    {sep}")

        base_layer = base_metrics.get(layer_name, {})
        cur_layer  = current_metrics[layer_name]

        for m in metric_order:
            v_base = base_layer.get(m, float('nan'))
            v_cur  = cur_layer.get(m, float('nan'))

            print(
                f"    {m:<{col_metric}}"
                f"{v_base:>{col_val}.4f}"
                f"{v_cur:>{col_val}.4f}"
            )

        print(f"    {sep}")


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = 'config/config.yaml'
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
    interface_config = get_interface_config(config['model']['arch'])

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
    train_loader, _ = get_data_loaders(config)

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

    # ============================================================
    # 7. 采集 base 模型的混淆指标基准
    # ============================================================
    print("\nEvaluating base model metrics as baseline...")
    with torch.no_grad():
        base_metrics_summary = evaluate_confuse_metrics(
            block_manager, key_manager, device, config['confuse']['rho']
        )

    print("\n" + "=" * 60)
    print("Base Model Metrics (Baseline)")
    print("=" * 60)
    print_metrics_table(base_metrics_summary, base_metrics_summary)
    print("=" * 60)

    # ============================================================
    # 8. 训练循环
    # ============================================================
    num_epochs = config['training']['epochs_confuse']
    lambda_conf = config['confuse']['lambda']

    print(f"\nStarting confuse training for {num_epochs} epochs...")
    print(f"Metrics interval: every {METRICS_INTERVAL} epoch(s)")
    print(f"Lambda (confuse weight): {lambda_conf}")
    print(f"Alpha_hi: {config['confuse']['alpha_hi']}, Alpha_lo: {config['confuse']['alpha_lo']}")
    print(f"Rho (selection ratio): {config['confuse']['rho']}\n")

    # 创建 metrics JSON 输出目录
    metrics_json_dir = os.path.join(
        config['paths']['log_dir'], 'metrics_per_epoch'
    )
    os.makedirs(metrics_json_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        # ---- 训练阶段（每轮都执行）----
        task_loss, conf_loss, _, train_acc = train_epoch(
            model, train_loader, criterion, confuse_loss, optimizer,
            device, lambda_conf, epoch, num_epochs
        )

        print(
            f"Epoch {epoch}: "
            f"task_loss={task_loss:.4f}, conf_loss={conf_loss:.4f}, "
            f"train_acc={train_acc:.2f}%"
        )

        # ---- 混淆指标评估 + 保存 JSON（每 METRICS_INTERVAL 轮）----
        if epoch % METRICS_INTERVAL == 0:
            print(f"\n{'=' * 60}")
            print(f"Confuse Metrics at Epoch {epoch}")
            print(f"{'=' * 60}")

            with torch.no_grad():
                metrics_summary = evaluate_confuse_metrics(
                    block_manager, key_manager, device, config['confuse']['rho']
                )

            # 控制台输出对比表
            print_metrics_table(base_metrics_summary, metrics_summary)

            # 保存当前 epoch 的 metrics 为 JSON
            json_path = os.path.join(
                metrics_json_dir, f"metrics_epoch_{epoch}.json"
            )
            serializable = {}
            for layer_name, data in metrics_summary.items():
                serializable[layer_name] = {k: float(v) for k, v in data.items()}
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            print(f"  Metrics saved to: {json_path}")

            print(f"{'=' * 60}\n")

    # ============================================================
    # 9. 保存最终模型
    # ============================================================
    confuse_model_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['confuse_model']
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'arch': config['model']['arch'],
        'num_classes': config['model']['num_classes'],
    }, confuse_model_path)

    print("\n" + "=" * 60)
    print("Confuse model training completed!")
    print(f"Model saved to: {confuse_model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()