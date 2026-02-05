"""
测试 locked 和 assembled 模型的准确率
验证 Lock/Assemble 机制的有效性：
  - locked 模型应该准确率很低（保护生效）
  - assembled 模型应该准确率恢复（可逆性验证）

运行方式:
    python test.py

输出:
  - 控制台: 逐模型的测试进度 + 结果 + 分析（与原版样式一致）
  - logs/test_log.csv: test_acc(locked), test_acc(assembled)
"""

import os
import sys
import csv
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_test_loader(config: dict):
    """
    创建测试数据加载器

    Returns:
        test_loader
    """
    mean = config['dataset']['mean']
    std = config['dataset']['std']

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    data_dir = config['dataset']['data_dir']

    testset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


def test_model(model, test_loader, device):
    """
    测试模型准确率

    Returns:
        accuracy (%), loss
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    print("\nTesting on MNIST test set (10,000 samples)...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 显示进度
            if (batch_idx + 1) % 20 == 0:
                acc = 100.0 * correct / total
                print(f"  Batch [{batch_idx+1}/{len(test_loader)}] Acc: {acc:.2f}%")

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


def print_results(model_tag: str, accuracy: float, avg_loss: float):
    """
    按原版样式打印单个模型的结果横幅 + 分析块

    Args:
        model_tag: 'locked' 或 'assembled'
        accuracy: 测试准确率 (%)
        avg_loss: 平均损失
    """
    print("\n" + "=" * 60)
    print(f"{model_tag.upper()} Model Test Results")
    print("=" * 60)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print("=" * 60)

    print("\n📊 Expected Performance & Analysis:")
    if model_tag == 'locked':
        print("  Expected: ~10% (random guess level for 10 classes)")
        print("  Purpose: Verify that protection mechanism works")
        print("  Analysis: Without the correct key, the model is unusable")
        if accuracy > 30:
            print("  ⚠️  Warning: Accuracy too high! Protection may be ineffective.")
            print("     Check that lock.py was run correctly.")
        elif accuracy < 5:
            print("  ⚠️  Warning: Accuracy suspiciously low (below random).")
            print("     Model might be broken. Check lock.py implementation.")
        else:
            print("  ✅ Protection effective: model is unusable without key")

    elif model_tag == 'assembled':
        print("  Expected: ~99% (recovered, should match confuse model)")
        print("  Purpose: Verify that recovery mechanism works")
        print("  Analysis: With correct key, performance is restored")
        if accuracy < 95:
            print("  ⚠️  Warning: Recovery failed or incomplete.")
            print("     Check that:")
            print("     - Correct key is used in assemble.py")
            print("     - Lock and assemble operations are inverse")
        elif accuracy >= 98:
            print("  ✅ Recovery successful: model performance restored")

    print("=" * 60)


def write_test_log(config: dict, acc_locked: float, acc_assembled: float):
    """
    写入 logs/test_log.csv

    每次运行追加一行（首次写 header）。
    字段: timestamp, test_acc(locked), test_acc(assembled)
    """
    log_dir = config['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, 'test_log.csv')

    file_exists = os.path.exists(csv_path)

    fieldnames = ['timestamp', 'test_acc(locked)', 'test_acc(assembled)']
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_acc(locked)': f"{acc_locked:.2f}",
        'test_acc(assembled)': f"{acc_assembled:.2f}",
    }

    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nTest log written to: {csv_path}")


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = 'config/config.yaml'
    config = load_config(config_path)

    device = torch.device(
        'cuda' if config['device']['cuda'] and torch.cuda.is_available() else 'cpu'
    )
    print("=" * 60)
    print("Lock / Assemble Verification Test (MNIST)")
    print("=" * 60)
    print(f"Device: {device}")

    # ============================================================
    # 2. 加载测试数据（只加载一次，两个模型共用）
    # ============================================================
    print("\nLoading test data...")
    test_loader = get_test_loader(config)
    print(f"Test samples: {len(test_loader.dataset)}")

    # ============================================================
    # 3. 检查两个模型文件是否存在
    # ============================================================
    locked_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['locked_model']
    )
    assembled_path = os.path.join(
        config['paths']['checkpoint_dir'],
        config['paths']['assembled_model']
    )

    missing = []
    if not os.path.exists(locked_path):
        missing.append(('locked', locked_path))
    if not os.path.exists(assembled_path):
        missing.append(('assembled', assembled_path))

    if missing:
        print("\n❌ Error: Required model files not found:")
        for tag, path in missing:
            print(f"   {tag}: {path}")
        print("\nPlease ensure you have run the full pipeline:")
        print("  1. python scripts/train_base.py")
        print("  2. python scripts/train_confuse.py")
        print("  3. python scripts/lock.py")
        print("  4. python scripts/assemble.py")
        return

    # ============================================================
    # 4. 测试 locked 模型
    # ============================================================
    print("\n" + "=" * 60)
    print("Testing LOCKED Model")
    print("=" * 60)
    print(f"Loading model from: {locked_path}")

    locked_model = load_model(
        locked_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    num_params = sum(p.numel() for p in locked_model.parameters())
    print(f"Model: {config['model']['arch']}")
    print(f"Parameters: {num_params:,}")

    acc_locked, loss_locked = test_model(locked_model, test_loader, device)
    print_results('locked', acc_locked, loss_locked)

    # 释放显存
    del locked_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ============================================================
    # 5. 测试 assembled 模型
    # ============================================================
    print("\n" + "=" * 60)
    print("Testing ASSEMBLED Model")
    print("=" * 60)
    print(f"Loading model from: {assembled_path}")

    assembled_model = load_model(
        assembled_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    num_params = sum(p.numel() for p in assembled_model.parameters())
    print(f"Model: {config['model']['arch']}")
    print(f"Parameters: {num_params:,}")

    acc_assembled, loss_assembled = test_model(assembled_model, test_loader, device)
    print_results('assembled', acc_assembled, loss_assembled)

    del assembled_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ============================================================
    # 6. 写入 CSV 日志
    # ============================================================
    write_test_log(config, acc_locked, acc_assembled)


if __name__ == '__main__':
    main()