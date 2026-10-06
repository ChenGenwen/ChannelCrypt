"""
批量执行所有非 ImageNet 实验组的 confuse 训练，仅记录训练耗时（hours）。

机制：本脚本直接覆写各组 config.yaml 的 data_dir + epochs_confuse 后再调
      train_confuse.py，运行完毕自动还原。各组脚本/配置无需任何修改。

超算部署：只挪这一个脚本到 ChannelCrypt/ 根目录，改下面的 DATA_DIR 即可。

输出: confuse_training_time.csv（与本脚本同目录），两列：实验组, 训练时间(h)
"""

import os
import sys
import re
import subprocess
import time
import csv

# ============================================================
# Config
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ 超算部署时只需改这一行 ★
DATA_DIR = '/seu_share2/home/huangjie/230258664/Model_IP_Protection/data'

FIXED_EPOCHS = 5  # 每组统一只跑 5 轮

GROUPS = [
    'resnet18-cifar-10',
    'resnet18-cifar-100',
    'resnet18-MNIST',
    'vgg16-BN-cifar-10',
    'vgg16-BN-cifar-100',
    'vgg16-BN-MNIST',
    'GoogLeNet-cifar-10',
    'GoogLeNet-cifar-100',
    'GoogLeNet-MNIST',
    'wideresnet50-2-cifar-10',
    'wideresnet50-2-cifar-100',
    'wideresnet50-2-MNIST',
]

OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'confuse_training_time.csv')


def patch_config(config_path: str, data_dir: str, epochs: int):
    """覆写 config.yaml 的 data_dir 和 epochs_confuse，返回原始内容"""
    with open(config_path, 'r', encoding='utf-8') as f:
        original = f.read()
    patched = re.sub(r'^(\s*data_dir:\s*).*', rf'\g<1>{data_dir}', original, flags=re.MULTILINE)
    patched = re.sub(r'^(\s*epochs_confuse:\s*).*', rf'\g<1>{epochs}', patched, flags=re.MULTILINE)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(patched)
    return original


def restore_config(config_path: str, original: str):
    """还原 config.yaml"""
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(original)


def run_group(group_name: str) -> tuple:
    """
    执行单个组的 confuse 训练。
    返回 (time_hours, ok: bool, error_msg)
    """
    group_dir = os.path.join(SCRIPT_DIR, group_name)
    script_path = os.path.join(group_dir, 'scripts', 'train_confuse.py')
    config_path = os.path.join(group_dir, 'config', 'config.yaml')

    if not os.path.isfile(script_path):
        return None, False, 'script not found'

    print(f"\n{'#' * 60}")
    print(f"  Group: {group_name}")
    print(f"{'#' * 60}")

    # 注入 data_dir + epochs
    original_config = patch_config(config_path, DATA_DIR, FIXED_EPOCHS)

    start = time.time()
    ok = True
    error = None

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=group_dir,
            capture_output=True,
            text=True,
            timeout=86400,
        )
        if result.returncode != 0:
            ok = False
            error = result.stderr[-500:] if result.stderr else 'unknown error'
            print(result.stderr[-1000:])

    except subprocess.TimeoutExpired:
        ok = False
        error = 'exceeded 24h'
    except Exception as e:
        ok = False
        error = str(e)
    finally:
        restore_config(config_path, original_config)

    elapsed = (time.time() - start) / 3600.0
    return round(elapsed, 2), ok, error


def main():
    print("=" * 60)
    print("  Batch Confuse Training — Timing Benchmark")
    print("=" * 60)
    print(f"DATA_DIR:     {DATA_DIR}")
    print(f"FIXED_EPOCHS: {FIXED_EPOCHS}")
    print(f"Groups:       {len(GROUPS)}")
    print(f"Output:       {OUTPUT_CSV}\n")

    f = open(OUTPUT_CSV, 'w', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow(['实验组', '训练时间(h)'])

    for i, group in enumerate(GROUPS, 1):
        print(f"\n[{i}/{len(GROUPS)}] {group}")
        time_hours, ok, error = run_group(group)

        if time_hours is not None:
            w.writerow([group, time_hours])
            f.flush()
            print(f"  => {time_hours} h")
        if not ok:
            print(f"  [FAIL] {error}")

    f.close()

    print(f"\n{'=' * 60}")
    print(f"  Done → {OUTPUT_CSV}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
