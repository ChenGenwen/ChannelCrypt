"""
Test Base Model Metrics (Baseline)
===================================
加载 base_resnet18_imagenet.pth，评估混淆指标基线（raw + normalized），
仅控制台输出，不保存任何文件。
"""

import os
import sys
import yaml
import torch

# 添加项目根目录到路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models import load_model, get_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager
from core.features import compute_layer_phi_list, normalize_phi
from utils.metrics import ConfuseMetrics

# ============================================================
# Config
# ============================================================
ADD_PHI4 = False  # 是否包含 φ4 谱峰值占比特征

METRIC_ORDER = [
    'homogeneity',
    'overall_similarity',
    'selected_similarity',
    'unselected_similarity',
    'cross_similarity',
]


def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate_confuse_metrics(block_manager, key_manager, device, rho):
    """
    评估混淆效果指标（raw + normalized 两组）
    """
    metrics_summary = {}

    for layer_name in block_manager.get_layer_names():
        info = block_manager.get_layer_info(layer_name)
        g = info['g']

        # 原始特征矩阵
        phi_raw = compute_layer_phi_list(
            block_manager, key_manager, layer_name, device,
            include_phi4=ADD_PHI4
        ).detach()

        # 归一化版本
        phi_norm = normalize_phi(phi_raw)

        # 选块集合
        S = key_manager.generate_selection(layer_name, g, rho)

        # 拆成 list
        raw_list = [phi_raw[i] for i in range(g)]
        norm_list = [phi_norm[i] for i in range(g)]

        # 原始特征指标
        raw_sim = ConfuseMetrics.compute_intra_similarity(raw_list, S)
        raw_homo = ConfuseMetrics.compute_layer_homogeneity(raw_list)

        # 归一化特征指标
        norm_sim = ConfuseMetrics.compute_intra_similarity(norm_list, S)
        norm_homo = ConfuseMetrics.compute_layer_homogeneity(norm_list)

        metrics_summary[layer_name] = {
            'raw': {
                'homogeneity': raw_homo,
                'overall_similarity': raw_sim['overall_similarity'],
                'selected_similarity': raw_sim['selected_similarity'],
                'unselected_similarity': raw_sim['unselected_similarity'],
                'cross_similarity': raw_sim['cross_similarity'],
            },
            'normalized': {
                'homogeneity': norm_homo,
                'overall_similarity': norm_sim['overall_similarity'],
                'selected_similarity': norm_sim['selected_similarity'],
                'unselected_similarity': norm_sim['unselected_similarity'],
                'cross_similarity': norm_sim['cross_similarity'],
            },
        }

    return metrics_summary


def print_metrics_table(metrics):
    """打印单组指标（raw | normalized 并列）"""
    col_metric = 28
    col_val = 18

    header = (
        f"{'Metric':<{col_metric}}"
        f"{'Raw':>{col_val}}"
        f"{'Normalized':>{col_val}}"
    )
    sep = '-' * len(header)

    layer_names = sorted(metrics.keys())

    for layer_name in layer_names:
        print(f"\n  {layer_name}:")
        print(f"    {sep}")
        print(f"    {header}")
        print(f"    {sep}")

        raw_data = metrics[layer_name]['raw']
        norm_data = metrics[layer_name]['normalized']

        for m in METRIC_ORDER:
            v_raw = raw_data.get(m, float('nan'))
            v_norm = norm_data.get(m, float('nan'))
            print(
                f"    {m:<{col_metric}}"
                f"{v_raw:>{col_val}.4f}"
                f"{v_norm:>{col_val}.4f}"
            )

        print(f"    {sep}")


def main():
    # ============================================================
    # 1. 加载配置
    # ============================================================
    config_path = os.path.join(SCRIPT_DIR, 'config', 'config.yaml')
    config = load_config(config_path)

    # ============================================================
    # 2. 设备
    # ============================================================
    device = torch.device('cuda' if config['device']['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ============================================================
    # 3. 加载 base 模型
    # ============================================================
    base_model_path = os.path.join(
        SCRIPT_DIR,
        config['paths']['checkpoint_dir'].lstrip('./'),
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
    # 4. 初始化组件（只需 BlockManager + KeyManager）
    # ============================================================
    print("\nInitializing components...")
    interface_config = get_interface_config(config['model']['arch'])
    block_manager = BlockManager(model, interface_config)
    key_manager = KeyManager(config['confuse']['key'])
    rho = config['confuse']['rho']

    layer_names = block_manager.get_layer_names()
    print(f"Layers: {layer_names}")

    # ============================================================
    # 5. 评估
    # ============================================================
    print("\nEvaluating base model metrics...")
    with torch.no_grad():
        metrics = evaluate_confuse_metrics(block_manager, key_manager, device, rho)

    # ============================================================
    # 6. 输出
    # ============================================================
    print("\n" + "=" * 72)
    print("  Base Model Metrics — base_resnet18_imagenet.pth")
    print("=" * 72)
    print_metrics_table(metrics)
    print("\n" + "=" * 72)
    print("Done.")


if __name__ == '__main__':
    main()
