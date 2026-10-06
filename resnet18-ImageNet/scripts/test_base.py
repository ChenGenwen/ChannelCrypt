"""
仅评估 base 模型的混淆指标（不训练），控制台输出对比表。
"""

import os
import sys
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import load_model, get_interface_config
from core.blocks import BlockManager
from core.crypto import KeyManager
from utils.metrics import ConfuseMetrics

ADD_PHI4 = False


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def evaluate_confuse_metrics(block_manager, key_manager, device, rho):
    """仅原始特征，不做归一化"""
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

        raw_sim = ConfuseMetrics.compute_intra_similarity(raw_list, S)
        raw_homo = ConfuseMetrics.compute_layer_homogeneity(raw_list)

        metrics_summary[layer_name] = {
            'homogeneity': raw_homo,
            **raw_sim,
        }
    return metrics_summary


def print_metrics_table(metrics):
    """单列输出（无对比基准）"""
    metric_order = [
        'homogeneity',
        'overall_similarity',
        'selected_similarity',
        'unselected_similarity',
        'cross_similarity',
    ]
    col_metric = 28
    col_val = 18

    header = f"{'Metric':<{col_metric}}{'Value':>{col_val}}"
    sep = '-' * len(header)

    layer_names = sorted(metrics.keys())
    for layer_name in layer_names:
        print(f"\n  {layer_name}:")
        print(f"    {sep}")
        print(f"    {header}")
        print(f"    {sep}")
        for m in metric_order:
            v = metrics[layer_name].get(m, float('nan'))
            print(f"    {m:<{col_metric}}{v:>{col_val}.4f}")
        print(f"    {sep}")


def main():
    config = load_config('config/config.yaml')
    device = torch.device('cuda' if config['device']['cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoint_path = r"D:\Model IP Protection\ChannelCrypt\resnet18-ImageNet\checkpoints\base_resnet18_imagenet.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading base model from: {checkpoint_path}")
    model = load_model(
        checkpoint_path,
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        device=device
    )
    print("Base model loaded successfully!")

    print("\nInitializing components...")
    interface_config = get_interface_config(config['model']['arch'])
    block_manager = BlockManager(model, interface_config)
    key_manager = KeyManager(config['confuse']['key'])

    rho = config['confuse']['rho']
    print(f"\n{'=' * 60}")
    print("Evaluating base model confuse metrics...")
    print(f"{'=' * 60}")

    with torch.no_grad():
        metrics = evaluate_confuse_metrics(block_manager, key_manager, device, rho)

    print_metrics_table(metrics)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
