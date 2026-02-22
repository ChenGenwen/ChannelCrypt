"""
EX-C2b — Conservative Attack Experiment
Plot test accuracy vs. epoch for 4 recovery ratios.

Inputs:
  EX-C2b1-resnet18-cifar10-fixed_dataset_10__.csv
  EX-C2b2-resnet18-cifar100-fixed_dataset_10__.csv

Outputs:
  EX-C2b1-resnet18-cifar10.pdf
  EX-C2b2-resnet18-cifar100.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# Config
# ============================================================
INPUT_DIR  = r"D:\Model IP Protection\locking model scheme\Experiments"
OUTPUT_DIR = INPUT_DIR

TASKS = [
    {
        'csv':    'EX-C2b1-resnet18-cifar10-fixed_dataset(10%).csv',
        'output': 'EX-C2b1-resnet18-cifar10.pdf',
        'title':  'ResNet18 on CIFAR-10',
        'ylabel': 'Test Accuracy (%)',
        'ylim':   (5, 40),
    },
    {
        'csv':    'EX-C2b2-resnet18-cifar100-fixed_dataset(10%).csv',
        'output': 'EX-C2b2-resnet18-cifar100.pdf',
        'title':  'ResNet18 on CIFAR-100',
        'ylabel': 'Test Accuracy (%)',
        'ylim':   (0, 60),
    },
]

# 4条曲线的颜色与标签（对应 acc(0%), acc(30%), acc(60%), acc(100%)）
COLORS = ['#eb9794', '#999dcb', '#7dbfa5', '#f4a460']
LABELS = ['0%', '30%', '60%', '100%']
COLS   = ['acc(0%)', 'acc(30%)', 'acc(60%)', 'acc(100%)']

# ============================================================
# Global style（与项目其他图保持一致）
# ============================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#333333',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.color': '#E5E5E5',
    'grid.linestyle': '-',
    'grid.linewidth': 0.4,
    'pdf.fonttype': 42,
    'lines.linewidth': 1.2,
})

# ============================================================
# Draw
# ============================================================
def draw_figure(task):
    csv_path = os.path.join(INPUT_DIR, task['csv'])
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    for col, color, label in zip(COLS, COLORS, LABELS):
        ax.plot(
            df['epoch'], df[col],
            color=color,
            label=label,
            linewidth=1.2,
            zorder=3,
        )

    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel(task['ylabel'], fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(*task['ylim'])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90, 100])

    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        frameon=True,
        fontsize=8,
        edgecolor='#cccccc',
        framealpha=0.95,
        loc='upper left',
        ncol=1,
    )

    out_path = os.path.join(OUTPUT_DIR, task['output'])    
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    for task in TASKS:
        draw_figure(task)
    print("Done.")