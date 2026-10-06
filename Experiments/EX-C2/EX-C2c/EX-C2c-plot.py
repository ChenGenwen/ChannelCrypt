"""
EX-C2c — Conservative Attack Experiment (Training Data Ratio)
Plot test accuracy vs. epoch for 9 training data ratios (10%–90%).

Inputs:
  EX-C2c1/EX-C2c1-resnet18-cifar10-fixed_shards_repermutation(50%).csv
  EX-C2c2/EX-C2c2-resnet18-cifar100-fixed_shards_repermutation(50%).csv

Outputs:
  EX-C2c1/EX-C2c1-resnet18-cifar10.pdf
  EX-C2c2/EX-C2c2-resnet18-cifar100.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# Config
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    {
        'csv':    os.path.join(SCRIPT_DIR, 'EX-C2c1',
                    'EX-C2c1-resnet18-cifar10-fixed_shards_repermutation(50%).csv'),
        'output': os.path.join(SCRIPT_DIR, 'EX-C2c1',
                    'EX-C2c1-resnet18-cifar10.pdf'),
        'ylabel': 'Accuracy (%)',
        'ylim':   (0, 82),
    },
    {
        'csv':    os.path.join(SCRIPT_DIR, 'EX-C2c2',
                    'EX-C2c2-resnet18-cifar100-fixed_shards_repermutation(50%).csv'),
        'output': os.path.join(SCRIPT_DIR, 'EX-C2c2',
                    'EX-C2c2-resnet18-cifar100.pdf'),
        'ylabel': 'Accuracy (%)',
        'ylim':   (0, 28),
    },
]

COLS   = ['Acc(10%)', 'Acc(20%)', 'Acc(30%)', 'Acc(40%)', 'Acc(50%)',
          'Acc(60%)', 'Acc(70%)', 'Acc(80%)', 'Acc(90%)']
LABELS = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
COLORS = ['#eb9794', '#f4a460', '#fcd5ae', '#7dbfa5', '#999dcb',
          '#6699cc', '#e891b5', '#8cc4a0', '#d4a6c8']

# Font sizes
AXIS_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 10

# ============================================================
# Global style — Times New Roman
# ============================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 9,
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
    df = pd.read_csv(task['csv'])

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    for col, color, label in zip(COLS, COLORS, LABELS):
        ax.plot(df['epoch'], df[col], color=color, label=label,
                linewidth=1.2, zorder=3)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(task['ylabel'], fontweight='bold', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0, 150)
    ax.set_ylim(*task['ylim'])
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])

    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    legend = ax.legend(
        frameon=True, fontsize=LEGEND_FONTSIZE, edgecolor='#333333',
        framealpha=0.95, loc='upper left', ncol=3,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.savefig(task['output'], bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {task['output']}")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    for task in TASKS:
        draw_figure(task)
    print("Done.")
