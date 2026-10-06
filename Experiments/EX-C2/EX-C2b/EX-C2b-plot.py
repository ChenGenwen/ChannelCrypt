"""
EX-C2b — Recovery Ratio Experiment
Plot test accuracy vs. epoch for 4 recovery ratios (0%, 30%, 60%, 100%).

Inputs:
  EX-C2b1/EX-C2b1-resnet18-cifar10.csv
  EX-C2b2/EX-C2b2-resnet18-cifar100.csv

Outputs:
  EX-C2b1/EX-C2b1-resnet18-cifar10.pdf
  EX-C2b2/EX-C2b2-resnet18-cifar100.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

# ============================================================
# Config
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    {
        'csv':    os.path.join(BASE_DIR, 'EX-C2b1',
                    'EX-C2b1-resnet18-cifar10.csv'),
        'output': os.path.join(BASE_DIR, 'EX-C2b1',
                    'EX-C2b1-resnet18-cifar10.pdf'),
        'ylabel': 'Accuracy (%)',
        'ylim':   (0, 52),
    },
    {
        'csv':    os.path.join(BASE_DIR, 'EX-C2b2',
                    'EX-C2b2-resnet18-cifar100.csv'),
        'output': os.path.join(BASE_DIR, 'EX-C2b2',
                    'EX-C2b2-resnet18-cifar100.pdf'),
        'ylabel': 'Accuracy (%)',
        'ylim':   (0, 62),
    },
]

# 4 curves
COLS   = ['0%', '30%', '60%', '100%']
LABELS = ['0%', '30%', '60%', '100%']
COLORS = ['#eb9794', '#999dcb', '#7dbfa5', '#e8b84b']

# Sampling
EPOCH_STEP = 5  # plot every N epochs

# Font sizes
AXIS_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 10

# ============================================================
# Global style — Times New Roman
# ============================================================
plt.rcParams.update({
    'font.family': 'Times New Roman',
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
    df = pd.read_csv(task['csv'])
    df = df[(df['epoch'] % EPOCH_STEP == 0) | (df['epoch'] == df['epoch'].min())]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    for col, color, label in zip(COLS, COLORS, LABELS):
        ax.plot(
            df['epoch'], df[col],
            color=color,
            label=label,
            linewidth=1.2,
            zorder=3,
        )

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
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        edgecolor='#333333',
        framealpha=0.95,
        loc='upper left',
        ncol=1,
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
