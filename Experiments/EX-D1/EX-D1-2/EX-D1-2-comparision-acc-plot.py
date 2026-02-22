"""
EX-D1-2 (Journal Single-Axis Version)
Accuracy comparison across datasets and methods.

Outputs:
  EX-D1-2-acc-resnet18.pdf
  EX-D1-2-acc-googlenet.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ============================================================
# Data
# ============================================================
DATASETS = ['MNIST', 'CIFAR-10', 'CIFAR-100', 'ImageNet']
METHODS  = ['Base', 'ChannelCrypt', 'EncryIP']

DATA = {
    'ResNet18': [
        [99.76, 99.74, 99.30],
        [95.09, 95.01, 90.94],
        [77.77, 75.97, 72.79],
        [89.10, 88.73, 80.96],
    ],
    'GoogLeNet': [
        [99.73, 99.74, 98.79],
        [94.57, 94.17, 85.08],
        [74.93, 75.09, 61.65],
        [89.86, 89.45, 85.86],
    ],
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Style
# ============================================================

COLORS = ['#eb9794', '#999dcb', '#fcd5ae']
EDGE_CLR = '#333333'
BAR_WIDTH = 0.22

GLOBAL_Y_RANGE = (60, 100)

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
})


# ============================================================
# Draw function
# ============================================================
def draw_figure(model_name, output_path):

    values = np.array(DATA[model_name])  # shape (datasets, methods)

    n_datasets = len(DATASETS)
    n_methods  = len(METHODS)

    x = np.arange(n_datasets)
    offsets = (np.arange(n_methods) - (n_methods - 1)/2) * BAR_WIDTH

    fig, ax = plt.subplots(figsize=(6.6, 3.2))

    y_lo, y_hi = GLOBAL_Y_RANGE
    span = y_hi - y_lo

    for i in range(n_methods):
        ax.bar(
            x + offsets[i],
            values[:, i],
            width=BAR_WIDTH,
            color=COLORS[i],
            edgecolor=EDGE_CLR,
            linewidth=0.4,
            label=METHODS[i],
            zorder=3
        )

        # value labels
        for j in range(n_datasets):
            v = values[j, i]
            ax.text(
                x[j] + offsets[i],
                v + span*0.008,
                f'{v:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                rotation=90,
                fontweight='bold'
            )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_ylim(y_lo, y_hi)

    ax.grid(axis='y')
    ax.set_axisbelow(True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.legend(
        frameon=True,
        fontsize=8,
        edgecolor='#cccccc',
        framealpha=0.95,
        loc='upper center',
        ncol=3,
        bbox_to_anchor=(0.5, 1.18)
    )

    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print('Saved:', output_path)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':

    draw_figure(
        'ResNet18',
        os.path.join(OUTPUT_DIR, 'EX-D1-2-acc-resnet18.pdf')
    )

    draw_figure(
        'GoogLeNet',
        os.path.join(OUTPUT_DIR, 'EX-D1-2-acc-googlenet.pdf')
    )

    print('Done.')