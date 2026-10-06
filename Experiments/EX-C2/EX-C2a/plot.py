"""
EX-C2a — Overall Similarity vs. Epoch (single figure)
CIFAR-10 / CIFAR-100 / ImageNet in one plot.
CIFAR-10 & CIFAR-100 only have data through epoch 5 → dashed extension to epoch 10.

Output: EX-C2a-overall_similarity.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ============================================================
# Config
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, 'overall_similarity.csv')
OUTPUT    = os.path.join(SCRIPT_DIR, 'EX-C2a-overall_similarity.pdf')

CURVES = [
    {
        'col':    'CIFAR-10_Overall_Similarity',
        'label':  'CIFAR-10',
        'color':  '#eb9794',
        'has_gap': True,   # 6–10 无数据，用虚线延伸
    },
    {
        'col':    'CIFAR-100_Overall_Similarity',
        'label':  'CIFAR-100',
        'color':  '#999dcb',
        'has_gap': True,
    },
    {
        'col':    'ImageNet_Overall_Similarity',
        'label':  'ImageNet',
        'color':  '#7dbfa5',
        'has_gap': False,  # 全程有数据
    },
]

# Font sizes
AXIS_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 13

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
    'lines.linewidth': 1.5,
})

# ============================================================
# Draw
# ============================================================
def main():
    df = pd.read_csv(CSV_PATH)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    for c in CURVES:
        col = c['col']
        sub = df[['Epoch', col]].dropna()

        if c['has_gap'] and len(sub) > 0:
            # 有数据的段：实线
            ax.plot(sub['Epoch'], sub[col],
                    color=c['color'], label=c['label'],
                    linestyle='-', linewidth=1.5, zorder=3)

            # 末点 → epoch 10 虚线延伸
            last_e = int(sub['Epoch'].iloc[-1])
            last_v = sub[col].iloc[-1]
            ax.plot([last_e, 10], [last_v, last_v],
                    color=c['color'], linestyle='--', linewidth=1.2, zorder=2)
        else:
            ax.plot(sub['Epoch'], sub[col],
                    color=c['color'], label=c['label'],
                    linestyle='-', linewidth=1.5, zorder=3)

    ax.set_xlabel('Epoch', fontweight='bold', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Overall Similarity', fontweight='bold', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.55, 1.03)
    ax.set_xticks(np.arange(0, 11, 1))

    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    legend = ax.legend(
        frameon=True, fontsize=LEGEND_FONTSIZE, edgecolor='#333333',
        framealpha=0.95, loc='lower right',
        handlelength=4.0, borderpad=1.0, labelspacing=0.8,
    )
    legend.get_frame().set_linewidth(0.6)

    fig.savefig(OUTPUT, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == '__main__':
    main()
