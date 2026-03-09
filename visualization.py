"""
AOC-IDS Training Visualization
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

PALETTE = {
    'blue':   '#1976D2',
    'green':  '#388E3C',
    'orange': '#F57C00',
    'red':    '#D32F2F',
    'cyan':   '#0097A7',
    'purple': '#7B1FA2',
}


def _setup_style():
    plt.rcParams.update({
        'figure.facecolor':    'white',
        'axes.facecolor':      '#FAFAFA',
        'axes.edgecolor':      '#CCCCCC',
        'axes.grid':           True,
        'grid.alpha':          0.3,
        'grid.linestyle':      '--',
        'grid.color':          '#CCCCCC',
        'font.size':           10,
        'axes.titlesize':      13,
        'axes.titleweight':    'bold',
        'axes.labelsize':      11,
        'legend.fontsize':     9,
        'legend.framealpha':   0.9,
        'figure.dpi':          150,
        'savefig.dpi':         150,
        'lines.linewidth':     2,
        'lines.markersize':    4,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
    })


def plot_training_summary(
    first_round_losses,
    online_losses,
    online_metrics,
    final_encoder,
    final_decoder,
    final_combined,
    y_test_true,
    y_test_pred,
    dataset,
    seed,
    save_dir='results',
):
    """
    Generate a comprehensive 3x2 training summary figure.

    Parameters
    ----------
    first_round_losses : list[float]
        Average loss per epoch during Stage-1 offline training.
    online_losses : list[float]
        Average loss per online step during Stage-2.
    online_metrics : dict[int, tuple]
        {step: (accuracy, precision, recall, f1)} per online step.
    final_encoder / final_decoder / final_combined : tuple
        (accuracy, precision, recall, f1) from final evaluation.
    y_test_true, y_test_pred : np.ndarray
        Ground-truth and predicted labels for the final test set.
    dataset : str
        Dataset name (e.g. 'nsl', 'unsw', 'cic').
    seed : int
        Random seed used.
    save_dir : str
        Output directory for the saved figure.
    """
    _setup_style()
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(
        3, 2, hspace=0.35, wspace=0.28,
        left=0.07, right=0.96, top=0.93, bottom=0.04,
    )
    fig.suptitle(
        f'AOC-IDS Training Report  \u00b7  {dataset.upper()}  \u00b7  seed={seed}',
        fontsize=16, fontweight='bold', y=0.97,
    )

    # ══════════════════════════════════════════════════════════════
    # (0, 0)  Stage-1  Offline Training Loss
    # ══════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0, 0])
    ep = range(1, len(first_round_losses) + 1)
    ax1.plot(ep, first_round_losses, color=PALETTE['blue'],
             marker='o', markersize=5, zorder=3)
    ax1.fill_between(ep, first_round_losses, alpha=0.08, color=PALETTE['blue'])
    ax1.set_title('Stage 1 \u2014 Offline Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CRC Loss')
    if first_round_losses:
        ax1.annotate(
            f'{first_round_losses[-1]:.4f}',
            xy=(len(first_round_losses), first_round_losses[-1]),
            xytext=(8, 8), textcoords='offset points',
            fontsize=9, color=PALETTE['blue'], fontweight='bold',
        )

    # ══════════════════════════════════════════════════════════════
    # (0, 1)  Stage-2  Online Training Loss
    # ══════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    steps = range(1, len(online_losses) + 1)
    ax2.plot(steps, online_losses, color=PALETTE['cyan'],
             alpha=0.45, linewidth=1, label='Raw', zorder=2)
    if len(online_losses) > 5:
        w = max(3, len(online_losses) // 15)
        smoothed = np.convolve(online_losses, np.ones(w) / w, mode='valid')
        offset = w // 2 + 1
        ax2.plot(range(offset, offset + len(smoothed)), smoothed,
                 color=PALETTE['red'], linewidth=2.2,
                 label='Smoothed', zorder=3)
    ax2.set_title('Stage 2 \u2014 Online Training Loss')
    ax2.set_xlabel('Online Step')
    ax2.set_ylabel('CRC Loss')
    ax2.legend(loc='upper right')

    # ══════════════════════════════════════════════════════════════
    # (1, :)  Online Metrics Evolution  (full width)
    # ══════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[1, :])
    if online_metrics:
        m_steps = sorted(online_metrics.keys())
        accs  = [online_metrics[s][0] for s in m_steps]
        precs = [online_metrics[s][1] for s in m_steps]
        recs  = [online_metrics[s][2] for s in m_steps]
        f1s   = [online_metrics[s][3] for s in m_steps]

        ax3.plot(m_steps, accs,  color=PALETTE['blue'],
                 marker='o', label='Accuracy')
        ax3.plot(m_steps, precs, color=PALETTE['green'],
                 marker='s', label='Precision')
        ax3.plot(m_steps, recs,  color=PALETTE['orange'],
                 marker='^', label='Recall')
        ax3.plot(m_steps, f1s,   color=PALETTE['red'],
                 marker='D', label='F1 Score')
        ax3.legend(loc='lower right', ncol=4)
    ax3.set_title(
        'Stage 2 \u2014 Batch-level Detection Metrics over Online Steps')
    ax3.set_xlabel('Online Step')
    ax3.set_ylabel('Score')
    ax3.set_ylim(-0.05, 1.1)

    # ══════════════════════════════════════════════════════════════
    # (2, 0)  Confusion Matrix Heatmap
    # ══════════════════════════════════════════════════════════════
    ax4 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_test_true, y_test_pred)
    im = ax4.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            color = 'white' if cm[r, c] > thresh else 'black'
            ax4.text(c, r, f'{cm[r, c]:,}',
                     ha='center', va='center', color=color,
                     fontsize=13, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Normal', 'Attack'])
    ax4.set_yticklabels(['Normal', 'Attack'])
    ax4.set_xlabel('Predicted Label')
    ax4.set_ylabel('True Label')
    ax4.set_title('Final Test \u2014 Confusion Matrix')

    # ══════════════════════════════════════════════════════════════
    # (2, 1)  Final Metrics Grouped Bar Chart
    # ══════════════════════════════════════════════════════════════
    ax5 = fig.add_subplot(gs[2, 1])
    names = ['Accuracy', 'Precision', 'Recall', 'F1']
    x_pos = np.arange(len(names))
    w = 0.22

    bars_en = ax5.bar(x_pos - w, final_encoder, w,
                      label='Encoder', color=PALETTE['blue'], alpha=0.85)
    bars_de = ax5.bar(x_pos, final_decoder, w,
                      label='Decoder', color=PALETTE['green'], alpha=0.85)
    bars_cb = ax5.bar(x_pos + w, final_combined, w,
                      label='Combined', color=PALETTE['red'], alpha=0.85)

    for bars in (bars_en, bars_de, bars_cb):
        for bar in bars:
            h = bar.get_height()
            ax5.annotate(
                f'{h:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            )

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(names)
    ax5.set_ylim(0, 1.18)
    ax5.set_ylabel('Score')
    ax5.set_title('Final Test \u2014 Encoder vs Decoder vs Combined')
    ax5.legend(loc='upper left')

    # ══════════════════════════════════════════════════════════════
    # Save
    # ══════════════════════════════════════════════════════════════
    save_path = os.path.join(save_dir, f'{dataset}_seed{seed}_summary.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f'\n{"=" * 60}')
    print(f'  [Visualization] Saved -> {save_path}')
    print(f'{"=" * 60}\n')
    return save_path
