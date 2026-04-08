"""
Visualization: Alpha vs Accuracy Curve for CLIP+GDA on Food-101
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from ablation study
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
top1 = [65.70, 68.63, 71.59, 74.11, 76.49, 78.46, 80.04, 81.00, 81.55, 81.32, 80.33]
top5 = [90.34, 91.78, 92.98, 94.12, 95.11, 95.78, 96.29, 96.75, 96.82, 96.68, 96.15]

output_dir = Path("../results/ablation_alpha_sweep")
output_dir.mkdir(parents=True, exist_ok=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Plot 1: Top-1 Accuracy ---
ax1.plot(alphas, top1, 'b-o', linewidth=2, markersize=8, label='Top-1 Accuracy')
ax1.axhline(y=80.33, color='green', linestyle='--', alpha=0.7, label='Zero-shot CLIP (80.33%)')
ax1.axhline(y=78.46, color='red', linestyle='--', alpha=0.7, label='Baseline α=0.5 (78.46%)')

# Mark optimal point
best_idx = np.argmax(top1)
ax1.scatter([alphas[best_idx]], [top1[best_idx]], color='gold', s=200, zorder=5, edgecolors='black', linewidths=2)
ax1.annotate(f'Optimal\nα={alphas[best_idx]}\n{top1[best_idx]}%', 
             xy=(alphas[best_idx], top1[best_idx]), 
             xytext=(alphas[best_idx]-0.15, top1[best_idx]-4),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black'))

ax1.set_xlabel('Alpha (α)', fontsize=12)
ax1.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
ax1.set_title('Effect of GDA Weighting on Food-101 Classification', fontsize=14, fontweight='bold')
ax1.set_xticks(alphas)
ax1.set_ylim([63, 84])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# Add annotations for extremes
ax1.annotate('Pure GDA\n(65.70%)', xy=(0, 65.70), xytext=(0.1, 67), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))
ax1.annotate('Pure CLIP\n(80.33%)', xy=(1.0, 80.33), xytext=(0.85, 78), fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))

# --- Plot 2: Both Top-1 and Top-5 ---
ax2.plot(alphas, top1, 'b-o', linewidth=2, markersize=8, label='Top-1 Accuracy')
ax2.plot(alphas, top5, 'g-s', linewidth=2, markersize=8, label='Top-5 Accuracy')

ax2.scatter([alphas[best_idx]], [top1[best_idx]], color='gold', s=150, zorder=5, edgecolors='black', linewidths=2)
ax2.scatter([alphas[best_idx]], [top5[best_idx]], color='gold', s=150, zorder=5, edgecolors='black', linewidths=2)

ax2.set_xlabel('Alpha (α)', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Top-1 and Top-5 Accuracy vs Alpha', fontsize=14, fontweight='bold')
ax2.set_xticks(alphas)
ax2.set_ylim([63, 100])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right')

# Add text box with key finding
textstr = 'Key Finding:\nα=0.0 (Pure GDA): 65.70%\nα=0.5 (Baseline): 78.46%\nα=0.8 (Optimal): 81.55%\nα=1.0 (Pure CLIP): 80.33%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(output_dir / 'alpha_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'alpha_curve.pdf', bbox_inches='tight')
print(f"Saved: {output_dir / 'alpha_curve.png'}")
print(f"Saved: {output_dir / 'alpha_curve.pdf'}")

plt.show()