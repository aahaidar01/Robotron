#!/usr/bin/env python3
"""
RL Training Analysis and Visualization Script (Curriculum Edition)
Auto-reads spawn positions from log and classifies by distance to target.

Usage:
  python3 analyze_training.py training_log.txt
  python3 analyze_training.py training_log.txt -o plots/run1
  python3 analyze_training.py training_log.txt -w 100
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import argparse
import math


# --- Auto-classification by distance to target ---
# Target at (0.0, 1.8). Boundaries tuned to match maze geometry:
#   dist < 1.35m → Easy      (short path, 0-1 turns)
#   dist < 2.1m  → Medium    (1 turn required)
#   dist < 3.5m  → Hard      (2 turns required)
#   dist >= 3.5m → Full Maze (3 turns, full maze traverse)

TARGET_POS = (0.0, 1.8)
DIST_BOUNDARIES = [(1.35, 'Easy'), (2.1, 'Medium'), (3.5, 'Hard')]
DEFAULT_LEVEL = 'Full Maze'

LEVEL_COLORS = {
    'Easy': '#2ecc71',
    'Medium': '#f39c12',
    'Hard': '#e74c3c',
    'Full Maze': '#8e44ad',
}
LEVEL_ORDER = ['Easy', 'Medium', 'Hard', 'Full Maze']


def classify_spawn(x, y):
    """Classify spawn difficulty based on Euclidean distance to target."""
    dist = math.sqrt((x - TARGET_POS[0])**2 + (y - TARGET_POS[1])**2)
    for boundary, level in DIST_BOUNDARIES:
        if dist < boundary:
            return level
    return DEFAULT_LEVEL


def discover_spawns(df):
    """Discover unique spawn positions from log data and classify them."""
    spawn_info = {}
    for idx in sorted(df['SpawnIdx'].unique()):
        if idx < 0:
            continue
        group = df[df['SpawnIdx'] == idx]
        x = group['SpawnX'].mode().iloc[0]
        y = group['SpawnY'].mode().iloc[0]
        level = classify_spawn(x, y)
        dist = math.sqrt((x - TARGET_POS[0])**2 + (y - TARGET_POS[1])**2)
        spawn_info[idx] = {
            'x': x, 'y': y, 'level': level, 'dist': dist,
            'label': f"({x:+.1f}, {y:+.1f})",
        }
    return spawn_info


def load_training_log(filepath):
    """Load and parse training log file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Episode,'):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not find CSV header in log file")

    metadata = {}
    for line in lines[:header_idx]:
        if 'Target:' in line:
            metadata['target'] = line.split('Target:')[1].strip()
        if 'Hyperparameters:' in line:
            metadata['hyperparams'] = line.split('Hyperparameters:')[1].strip()

    df = pd.read_csv(filepath, skiprows=header_idx)
    return df, metadata


def compute_metrics(df, window=50):
    """Compute rolling metrics for analysis."""
    metrics = pd.DataFrame()
    metrics['episode'] = df['Episode']
    metrics['reward'] = df['Reward']
    metrics['steps'] = df['Steps']
    metrics['success'] = df['Success']
    metrics['collision'] = df['Collision']
    metrics['timeout'] = df['Timeout']
    metrics['epsilon'] = df['Epsilon']
    metrics['alpha'] = df['Alpha']
    metrics['spawn_idx'] = df['SpawnIdx']

    metrics['reward_ma'] = df['Reward'].rolling(window=window, min_periods=1).mean()
    metrics['steps_ma'] = df['Steps'].rolling(window=window, min_periods=1).mean()
    metrics['success_rate'] = df['Success'].rolling(window=window, min_periods=1).mean() * 100
    metrics['collision_rate'] = df['Collision'].rolling(window=window, min_periods=1).mean() * 100

    metrics['fwd_pct'] = df['Fwd']
    metrics['left_pct'] = df['Left']
    metrics['right_pct'] = df['Right']

    metrics['cumulative_success'] = df['Success'].cumsum()
    metrics['cumulative_success_rate'] = metrics['cumulative_success'] / metrics['episode'] * 100

    return metrics


def rolling_success_rate(eps_df, window):
    """Compute rolling success rate for a subset of episodes."""
    if len(eps_df) < window:
        return [], []
    successes = eps_df['Success'].values
    episode_nums = eps_df['Episode'].values
    rates = []
    x_vals = []
    for i in range(window, len(successes) + 1):
        rates.append(np.mean(successes[i - window:i]) * 100)
        x_vals.append(episode_nums[i - 1])
    return x_vals, rates


# =====================================================================
# PLOTS
# =====================================================================

def plot_reward(metrics, metadata, output_dir, base_name, window):
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')
    ax.plot(metrics['episode'], metrics['reward'], alpha=0.3, color='blue', label='Episode Reward')
    ax.plot(metrics['episode'], metrics['reward_ma'], color='blue', linewidth=2, label=f'Moving Avg ({window})')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=300, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Reward' + title_suffix)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_1_reward.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_1_reward.png")
    plt.close(fig)


def plot_success_collision(metrics, metadata, output_dir, base_name, window):
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')
    ax.plot(metrics['episode'], metrics['success_rate'], color='green', linewidth=2, label='Success Rate')
    ax.plot(metrics['episode'], metrics['collision_rate'], color='red', linewidth=2, label='Collision Rate')
    ax.fill_between(metrics['episode'], metrics['success_rate'], alpha=0.3, color='green')
    ax.fill_between(metrics['episode'], metrics['collision_rate'], alpha=0.3, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rate (%)')
    ax.set_title(f'Success/Collision Rate (Rolling Window={window})' + title_suffix)
    ax.legend(loc='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_2_success_rate.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_2_success_rate.png")
    plt.close(fig)


def plot_steps(metrics, metadata, output_dir, base_name, window):
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')
    ax.plot(metrics['episode'], metrics['steps'], alpha=0.3, color='purple', label='Steps')
    ax.plot(metrics['episode'], metrics['steps_ma'], color='purple', linewidth=2, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode (Lower = More Efficient)' + title_suffix)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_3_steps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_3_steps.png")
    plt.close(fig)


def plot_hyperparams(metrics, metadata, output_dir, base_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')
    ax.plot(metrics['episode'], metrics['epsilon'], color='orange', linewidth=2, label='Epsilon (Exploration)')
    ax.plot(metrics['episode'], metrics['alpha'], color='cyan', linewidth=2, label='Alpha (Learning Rate)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Value')
    ax.set_title('Hyperparameter Decay' + title_suffix)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_4_hyperparams.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_4_hyperparams.png")
    plt.close(fig)


def plot_actions(metrics, metadata, output_dir, base_name, window):
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')
    fwd_ma = metrics['fwd_pct'].rolling(window=window, min_periods=1).mean()
    left_ma = metrics['left_pct'].rolling(window=window, min_periods=1).mean()
    right_ma = metrics['right_pct'].rolling(window=window, min_periods=1).mean()
    ax.plot(metrics['episode'], fwd_ma, label='Forward', linewidth=2)
    ax.plot(metrics['episode'], left_ma, label='Left', linewidth=2)
    ax.plot(metrics['episode'], right_ma, label='Right', linewidth=2)
    ax.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random (33%)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Action %')
    ax.set_title('Action Distribution (Rolling Avg)' + title_suffix)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 90)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_5_actions.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_5_actions.png")
    plt.close(fig)


def plot_cumulative(metrics, metadata, output_dir, base_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')
    ax.plot(metrics['episode'], metrics['cumulative_success_rate'], color='darkgreen', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate (%)')
    ax.set_title('Overall Success Rate Over Time' + title_suffix)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_6_cumulative.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_6_cumulative.png")
    plt.close(fig)


def plot_success_by_difficulty(df, spawn_info, metadata, output_dir, base_name, window):
    """Rolling success rate per difficulty level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')

    for level in LEVEL_ORDER:
        spawn_ids = [k for k, v in spawn_info.items() if v['level'] == level]
        if not spawn_ids:
            continue
        level_df = df[df['SpawnIdx'].isin(spawn_ids)].reset_index(drop=True)
        if len(level_df) < 10:
            continue
        w = min(window, max(5, len(level_df) // 5))
        x, y = rolling_success_rate(level_df, w)
        if x:
            ax.plot(x, y, label=f'{level} (n={len(level_df)})',
                    color=LEVEL_COLORS[level], linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate by Difficulty Level (Rolling {window})' + title_suffix)
    ax.legend(loc='upper left')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_7_difficulty_success.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_7_difficulty_success.png")
    plt.close(fig)


def plot_spawn_bars(df, spawn_info, metadata, output_dir, base_name):
    """Per-spawn success vs collision bar chart."""
    fig, ax = plt.subplots(figsize=(max(12, len(spawn_info) * 1.5), 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')

    spawn_ids = sorted(spawn_info.keys())
    labels = []
    success_rates = []
    collision_rates = []
    bar_colors = []

    for idx in spawn_ids:
        group = df[df['SpawnIdx'] == idx]
        n = len(group)
        s = group['Success'].sum()
        c = group['Collision'].sum()
        info = spawn_info[idx]
        success_rates.append(s / n * 100 if n > 0 else 0)
        collision_rates.append(c / n * 100 if n > 0 else 0)
        labels.append(f"S{idx} {info['label']}\n{info['level']} ({n} eps)")
        bar_colors.append(LEVEL_COLORS.get(info['level'], 'gray'))

    x = np.arange(len(labels))
    w = 0.35

    bars_s = ax.bar(x - w / 2, success_rates, w, color=bar_colors, edgecolor='white',
                    linewidth=0.5, label='Success %')
    bars_c = ax.bar(x + w / 2, collision_rates, w, color='#e74c3c',
                    edgecolor='white', linewidth=0.5, label='Collision %', alpha=0.45)

    for bar, rate in zip(bars_s, success_rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar, rate in zip(bars_c, collision_rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=7, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('Spawn Position')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Success vs Collision Rate by Spawn Position' + title_suffix)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [Patch(facecolor=LEVEL_COLORS[l], label=l) for l in LEVEL_ORDER]
    legend_elements.append(Patch(facecolor='#e74c3c', alpha=0.45, label='Collision'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_8_spawn_bars.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_8_spawn_bars.png")
    plt.close(fig)


def plot_spawn_bars_last_n(df, spawn_info, metadata, output_dir, base_name, n=500):
    """Per-spawn success vs collision bar chart for the LAST n episodes only."""
    last_df = df.tail(n)
    actual_n = len(last_df)

    fig, ax = plt.subplots(figsize=(max(12, len(spawn_info) * 1.5), 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')

    spawn_ids = sorted(spawn_info.keys())
    labels = []
    success_rates = []
    collision_rates = []
    bar_colors = []

    for idx in spawn_ids:
        group = last_df[last_df['SpawnIdx'] == idx]
        n_eps = len(group)
        s = group['Success'].sum()
        c = group['Collision'].sum()
        info = spawn_info[idx]
        success_rates.append(s / n_eps * 100 if n_eps > 0 else 0)
        collision_rates.append(c / n_eps * 100 if n_eps > 0 else 0)
        labels.append(f"S{idx} {info['label']}\n{info['level']} ({n_eps} eps)")
        bar_colors.append(LEVEL_COLORS.get(info['level'], 'gray'))

    x = np.arange(len(labels))
    w = 0.35

    bars_s = ax.bar(x - w / 2, success_rates, w, color=bar_colors, edgecolor='white',
                    linewidth=0.5, label='Success %')
    bars_c = ax.bar(x + w / 2, collision_rates, w, color='#e74c3c',
                    edgecolor='white', linewidth=0.5, label='Collision %', alpha=0.45)

    for bar, rate in zip(bars_s, success_rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar, rate in zip(bars_c, collision_rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{rate:.0f}%', ha='center', va='bottom', fontsize=7, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel('Spawn Position')
    ax.set_ylabel('Rate (%)')
    ax.set_title(f'Success vs Collision Rate by Spawn — Last {actual_n} Episodes' + title_suffix)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [Patch(facecolor=LEVEL_COLORS[l], label=l) for l in LEVEL_ORDER]
    legend_elements.append(Patch(facecolor='#e74c3c', alpha=0.45, label='Collision'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_11_spawn_bars_last{actual_n}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_11_spawn_bars_last{actual_n}.png")
    plt.close(fig)

def plot_per_spawn_success_collision(df, spawn_info, metadata, output_dir, base_name, window=30):
    """Individual rolling success AND collision curve per spawn (dynamic grid)."""
    n_spawns = len(spawn_info)
    cols = min(4, n_spawns)
    rows = (n_spawns + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), sharey=True, squeeze=False)
    title_suffix = metadata.get('hyperparams', '')
    fig.suptitle(f'Per-Spawn Success & Collision Rate (Rolling {window})\n{title_suffix}',
                 fontsize=12, fontweight='bold')

    spawn_ids = sorted(spawn_info.keys())

    for i, idx in enumerate(spawn_ids):
        ax = axes[i // cols, i % cols]
        spawn_df = df[df['SpawnIdx'] == idx].reset_index(drop=True)
        n = len(spawn_df)
        info = spawn_info[idx]

        if n >= 5:
            w = min(window, max(5, n // 5))

            x_s, y_s = rolling_success_rate(spawn_df, w)
            if x_s:
                ax.plot(x_s, y_s, color='green', linewidth=1.5, label='Success')
                ax.fill_between(x_s, y_s, alpha=0.15, color='green')

            coll_df = spawn_df.copy()
            coll_df['Success'] = coll_df['Collision']
            x_c, y_c = rolling_success_rate(coll_df, w)
            if x_c:
                ax.plot(x_c, y_c, color='red', linewidth=1.2, alpha=0.7, label='Collision')
                ax.fill_between(x_c, y_c, alpha=0.1, color='red')

        s = spawn_df['Success'].sum() if n > 0 else 0
        c = spawn_df['Collision'].sum() if n > 0 else 0
        ax.set_title(f'S{idx} {info["label"]} — {info["level"]}\n'
                     f'S:{s}/{n} ({s/n*100:.0f}%)  C:{c}/{n} ({c/n*100:.0f}%)' if n > 0
                     else f'S{idx} — No data', fontsize=9)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Episode', fontsize=8)
        if i % cols == 0:
            ax.set_ylabel('Rate (%)', fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    # Hide unused subplots
    for i in range(len(spawn_ids), rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_9_per_spawn_timeline.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_9_per_spawn_timeline.png")
    plt.close(fig)


def plot_full_maze_focus(df, spawn_info, metadata, output_dir, base_name):
    """Dedicated full maze spawns analysis (3-panel)."""
    full_ids = [k for k, v in spawn_info.items() if v['level'] == 'Full Maze']
    if not full_ids:
        return

    full_df = df[df['SpawnIdx'].isin(full_ids)].reset_index(drop=True)
    if len(full_df) < 10:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title_suffix = metadata.get('hyperparams', '')
    id_str = ', '.join(f'S{i}' for i in sorted(full_ids))
    fig.suptitle(f'Full Maze Spawns Focus ({id_str}) — {len(full_df)} episodes\n{title_suffix}',
                 fontsize=12, fontweight='bold')

    color = LEVEL_COLORS['Full Maze']

    # Panel 1: Rolling success rate
    ax1 = axes[0]
    w = min(30, max(5, len(full_df) // 5))
    x, y = rolling_success_rate(full_df, w)
    if x:
        ax1.plot(x, y, color=color, linewidth=2)
        ax1.fill_between(x, y, alpha=0.2, color=color)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title(f'Success Rate (Rolling {w})')
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Steps distribution box plot
    ax2 = axes[1]
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    for outcome, col, c in [('Success', 'Success', '#2ecc71'),
                              ('Collision', 'Collision', '#e74c3c'),
                              ('Timeout', 'Timeout', '#f39c12')]:
        steps = full_df[full_df[col] == 1]['Steps']
        if len(steps) > 0:
            data_to_plot.append(steps.values)
            labels_to_plot.append(f'{outcome}\n(n={len(steps)})')
            colors_to_plot.append(c)
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, tick_labels=labels_to_plot, patch_artist=True)
        for patch, c in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Steps')
    ax2.set_title('Steps Distribution by Outcome')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Reward scatter
    ax3 = axes[2]
    outcome_colors = []
    for _, row in full_df.iterrows():
        if row['Success'] == 1:
            outcome_colors.append('#2ecc71')
        elif row['Collision'] == 1:
            outcome_colors.append('#e74c3c')
        else:
            outcome_colors.append('#f39c12')
    ax3.scatter(full_df['Episode'], full_df['Reward'], alpha=0.4, s=15, c=outcome_colors)
    if len(full_df) >= 20:
        reward_ma = full_df['Reward'].rolling(window=min(20, len(full_df) // 3), min_periods=1).mean()
        ax3.plot(full_df['Episode'].values, reward_ma.values, color=color, linewidth=2, label='Rolling avg')
        ax3.legend()
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward (green=S  red=C  orange=T)')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_10_full_maze_focus.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_10_full_maze_focus.png")
    plt.close(fig)


# =====================================================================
# TEXT SUMMARY
# =====================================================================

def print_summary(df, spawn_info):
    total = len(df)
    successes = df['Success'].sum()
    collisions = df['Collision'].sum()
    timeouts = df['Timeout'].sum()

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nTotal Episodes: {total}")
    print(f"  Successes:  {successes} ({successes/total*100:.1f}%)")
    print(f"  Collisions: {collisions} ({collisions/total*100:.1f}%)")
    print(f"  Timeouts:   {timeouts} ({timeouts/total*100:.1f}%)")

    print(f"\nReward: mean={df['Reward'].mean():.1f}, std={df['Reward'].std():.1f}, "
          f"min={df['Reward'].min():.1f}, max={df['Reward'].max():.1f}")
    print(f"Steps:  mean={df['Steps'].mean():.1f}, std={df['Steps'].std():.1f}")

    # Performance by phase
    phases = [("First 100", df.head(100)), ("Last 100", df.tail(100))]
    if len(df) >= 200:
        phases.insert(1, ("Episodes 100-200", df.iloc[100:200]))

    print(f"\nPerformance by Phase:")
    print(f"{'Phase':<20} {'Success%':>10} {'Avg Reward':>12} {'Avg Steps':>10}")
    print("-" * 55)
    for name, phase_df in phases:
        print(f"{name:<20} {phase_df['Success'].mean()*100:>10.1f} "
              f"{phase_df['Reward'].mean():>12.1f} {phase_df['Steps'].mean():>10.1f}")

    # Discovered spawns
    print(f"\n{'='*60}")
    print("DISCOVERED SPAWNS (auto-classified by distance to target)")
    print("=" * 60)

    for idx in sorted(spawn_info.keys()):
        info = spawn_info[idx]
        group = df[df['SpawnIdx'] == idx]
        n = len(group)
        if n == 0:
            continue
        s = group['Success'].sum()
        c = group['Collision'].sum()

        print(f"\n  S{idx} {info['label']} — {info['level']} (dist={info['dist']:.2f}m)")
        print(f"    Episodes: {n}  |  Success: {s}/{n} ({s/n*100:.1f}%)  |  Collision: {c}/{n} ({c/n*100:.1f}%)")
        print(f"    Avg Steps: {group['Steps'].mean():.1f}  |  Avg Reward: {group['Reward'].mean():.1f}")

        if n >= 20:
            block = min(20, n // 3)
            first_rate = group.head(block)['Success'].mean() * 100
            last_rate = group.tail(block)['Success'].mean() * 100
            print(f"    Trend: First {block} -> {first_rate:.0f}%  |  Last {block} -> {last_rate:.0f}%")

    # Per-difficulty
    print(f"\n{'='*60}")
    print("PER-DIFFICULTY-LEVEL SUMMARY")
    print("=" * 60)

    for level in LEVEL_ORDER:
        spawn_ids = [k for k, v in spawn_info.items() if v['level'] == level]
        if not spawn_ids:
            continue
        group = df[df['SpawnIdx'].isin(spawn_ids)]
        n = len(group)
        if n == 0:
            continue
        s = group['Success'].sum()
        spawns_str = ', '.join(f'S{i}' for i in sorted(spawn_ids))
        print(f"  {level:10s} [{spawns_str}]: {s}/{n} success ({s/n*100:.1f}%) | "
              f"Avg Steps: {group['Steps'].mean():.1f}")

    # Learning progress windows
    print(f"\n{'='*60}")
    print("LEARNING PROGRESS OVER TIME")
    print("=" * 60)

    window = max(100, total // 10)
    num_windows = min(10, total // window)

    for i in range(num_windows):
        start = i * window
        end = start + window
        block = df.iloc[start:end]
        s = block['Success'].sum()
        n = len(block)
        eps = block['Epsilon'].iloc[-1]

        rates = {}
        for level in LEVEL_ORDER:
            spawn_ids = [k for k, v in spawn_info.items() if v['level'] == level]
            lvl_block = block[block['SpawnIdx'].isin(spawn_ids)]
            rates[level] = lvl_block['Success'].mean() * 100 if len(lvl_block) > 0 else 0

        print(f"  Ep {start+1:5d}-{end:5d}: "
              f"Overall {s/n*100:5.1f}% | "
              f"Easy {rates['Easy']:5.1f}% | Med {rates['Medium']:5.1f}% | "
              f"Hard {rates['Hard']:5.1f}% | Full {rates['Full Maze']:5.1f}% | "
              f"e={eps:.3f}")

    # Full maze focus
    full_ids = [k for k, v in spawn_info.items() if v['level'] == 'Full Maze']
    if full_ids:
        full_df = df[df['SpawnIdx'].isin(full_ids)]
        if len(full_df) > 0:
            id_str = ', '.join(f'S{i}' for i in sorted(full_ids))
            print(f"\n{'='*60}")
            print(f"FULL MAZE FOCUS ({id_str}) — {len(full_df)} episodes")
            print("=" * 60)

            s = full_df['Success'].sum()
            print(f"  Overall: {s}/{len(full_df)} success ({s/len(full_df)*100:.1f}%)")

            if len(full_df) >= 10:
                print(f"  First 10: {full_df.head(10)['Success'].sum()}/10  |  "
                      f"Last 10: {full_df.tail(10)['Success'].sum()}/10")

            success_rows = full_df[full_df['Success'] == 1]
            if len(success_rows) > 0:
                first_ep = success_rows.iloc[0]['Episode']
                idx_in_full = full_df.index.get_loc(success_rows.index[0])
                print(f"  First success: episode {int(first_ep)} (after {idx_in_full+1} attempts)")
            else:
                print("  No successes yet from full maze spawns")

            block_size = max(10, len(full_df) // 15)
            for i in range(0, len(full_df), block_size):
                block = full_df.iloc[i:i + block_size]
                if len(block) < 5:
                    break
                bs = block['Success'].sum()
                bn = len(block)
                print(f"  Attempts {i+1:3d}-{i+bn:3d}: {bs}/{bn} ({bs/bn*100:5.1f}%) | "
                      f"Avg steps: {block['Steps'].mean():.0f}")

    # Convergence
    print(f"\n{'='*60}")
    print("CONVERGENCE INDICATORS")
    print("=" * 60)
    last_50 = df.tail(50)
    last_50_sr = last_50['Success'].mean() * 100
    last_50_std = last_50['Reward'].std()
    print(f"  Last 50 episodes success rate: {last_50_sr:.1f}%")
    print(f"  Last 50 episodes reward std:   {last_50_std:.1f}")
    if last_50_std < 100 and last_50_sr > 70:
        print("  Status: Likely CONVERGED")
    elif last_50_sr > 50:
        print("  Status: LEARNING, not yet converged")
    else:
        print("  Status: Still in EXPLORATION phase")

    print("\n" + "=" * 60)


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze RL training logs (Curriculum Edition)')
    parser.add_argument('logfile', type=str, help='Path to training log file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for plots (default: current dir)')
    parser.add_argument('--window', '-w', type=int, default=50,
                        help='Rolling window size (default: 50)')

    args = parser.parse_args()

    print(f"Loading: {args.logfile}")
    df, metadata = load_training_log(args.logfile)
    print(f"Loaded {len(df)} episodes")

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = 'training'
    else:
        output_dir = Path('.')
        base_name = Path(args.logfile).stem

    # Discover and classify spawns from data
    spawn_info = discover_spawns(df)
    print(f"\nDiscovered {len(spawn_info)} spawn positions:")
    for idx in sorted(spawn_info.keys()):
        info = spawn_info[idx]
        n = len(df[df['SpawnIdx'] == idx])
        print(f"  S{idx} {info['label']} -> {info['level']} (dist={info['dist']:.2f}m, {n} eps)")

    # Compute metrics
    metrics = compute_metrics(df, window=args.window)

    # Text summary
    print_summary(df, spawn_info)

    # Generate all plots
    print(f"\nGenerating plots (window={args.window})...")
    plot_reward(metrics, metadata, output_dir, base_name, args.window)
    plot_success_collision(metrics, metadata, output_dir, base_name, args.window)
    plot_steps(metrics, metadata, output_dir, base_name, args.window)
    plot_hyperparams(metrics, metadata, output_dir, base_name)
    plot_actions(metrics, metadata, output_dir, base_name, args.window)
    plot_cumulative(metrics, metadata, output_dir, base_name)
    plot_success_by_difficulty(df, spawn_info, metadata, output_dir, base_name, args.window)
    plot_spawn_bars(df, spawn_info, metadata, output_dir, base_name)
    plot_spawn_bars_last_n(df, spawn_info, metadata, output_dir, base_name, n=500)
    plot_per_spawn_success_collision(df, spawn_info, metadata, output_dir, base_name,
                                     window=min(30, args.window))
    plot_full_maze_focus(df, spawn_info, metadata, output_dir, base_name)

    print(f"\nDone! 10 plots saved to: {output_dir}/")


if __name__ == '__main__':
    main()