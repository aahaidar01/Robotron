#!/usr/bin/env python3
"""
RL Training Analysis and Visualization Script (Curriculum Edition)
Analyzes Q-learning training logs and generates evaluation plots.

Usage: python3 analyze_training.py training_logs/training_log_XXXXX.txt
       python3 analyze_training.py training_logs/training_log_XXXXX.txt -o plots/run1
       python3 analyze_training.py training_logs/training_log_XXXXX.txt -w 100
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import argparse


# --- Spawn and difficulty config ---
SPAWN_NAMES = {
    0: "Easy:  Right Zone3  (0.5, 1.5)",
    1: "Easy:  Center Zone3 (0.0, 1.5)",
    2: "Med:   Center Zone2 (0.0, 0.5)",
    3: "Med:   Left Zone2   (-0.5, 0.5)",
    4: "Hard:  Left Zone1   (-0.5,-0.6)",
    5: "Hard:  Right Zone1  (0.5,-0.6)",
    6: "Full:  Near gap     (0.5,-1.8)",
    7: "Full:  Original     (-0.5,-2.1)",
}

LEVEL_MAP = {0: 'Easy', 1: 'Easy', 2: 'Medium', 3: 'Medium',
             4: 'Hard', 5: 'Hard', 6: 'Full Maze', 7: 'Full Maze'}

LEVEL_COLORS = {
    'Easy': '#2ecc71',
    'Medium': '#f39c12',
    'Hard': '#e74c3c',
    'Full Maze': '#8e44ad',
}


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
# ORIGINAL PLOTS (same style as previous script)
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
    ax.set_ylim(0, 60)
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


# =====================================================================
# CURRICULUM-SPECIFIC PLOTS
# =====================================================================

def plot_success_by_difficulty(df, metadata, output_dir, base_name, window):
    """Rolling success rate per difficulty level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')

    for level in ['Easy', 'Medium', 'Hard', 'Full Maze']:
        spawn_ids = [k for k, v in LEVEL_MAP.items() if v == level]
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


def plot_spawn_bars(df, metadata, output_dir, base_name):
    """Per-spawn success vs collision bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    title_suffix = '\n' + metadata.get('hyperparams', '')

    spawn_ids = sorted([i for i in df['SpawnIdx'].unique() if i >= 0])
    labels = []
    success_rates = []
    collision_rates = []
    bar_colors = []

    for idx in spawn_ids:
        group = df[df['SpawnIdx'] == idx]
        n = len(group)
        s = group['Success'].sum()
        c = group['Collision'].sum()
        success_rates.append(s / n * 100 if n > 0 else 0)
        collision_rates.append(c / n * 100 if n > 0 else 0)
        labels.append(f"S{idx}\n({n} eps)")
        bar_colors.append(LEVEL_COLORS.get(LEVEL_MAP.get(idx, ''), 'gray'))

    x = np.arange(len(labels))
    w = 0.35

    bars_s = ax.bar(x - w / 2, success_rates, w, color=bar_colors, edgecolor='white',
                    linewidth=0.5, label='Success %')
    # Faded version of same color for collision bars
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
    ax.set_xticklabels(labels)
    ax.set_xlabel('Spawn Position')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Success vs Collision Rate by Spawn Position' + title_suffix)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')

    legend_elements = [Patch(facecolor=LEVEL_COLORS[l], label=l) for l in ['Easy', 'Medium', 'Hard', 'Full Maze']]
    legend_elements.append(Patch(facecolor='#e74c3c', alpha=0.45, label='Collision'))
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_8_spawn_bars.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_8_spawn_bars.png")
    plt.close(fig)


def plot_per_spawn_success_collision(df, metadata, output_dir, base_name, window=30):
    """Individual rolling success AND collision curve per spawn (2x4 grid)."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    title_suffix = metadata.get('hyperparams', '')
    fig.suptitle(f'Per-Spawn Success & Collision Rate (Rolling {window})\n{title_suffix}',
                 fontsize=12, fontweight='bold')

    for idx in range(8):
        ax = axes[idx // 4, idx % 4]
        spawn_df = df[df['SpawnIdx'] == idx].reset_index(drop=True)
        n = len(spawn_df)
        level = LEVEL_MAP.get(idx, 'Unknown')
        color = LEVEL_COLORS.get(level, 'gray')

        if n >= 5:
            w = min(window, max(5, n // 5))

            # Success
            x_s, y_s = rolling_success_rate(spawn_df, w)
            if x_s:
                ax.plot(x_s, y_s, color=color, linewidth=1.5, label='Success')
                ax.fill_between(x_s, y_s, alpha=0.15, color=color)

            # Collision — reuse rolling_success_rate on Collision column
            coll_df = spawn_df.copy()
            coll_df['Success'] = coll_df['Collision']
            x_c, y_c = rolling_success_rate(coll_df, w)
            if x_c:
                ax.plot(x_c, y_c, color='red', linewidth=1.2, alpha=0.7, label='Collision')
                ax.fill_between(x_c, y_c, alpha=0.1, color='red')

        s = spawn_df['Success'].sum() if n > 0 else 0
        c = spawn_df['Collision'].sum() if n > 0 else 0
        ax.set_title(f'S{idx} — {level}\n✓{s}/{n} ({s/n*100:.0f}%)  ✗{c}/{n} ({c/n*100:.0f}%)' if n > 0
                     else f'S{idx} — No data', fontsize=9)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Episode', fontsize=8)
        if idx % 4 == 0:
            ax.set_ylabel('Rate (%)', fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(output_dir / f'{base_name}_9_per_spawn_timeline.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {base_name}_9_per_spawn_timeline.png")
    plt.close(fig)


def plot_full_maze_focus(df, metadata, output_dir, base_name):
    """Dedicated full maze spawns analysis (3-panel)."""
    full_df = df[df['SpawnIdx'].isin([6, 7])].reset_index(drop=True)
    if len(full_df) < 10:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    title_suffix = metadata.get('hyperparams', '')
    fig.suptitle(f'Full Maze Spawns Focus (S6 & S7) — {len(full_df)} episodes\n{title_suffix}',
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
    success_steps = full_df[full_df['Success'] == 1]['Steps']
    collision_steps = full_df[full_df['Collision'] == 1]['Steps']
    timeout_steps = full_df[full_df['Timeout'] == 1]['Steps']
    data_to_plot = []
    labels_to_plot = []
    colors_to_plot = []
    if len(success_steps) > 0:
        data_to_plot.append(success_steps.values)
        labels_to_plot.append(f'Success\n(n={len(success_steps)})')
        colors_to_plot.append('#2ecc71')
    if len(collision_steps) > 0:
        data_to_plot.append(collision_steps.values)
        labels_to_plot.append(f'Collision\n(n={len(collision_steps)})')
        colors_to_plot.append('#e74c3c')
    if len(timeout_steps) > 0:
        data_to_plot.append(timeout_steps.values)
        labels_to_plot.append(f'Timeout\n(n={len(timeout_steps)})')
        colors_to_plot.append('#f39c12')
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, tick_labels=labels_to_plot, patch_artist=True)
        for patch, c in zip(bp['boxes'], colors_to_plot):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
    ax2.set_ylabel('Steps')
    ax2.set_title('Steps Distribution by Outcome')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Reward scatter colored by outcome
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
        ax3.plot(full_df['Episode'].values, reward_ma.values, color=color, linewidth=2,
                 label='Rolling avg')
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

def print_summary(df):
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

    print(f"\nReward Statistics:")
    print(f"  Mean:   {df['Reward'].mean():.1f}")
    print(f"  Std:    {df['Reward'].std():.1f}")
    print(f"  Min:    {df['Reward'].min():.1f}")
    print(f"  Max:    {df['Reward'].max():.1f}")

    print(f"\nSteps Statistics:")
    print(f"  Mean:   {df['Steps'].mean():.1f}")
    print(f"  Std:    {df['Steps'].std():.1f}")

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

    # Per-spawn
    print(f"\n{'='*60}")
    print("PER-SPAWN PERFORMANCE")
    print("=" * 60)

    for idx in sorted(df['SpawnIdx'].unique()):
        if idx < 0:
            continue
        group = df[df['SpawnIdx'] == idx]
        n = len(group)
        s = group['Success'].sum()
        c = group['Collision'].sum()
        name = SPAWN_NAMES.get(idx, f"Spawn {idx}")

        print(f"\n  Spawn {idx} — {name}")
        print(f"    Episodes: {n}  |  Success: {s}/{n} ({s/n*100:.1f}%)  |  Collision: {c}/{n} ({c/n*100:.1f}%)")
        print(f"    Avg Steps: {group['Steps'].mean():.1f}  |  Avg Reward: {group['Reward'].mean():.1f}")

        if n >= 20:
            block = min(20, n // 3)
            first_rate = group.head(block)['Success'].mean() * 100
            last_rate = group.tail(block)['Success'].mean() * 100
            print(f"    Trend: First {block} → {first_rate:.0f}%  |  Last {block} → {last_rate:.0f}%")

    # Per-difficulty
    print(f"\n{'='*60}")
    print("PER-DIFFICULTY-LEVEL SUMMARY")
    print("=" * 60)

    for level in ['Easy', 'Medium', 'Hard', 'Full Maze']:
        spawn_ids = [k for k, v in LEVEL_MAP.items() if v == level]
        group = df[df['SpawnIdx'].isin(spawn_ids)]
        n = len(group)
        if n == 0:
            continue
        s = group['Success'].sum()
        print(f"  {level:10s}: {s}/{n} success ({s/n*100:.1f}%) | Avg Steps: {group['Steps'].mean():.1f}")

    # Learning progress windows with per-level breakdown
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
        for level in ['Easy', 'Medium', 'Hard', 'Full Maze']:
            spawn_ids = [k for k, v in LEVEL_MAP.items() if v == level]
            lvl_block = block[block['SpawnIdx'].isin(spawn_ids)]
            rates[level] = lvl_block['Success'].mean() * 100 if len(lvl_block) > 0 else 0

        print(f"  Ep {start+1:5d}-{end:5d}: "
              f"Overall {s/n*100:5.1f}% | "
              f"Easy {rates['Easy']:5.1f}% | Med {rates['Medium']:5.1f}% | "
              f"Hard {rates['Hard']:5.1f}% | Full {rates['Full Maze']:5.1f}% | "
              f"ε={eps:.3f}")

    # Full maze focus
    full_df = df[df['SpawnIdx'].isin([6, 7])]
    if len(full_df) > 0:
        print(f"\n{'='*60}")
        print(f"FULL MAZE SPAWNS FOCUS (S6 & S7) — {len(full_df)} episodes")
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
        print("  Status: Likely CONVERGED ✓")
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

    metrics = compute_metrics(df, window=args.window)

    # Text summary
    print_summary(df)

    # Generate all plots
    print(f"\nGenerating plots (window={args.window})...")

    # Original 6 plots (same style as previous script)
    plot_reward(metrics, metadata, output_dir, base_name, args.window)
    plot_success_collision(metrics, metadata, output_dir, base_name, args.window)
    plot_steps(metrics, metadata, output_dir, base_name, args.window)
    plot_hyperparams(metrics, metadata, output_dir, base_name)
    plot_actions(metrics, metadata, output_dir, base_name, args.window)
    plot_cumulative(metrics, metadata, output_dir, base_name)

    # New curriculum-specific plots
    plot_success_by_difficulty(df, metadata, output_dir, base_name, args.window)
    plot_spawn_bars(df, metadata, output_dir, base_name)
    plot_per_spawn_success_collision(df, metadata, output_dir, base_name, window=min(30, args.window))
    plot_full_maze_focus(df, metadata, output_dir, base_name)

    print(f"\n✅ Done! 10 plots saved to: {output_dir}/")


if __name__ == '__main__':
    main()