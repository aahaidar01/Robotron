#!/usr/bin/env python3
"""
RL Training Analysis and Visualization Script
Analyzes Q-learning training logs and generates evaluation plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_training_log(filepath):
    """Load and parse training log file."""
    # Skip header lines until we find the CSV header
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the CSV header line
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Episode,'):
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find CSV header in log file")
    
    # Parse metadata from header
    metadata = {}
    for line in lines[:header_idx]:
        if 'Target:' in line:
            metadata['target'] = line.split('Target:')[1].strip()
        if 'Hyperparameters:' in line:
            metadata['hyperparams'] = line.split('Hyperparameters:')[1].strip()
    
    # Load CSV data
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
    
    # Rolling averages
    metrics['reward_ma'] = df['Reward'].rolling(window=window, min_periods=1).mean()
    metrics['steps_ma'] = df['Steps'].rolling(window=window, min_periods=1).mean()
    metrics['success_rate'] = df['Success'].rolling(window=window, min_periods=1).mean() * 100
    metrics['collision_rate'] = df['Collision'].rolling(window=window, min_periods=1).mean() * 100
    
    # Action distribution
    metrics['fwd_pct'] = df['Fwd']
    metrics['left_pct'] = df['Left']
    metrics['right_pct'] = df['Right']
    
    # Cumulative success
    metrics['cumulative_success'] = df['Success'].cumsum()
    metrics['cumulative_success_rate'] = metrics['cumulative_success'] / metrics['episode'] * 100
    
    return metrics


def plot_training_analysis(metrics, metadata, output_path=None):
    """Generate separate training analysis plots."""
    
    # Create output directory from output_path
    if output_path:
        output_dir = Path(output_path).parent
        base_name = Path(output_path).stem
    else:
        output_dir = Path('.')
        base_name = 'training'
    
    figures = []
    title_suffix = '\n' + metadata.get('hyperparams', '')
    
    # 1. Reward over time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(metrics['episode'], metrics['reward'], alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(metrics['episode'], metrics['reward_ma'], color='blue', linewidth=2, label='Moving Avg (50)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=300, color='green', linestyle='--', alpha=0.5, label='Success Threshold')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Reward' + title_suffix)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig1.savefig(output_dir / f'{base_name}_1_reward.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {base_name}_1_reward.png")
    figures.append(fig1)
    
    # 2. Success Rate over time
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(metrics['episode'], metrics['success_rate'], color='green', linewidth=2, label='Success Rate')
    ax2.plot(metrics['episode'], metrics['collision_rate'], color='red', linewidth=2, label='Collision Rate')
    ax2.fill_between(metrics['episode'], metrics['success_rate'], alpha=0.3, color='green')
    ax2.fill_between(metrics['episode'], metrics['collision_rate'], alpha=0.3, color='red')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Success/Collision Rate (Rolling Window=50)' + title_suffix)
    ax2.legend(loc='right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    plt.tight_layout()
    if output_path:
        fig2.savefig(output_dir / f'{base_name}_2_success_rate.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {base_name}_2_success_rate.png")
    figures.append(fig2)
    
    # 3. Steps per episode
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(metrics['episode'], metrics['steps'], alpha=0.3, color='purple', label='Steps')
    ax3.plot(metrics['episode'], metrics['steps_ma'], color='purple', linewidth=2, label='Moving Avg (50)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps')
    ax3.set_title('Steps per Episode (Lower = More Efficient)' + title_suffix)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig3.savefig(output_dir / f'{base_name}_3_steps.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {base_name}_3_steps.png")
    figures.append(fig3)
    
    # 4. Epsilon and Alpha decay
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(metrics['episode'], metrics['epsilon'], color='orange', linewidth=2, label='Epsilon (Exploration)')
    ax4.plot(metrics['episode'], metrics['alpha'], color='cyan', linewidth=2, label='Alpha (Learning Rate)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Value')
    ax4.set_title('Hyperparameter Decay' + title_suffix)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    plt.tight_layout()
    if output_path:
        fig4.savefig(output_dir / f'{base_name}_4_hyperparams.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {base_name}_4_hyperparams.png")
    figures.append(fig4)
    
    # 5. Action Distribution over time
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    window = 50
    fwd_ma = metrics['fwd_pct'].rolling(window=window, min_periods=1).mean()
    left_ma = metrics['left_pct'].rolling(window=window, min_periods=1).mean()
    right_ma = metrics['right_pct'].rolling(window=window, min_periods=1).mean()
    
    ax5.plot(metrics['episode'], fwd_ma, label='Forward', linewidth=2)
    ax5.plot(metrics['episode'], left_ma, label='Left', linewidth=2)
    ax5.plot(metrics['episode'], right_ma, label='Right', linewidth=2)
    ax5.axhline(y=33.33, color='gray', linestyle='--', alpha=0.5, label='Random (33%)')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Action %')
    ax5.set_title('Action Distribution (Rolling Avg)' + title_suffix)
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 60)
    plt.tight_layout()
    if output_path:
        fig5.savefig(output_dir / f'{base_name}_5_actions.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {base_name}_5_actions.png")
    figures.append(fig5)
    
    # 6. Cumulative Success Rate
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    ax6.plot(metrics['episode'], metrics['cumulative_success_rate'], color='darkgreen', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Cumulative Success Rate (%)')
    ax6.set_title('Overall Success Rate Over Time' + title_suffix)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 100)
    plt.tight_layout()
    if output_path:
        fig6.savefig(output_dir / f'{base_name}_6_cumulative.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {base_name}_6_cumulative.png")
    figures.append(fig6)
    
    plt.show()
    return figures


def print_summary_statistics(df, metrics):
    """Print summary statistics of training."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY STATISTICS")
    print("=" * 60)
    
    total_episodes = len(df)
    total_success = df['Success'].sum()
    total_collision = df['Collision'].sum()
    total_timeout = df['Timeout'].sum()
    
    print(f"\nTotal Episodes: {total_episodes}")
    print(f"  Successes:  {total_success} ({100*total_success/total_episodes:.1f}%)")
    print(f"  Collisions: {total_collision} ({100*total_collision/total_episodes:.1f}%)")
    print(f"  Timeouts:   {total_timeout} ({100*total_timeout/total_episodes:.1f}%)")
    
    print(f"\nReward Statistics:")
    print(f"  Mean:   {df['Reward'].mean():.1f}")
    print(f"  Std:    {df['Reward'].std():.1f}")
    print(f"  Min:    {df['Reward'].min():.1f}")
    print(f"  Max:    {df['Reward'].max():.1f}")
    
    print(f"\nSteps Statistics:")
    print(f"  Mean:   {df['Steps'].mean():.1f}")
    print(f"  Std:    {df['Steps'].std():.1f}")
    
    # Performance in different phases
    phases = [
        ("First 100", df.head(100)),
        ("Last 100", df.tail(100)),
    ]
    
    if len(df) >= 200:
        phases.insert(1, ("Episodes 100-200", df.iloc[100:200]))
    
    print(f"\nPerformance by Phase:")
    print(f"{'Phase':<20} {'Success%':>10} {'Avg Reward':>12} {'Avg Steps':>10}")
    print("-" * 55)
    
    for name, phase_df in phases:
        success_rate = phase_df['Success'].mean() * 100
        avg_reward = phase_df['Reward'].mean()
        avg_steps = phase_df['Steps'].mean()
        print(f"{name:<20} {success_rate:>10.1f} {avg_reward:>12.1f} {avg_steps:>10.1f}")
    
    # Convergence analysis
    print(f"\nConvergence Indicators:")
    last_50_success = df.tail(50)['Success'].mean() * 100
    last_50_reward_std = df.tail(50)['Reward'].std()
    print(f"  Last 50 episodes success rate: {last_50_success:.1f}%")
    print(f"  Last 50 episodes reward std:   {last_50_reward_std:.1f}")
    
    if last_50_reward_std < 100 and last_50_success > 70:
        print("  Status: Likely CONVERGED ✓")
    elif last_50_success > 50:
        print("  Status: LEARNING, not yet converged")
    else:
        print("  Status: Still in EXPLORATION phase")


def main():
    parser = argparse.ArgumentParser(description='Analyze RL training logs')
    parser.add_argument('logfile', type=str, help='Path to training log file')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output path for plot image')
    parser.add_argument('--window', '-w', type=int, default=50, help='Rolling window size for metrics')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading training log: {args.logfile}")
    df, metadata = load_training_log(args.logfile)
    print(f"Loaded {len(df)} episodes")