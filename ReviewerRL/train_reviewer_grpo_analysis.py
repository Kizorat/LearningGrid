import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import ast
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)


GENERATE_ALL_PLOTS = True


def map_type(env_name):
    env_name = str(env_name)  
    if "LavaCrossing" in env_name:
        return "LavaCrossing"
    if "DoorKey" in env_name:
        return "DoorKey"
    if "Empty" in env_name:
        return "Empty"
    return "Other"


def detect_dataset_format(df):
    # Check for GRPO offline format
    grpo_columns = ['episode', 'env_name', 'chosen_sequence', 'rejected_sequence', 
                    'chosen_reward', 'rejected_reward']
    if all(col in df.columns for col in grpo_columns):
        return 'grpo_offline'
    
    # Check for old sequence format
    sequence_columns = ['sequenza_helper_suggerita', 'sequenza_eseguita', 
                        'sequence_score_model', 'coherence']
    if all(col in df.columns for col in sequence_columns):
        return 'sequences'
    
    return 'single_actions'


def parse_action_list(action_str):
    if pd.isna(action_str):
        return []
    
    try:
        # parsing with ast.literal_eval (handles both list and string formats)
        return ast.literal_eval(str(action_str))
    except:
        try:
            # Fallback: manual parsing
            action_str = str(action_str).strip()
            if action_str.startswith('[') and action_str.endswith(']'):
                action_str = action_str[1:-1]
            actions = [a.strip().strip("'\"") for a in action_str.split(',') if a.strip()]
            return actions
        except:
            return []


def compute_grpo_metrics(df, dataset_format='single_actions'):
    metrics = {}
    
    # general reward metrics
    if dataset_format == 'grpo_offline':
        # GRPO offline format has chosen_reward and rejected_reward
        if 'chosen_reward' in df.columns:
            metrics['avg_chosen_reward'] = df['chosen_reward'].mean()
            metrics['std_chosen_reward'] = df['chosen_reward'].std()
            metrics['min_chosen_reward'] = df['chosen_reward'].min()
            metrics['max_chosen_reward'] = df['chosen_reward'].max()
        if 'rejected_reward' in df.columns:
            metrics['avg_rejected_reward'] = df['rejected_reward'].mean()
            metrics['std_rejected_reward'] = df['rejected_reward'].std()
        if 'reward_margin' in df.columns:
            metrics['avg_reward_margin'] = df['reward_margin'].mean()
            metrics['std_reward_margin'] = df['reward_margin'].std()
    elif 'reward' in df.columns:
        metrics['avg_reward'] = df['reward'].mean()
        metrics['std_reward'] = df['reward'].std()
        metrics['min_reward'] = df['reward'].min()
        metrics['max_reward'] = df['reward'].max()
    elif 'reward_totale' in df.columns:
        metrics['avg_reward'] = df['reward_totale'].mean()
        metrics['std_reward'] = df['reward_totale'].std()
        metrics['min_reward'] = df['reward_totale'].min()
        metrics['max_reward'] = df['reward_totale'].max()
    
    # GRPO offline specific metrics
    if dataset_format == 'grpo_offline':
        if 'chosen_coherence' in df.columns:
            metrics['avg_chosen_coherence'] = df['chosen_coherence'].mean()
            metrics['std_chosen_coherence'] = df['chosen_coherence'].std()
        if 'rejected_coherence' in df.columns:
            metrics['avg_rejected_coherence'] = df['rejected_coherence'].mean()
        if 'sequence_length' in df.columns:
            metrics['avg_sequence_length'] = df['sequence_length'].mean()
            metrics['std_sequence_length'] = df['sequence_length'].std()
        if 'matches_astar' in df.columns:
            metrics['astar_match_rate'] = df['matches_astar'].mean()
    
    # Sequence-specific metrics
    elif dataset_format == 'sequences':
        if 'coherence' in df.columns:
            metrics['avg_coherence'] = df['coherence'].mean()
            metrics['std_coherence'] = df['coherence'].std()
        
        if 'sequence_score_model' in df.columns:
            metrics['avg_sequence_score'] = df['sequence_score_model'].mean()
            metrics['std_sequence_score'] = df['sequence_score_model'].std()
        
        # Calculate accuracy with respect to A*
        if 'sequenza_helper_suggerita' in df.columns and 'sequenza_astar_ottimale' in df.columns:
            matches = 0
            total = 0
            for _, row in df.iterrows():
                helper_seq = parse_action_list(row['sequenza_helper_suggerita'])
                astar_seq = parse_action_list(row['sequenza_astar_ottimale'])
                if astar_seq and len(astar_seq) > 0:
                    total += 1
                    if helper_seq == astar_seq:
                        matches += 1
            
            metrics['astar_match_rate'] = matches / total if total > 0 else 0.0
    
    # Single action metrics
    if dataset_format == 'single_actions' and 'is_correct' in df.columns:
        # Convert is_correct from True/False or 0/1
        try:
            is_correct = df['is_correct'].apply(lambda x: 1 if str(x).lower() == 'true' or x == 1 else 0)
            metrics['accuracy'] = is_correct.mean()
        except:
            pass
    
    return metrics


def save_reward_evolution_plot(df, title, out_path, dataset_format='single_actions'):
    plt.figure(figsize=(14, 7))
    
    # Determine episode column name
    episode_col = 'episode' if 'episode' in df.columns else 'episodio'
    
    if dataset_format == 'grpo_offline' and 'chosen_reward' in df.columns:
        # GRPO offline format: plot chosen vs rejected rewards aggregated by episode
        episode_data = df.groupby(episode_col).agg({
            'chosen_reward': 'mean',
            'rejected_reward': 'mean'
        }).reset_index()
        
        episodes = episode_data[episode_col].values
        chosen_rewards = episode_data['chosen_reward'].values
        rejected_rewards = episode_data['rejected_reward'].values
        
        n_episodes = len(episodes)
        
        if n_episodes > 500:
            window = max(50, n_episodes // 100)
            
            # Plot raw data with high transparency
            plt.plot(episodes, chosen_rewards, alpha=0.15, color='green', linewidth=0.5, label='Chosen (raw)')
            plt.plot(episodes, rejected_rewards, alpha=0.15, color='red', linewidth=0.5, label='Rejected (raw)')
            
            # Plot smoothed trends prominently
            chosen_smooth = pd.Series(chosen_rewards).rolling(window=window, center=True, min_periods=1).mean()
            rejected_smooth = pd.Series(rejected_rewards).rolling(window=window, center=True, min_periods=1).mean()
            plt.plot(episodes, chosen_smooth, color='darkgreen', linewidth=3, 
                    alpha=0.9, label=f'Chosen (MA-{window})')
            plt.plot(episodes, rejected_smooth, color='darkred', linewidth=3, 
                    alpha=0.9, label=f'Rejected (MA-{window})')
        elif n_episodes > 100:
            window = max(10, n_episodes // 20)
            plt.plot(episodes, chosen_rewards, alpha=0.4, color='green', linewidth=1, label='Chosen')
            plt.plot(episodes, rejected_rewards, alpha=0.4, color='red', linewidth=1, label='Rejected')
            
            chosen_smooth = pd.Series(chosen_rewards).rolling(window=window, center=True, min_periods=1).mean()
            rejected_smooth = pd.Series(rejected_rewards).rolling(window=window, center=True, min_periods=1).mean()
            plt.plot(episodes, chosen_smooth, color='darkgreen', linewidth=2.5, 
                    alpha=0.9, linestyle='--', label=f'Chosen (MA-{window})')
            plt.plot(episodes, rejected_smooth, color='darkred', linewidth=2.5, 
                    alpha=0.9, linestyle='--', label=f'Rejected (MA-{window})')
        else:
            marker_style = 'o' if n_episodes < 50 else None
            markersize = 4 if n_episodes < 50 else 0
            plt.plot(episodes, chosen_rewards, marker=marker_style, markersize=markersize, 
                    alpha=0.7, label='Chosen', color='green', linewidth=2)
            plt.plot(episodes, rejected_rewards, marker=marker_style, markersize=markersize, 
                    alpha=0.7, label='Rejected', color='red', linewidth=2)
        
        plt.legend(loc='best')
        
    elif dataset_format == 'sequences' and 'reward_totale' in df.columns:
        # Group by episode and calculate mean reward
        episode_rewards = df.groupby(episode_col)['reward_totale'].mean()
        episodes = episode_rewards.index.values
        rewards = episode_rewards.values
        
        n_episodes = len(episodes)
        
        if n_episodes > 500:
            # Large dataset: show smoothed trend prominently
            window = max(50, n_episodes // 100)
            plt.plot(episodes, rewards, alpha=0.2, color='#1f77b4', linewidth=0.5, label='Raw')
            rewards_smooth = pd.Series(rewards).rolling(window=window, center=True, min_periods=1).mean()
            plt.plot(episodes, rewards_smooth, color='darkblue', linewidth=3, 
                    alpha=0.9, label=f'Moving Average (window={window})')
            plt.legend(loc='best')
        elif n_episodes > 100:
            # Medium dataset
            window = max(10, n_episodes // 20)
            plt.plot(episodes, rewards, alpha=0.5, color='#1f77b4', linewidth=1, label='Raw')
            rewards_smooth = pd.Series(rewards).rolling(window=window, center=True, min_periods=1).mean()
            plt.plot(episodes, rewards_smooth, color='darkblue', linewidth=2.5, 
                    alpha=0.9, linestyle='--', label=f'Trend (MA-{window})')
            plt.legend(loc='best')
        else:
            # Small dataset
            marker_style = 'o' if n_episodes < 50 else None
            markersize = 4 if n_episodes < 50 else 0
            plt.plot(episodes, rewards, marker=marker_style, markersize=markersize, 
                    alpha=0.7, color='#1f77b4', linewidth=2)
            
    elif 'reward' in df.columns:
        # Group by episode and calculate mean reward
        episode_rewards = df.groupby(episode_col)['reward'].mean()
        episodes = episode_rewards.index.values
        rewards = episode_rewards.values
        
        n_episodes = len(episodes)
        
        if n_episodes > 500:
            window = max(50, n_episodes // 100)
            plt.plot(episodes, rewards, alpha=0.2, color='#1f77b4', linewidth=0.5, label='Raw')
            rewards_smooth = pd.Series(rewards).rolling(window=window, center=True, min_periods=1).mean()
            plt.plot(episodes, rewards_smooth, color='darkblue', linewidth=3, 
                    alpha=0.9, label=f'Moving Average (window={window})')
            plt.legend(loc='best')
        elif n_episodes > 100:
            window = max(10, n_episodes // 20)
            plt.plot(episodes, rewards, alpha=0.5, color='#1f77b4', linewidth=1, label='Raw')
            rewards_smooth = pd.Series(rewards).rolling(window=window, center=True, min_periods=1).mean()
            plt.plot(episodes, rewards_smooth, color='darkblue', linewidth=2.5, 
                    alpha=0.9, linestyle='--', label=f'Trend (MA-{window})')
            plt.legend(loc='best')
        else:
            marker_style = 'o' if n_episodes < 50 else None
            markersize = 4 if n_episodes < 50 else 0
            plt.plot(episodes, rewards, marker=marker_style, markersize=markersize, 
                    alpha=0.7, color='#1f77b4', linewidth=2)
    else:
        print(f"No reward data found for plot: {out_path}")
        return
    
    
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Average Reward per Episode", fontsize=11)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved: {out_path}")


def save_coherence_evolution_plot(df, title, out_path, dataset_format='single_actions'):
    # Determine episode column name
    episode_col = 'episode' if 'episode' in df.columns else 'episodio'
    
    plt.figure(figsize=(14, 7))
    
    if dataset_format == 'grpo_offline' and 'chosen_coherence' in df.columns:
        # GRPO offline format
        episode_coherence = df.groupby(episode_col)['chosen_coherence'].mean()
        episodes = episode_coherence.index.values
        coherence = episode_coherence.values
    elif 'coherence' in df.columns:
        # Group by episode and calculate mean coherence
        episode_coherence = df.groupby(episode_col)['coherence'].mean()
        episodes = episode_coherence.index.values
        coherence = episode_coherence.values
    else:
        return
    
    n_episodes = len(episodes)
    
    if n_episodes > 500:
        window = max(50, n_episodes // 100)
        plt.plot(episodes, coherence, alpha=0.2, color='green', linewidth=0.5, label='Raw')
        coherence_smooth = pd.Series(coherence).rolling(window=window, center=True, min_periods=1).mean()
        plt.plot(episodes, coherence_smooth, color='darkgreen', linewidth=3, 
                alpha=0.9, label=f'Moving Average (window={window})')
        plt.legend(loc='best')
    elif n_episodes > 100:
        window = max(10, n_episodes // 20)
        plt.plot(episodes, coherence, alpha=0.5, color='green', linewidth=1, label='Raw')
        coherence_smooth = pd.Series(coherence).rolling(window=window, center=True, min_periods=1).mean()
        plt.plot(episodes, coherence_smooth, color='darkgreen', linewidth=2.5, 
                alpha=0.9, linestyle='--', label=f'Trend (MA-{window})')
        plt.legend(loc='best')
    else:
        marker_style = 'o' if n_episodes < 50 else None
        markersize = 4 if n_episodes < 50 else 0
        plt.plot(episodes, coherence, marker=marker_style, markersize=markersize, 
                alpha=0.7, color='green', linewidth=2)
    
    plt.xlabel("Episode", fontsize=11)
    plt.ylabel("Average Coherence per Episode", fontsize=11)
    plt.ylim(0, 1)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"Saved: {out_path}")


def save_bar_grpo_metrics(metrics, title, out_path):
    if not metrics:
        return
    
    # Select metrics to display
    display_metrics = {}
    
    if 'avg_reward' in metrics:
        display_metrics['Avg Reward'] = metrics['avg_reward']
    if 'avg_coherence' in metrics:
        display_metrics['Avg Coherence'] = metrics['avg_coherence']
    if 'avg_sequence_score' in metrics:
        display_metrics['Avg Seq Score'] = metrics['avg_sequence_score']
    if 'astar_match_rate' in metrics:
        display_metrics['A* Match Rate'] = metrics['astar_match_rate']
    if 'accuracy' in metrics:
        display_metrics['Accuracy'] = metrics['accuracy']
    
    if not display_metrics:
        return
    
    plt.figure(figsize=(10, 6))
    names = list(display_metrics.keys())
    values = list(display_metrics.values())
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    plt.bar(names, values, color=colors[:len(names)])
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def save_reward_distribution_plot(df, title, out_path, dataset_format='single_actions'):
    plt.figure(figsize=(10, 6))
    
    if dataset_format == 'grpo_offline' and 'reward_margin' in df.columns:
        # For GRPO, plot reward margin distribution
        rewards = df['reward_margin'].values
        plt.xlabel("Reward Margin (Chosen - Rejected)")
    elif dataset_format == 'sequences' and 'reward_totale' in df.columns:
        rewards = df['reward_totale'].values
        plt.xlabel("Reward")
    elif 'reward' in df.columns:
        rewards = df['reward'].values
        plt.xlabel("Reward")
    else:
        return
    
    plt.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    plt.ylabel("Frequency")
    plt.title(title)
    plt.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def save_sequence_length_plot(df, title, out_path):
    if 'sequenza_helper_suggerita' not in df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Calculate sequence lengths
    lengths = []
    for _, row in df.iterrows():
        seq = parse_action_list(row['sequenza_helper_suggerita'])
        lengths.append(len(seq))
    
    if not lengths:
        return
    
    plt.hist(lengths, bins=range(1, max(lengths) + 2), alpha=0.7, edgecolor='black', align='left')
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.xticks(range(1, max(lengths) + 1))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def save_metrics_report(metrics, out_path):
    if not metrics:
        return
    
    # Converti numpy types a Python natives
    report = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            report[k] = float(v)
        else:
            report[k] = v
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved: {out_path}")


def run_grpo_post_training_analysis(csv_path, plot_dir, force_degenerate=False):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return
    
    print(f"GRPO POST-TRAINING ANALYSIS")
    print(f"CSV:   {csv_path}")
    print(f"Plots: {plot_dir}")
    
    df = pd.read_csv(csv_path)
    
    # Detect dataset format
    dataset_format = detect_dataset_format(df)
    print(f"Dataset format: {dataset_format}")
    
    # Determine column names based on format
    episode_col = 'episode' if 'episode' in df.columns else 'episodio'
    env_col = 'env_name' if 'env_name' in df.columns else 'Environment'
    
    # Add map_type column
    df["map_type"] = df[env_col].apply(map_type)
    
    allow_degenerate = force_degenerate or GENERATE_ALL_PLOTS
    
    # Dataset overview
    print("\n--- DATASET OVERVIEW ---")
    print(f"Total samples: {len(df)}")
    print(f"Episodes: {df[episode_col].nunique()}")
    print(f"Environments: {df[env_col].nunique()}")
    
    grouped = df.groupby(["map_type", env_col]).size().reset_index(name='samples')
    print(grouped)
    

    print("\n[1/5] Analyzing ALL ENVIRONMENTS...")
    metrics = compute_grpo_metrics(df, dataset_format)
    
    if metrics:
        save_bar_grpo_metrics(
            metrics, "GRPO Metrics – All Environments",
            os.path.join(plot_dir, "bar_GRPO_All_Environments.png")
        )
        save_metrics_report(
            metrics,
            os.path.join(plot_dir, "metrics_GRPO_All_Environments.json")
        )
    
    # Plot evoluzione reward
    save_reward_evolution_plot(
        df, "Reward Evolution – All Environments",
        os.path.join(plot_dir, "reward_evolution_All_Environments.png"),
        dataset_format
    )
    
    # Plot distribuzione reward
    save_reward_distribution_plot(
        df, "Reward Distribution – All Environments",
        os.path.join(plot_dir, "reward_distribution_All_Environments.png"),
        dataset_format
    )
    
    # evolution plot for coherence 
    if dataset_format in ['sequences', 'grpo_offline']:
        save_coherence_evolution_plot(
            df, "Coherence Evolution – All Environments",
            os.path.join(plot_dir, "coherence_evolution_All_Environments.png"),
            dataset_format
        )
        if dataset_format == 'sequences':
            save_sequence_length_plot(
                df, "Sequence Length Distribution – All Environments",
                os.path.join(plot_dir, "sequence_length_All_Environments.png")
            )
    
    # for each map type
    print("\n[2/5] Analyzing by MAP TYPE...")
    for m in df["map_type"].unique():
        sub = df[df["map_type"] == m]
        print(f"  Processing {m}: {len(sub)} samples")
        
        if len(sub) < 5:
            print(f"Skipping {m} (not enough samples: {len(sub)})")
            continue
        
        metrics = compute_grpo_metrics(sub, dataset_format)
        
        if metrics:
            save_bar_grpo_metrics(
                metrics, f"GRPO Metrics – {m}",
                os.path.join(plot_dir, f"bar_GRPO_{m}.png")
            )
            save_metrics_report(
                metrics,
                os.path.join(plot_dir, f"metrics_GRPO_{m}.json")
            )
        
        save_reward_evolution_plot(
            sub, f"Reward Evolution – {m}",
            os.path.join(plot_dir, f"reward_evolution_{m}.png"),
            dataset_format
        )
        
        save_reward_distribution_plot(
            sub, f"Reward Distribution – {m}",
            os.path.join(plot_dir, f"reward_distribution_{m}.png"),
            dataset_format
        )
        
        if dataset_format in ['sequences', 'grpo_offline']:
            save_coherence_evolution_plot(
                sub, f"Coherence Evolution – {m}",
                os.path.join(plot_dir, f"coherence_evolution_{m}.png"),
                dataset_format
            )
            if dataset_format == 'sequences':
                save_sequence_length_plot(
                    sub, f"Sequence Length Distribution – {m}",
                    os.path.join(plot_dir, f"sequence_length_{m}.png")
                )
    
    # for each environment
    print("\n[3/5] Analyzing by ENVIRONMENT...")
    for (m, env), sub in df.groupby(["map_type", env_col]):
        print(f"  Processing {env} ({m}): {len(sub)} samples")
        
        if len(sub) < 3:
            print(f"Skipping {env}: too few samples ({len(sub)})")
            continue
        
        safe_m = str(m).replace(" ", "_")
        safe_env = str(env).replace(":", "_").replace("/", "_").replace(" ", "_")
        
        metrics = compute_grpo_metrics(sub, dataset_format)
        
        if metrics:
            save_bar_grpo_metrics(
                metrics,
                f"GRPO Metrics – {m} | {env}",
                os.path.join(plot_dir, f"bar_GRPO_{safe_m}_{safe_env}.png")
            )
            save_metrics_report(
                metrics,
                os.path.join(plot_dir, f"metrics_GRPO_{safe_m}_{safe_env}.json")
            )
        
        save_reward_evolution_plot(
            sub, f"Reward Evolution – {m} | {env}",
            os.path.join(plot_dir, f"reward_evolution_{safe_m}_{safe_env}.png"),
            dataset_format
        )
        
        save_reward_distribution_plot(
            sub, f"Reward Distribution – {m} | {env}",
            os.path.join(plot_dir, f"reward_distribution_{safe_m}_{safe_env}.png"),
            dataset_format
        )
        
        if dataset_format in ['sequences', 'grpo_offline']:
            save_coherence_evolution_plot(
                sub, f"Coherence Evolution – {m} | {env}",
                os.path.join(plot_dir, f"coherence_evolution_{safe_m}_{safe_env}.png"),
                dataset_format
            )
    
    #A* vs Helper Comparison (sequences only) 
    if dataset_format == 'sequences':
        print("\n[4/5] Analyzing HELPER vs A* COMPARISON...")
        
        # Calculate matching rate per episode
        if 'sequenza_helper_suggerita' in df.columns and 'sequenza_astar_ottimale' in df.columns:
            episode_match_rates = []
            episodes = []
            
            for ep in df[episode_col].unique():
                ep_df = df[df[episode_col] == ep]
                matches = 0
                total = 0
                
                for _, row in ep_df.iterrows():
                    helper_seq = parse_action_list(row['sequenza_helper_suggerita'])
                    astar_seq = parse_action_list(row['sequenza_astar_ottimale'])
                    if astar_seq and len(astar_seq) > 0:
                        total += 1
                        if helper_seq == astar_seq:
                            matches += 1
                
                if total > 0:
                    episode_match_rates.append(matches / total)
                    episodes.append(ep)
            
            if episode_match_rates:
                plt.figure(figsize=(12, 6))
                plt.plot(episodes, episode_match_rates, marker='o', markersize=3, alpha=0.7, color='purple')
                plt.xlabel("Episode")
                plt.ylabel("A* Match Rate")
                plt.ylim(0, 1)
                plt.title("Helper Alignment with A* Over Time")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, "astar_match_evolution_All_Environments.png"))
                plt.close()
                print(f"Saved: astar_match_evolution_All_Environments.png")
    
    print("\n[5/5] Analysis complete!")
    
    print(f"All plots saved in: {plot_dir}")


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Analizza dataset GRPO del Reviewer e genera plot")
    parser.add_argument("--csv", type=str, required=True, help="Path al CSV del dataset GRPO")
    parser.add_argument("--plots", type=str, required=True, help="Directory per salvare i plot")
    parser.add_argument("--force-degenerate", action="store_true", 
                       help="Genera plot anche con pochi dati")
    args = parser.parse_args()
    
    os.makedirs(args.plots, exist_ok=True)
    run_grpo_post_training_analysis(args.csv, args.plots, force_degenerate=args.force_degenerate)
