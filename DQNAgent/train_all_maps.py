import time
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import gymnasium as gym
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# Importiamo le classi esistenti
from agent import DQNAgent
from enviroment import DynamicMiniGridWrapper
from train_npc import make_env, detect_task_type

# Variabile globale per gestire lo sblocco dai timeout
timeout_streak = 0

# Contatori globali per statistiche Crossing (come in train_npc.py)
lava_death_count = 0
crossing_timeout_count = 0
crossing_goal_count = 0

def run_training_suite():
    # 1. SCHEDULE DEL CURRICULUM
    map_schedule = [
        # Fase 1: Mappe Vuote (Apprendimento movimento base)
        {"id": "MiniGrid-Empty-5x5-v0", "episodes": 100},
        {"id": "MiniGrid-Empty-8x8-v0", "episodes": 150},
        {"id": "MiniGrid-Empty-16x16-v0", "episodes": 300}, 
        
        # Fase 2: Lava Crossing (Apprendimento evitamento ostacoli)
        {"id": "MiniGrid-LavaCrossingS9N1-v0", "episodes": 300},
        {"id": "MiniGrid-LavaCrossingS9N3-v0", "episodes": 300},
        {"id": "MiniGrid-LavaCrossingS11N5-v0", "episodes": 300},

        # Fase 3: DoorKey (Apprendimento manipolazione oggetti)
        {"id": "MiniGrid-DoorKey-5x5-v0", "episodes": 200},
        {"id": "MiniGrid-DoorKey-8x8-v0", "episodes": 250},
        {"id": "MiniGrid-DoorKey-16x16-v0", "episodes": 300}, 
    ]

    # Inizializzazione Agente - DQNAgent standard come train_npc.py
    dummy_env = DynamicMiniGridWrapper(make_env(map_schedule[0]["id"]), "Empty")
    agent = DQNAgent(
        dummy_env.state_size, 
        dummy_env.action_size, 
        dueling=False,  # DQN standard, non dueling
        prioritized=False,
        batch_size=64
    )
    
    global_stats = []
    
    print(f"{'='*60}")
    print(f"STARTING REWRITTEN CURRICULUM TRAINING v2.1")
    print(f"{'='*60}\n")

    for idx, config in enumerate(map_schedule):
        env_id = config["id"]
        n_episodes = config["episodes"]
        task_type = detect_task_type(env_id)
        
        env_gym = make_env(env_id)
        env = DynamicMiniGridWrapper(env_gym, task_type)
        
        # --- CONFIGURAZIONE DINAMICA PARAMETRI ---
        if task_type == "Crossing":
            # Parametri ESATTI da train_npc.py
            agent.epsilon = 1.0 
            agent.epsilon_decay = 0.998 
            agent.epsilon_min = 0.05
            base_lr = 0.00025  # Default DQNAgent, come train_npc.py
            agent.gamma = 0.99
            
            # FIX: Reset contatori per OGNI mappa Crossing
            global lava_death_count, crossing_timeout_count, crossing_goal_count
            lava_death_count = 0
            crossing_timeout_count = 0
            crossing_goal_count = 0
            
            # Reset Buffer per OGNI mappa Crossing (ogni mappa ha layout diverso!)
            print(f"\n>>> LAVA CROSSING RESET: Buffer, Priorità e Contatori...")
            agent.replay_buffer.clear()
            if hasattr(agent, 'priorities'):
                agent.priorities.clear()
            
            print(f">>> Crossing config: epsilon=1.0, decay=0.998, min=0.05, gamma=0.99, lr=0.00025")

        elif task_type == "DoorKey":
            # Verifica se è la PRIMA mappa DoorKey (transizione da Crossing)
            prev_task = detect_task_type(map_schedule[idx-1]["id"]) if idx > 0 else None
            is_first_doorkey = (prev_task != "DoorKey")
            
            if is_first_doorkey:
                # Reset COMPLETO quando si entra in fase DoorKey
                print(f"\n>>> DOORKEY PHASE START: Reset buffer per nuovo task!")
                agent.replay_buffer.clear()
                if hasattr(agent, 'priorities'):
                    agent.priorities.clear()
            
            # Configurazione ANTI-OVERFITTING per DoorKey con ESPLORAZIONE BILANCIATA
            if "16x16" in env_id:
                agent.epsilon = 1.0 if is_first_doorkey else 0.85  # Esplorazione alta
                agent.epsilon_decay = 0.998  # Decay più equilibrato
                agent.epsilon_min = 0.15  # Minimo ragionevole
                base_lr = 0.0002  # LR più basso per stabilità
                agent.gamma = 0.99
                agent.batch_size = 32  # Batch size ridotto
                print(f">>> DoorKey 16x16 BALANCED: epsilon={agent.epsilon}, decay=0.998, min=0.15, batch=32")
            elif "8x8" in env_id:
                agent.epsilon = 1.0 if is_first_doorkey else 0.80
                agent.epsilon_decay = 0.998  # Decay bilanciato
                agent.epsilon_min = 0.15  # Esplorazione moderata
                base_lr = 0.0002
                agent.gamma = 0.99
                agent.batch_size = 32
                print(f">>> DoorKey 8x8 BALANCED: epsilon={agent.epsilon}, decay=0.998, min=0.15, batch=32")
            else:  # 5x5
                agent.epsilon = 1.0 if is_first_doorkey else 0.75
                agent.epsilon_decay = 0.997  # Decay moderato
                agent.epsilon_min = 0.20  # Più esplorazione per 5x5 (ambiente piccolo)
                base_lr = 0.00025
                agent.gamma = 0.98  # Gamma più alto per long-term
                agent.batch_size = 32
                print(f">>> DoorKey 5x5 BALANCED: epsilon={agent.epsilon}, decay=0.997, min=0.20, batch=32")

        elif "16x16" in env_id:
            # Setup per mappe grandi Empty: meno esplorazione casuale
            agent.epsilon = 0.8
            agent.epsilon_decay = 0.992 
            agent.epsilon_min = 0.05
            base_lr = 0.0002
            agent.gamma = 0.999
            print(f"\n>>> LARGE MAP SETUP (16x16 Empty)")

        elif "8x8" in env_id:
            # Mappe 8x8 Empty
            agent.epsilon = 0.8 
            agent.epsilon_decay = 0.996 
            agent.epsilon_min = 0.05
            base_lr = 0.00025
            agent.gamma = 0.99

        else: # Mappe 5x5 Empty
            agent.epsilon = 1.0 if idx == 0 else 0.6
            agent.epsilon_decay = 0.995 
            agent.epsilon_min = 0.10
            base_lr = 0.0003
            agent.gamma = 0.95

        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = base_lr
            
        # Scheduler solo per non-Crossing (in train_npc.py non c'è scheduler)
        if task_type == "Crossing":
            scheduler = None
        else:
            scheduler = ReduceLROnPlateau(agent.optimizer, mode='max', factor=0.5, patience=40)

        env_rewards = []
        env_success = []
        
        for e in range(n_episodes):
            run_single_episode(agent, env, env_id, task_type, global_stats, env_rewards, env_success, e, n_episodes, scheduler)

    # PART 2: Fase finale mista per consolidamento
    run_grand_final(agent, map_schedule, global_stats)

def run_single_episode(agent, env, env_id, task_type, global_stats, rewards_list, success_list, e, n_episodes, scheduler, is_mixed=False):
    global timeout_streak, lava_death_count, crossing_timeout_count, crossing_goal_count
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    episode_loss = 0
    loss_count = 0
    
    while not done:
        steps += 1
        valid_actions = env.get_valid_actions()
        
        action = agent.act(state, valid_actions)
        next_state, reward, done, info = env.step(action)
        
        # Sincronizzazione reward morte Lava con environment.py per stabilità
        if task_type == "Crossing" and done and not info.get("goal_reached", False) and not info.get("timeout", False):
            reward = -50.0 

        # Maschera valida di default per il buffer
        valid_mask = [1 if a in valid_actions else 0 for a in range(env.action_size)]
        agent.remember(state, action, reward, next_state, done, valid_mask, [1]*env.action_size)
        
        state = next_state
        total_reward += reward
        
        # Training con frequenza adattiva (ogni 8 step per DoorKey, ogni 4 per altri)
        train_freq = 8 if task_type == "DoorKey" else 4
        if steps % train_freq == 0:
            loss = agent.replay(n_steps=1)
            if loss is not None:
                episode_loss += loss
                loss_count += 1
    
    # --- GESTIONE LOGICHE POST-EPISODIO ---
    
    # Le logiche di seed/contatori NON devono partire durante la fase mixed
    if not is_mixed:
        # 1. Anti-Timeout Boost
        if info.get("timeout", False):
            timeout_streak += 1
        else:
            timeout_streak = 0
        
        if timeout_streak >= 15:
            # Se siamo bloccati da 15 episodi, scuotiamo l'esplorazione
            agent.epsilon = min(agent.epsilon + 0.25, 1.0)
            timeout_streak = 0
            print(f"    !!! TIMEOUT STREAK DETECTED: Epsilon boosted to {agent.epsilon:.2f}")

        # 2. Logica specifica Crossing (come in train_npc.py)
        if task_type == "Crossing":
            if info.get("goal_reached", False):
                crossing_goal_count += 1
            if info.get("lava_death", False):
                lava_death_count += 1
                # CRITICO: Accelera epsilon decay dopo morte nella lava (come train_npc.py)
                agent.epsilon *= 0.99  
            if info.get("timeout", False):
                crossing_timeout_count += 1

    # 3. Replay extra ogni 5 episodi (come train_npc.py) - DISABILITATO per DoorKey
    # NOTA: Il decay epsilon avviene GIÀ dentro DQNAgent.replay()
    if (e + 1) % 5 == 0 and task_type != "DoorKey":
        agent.replay(n_steps=1)
    
    # 3b. Pulizia periodica buffer per DoorKey (anti-overfitting) - ridotta frequenza
    if not is_mixed and task_type == "DoorKey" and (e + 1) % 100 == 0:
        # Mantieni solo le ultime 5000 esperienze più recenti
        if len(agent.replay_buffer) > 5000:
            # Rimuovi le esperienze più vecchie dal deque
            remove_count = len(agent.replay_buffer) - 5000
            for _ in range(remove_count):
                agent.replay_buffer.popleft()
            if hasattr(agent, 'priorities') and len(agent.priorities) > 0:
                # Rimuovi anche dalle priorities se esistono
                for _ in range(min(remove_count, len(agent.priorities))):
                    agent.priorities.popleft() if hasattr(agent.priorities, 'popleft') else agent.priorities.pop(0)
            print(f"    >>> Buffer cleanup: rimossi {remove_count} vecchi sample (anti-overfitting)")

    # 4. Log e statistiche
    avg_loss = episode_loss / loss_count if loss_count > 0 else 0
    success = 1 if info.get("goal_reached", False) else 0
    rewards_list.append(total_reward)
    success_list.append(success)
    
    if scheduler and len(rewards_list) >= 10:
        scheduler.step(np.mean(rewards_list[-10:]))
    
    global_stats.append({
        "env_id": env_id,
        "task_type": task_type,
        "reward": total_reward,
        "success": success,
        "epsilon": agent.epsilon,
        "steps": steps,
        "phase": "Mixed" if is_mixed else "Curriculum"
    })

    # Logging più frequente per Crossing (ogni 20 ep), standard per altri (ogni 50 ep)
    log_freq = 20 if task_type == "Crossing" else 50
    if (e+1) % log_freq == 0:
        prefix = "[MIXED]" if is_mixed else "    "
        # Logging dettagliato per Crossing (come in train_npc.py)
        if task_type == "Crossing" and not is_mixed:
            total_ep = e + 1
            success_rate = (crossing_goal_count / total_ep) * 100
            death_rate = (lava_death_count / total_ep) * 100
            timeout_rate = (crossing_timeout_count / total_ep) * 100
            print(f"{prefix} Ep {e+1}/{n_episodes} | Map: {env_id}")
            print(f"       R: {np.mean(rewards_list[-50:]):.1f} | Success: {success_rate:.1f}% | Deaths: {death_rate:.1f}% | Timeouts: {timeout_rate:.1f}% | Eps: {agent.epsilon:.3f}")
        else:
            print(f"{prefix} Ep {e+1}/{n_episodes} | Map: {env_id} | R: {np.mean(rewards_list[-50:]):.1f} | S: {np.mean(success_list[-50:])*100:.0f}% | Eps: {agent.epsilon:.2f}")

def run_grand_final(agent, map_schedule, global_stats):
    """Esegue un training finale su mappe casuali per consolidare l'apprendimento."""
    print(f"\n{'='*60}")
    print(f"PHASE 4: GRAND FINAL (Mixed Consolidation)")
    print(f"{'='*60}\n")
    
    all_env_ids = [c["id"] for c in map_schedule]
    agent.epsilon = 0.15 # Esplorazione bilanciata per generalizzazione
    agent.batch_size = 32  # Batch size ridotto per mixed
    
    # Abbassa LR per fine-tuning
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = 0.00001
    
    mixed_episodes = 600
    # Liste persistenti per statistiche della fase mixed
    mixed_rewards = []
    mixed_success = []
    
    for e in range(mixed_episodes):
        rid = random.choice(all_env_ids)
        tt = detect_task_type(rid)
        env = DynamicMiniGridWrapper(make_env(rid), tt)
        env.skip_seed_logic = True  # Disabilita logica seed per fase Mixed
        run_single_episode(agent, env, rid, tt, global_stats, mixed_rewards, mixed_success, e, mixed_episodes, None, is_mixed=True)
    
    # Crea cartella con timestamp per salvare modello e grafici
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = os.path.join(os.path.dirname(__file__), "Model", timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "master_model_universal.pth")
    agent.save(model_path)
    print(f"\nModello Universale salvato in: {model_path}")
    generate_plots(global_stats, model_dir)

def generate_plots(stats, output_dir="."):
    """Genera e salva i grafici di performance."""
    df = pd.DataFrame(stats)
    df.to_csv(os.path.join(output_dir, "training_stats_universal.csv"), index=False)
    sns.set_theme(style="darkgrid")
    
    # --- GRAFICO 1: Reward Trend con media mobile ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Reward Trend con smoothing (media mobile 50 episodi)
    ax1 = axes[0, 0]
    for task in df['task_type'].unique():
        task_data = df[df['task_type'] == task]['reward'].reset_index(drop=True)
        # Media mobile per smoothing
        smoothed = task_data.rolling(window=50, min_periods=1).mean()
        ax1.plot(smoothed, label=task, alpha=0.8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title("Reward Trend (Media Mobile 50 ep)")
    ax1.set_xlabel("Episodi")
    ax1.set_ylabel("Reward")
    ax1.legend()
    
    # --- GRAFICO 2: Success Rate per task type ---
    ax2 = axes[0, 1]
    for task in df['task_type'].unique():
        task_data = df[df['task_type'] == task]['success'].reset_index(drop=True)
        smoothed = task_data.rolling(window=50, min_periods=1).mean() * 100
        ax2.plot(smoothed, label=task, alpha=0.8)
    ax2.set_title("Success Rate % (Media Mobile 50 ep)")
    ax2.set_xlabel("Episodi")
    ax2.set_ylabel("Success %")
    ax2.set_ylim(0, 105)
    ax2.legend()
    
    # --- GRAFICO 3: Epsilon Decay ---
    ax3 = axes[1, 0]
    ax3.plot(df['epsilon'], alpha=0.7, color='purple')
    ax3.set_title("Epsilon Decay")
    ax3.set_xlabel("Episodi")
    ax3.set_ylabel("Epsilon")
    
    # --- GRAFICO 4: Steps per Episodio ---
    ax4 = axes[1, 1]
    for task in df['task_type'].unique():
        task_data = df[df['task_type'] == task]['steps'].reset_index(drop=True)
        smoothed = task_data.rolling(window=50, min_periods=1).mean()
        ax4.plot(smoothed, label=task, alpha=0.8)
    ax4.set_title("Steps per Episodio (Media Mobile 50 ep)")
    ax4.set_xlabel("Episodi")
    ax4.set_ylabel("Steps")
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "universal_metrics_overview.png"), dpi=150)
    plt.close()
    
    # --- GRAFICO SEPARATO: Reward Medio per Task ---
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = df.groupby('task_type').agg({
        'reward': 'mean'
    }).reset_index()
    summary = summary.sort_values('reward', ascending=False)
    
    # Colori distinti per ogni task
    task_colors_map = {'Empty': 'royalblue', 'Crossing': 'crimson', 'DoorKey': 'forestgreen'}
    colors = [task_colors_map.get(task, 'steelblue') for task in summary['task_type']]
    
    bars = ax.bar(summary['task_type'], summary['reward'], alpha=0.8, color=colors)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Reward Medio', fontsize=12)
    ax.set_title("Reward Medio per Task Type", fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "universal_reward_by_task.png"), dpi=150)
    plt.close()
    
    # --- GRAFICO SEPARATO: Success Rate per Task ---
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = df.groupby('task_type').agg({
        'success': 'mean'
    }).reset_index()
    summary['success'] = summary['success'] * 100
    summary = summary.sort_values('success', ascending=False)
    
    # Colori distinti per ogni task
    task_colors_map = {'Empty': 'royalblue', 'Crossing': 'crimson', 'DoorKey': 'forestgreen'}
    colors = [task_colors_map.get(task, 'green') for task in summary['task_type']]
    
    bars = ax.bar(summary['task_type'], summary['success'], alpha=0.8, color=colors)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title("Success Rate per Task Type", fontsize=14)
    ax.set_ylim(0, 105)
    
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "universal_success_by_task.png"), dpi=150)
    plt.close()
    
    # --- GRAFICO SEPARATO: Steps Medi per Task ---
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = df.groupby('task_type').agg({
        'steps': 'mean'
    }).reset_index()
    summary = summary.sort_values('steps', ascending=True)
    
    # Colori distinti per ogni task
    task_colors_map = {'Empty': 'royalblue', 'Crossing': 'crimson', 'DoorKey': 'forestgreen'}
    colors = [task_colors_map.get(task, 'orange') for task in summary['task_type']]
    
    bars = ax.bar(summary['task_type'], summary['steps'], alpha=0.8, color=colors)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_ylabel('Steps Medi', fontsize=12)
    ax.set_title("Steps Medi per Task Type", fontsize=14)
    
    # Aggiungi valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "universal_steps_by_task.png"), dpi=150)
    plt.close()
    
    # --- GRAFICO SEPARATO: Reward Curriculum per Task Type ---
    fig, ax = plt.subplots(figsize=(14, 6))
    curriculum_df = df[df['phase'] == 'Curriculum']
    
    # Colori per task types
    task_colors = {
        'Empty': 'royalblue',
        'Crossing': 'crimson',
        'DoorKey': 'forestgreen'
    }
    
    if len(curriculum_df) > 0:
        episode_counter = 0
        for task in ['Empty', 'Crossing', 'DoorKey']:
            task_data = curriculum_df[curriculum_df['task_type'] == task]
            if len(task_data) > 0:
                rewards = task_data['reward'].reset_index(drop=True)
                smoothed = rewards.rolling(window=50, min_periods=1).mean()
                x_range = range(episode_counter, episode_counter + len(smoothed))
                ax.plot(x_range, smoothed, label=f'{task}', 
                       alpha=0.8, linewidth=2, color=task_colors[task])
                episode_counter += len(smoothed)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title("Reward Curriculum Learning per Task Type")
    ax.set_xlabel("Episodi")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "universal_curriculum_by_task.png"), dpi=150)
    plt.close()
    
    # --- GRAFICO SEPARATO: Reward Fase Mixed ---
    fig, ax = plt.subplots(figsize=(12, 6))
    mixed_df = df[df['phase'] == 'Mixed']
    
    if len(mixed_df) > 0:
        mixed_rewards = mixed_df['reward'].reset_index(drop=True)
        mixed_smooth = mixed_rewards.rolling(window=50, min_periods=1).mean()
        ax.plot(mixed_smooth, alpha=0.8, linewidth=2, color='purple', label='Mixed Phase')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title("Reward Mixed Consolidation Phase")
        ax.set_xlabel("Episodi Mixed")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "universal_mixed_phase.png"), dpi=150)
    plt.close()
    
    print(f"\nGrafici generati in {output_dir}:")
    print("  - training_stats_universal.csv")
    print("  - universal_metrics_overview.png (4 grafici: Reward, Success, Epsilon, Steps)")
    print("  - universal_reward_by_task.png (Reward medio per task)")
    print("  - universal_success_by_task.png (Success rate per task)")
    print("  - universal_steps_by_task.png (Steps medi per task)")
    print("  - universal_curriculum_by_task.png (Reward curriculum colorato per task)")
    print("  - universal_mixed_phase.png (Reward fase mixed)")

if __name__ == "__main__":
    run_training_suite()