import argparse
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from enviroment import DynamicMiniGridWrapper
from agent import DQNAgent
import csv
import time

# Rileva il tipo di task in base all'ID dell'ambiente
def detect_task_type(env_id):
    if "DoorKey" in env_id:
        return "DoorKey"
    if "Crossing" in env_id or "LavaCrossing" in env_id:
        return "Crossing"
    return "Empty"

# Crea l'ambiente con la giusta wrapper
def make_env(env_id, render=False):
    env = gym.make(env_id, render_mode=("human" if render else None))
    env = FullyObsWrapper(env) # Osservazioni complete
    return env

# Funzione principale di training
def train_npc(task_name, episodes=300, batch_size=64,
              replay_every=5, turbo=False, render=False, csv_out=None, dueling=True, prioritized=False):


    env_gym = make_env(task_name, render=render)
    task_type = detect_task_type(task_name)
    env = DynamicMiniGridWrapper(env_gym, task_type)


    agent = DQNAgent(env.state_size, env.action_size, dueling=dueling, prioritized=prioritized)
    agent.batch_size = batch_size
    
    # Modifiche specifiche per task
    if task_type == "Crossing":
        # Crossing: Prudenza massima, decay lento
        agent.epsilon_decay = 0.998  
        agent.epsilon_min = 0.05
        print(f"Crossing mode: epsilon_decay={agent.epsilon_decay}, epsilon_min={agent.epsilon_min}")
    
    elif task_type == "Empty":
        agent.epsilon_decay = 0.9995 
        agent.epsilon_min = 0.05
        print(f"Empty mode: epsilon_decay={agent.epsilon_decay}, epsilon_min={agent.epsilon_min}")
        
    else:
        if "16x16" in env_gym.spec.id:
            agent.epsilon_decay = 0.99995
            agent.epsilon_min = 0.05
            print(f"DoorKey mode: epsilon_decay={agent.epsilon_decay}, epsilon_min={agent.epsilon_min}")
        else:
            # DoorKey e altri
            agent.epsilon_decay = 0.995
            agent.epsilon_min = 0.10
            print(f"DoorKey mode: epsilon_decay={agent.epsilon_decay}, epsilon_min={agent.epsilon_min}")

    #Se esiste un CSV
    if csv_out:
        csvfile = open(csv_out, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "step", "action", "reward", "done"])
    else:
        writer = None

    # Tracking per DoorKey
    goal_count = 0
    phase_1_count = 0  # Chiave raccolta
    phase_2_count = 0  # Porta aperta
    
    # Tracking per Crossing 
    lava_death_count = 0
    timeout_count = 0

    for e in range(episodes):

        state = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Tracking fasi per questo episodio
        episode_phase1_done = False
        episode_phase2_done = False

        while not done:
            step += 1
            # Replay solo ogni 4 step per stabilità
            if step % 4 == 0:
                agent.replay()

            # Scelta dell'azione valida
            valid_actions = env.get_valid_actions()

            valid_mask = [1 if a in valid_actions else 0 for a in range(env.action_size)]
            action = agent.act(state, valid_actions)

            # step env
            next_state, reward, done, info = env.step(action)
            total_reward += reward

            next_valid_actions=env.get_valid_actions()
            next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(env.action_size)]


            agent.remember(state, action, reward, next_state, done, valid_mask,next_valid_mask)
            state = next_state


            if render:
                env.render()
                time.sleep(0.02)

            # log to CSV
            if writer:
                writer.writerow([e+1, step, action, reward, done])
            
            # Tracking fasi DoorKey 
            if task_type == "DoorKey":
                if info.get("phase_1_complete", False):
                    episode_phase1_done = True
                if info.get("phase_2_complete", False):
                    episode_phase2_done = True

        # Conteggio fasi una volta per episodio
        if task_type == "DoorKey":
            if episode_phase1_done:
                phase_1_count += 1
            if episode_phase2_done:
                phase_2_count += 1

        # Conteggio successi
        if info.get("goal_reached", False):
            goal_count += 1
        
        # Conteggio morti e timeout per Crossing
        if task_type == "Crossing":
            if info.get("lava_death", False):
                lava_death_count += 1
                # Accelera epsilon decay dopo morte nella lava
                agent.epsilon *= 0.99  
            if info.get("timeout", False):
                timeout_count += 1

        #Per il replay
        if (e + 1) % replay_every == 0:
            if turbo:
                agent.replay(n_steps=8)
            else:
                agent.replay(n_steps=1)

        # Log con metriche specifiche per task
        if task_type == "DoorKey" and (e + 1) % 20 == 0:
            success_rate = (goal_count / (e + 1)) * 100
            phase1_rate = (phase_1_count / (e + 1)) * 100
            phase2_rate = (phase_2_count / (e + 1)) * 100
            print(f"[{task_name}] Episode {e+1}/{episodes}")
            print(f"Reward: {total_reward:.2f}")
            print(f"Success Rate (Goal): {success_rate:.1f}%")
            print(f"Phase 1 (Key): {phase1_rate:.1f}%")
            print(f"Phase 2 (Door): {phase2_rate:.1f}%")
            print(f"Epsilon: {agent.epsilon:.3f}")
        elif task_type == "Crossing" and (e + 1) % 20 == 0:
            success_rate = (goal_count / (e + 1)) * 100
            death_rate = (lava_death_count / (e + 1)) * 100
            timeout_rate = (timeout_count / (e + 1)) * 100
            print(f"[{task_name}] Episode {e+1}/{episodes}")
            print(f"Reward: {total_reward:.2f}")
            print(f"Success Rate (Goal): {success_rate:.1f}%")
            print(f"Lava Deaths: {death_rate:.1f}%")
            print(f"Timeouts: {timeout_rate:.1f}%")
            print(f"Epsilon: {agent.epsilon:.3f}")
        else:
            print(f"[{task_name}] Episode {e+1}/{episodes} | Reward {total_reward:.2f} | Epsilon {agent.epsilon:.3f}")

    if csv_out:
        csvfile.close()

    # Riepilogo finale per DoorKey
    if task_type == "DoorKey":
        final_success_rate = (goal_count / episodes) * 100
        final_phase1_rate = (phase_1_count / episodes) * 100
        final_phase2_rate = (phase_2_count / episodes) * 100
        
        print(f"\n{'='*60}")
        print(f"RIEPILOGO TRAINING DoorKey")
        print(f"{'='*60}")
        print(f"Ambiente: {task_name}")
        print(f"Episodi totali: {episodes}")
        print(f"├─ Goal raggiunti: {goal_count}/{episodes} ({final_success_rate:.1f}%)")
        print(f"├─ Fase 1 (Chiave): {phase_1_count}/{episodes} ({final_phase1_rate:.1f}%)")
        print(f"├─ Fase 2 (Porta): {phase_2_count}/{episodes} ({final_phase2_rate:.1f}%)")
        print(f"└─ Epsilon finale: {agent.epsilon:.3f}")
        print(f"{'='*60}\n")
    
    # Riepilogo finale per Crossing
    if task_type == "Crossing":
        final_success_rate = (goal_count / episodes) * 100
        final_death_rate = (lava_death_count / episodes) * 100
        final_timeout_rate = (timeout_count / episodes) * 100
        
        print(f"\n{'='*60}")
        print(f"RIEPILOGO TRAINING LavaCrossing")
        print(f"{'='*60}")
        print(f"Ambiente: {task_name}")
        print(f"Episodi totali: {episodes}")
        print(f"Goal raggiunti: {goal_count}/{episodes} ({final_success_rate:.1f}%)")
        print(f"Morti nella lava: {lava_death_count}/{episodes} ({final_death_rate:.1f}%)")
        print(f"Timeout: {timeout_count}/{episodes} ({final_timeout_rate:.1f}%)")
        print(f"Epsilon finale: {agent.epsilon:.3f}")
        print(f"{'='*60}\n")

    return agent, env




# Main function per parsing argomenti
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_every", type=int, default=5)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--dueling", action="store_true", default=True, help="Usa Dueling DQN")
    parser.add_argument("--prioritized", action="store_true", help="Usa Prioritized Experience Replay")
    parser.add_argument("--curriculum", action="store_true", help="Usa Curriculum Learning (Empty -> LavaCrossing)")
    args = parser.parse_args()

    # Se curriculum è attivo, importa e lancia curriculum_train
    if args.curriculum:
        print("\n Attivazione CURRICULUM LEARNING\n")
        from curriculum_train import curriculum_learning
        curriculum_learning(
            dueling=args.dueling,
            prioritized=args.prioritized,
            render=args.render
        )
        exit(0)

    env_options = {
    "1": {
        "label": "Crossing",
        "ids": [
            "MiniGrid-LavaCrossingS9N1-v0",
            "MiniGrid-LavaCrossingS9N3-v0",
            "MiniGrid-LavaCrossingS11N5-v0",
        ],
    },
    "2": {
        "label": "DoorKey",
        "ids": [
            "MiniGrid-DoorKey-5x5-v0",
            "MiniGrid-DoorKey-8x8-v0",
            "MiniGrid-DoorKey-16x16-v0",
        ],
    },
    "3": {
        "label": "Empty",
        "ids": [
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-Empty-8x8-v0",
            "MiniGrid-Empty-16x16-v0",
        ],
    },
}

    # Scelta dell'ambiente
    while True:
        choice = input("Scegli l'ambiente (1=Crossing, 2=DoorKey, 3=Empty): ").strip()
        if choice in env_options:
            break
        print("Scelta non valida, inserisci 1, 2 o 3.")

        # Scelta della dimensione
    while True:
        size = input("Scegli la dimensione della mappa (1=piccolo, 2=medio, 3=grande): ").strip()
        if size in ("1", "2", "3"):
            break
        print("Scelta non valida, inserisci 1, 2 o 3.")

    env_id = env_options[choice]["ids"][int(size) - 1]
    print(f"Ambiente selezionato: {env_options[choice]['label']} -> {env_id}")



    csv_path = f"{env_id.replace(':','_')}_log.csv" if args.csv else None

    train_npc(
        env_id,
        episodes=args.episodes,
        batch_size=args.batch_size,
        replay_every=args.replay_every,
        turbo=args.turbo,
        render=args.render,
        csv_out=csv_path,
        dueling=args.dueling,
        prioritized=args.prioritized
    )
