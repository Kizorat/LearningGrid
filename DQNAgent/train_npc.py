# train_npc.py
import argparse
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper
from enviroment import DynamicMiniGridWrapper
from agent import DQNAgent
import csv
import time


def detect_task_type(env_id):
    if "DoorKey" in env_id:
        return "DoorKey"
    if "Crossing" in env_id or "LavaCrossing" in env_id:
        return "Crossing"
    return "Empty"


def make_env(env_id, render=False):
    env = gym.make(env_id, render_mode=("human" if render else None))
    env = ImgObsWrapper(env)     
    return env


def train_npc(task_name, episodes=300, batch_size=64,
              replay_every=5, turbo=False, render=False, csv_out=None):


    env_gym = make_env(task_name, render=render)
    task_type = detect_task_type(task_name)
    env = DynamicMiniGridWrapper(env_gym, task_type)


    agent = DQNAgent(env.state_size, env.action_size)
    agent.batch_size = batch_size

    #Se esiste un CSV
    if csv_out:
        csvfile = open(csv_out, "w", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "step", "action", "reward", "done"])
    else:
        writer = None



    for e in range(episodes):

        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            step += 1
            agent.replay()

            # Scelta dell'azione valida
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions)

            # step env
            next_state, reward, done, info = env.step(action)
            total_reward += reward


            agent.remember(state, action, reward, next_state, done)
            state = next_state


            if render:
                env.render()
                time.sleep(0.02)

            # log to CSV
            if writer:
                writer.writerow([e+1, step, action, reward, done])


        #Per il replay
        if (e + 1) % replay_every == 0:
            if turbo:
                agent.replay(n_steps=8)
            else:
                agent.replay(n_steps=1)

        print(f"[{task_name}] Episode {e+1}/{episodes} | Reward {total_reward:.2f} | Epsilon {agent.epsilon:.3f}")

    if csv_out:
        csvfile.close()

    return agent, env





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_every", type=int, default=5)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    tasks = [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-LavaCrossingS9N1-v0"
    ]

    for task in tasks:
        print(f"\n===== Training task: {task} =====")

        csv_path = f"{task.replace(':','_')}_log.csv" if args.csv else None

        train_npc(
            task,
            episodes=args.episodes,
            batch_size=args.batch_size,
            replay_every=args.replay_every,
            turbo=args.turbo,
            render=args.render,
            csv_out=csv_path
        )
