"""
Versione ottimizzata:
- Pianificazione A* + conversione path->azioni (con rotazioni corrette)
- LLM opzionale + prompt esplicito + ASCII grid
- Minimizzazione delle chiamate LLM e debug output opzionale
- Minor frequenza di replay per velocità
"""

import argparse
import time
import json
import heapq
import numpy as np
import torch
import re
import ollama
import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
from DQNAgent.agent import DQNAgent
from DQNAgent.enviroment import DynamicMiniGridWrapper
import matplotlib.pyplot as plt

# ---------------- Config ----------------
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
USE_LLAMA_CPP = False   # se hai llama_cpp configurato, puoi attivarlo e modificare LLMHelper di conseguenza
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VALID_ACTIONS = ["forward", "left", "right"]
DIR2VEC = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}  # 0=east,1=south,2=west,3=north

# ---------------- Utils ----------------
def flatten_obs(obs):
    if isinstance(obs, dict) and 'image' in obs:
        return obs['image'].astype(np.float32).ravel() / 255.0
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            if isinstance(v, np.ndarray):
                parts.append(v.ravel())
            elif isinstance(v, (int, float)):
                parts.append(np.array([v]))
        return np.concatenate(parts).astype(np.float32)
    return np.array(obs, dtype=np.float32).ravel()

def simulate_actions(agent_pos, agent_dir, actions, grid):
    pos = list(agent_pos)
    dir_ = int(agent_dir)
    for a in actions:
        if a == "left":
            dir_ = (dir_-1)%4
        elif a == "right":
            dir_ = (dir_+1)%4
        elif a == "forward":
            dx, dy = DIR2VEC[dir_]
            nx, ny = pos[0]+dx, pos[1]+dy
            if not (0<=nx<grid.width and 0<=ny<grid.height):
                return False
            cell = grid.get(nx,ny)
            if cell is not None and getattr(cell, "type", None) in ["wall","lava"]:
                return False
            pos = [nx, ny]
        else:
            return False
    return True

# ---------------- A* pathfinder (grid cells only) ----------------
import heapq

import heapq

def astar_path(grid, start, goal):
    """
    Trova la prima cella libera verso il goal muovendosi SOLO in orizzontale
    o verticale. Da quella cella parte A*.
    Il path restituito inizia sempre dallo start.
    """

    W, H = grid.width, grid.height

    def passable(x, y):
        if not (0 <= x < W and 0 <= y < H):
            return False
        c = grid.get(x, y)
        return (c is None) or (getattr(c, "type", None) not in ["wall", "lava"])

    sx, sy = start
    gx, gy = goal

    # ------------------------------------------------------------
    # 1. Scegliamo l’asse su cui muoverci
    # ------------------------------------------------------------
    dx = 0
    dy = 0

    if abs(gx - sx) >= abs(gy - sy):
        # Muoviamoci lungo X
        dx = 1 if gx > sx else -1
        dy = 0
    else:
        # Muoviamoci lungo Y
        dx = 0
        dy = 1 if gy > sy else -1

    # ------------------------------------------------------------
    # 2. Cerca la prima cella libera lungo l’asse scelto
    # ------------------------------------------------------------
    cur_x, cur_y = sx, sy
    first_free = None

    while True:
        cur_x += dx
        cur_y += dy

        # fuori mappa → niente path valido
        if not (0 <= cur_x < W and 0 <= cur_y < H):
            return []

        if passable(cur_x, cur_y):
            first_free = (cur_x, cur_y)
            break

    if first_free is None:
        return []

    # ------------------------------------------------------------
    # 3. A* parte da first_free
    # ------------------------------------------------------------
    new_start = first_free

    open_heap = []
    heapq.heappush(open_heap, (0, new_start[0], new_start[1]))
    came_from = {new_start: None}
    gscore = {new_start: 0}

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------
    # 4. A*
    # ------------------------------------------------------------
    while open_heap:
        _, x, y = heapq.heappop(open_heap)

        if (x, y) == (gx, gy):
            # Ricostruzione
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()

            # Prepend start
            return [start] + path

        for mx, my in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + mx, y + my
            if not passable(nx, ny):
                continue

            tentative = gscore[(x, y)] + 1
            if (nx, ny) not in gscore or tentative < gscore[(nx, ny)]:
                gscore[(nx, ny)] = tentative
                priority = tentative + h((nx, ny), (gx, gy))
                heapq.heappush(open_heap, (priority, nx, ny))
                came_from[(nx, ny)] = (x, y)

    return []



# ---------------- conversione path->azioni con rotazioni corrette ----------------
def rotate_to(curr_dir, desired_dir):
    """Restituisce lista di 'left'/'right' per ruotare dal curr_dir al desired_dir
       usando il numero minimo di rotazioni (90deg)."""
    diff = (desired_dir - curr_dir) % 4
    if diff == 0:
        return []
    if diff == 1:
        return ["right"]
    if diff == 2:
        # due rotazioni equivalenti; scegli left,left (o right,right). qui uso right,right
        return ["right","right"]
    if diff == 3:
        return ["left"]
    return []

def front_cell(grid, pos, direction):
    DIR2VEC = {0: (1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
    dx, dy = DIR2VEC[direction]
    fx, fy = pos[0] + dx, pos[1] + dy
    if not (0 <= fx < grid.width and 0 <= fy < grid.height):
        return fx, fy, "wall"
    cell = grid.get(fx, fy)
    ctype = "empty" if cell is None else cell.type
    return fx, fy, ctype

def pos_to_dir(from_pos, to_pos):
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    for d,(vx,vy) in DIR2VEC.items():
        if (dx,dy) == (vx,vy):
            return d
    return None

def path_to_actions(path, start_dir):
    """path: list di pos (x,y). start_dir: direzione corrente dell'agente.
       Restituisce lista azioni ["left","forward",...] che seguono il path."""
    if not path or len(path) == 1:
        return []
    actions = []
    dir_ = start_dir
    for i in range(len(path)-1):
        cur = path[i]
        nxt = path[i+1]
        desired_dir = pos_to_dir(cur, nxt)
        if desired_dir is None:
            # salto anomalia
            continue
        actions += rotate_to(dir_, desired_dir)
        actions += ["forward"]
        dir_ = desired_dir
    return actions

# ---------------- ASCII Grid builder ----------------
def build_ascii_grid(grid, agent_pos, goal_pos):
    rows = []
    for y in range(1, grid.height - 1):  # Skip top and bottom rows (walls)
        row = []
        for x in range(1, grid.width - 1):  # Skip left and right columns (walls)
            if (x,y) == tuple(agent_pos):
                row.append("A")
            elif (x,y) == tuple(goal_pos):
                row.append("G")
            else:
                c = grid.get(x,y)
                if c is None:
                    row.append(".")
                else:
                    t = getattr(c,"type", "?")
                    if t == "wall": row.append("#")
                    elif t == "lava": row.append("~")
                    else: row.append(".")
        rows.append(" ".join(row))
    # ritorna stringa con y crescente (top->bottom)
    return "\n".join(rows)

# ---------------- LLM Helper (ottimizzato) ----------------
class LLMHelper:
    def __init__(self, model_name="mistral:7b", verbose=False):
        self.verbose = verbose
        self.model_name = model_name

    def build_prompt(self, grid_ascii, agent_pos, agent_dir, goal_pos, mid_cell, max_actions, grid):
        path = astar_path(grid, tuple(agent_pos), tuple(goal_pos))
        mid_cell_str = f"Key mid cell: {mid_cell}" if mid_cell is not None else ""
        
        # Convert numpy types to Python int
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        agent_dir = int(agent_dir)
        agent_dir_str = ["east", "south", "west", "north"][agent_dir]
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        path = [(int(p[0]), int(p[1])) for p in path]
        
        # Convert ASCII grid into structured JSON grid
        # Assumes grid_ascii is something like:
        # ". . .\n. A .\n. . G"
        rows = [
            [cell for cell in row.split()]
            for row in grid_ascii.strip().split("\n")
        ]

        # Replace agent symbol with direction indicator
        agent_dir_symbols = {0: ">", 1: "v", 2: "<", 3: "^"}  # 0=east, 1=south, 2=west, 3=north
        agent_symbol = agent_dir_symbols.get(agent_dir, "A")
        
        for i, row in enumerate(rows):
            for j, cell in enumerate(row):
                if cell == "A":
                    rows[i][j] = agent_symbol
                    break

        # Build JSON structure with single-line rows
        grid_json = {
            "rows": len(rows),
            "cols": len(rows[0]) if rows else 0,
            "grid": [" ".join(row) for row in rows]  # Join each row into a single string
        }

        prompt = (
            "You are a controller that must produce the shortest valid action sequence for an agent moving in a grid (positions = columns, rows).\n"
            "Your output must be deterministic, minimal and contain only the final list of actions.\n"
            f"Grid ASCII JSON (A=agent, G=goal, #=wall, ~=lava, .=free):\n{json.dumps(grid_json, indent=2)}\n\n"
            f"Agent position: {agent_pos}, direction: {agent_dir_str}\n"
            f"Goal position: {goal_pos}\n"
            f"{mid_cell_str}\n"
            f"Valid actions: ['forward', 'left', 'right'] (left and right 90° turn the agent's direction without moving it)\n"
            f"Path from agent to goal (cells): {path}\n"
            f"Max suggested actions: {max_actions}\n\n"
            "IMPORTANT:\n"
            "- Output ONLY a Python list of actions, e.g. ['left','forward','right']\n"
            "- Do not include explanations.\n"
            "Example grid:\n"
            ". . .\n"
            ". ^ .\n"
            ". . G\n"
            "Example output:\n"
            "['right', 'forward', 'right', 'forward']\n"
        )
        print(prompt)
        return prompt

    def suggest_actions(self, grid, agent_pos, agent_dir, goal_pos, mid_cell, max_actions=10):
        grid_ascii = build_ascii_grid(grid, agent_pos, goal_pos)
        prompt = self.build_prompt(grid_ascii, agent_pos, agent_dir, goal_pos, mid_cell, max_actions, grid)

        if self.verbose:
            print("--- PROMPT ---\n", prompt)

        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt
            )
            raw = response["response"]
            print(raw)
            if self.verbose:
                print("--- RAW LLM RESPONSE ---\n", raw)
        except Exception as e:
            print("Errore Ollama:", e)
            return []

        # Parsing Python list
        m = re.search(r"\[([^\]]*)\]", raw)
        if not m:
            return []

        tokens = [t.strip().strip("'\"") for t in m.group(1).split(",")]
        valid = ["forward", "left", "right"]

        seq = [tok for tok in tokens if tok in valid][:max_actions]

        # Safety check
        safe = []
        sim_pos = tuple(agent_pos)
        sim_dir = int(agent_dir)
        for s in seq:
            if simulate_actions(sim_pos, sim_dir, [s], grid):
                safe.append(s)
                if s == "forward":
                    dx,dy = DIR2VEC[sim_dir]
                    sim_pos = (sim_pos[0] + dx, sim_pos[1] + dy)
                elif s == "left":
                    sim_dir = (sim_dir - 1) % 4
                elif s == "right":
                    sim_dir = (sim_dir + 1) % 4
            else:
                break

        return safe

# ---------------- Training Loop ----------------
def train(env_wrapper, episodes=5, use_llm_helper=True, render_env=False,
          max_steps_before_llm=4, max_llm_actions_factor=2, replay_frequency=4, verbose=False):
    
    example_state = flatten_obs(env_wrapper.get_state())
    agent = DQNAgent(len(example_state), env_wrapper.action_size)
    helper = LLMHelper(verbose=verbose) if use_llm_helper else None
    
    stats = {"episode_rewards":[],"episode_moves":[],"llm_suggestions_used":[],"llm_hallucinations":[]}
    
    for ep in range(episodes):
        state_raw = env_wrapper.reset()
        state = flatten_obs(state_raw)
        done = False
        total_reward = 0
        moves = 0
        steps_since_llm = 0
        ep_sugg = 0
        ep_hall = 0
        step_counter = 0
        
        while not done:
            valid_actions = env_wrapper.get_valid_actions()
            use_llm = use_llm_helper and steps_since_llm >= max_steps_before_llm

            if use_llm:
                agent_pos = env_wrapper.base_env.agent_pos
                agent_dir = env_wrapper.base_env.agent_dir
                goal_pos = env_wrapper.base_env.goal_pos
                mid_cell,_ = find_mid_cell_and_path(env_wrapper.base_env.grid, agent_pos, goal_pos)

                plan_actions = []

                # 1) Chiamata principale all'LLM
                if helper is not None:
                    max_actions = max(1,int(max_llm_actions_factor*(abs(agent_pos[0]-goal_pos[0])+abs(agent_pos[1]-goal_pos[1])))) + 1
                    plan_actions = helper.suggest_actions(env_wrapper.base_env.grid, agent_pos, agent_dir, goal_pos, mid_cell, max_actions=max_actions)

                # 2) LLM non produce azioni valide
                if not plan_actions:
                    print("LLM non ha suggerito azioni valide.")
                    ep_hall += 1

                # 3) applica le azioni pianificate (o single-step fallback)
                for s in plan_actions:
                    action_map={"left":0,"right":1,"forward":2}
                    if s not in action_map:
                        continue
                    action = action_map[s]

                    ok = simulate_actions(env_wrapper.base_env.agent_pos, env_wrapper.base_env.agent_dir, [s], env_wrapper.base_env.grid)
                    if not ok:
                        # Hallucination: l'azione non è eseguibile nello stato reale
                        ep_hall += 1
                        action = agent.act(state, valid_actions)
                    else:
                        ep_sugg += 1

                    next_state_raw, reward, done, info = env_wrapper.step(action)
                    next_state = flatten_obs(next_state_raw)
                    valid_mask = [1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                    next_valid_actions=env_wrapper.get_valid_actions()
                    next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                    agent.remember(state, action, reward, next_state, done, valid_mask, next_valid_mask)
                    
                    # replay a intervalli per velocità
                    if step_counter % replay_frequency == 0:
                        agent.replay()
                    
                    state = next_state
                    total_reward += reward
                    moves += 1
                    step_counter += 1

                    if done: break

                steps_since_llm = 0  # reset dopo aver usato LLM/A*
            else:
                # DQN step normale
                action = agent.act(state, valid_actions)
                next_state_raw, reward, done, info = env_wrapper.step(action)
                next_state = flatten_obs(next_state_raw)
                valid_mask = [1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                next_valid_actions=env_wrapper.get_valid_actions()
                next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                agent.remember(state, action, reward, next_state, done, valid_mask, next_valid_mask)
                if step_counter % replay_frequency == 0:
                    agent.replay()
                state = next_state
                total_reward += reward
                moves += 1
                step_counter += 1
                steps_since_llm += 1

            if render_env and moves%5==0:
                env_wrapper.render()
                time.sleep(0.01)

        stats["episode_rewards"].append(total_reward)
        stats["episode_moves"].append(moves)
        stats["llm_suggestions_used"].append(ep_sugg)
        stats["llm_hallucinations"].append(ep_hall)
        print(f"Ep {ep+1}/{episodes}: reward={total_reward:.2f}, moves={moves}, llm_sugg={ep_sugg}, llm_hall={ep_hall}")
    
    return stats

# ---------------- Mid-cell finder (mantenuto) ----------------
def find_mid_cell_and_path(grid, agent_pos, goal_pos):
    safe_cells = [(x,y) for x in range(grid.width) for y in range(grid.height)
                  if grid.get(x,y) is None or getattr(grid.get(x,y),"type",None) not in ["wall","lava"]]
    
    key_mid_cells = []
    for x,y in safe_cells:
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<grid.width and 0<=ny<grid.height:
                c = grid.get(nx,ny)
                if c is not None and getattr(c,"type",None)=="lava":
                    key_mid_cells.append((x,y))
                    break
    mid_cell = None
    min_neighbors = 9
    for x,y in key_mid_cells:
        neighbors = sum((x+dx,y+dy) in safe_cells for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])
        if neighbors<min_neighbors:
            min_neighbors = neighbors
            mid_cell = (x,y)
    return mid_cell, []

# ---------------- Plot ----------------
def plot_stats(stats):
    plt.figure(figsize=(8,4))
    plt.plot(stats["episode_rewards"])
    plt.title("Rewards per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("plots\\rewards.png")
    
    plt.figure(figsize=(8,4))
    plt.plot(stats["episode_moves"])
    plt.title("Moves per episode")
    plt.xlabel("Episode")
    plt.ylabel("Moves")
    plt.tight_layout()
    plt.savefig("plots\\moves.png")
    
    plt.figure(figsize=(8,4))
    plt.plot(stats["llm_suggestions_used"], label="sugg used")
    plt.plot(stats["llm_hallucinations"], label="hallucinations")
    plt.legend()
    plt.title("LLM suggestions / hallucinations")
    plt.tight_layout()
    plt.savefig("plots\\llm_stats.png")

# ---------------- Main ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10, help="massimo step tra chiamate LLM")
    parser.add_argument("--replay-freq", type=int, default=4, help="quante volte chiamare agent.replay() (meno = più veloce)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    

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

    env = gym.make(env_id, render_mode="human" if args.render else None)
    env = FullyObsWrapper(env)
    env_wrapper = DynamicMiniGridWrapper(env, task_type="Empty")
    
    stats = train(
        env_wrapper, 
        episodes=args.episodes, 
        use_llm_helper=args.llm, 
        render_env=args.render, 
        max_steps_before_llm=args.max_steps,
        replay_frequency=args.replay_freq,
        verbose=args.verbose
    )
    plot_stats(stats)
