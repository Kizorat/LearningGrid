"""
Versione ottimizzata:
- Pianificazione A* + conversione path->azioni (con rotazioni corrette)
- LLM opzionale + prompt esplicito + ASCII grid
- Minimizzazione delle chiamate LLM e debug output opzionale
- Minor frequenza di replay per velocità
"""

import argparse
import time
import heapq
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
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
def astar_path(grid, start, goal):
    """Restituisce una lista di pos (x,y) dal start al goal (incluso),
    o [] se non esiste path. Considera come bloccanti wall/lava."""
    W, H = grid.width, grid.height
    def passable(x,y):
        if not (0<=x<W and 0<=y<H): return False
        c = grid.get(x,y)
        return (c is None) or (getattr(c,"type",None) not in ["wall","lava"])
    sx, sy = start
    gx, gy = goal
    open_heap = []
    heapq.heappush(open_heap, (0, sx, sy))
    came_from = {(sx,sy): None}
    gscore = { (sx,sy): 0 }
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    while open_heap:
        _, x, y = heapq.heappop(open_heap)
        if (x,y) == (gx,gy):
            # ricostruisci path
            path = []
            cur = (x,y)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            return list(reversed(path))
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x+dx, y+dy
            if not passable(nx,ny): continue
            tentative = gscore[(x,y)] + 1
            if (nx,ny) not in gscore or tentative < gscore[(nx,ny)]:
                gscore[(nx,ny)] = tentative
                priority = tentative + h((nx,ny),(gx,gy))
                heapq.heappush(open_heap, (priority, nx, ny))
                came_from[(nx,ny)] = (x,y)
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
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
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
    def __init__(self, hf_model=HF_MODEL, device=DEVICE, verbose=False):
        self.verbose = verbose
        self.device = device
        # caricamento HF (se disponibile)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model = AutoModelForCausalLM.from_pretrained(hf_model)
            if torch.cuda.is_available() and device == "cuda":
                self.model.to("cuda")
            else:
                self.model.to("cpu")
        except Exception as e:
            print("Attenzione: non sono riuscito a caricare il modello HF. LLM non disponibile.", e)
            self.tokenizer = None
            self.model = None

    def build_prompt(self, grid_ascii, agent_pos, agent_dir, goal_pos, mid_cell, max_actions):
        prompt = (
            "You are an assistant that must propose a sequence of safe actions for an agent in a grid.\n"
            f"Grid ASCII (A=agent, G=goal, #=wall, ~=lava, .=free):\n{grid_ascii}\n\n"
            f"Agent position: {agent_pos}, direction: {agent_dir}\n"
            f"Goal position: {goal_pos}\n"
            f"Key mid cell (bridge across lava): {mid_cell}\n"
            f"Valid actions: {VALID_ACTIONS}\n"
            f"Max suggested actions: {max_actions}\n\n"
            "IMPORTANT:\n"
            "- 'forward' = move 1 cell in current facing direction\n"
            "- 'left' = rotate agent 90 degrees to the LEFT (no movement)\n"
            "- 'right' = rotate agent 90 degrees to the RIGHT (no movement)\n"
            "- You MUST output ONLY a Python list of actions using only these tokens, e.g. ['left','forward','forward']\n"
            "- Use the minimal number of actions necessary to reach the goal; avoid walls and lava.\n"
            "- No extra text or explanation. If you cannot find a safe move, return []\n"
        )
        return prompt

    def suggest_actions(self, grid, agent_pos, agent_dir, goal_pos, mid_cell, max_actions=10):
        # Se il modello non è caricato -> fallback immediato
        if self.model is None or self.tokenizer is None:
            return []

        grid_ascii = build_ascii_grid(grid, agent_pos, goal_pos)
        prompt = self.build_prompt(grid_ascii, agent_pos, agent_dir, goal_pos, mid_cell, max_actions)
        if self.verbose:
            print("--- PROMPT ---\n", prompt)
    
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        try:
            # uso deterministico (no sampling), pochi token
            gen = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id
            )
            raw = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            if self.verbose:
                print("--- RAW LLM RESPONSE ---\n", raw)
        except Exception as e:
            if self.verbose:
                print("LLM generation error:", e)
            raw = ""

        # parsing robusto: trova prima lista python-like
        m = re.search(r"\[([^\]]*)\]", raw)
        seq = []
        if m:
            items = m.group(1).split(",")
            for it in items:
                tok = it.strip().strip("'\"")
                if tok in VALID_ACTIONS:
                    seq.append(tok)
                    if len(seq) >= max_actions:
                        break

        # verifica di sicurezza: simula ogni azione e scarta quelle che portano a collisione
        safe_seq = []
        sim_pos = tuple(agent_pos)
        sim_dir = int(agent_dir)
        for s in seq:
            if simulate_actions(sim_pos, sim_dir, [s], grid):
                safe_seq.append(s)
                # update sim state
                if s == "forward":
                    dx,dy = DIR2VEC[sim_dir]
                    sim_pos = (sim_pos[0]+dx, sim_pos[1]+dy)
                elif s == "left":
                    sim_dir = (sim_dir-1)%4
                elif s == "right":
                    sim_dir = (sim_dir+1)%4
            else:
                # interruzione se LLM ha suggerito mossa non valida
                break

        return safe_seq[:max_actions]

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
                
                # 1) tentativo deterministico: A* path + conversione in azioni
                path = astar_path(env_wrapper.base_env.grid, tuple(agent_pos), tuple(goal_pos))
                plan_actions = []
                if path:
                    plan_actions = path_to_actions(path, agent_dir)
                    # taglia alle max azioni desiderate
                    max_actions = max(1, int(max_llm_actions_factor*(abs(agent_pos[0]-goal_pos[0])+abs(agent_pos[1]-goal_pos[1]))))
                    plan_actions = plan_actions[:max_actions]
                
                # 2) se A* non trova path, prova LLM (fallback)
                suggestions = []
                if not plan_actions and helper is not None:
                    max_actions = max(1,int(max_llm_actions_factor*(abs(agent_pos[0]-goal_pos[0])+abs(agent_pos[1]-goal_pos[1]))))
                    suggestions = helper.suggest_actions(env_wrapper.base_env.grid, agent_pos, agent_dir, goal_pos, mid_cell, max_actions=max_actions)
                    plan_actions = suggestions

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
                    agent.remember(state, action, reward, next_state, done)
                    
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
                agent.remember(state, action, reward, next_state, done)
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
    plt.savefig("rewards.png")
    
    plt.figure(figsize=(8,4))
    plt.plot(stats["episode_moves"])
    plt.title("Moves per episode")
    plt.xlabel("Episode")
    plt.ylabel("Moves")
    plt.tight_layout()
    plt.savefig("moves.png")
    
    plt.figure(figsize=(8,4))
    plt.plot(stats["llm_suggestions_used"], label="sugg used")
    plt.plot(stats["llm_hallucinations"], label="hallucinations")
    plt.legend()
    plt.title("LLM suggestions / hallucinations")
    plt.tight_layout()
    plt.savefig("llm_stats.png")

# ---------------- Main ----------------
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--replay-freq", type=int, default=4, help="quante volte chiamare agent.replay() (meno = più veloce)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human" if args.render else None)
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
