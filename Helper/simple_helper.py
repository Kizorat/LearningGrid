"""
MiniGrid RL Helper ottimizzato con LLM opzionale e A* backup.
Versione compatta, con debug e stampe cprint.
INTEGRAZIONE UI: Supporta visual_game.py per layout split-screen con Buffer Episodi.
FIX: 
- Coordinate pulite (int) nel display.
- Timing: 3s pre-acquisizione, 3s lettura piano.
- Max 5 azioni.
"""
import sys
import os
from pathlib import Path

# Aggiungi la directory root al path per permettere gli import
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import argparse
import time
import json
import heapq
import numpy as np
import torch
import re
import ollama
import gymnasium as gym
import csv
from datetime import datetime
from minigrid.wrappers import FullyObsWrapper
from DQNAgent.agent import DQNAgent
from DQNAgent.enviroment import DynamicMiniGridWrapper
from Dataset.dataset_generator import path_to_interaction
import matplotlib.pyplot as plt

# ---------------- cprint (SPOSTATO SOPRA GLI IMPORT UI) ----------------
class cprint:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'

    @staticmethod
    def info(*args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args)
        print(f"{cprint.OKBLUE}{msg}{cprint.ENDC}", end=end)

    @staticmethod
    def success(*args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args)
        print(f"{cprint.OKGREEN}{msg}{cprint.ENDC}", end=end)

    @staticmethod
    def warn(*args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args)
        print(f"{cprint.WARNING}{msg}{cprint.ENDC}", end=end)

    @staticmethod
    def error(*args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args)
        print(f"{cprint.FAIL}{msg}{cprint.ENDC}", end=end)

    @staticmethod
    def highlight(*args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args)
        print(f"{cprint.HEADER}{msg}{cprint.ENDC}", end=end)

# --- IMPORT UI (DOPO CPRINT) ---
try:
    from UI.visual_game import LayoutManager
except ImportError as e:
    print(f"{cprint.FAIL}ATTENZIONE: Errore import UI: {e}. Il render grafico non funzionerà.{cprint.ENDC}")
    LayoutManager = None

# ---------------- Config ----------------
VALID_ACTIONS = ["forward", "left", "right"]
VALID_ACTIONS_DOORKEY = ["forward", "left", "right", "pickup", "toggle"]
# MiniGrid coordinate system: 0=right, 1=down, 2=left, 3=up
DIR2VEC = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Utils ----------------
def flatten_obs(obs):
    if isinstance(obs, dict) and 'image' in obs:
        return obs['image'].astype(np.float32).ravel()/255.0
    if isinstance(obs, dict):
        parts=[]
        for v in obs.values():
            if isinstance(v, np.ndarray):
                parts.append(v.ravel())
            elif isinstance(v,(int,float)):
                parts.append(np.array([v]))
        return np.concatenate(parts).astype(np.float32)
    return np.array(obs,dtype=np.float32).ravel()

def simulate_actions(agent_pos, agent_dir, actions, grid, base_env=None, env_type="empty"):
    """Simula le azioni e verifica se sono tutte valide senza cambiare lo stato reale"""
    pos = [int(agent_pos[0]), int(agent_pos[1])]
    dir_ = int(agent_dir)
    
    for a in actions:
        if a == "left":
            dir_ = (dir_ - 1) % 4
        elif a == "right":
            dir_ = (dir_ + 1) % 4
        elif a == "forward":
            dx, dy = DIR2VEC[dir_]
            nx, ny = pos[0] + dx, pos[1] + dy
            
            if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                return False
            
            cell = grid.get(nx, ny)
            if cell is not None:
                cell_type = getattr(cell, "type", None)
                if cell_type in ["wall", "lava"]:
                    return False
                if cell_type == "door" and not getattr(cell, "is_open", False):
                    return False
                if cell_type == "key":
                    return False
            
            pos = [nx, ny]
        elif a == "pickup":
            if env_type != "DoorKey":
                return False
            dx, dy = DIR2VEC[dir_]
            front_x, front_y = pos[0] + dx, pos[1] + dy
            if 0 <= front_x < grid.width and 0 <= front_y < grid.height:
                cell = grid.get(front_x, front_y)
                if cell is None or getattr(cell, "type", None) != "key":
                    return False
            else:
                return False
        elif a == "toggle":
            if env_type != "DoorKey":
                return False
            dx, dy = DIR2VEC[dir_]
            front_x, front_y = pos[0] + dx, pos[1] + dy
            if 0 <= front_x < grid.width and 0 <= front_y < grid.height:
                cell = grid.get(front_x, front_y)
                if cell is None or getattr(cell, "type", None) != "door":
                    return False
            else:
                return False
        else:
            return False
    
    return True

# ---------------- A* Pathfinding ----------------
def astar_path(grid, start, goal):
    W,H=grid.width,grid.height
    def passable(x,y):
        if not (0<=x<W and 0<=y<H): return False
        c=grid.get(x,y)
        return (c is None) or getattr(c,"type",None) not in ["wall","lava"]

    sx,sy=start
    gx,gy=goal

    dx=1 if gx>sx else -1 if gx<sx else 0
    dy=1 if gy>sy else -1 if gy<sy else 0

    cur_x,cur_y=sx,sy
    first_free=None
    while True:
        cur_x+=dx
        cur_y+=dy
        if not (0<=cur_x<W and 0<=cur_y<H): return []
        if passable(cur_x,cur_y):
            first_free=(cur_x,cur_y)
            break
    if first_free is None: return []

    new_start=first_free
    open_heap=[]
    heapq.heappush(open_heap,(0,new_start[0],new_start[1]))
    came_from={new_start:None}
    gscore={new_start:0}
    h=lambda a,b: abs(a[0]-b[0])+abs(a[1]-b[1])

    while open_heap:
        _,x,y=heapq.heappop(open_heap)
        if (x,y)==(gx,gy):
            path=[]
            cur=(x,y)
            while cur is not None:
                path.append(cur)
                cur=came_from[cur]
            path.reverse()
            return [start]+path
        for mx,my in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+mx,y+my
            if not passable(nx,ny): continue
            tentative=gscore[(x,y)]+1
            if (nx,ny) not in gscore or tentative<gscore[(nx,ny)]:
                gscore[(nx,ny)]=tentative
                heapq.heappush(open_heap,(tentative+h((nx,ny),(gx,gy)),nx,ny))
                came_from[(nx,ny)]=(x,y)
    return []

def pos_to_dir(from_pos,to_pos):
    dx=to_pos[0]-from_pos[0]
    dy=to_pos[1]-from_pos[1]
    for d,(vx,vy) in DIR2VEC.items():
        if (dx,dy)==(vx,vy):
            return d
    return None

def rotate_to(curr_dir,desired_dir):
    diff=(desired_dir-curr_dir)%4
    if diff==0: return []
    if diff==1: return ["right"]
    if diff==2: return ["right","right"]
    if diff==3: return ["left"]
    return []

def path_to_actions(path,start_dir):
    if not path or len(path)==1: return []
    actions=[]
    dir_=start_dir
    for i in range(len(path)-1):
        cur=path[i]
        nxt=path[i+1]
        desired_dir=pos_to_dir(cur,nxt)
        if desired_dir is None: continue
        actions+=rotate_to(dir_,desired_dir)
        actions+=["forward"]
        dir_=desired_dir
    return actions

def build_ascii_grid(grid, agent_pos, goal_pos):
    """Costruisce una rappresentazione ASCII della griglia per debugging"""
    agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
    goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
    
    rows = []
    for y in range(grid.height):
        row = []
        for x in range(grid.width):
            if (x, y) == agent_pos:
                row.append("A")
            elif (x, y) == goal_pos:
                row.append("G")
            else:
                c = grid.get(x, y)
                if c is None:
                    row.append(".")
                else:
                    t = getattr(c, "type", "?")
                    if t == "wall":
                        row.append("#")
                    elif t == "lava":
                        row.append("~")
                    elif t == "door":
                        # Mostra stato porta: D=chiusa, O=aperta
                        is_open = getattr(c, "is_open", False)
                        row.append("O" if is_open else "D")
                    elif t == "key":
                        row.append("K")
                    else:
                        row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)

# ---------------- DoorKey Helper Functions ----------------
def find_key_position(grid):
    """Trova la posizione della chiave nella griglia"""
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and getattr(cell, "type", None) == "key":
                return (x, y)
    return None

def find_door_position(grid):
    """Trova la posizione della porta nella griglia"""
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and getattr(cell, "type", None) == "door":
                return (x, y)
    return None

def is_door_open(grid):
    """Verifica se la porta è aperta"""
    door_pos = find_door_position(grid)
    if door_pos:
        cell = grid.get(door_pos[0], door_pos[1])
        if cell:
            return getattr(cell, "is_open", False)
    return False

def agent_has_key(base_env):
    """Verifica se l'agente ha la chiave"""
    carrying = getattr(base_env, "carrying", None)
    if carrying and getattr(carrying, "type", None) == "key":
        return True
    return False

def get_cell_in_front(agent_pos, agent_dir, grid):
    """Restituisce la cella davanti all'agente"""
    dx, dy = DIR2VEC[agent_dir]
    front_x = int(agent_pos[0]) + dx
    front_y = int(agent_pos[1]) + dy
    if 0 <= front_x < grid.width and 0 <= front_y < grid.height:
        return grid.get(front_x, front_y), (front_x, front_y)
    return None, None

def get_doorkey_phase(base_env, grid):
    """
    Determina la fase corrente per DoorKey:
    - 'get_key': deve raccogliere la chiave
    - 'open_door': ha la chiave, deve aprire la porta
    - 'go_to_goal': porta aperta, vai al goal
    """
    has_key = agent_has_key(base_env)
    door_open = is_door_open(grid)
    
    if not has_key and find_key_position(grid) is not None:
        return 'get_key'
    elif has_key and not door_open:
        return 'open_door'
    else:
        return 'go_to_goal'

def is_agent_in_front_of(agent_pos, agent_dir, target_pos):
    """Verifica se l'agente è posizionato di fronte al target"""
    if target_pos is None:
        return False
    dx, dy = DIR2VEC[agent_dir]
    front_x = int(agent_pos[0]) + dx
    front_y = int(agent_pos[1]) + dy
    return (front_x, front_y) == target_pos

# ---------- Pathfinding functions from dataset_generator ----------
def heuristic(pos, goal):
    """Manhattan distance heuristic for A*"""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def valid_grid_cell(grid, pos, ignore_obstacles=None):
    """Check if a grid cell is valid and traversable"""
    if ignore_obstacles is None:
        ignore_obstacles = []
    
    x, y = int(pos[0]), int(pos[1])
    
    # Verifica limiti griglia
    if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
        return False
    
    cell = grid.get(x, y)
    
    # Se la cella è vuota (None), è valida
    if cell is None:
        return True
    
    # Verifica ostacoli (wall, lava)
    cell_type = getattr(cell, "type", None)
    if cell_type in ["wall", "lava"]:
        return False
    
    # La chiave è un oggetto fisico che occupa la cella
    if cell_type == "key":
        return False
    
    # Per DoorKey: porta chiusa è un ostacolo, porta aperta è passabile
    if cell_type == "door":
        is_open = getattr(cell, "is_open", False)
        if not is_open:
            return False  # Porta chiusa blocca il passaggio
    
    return True

def get_turn_actions(current_dir, desired_dir):
    """Get rotation actions to face desired direction"""
    diff = (desired_dir - current_dir) % 4
    if diff == 0: return []
    if diff == 1: return ["right"]
    if diff == 2: return ["right", "right"]
    if diff == 3: return ["left"]
    return []

def astar_grid(grid, start_pos, start_dir, goal_pos, ignore_obstacles=None):
    """A* search algorithm for grid navigation con direzione"""
    start_pos = (int(start_pos[0]), int(start_pos[1]))
    goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
    start_dir = int(start_dir)
    
    start = (start_pos, start_dir)
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {start: None}
    action_from = {start: None}
    cost_so_far = {start: 0}
    
    best_end_state = None

    while frontier:
        _, (pos, direction) = heapq.heappop(frontier)

        if pos == goal_pos:
            best_end_state = (pos, direction)
            break

        x, y = pos
        successors = [
            ("left",  (pos, (direction - 1) % 4)),
            ("right", (pos, (direction + 1) % 4)),
            ("forward", ((x + DIR2VEC[direction][0], y + DIR2VEC[direction][1]), direction)),
        ]

        for action, new_state in successors:
            new_pos, new_dir = new_state

            if action == "forward":
                if not valid_grid_cell(grid, new_pos, ignore_obstacles):
                    continue

            new_cost = cost_so_far[(pos, direction)] + 1

            if (new_pos, new_dir) not in cost_so_far or new_cost < cost_so_far[(new_pos, new_dir)]:
                cost_so_far[(new_pos, new_dir)] = new_cost
                priority = new_cost + heuristic(new_pos, goal_pos)
                heapq.heappush(frontier, (priority, (new_pos, new_dir)))
                came_from[(new_pos, new_dir)] = (pos, direction)
                action_from[(new_pos, new_dir)] = action

    if best_end_state is None:
        return None, None

    actions = []
    cur = best_end_state
    while action_from[cur] is not None:
        actions.append(action_from[cur])
        cur = came_from[cur]
    actions.reverse()
    
    return actions, best_end_state

def path_to_interaction_grid(grid, start_pos, start_dir, target_pos, ignore_obstacles=None):
    """Find path to a position with directional constraints"""
    tx, ty = target_pos
    neighbors = [
        ((tx - 1, ty), 0),
        ((tx, ty - 1), 1),
        ((tx + 1, ty), 2),
        ((tx, ty + 1), 3)
    ]
    
    best_path = None
    best_state = None
    min_len = float('inf')

    for n_pos, req_dir in neighbors:
        if valid_grid_cell(grid, n_pos, ignore_obstacles):
            path, end_state = astar_grid(grid, start_pos, start_dir, n_pos, ignore_obstacles)
            
            if end_state:
                curr_dir = end_state[1]
                turn_actions = get_turn_actions(curr_dir, req_dir)
                full_path = path + turn_actions
                
                if len(full_path) < min_len:
                    min_len = len(full_path)
                    best_path = full_path
                    best_state = (n_pos, req_dir)

    return best_path, best_state

def get_optimal_path_length(grid, agent_pos, agent_dir, goal_pos, env_type="empty"):
    """Calculate the length of optimal path to goal using A* algorithm"""
    try:
        path, _ = astar_grid(grid, agent_pos, agent_dir, goal_pos)
        return len(path) if path else 0
    except:
        return 0

# ---------------- LLM Helper ----------------
class LLMHelper:
    def __init__(self, model_name="mistral:7b", verbose=False, env_type="empty"):
        self.verbose=verbose
        self.model_name=model_name
        self.env_type=env_type
        # Tracking per DoorKey - per rilevare allucinazioni
        self.pickup_executed = False
        self.toggle_executed = False
        # Tracking chiamate helper per logging
        self.call_number = 0
        self.csv_writer = None
        self.csv_file = None

    def reset_doorkey_state(self):
        """Reset stato DoorKey per nuovo episodio"""
        self.pickup_executed = False
        self.toggle_executed = False
    
    def init_logging(self, env_name):
        """Inizializza il logging CSV in una cartella con timestamp"""
        # Crea cartella con data e ora
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = ROOT_DIR / "Helper" / "Logs" / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Crea file CSV
        csv_path = log_dir / "helper_calls.csv"
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        # Scrivi header
        self.csv_writer.writerow(["Environment", "Episodio", "Chiamata_numero", "Numero_azioni_suggerite", "Azioni_parsate"])
        self.csv_file.flush()
        
        self.env_name = env_name
        self.current_episode = 0
        
        if self.verbose:
            cprint.success(f"Logging inizializzato in: {csv_path}")
    
    def reset_episode(self, episode_num):
        """Reset contatori per nuovo episodio"""
        self.call_number = 0
        self.current_episode = episode_num
        self.reset_doorkey_state()
        
        if self.verbose:
            cprint.info(f"Episodio {episode_num}: contatori azzerati")
    
    def log_call(self, num_suggested, num_parsed):
        """Registra una chiamata dell'helper nel CSV"""
        self.call_number += 1
        
        if self.csv_writer:
            self.csv_writer.writerow([
                self.env_name,
                self.current_episode,
                self.call_number,
                num_suggested,
                num_parsed
            ])
            self.csv_file.flush()
            
            if self.verbose:
                cprint.info(f"Logged call #{self.call_number}: {num_suggested} suggested, {num_parsed} parsed")
    
    def close_logging(self):
        """Chiude il file CSV"""
        if self.csv_file:
            self.csv_file.close()
            if self.verbose:
                cprint.success("Logging chiuso")

    def get_cell_info(self, grid, x, y):
        """Restituisce info sulla cella"""
        if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
            return "wall"
        cell = grid.get(x, y)
        if cell is None:
            return "empty"
        cell_type = getattr(cell, "type", "unknown")
        # Per le porte, indica anche lo stato
        if cell_type == "door":
            is_open = getattr(cell, "is_open", False)
            return "door_open" if is_open else "door_closed"
        return cell_type

    def build_prompt(self, grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid):
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        # Direzioni: 0=right, 1=down, 2=left, 3=up
        dir_names = ["right", "down", "left", "up"]
        agent_dir_str = dir_names[agent_dir]
        
        # Calcola cosa c'è in ogni direzione rispetto all'agente
        ax, ay = agent_pos
        directions_info = {
            "right": self.get_cell_info(grid, ax+1, ay),
            "down": self.get_cell_info(grid, ax, ay+1),
            "left": self.get_cell_info(grid, ax-1, ay),
            "up": self.get_cell_info(grid, ax, ay-1)
        }
        
        # Cosa c'è davanti all'agente (nella direzione in cui guarda)
        front_cell = directions_info[agent_dir_str]
        can_move_forward = front_cell in ["empty", "goal"]
        
        # Calcola direzione verso il goal
        dx = goal_pos[0] - agent_pos[0]
        dy = goal_pos[1] - agent_pos[1]
        
        # Calcola quale direzione deve guardare per andare verso il goal
        needed_facing = []
        if dx > 0: needed_facing.append("right")
        if dx < 0: needed_facing.append("left")
        if dy > 0: needed_facing.append("down")
        if dy < 0: needed_facing.append("up")
        
        # Calcola come ruotare per guardare verso il goal
        def get_rotation_to(current, target):
            dirs = ["right", "down", "left", "up"]
            curr_idx = dirs.index(current)
            tgt_idx = dirs.index(target)
            diff = (tgt_idx - curr_idx) % 4
            if diff == 0: return "already facing"
            if diff == 1: return "rotate right once"
            if diff == 2: return "rotate right twice OR left twice"
            if diff == 3: return "rotate left once"
            return ""
        
        rotation_hints = []
        for nf in needed_facing:
            rotation_hints.append(f"To face {nf}: {get_rotation_to(agent_dir_str, nf)}")
        
        # Calcola quanti forward servono in ogni direzione
        steps_needed = []
        if dx != 0: steps_needed.append(f"{abs(dx)} steps {'right' if dx > 0 else 'left'}")
        if dy != 0: steps_needed.append(f"{abs(dy)} steps {'down' if dy > 0 else 'up'}")
        
        # Calcola il percorso ottimale con A* per darlo come hint all'LLM
        try:
            optimal_path, _ = astar_grid(grid, agent_pos, agent_dir, goal_pos)
            # Limita a 5 azioni
            optimal_path = optimal_path[:5] if optimal_path else None
            optimal_hint = f"OPTIMAL PATH (use this!): {optimal_path}" if optimal_path else ""
        except:
            optimal_hint = ""
        
        prompt = f"""You are a path planner. Output the EXACT actions to reach the goal.

PROBLEM: Agent at {agent_pos} facing {agent_dir_str}, goal at {goal_pos}.

{optimal_hint}

RULES:
- "forward" moves 1 cell in facing direction
- "left"/"right" rotate the agent (don't move)
- Facing {agent_dir_str}, so "forward" goes {agent_dir_str}
- In front: {front_cell} {"(BLOCKED!)" if not can_move_forward else "(clear)"}
LIMIT: Suggest MAX {max_actions} actions.

{{"actions": [...]}}"""
        
        if self.verbose:
            cprint.info("Agent position:", agent_pos)
            cprint.info("Goal position:", goal_pos)
            cprint.info("Agent direction:", agent_dir_str)
            cprint.info("Front cell:", front_cell, "(can move)" if can_move_forward else "(BLOCKED)")
            if optimal_hint:
                cprint.info("Optimal path hint:", optimal_hint)
            cprint.info("Max actions:", max_actions)
            cprint.info("GRID:")
            for line in grid_ascii.split("\n"):
                cprint.info(line)
        
        return prompt

    def build_prompt_doorkey(self, grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid, base_env):
        """Costruisce il prompt per DoorKey con fasi: chiave -> porta -> goal"""
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        dir_names = ["right", "down", "left", "up"]
        agent_dir_str = dir_names[agent_dir]
        
        # Stato corrente DoorKey
        phase = get_doorkey_phase(base_env, grid)
        has_key = agent_has_key(base_env)
        door_open = is_door_open(grid)
        key_pos = find_key_position(grid)
        door_pos = find_door_position(grid)
        
        # Cosa c'è davanti all'agente
        front_cell, front_pos = get_cell_in_front(agent_pos, agent_dir, grid)
        front_type = self.get_cell_info(grid, front_pos[0], front_pos[1]) if front_pos else "wall"
        
        # Determina il target corrente e le azioni necessarie
        if phase == 'get_key':
            current_target = key_pos
            target_desc = f"KEY at {key_pos}"
            phase_instruction = "PHASE 1: Go to the key and use 'pickup' when facing it."
            required_final_action = "pickup"
        elif phase == 'open_door':
            current_target = door_pos
            target_desc = f"DOOR at {door_pos}"
            phase_instruction = "PHASE 2: Go to the door and use 'toggle' when facing it to open it."
            required_final_action = "toggle"
        else:  # go_to_goal - FASE 3 OTTIMIZZATA
            current_target = goal_pos
            target_desc = f"GOAL at {goal_pos}"
            # Istruzione soft: informa che la via è libera
            phase_instruction = f"PHASE 3: The door is OPEN. Proceed to the goal at {goal_pos}."
            required_final_action = None
        
        # Calcola path ottimale verso il target corrente
        try:
            if current_target:
                # Per chiave e porta, vai alla cella adiacente
                if phase in ['get_key', 'open_door']:
                    # Check if agent is ALREADY in position to interact
                    if is_agent_in_front_of(agent_pos, agent_dir, current_target):
                        # Agent is already facing the target, just need the action
                        path_actions = [required_final_action]
                        if self.verbose:
                            cprint.success(f"Agent già in posizione per {required_final_action}!")
                    else:
                        # Need to navigate to target
                        path_actions, end_state = path_to_interaction_grid(grid, agent_pos, agent_dir, current_target)
                        if path_actions:
                            path_actions = path_actions + [required_final_action]
                        elif end_state is None and self.verbose:
                            cprint.warn(f"Impossibile trovare path verso {current_target}")
                    hint_label = "SUGGESTED PATH"
                else:
                    # --- FASE 3: Calcolo A* puro verso il goal ---
                    # Usiamo "Navigation Hint" per non essere troppo autoritari
                    path_actions, _ = astar_grid(grid, agent_pos, agent_dir, current_target)
                    hint_label = "NAVIGATION HINT (Shortest Path)"
                
                # Limita a 5 azioni
                path_actions = path_actions[:5] if path_actions else None
                optimal_hint = f"{hint_label}: {path_actions}" if path_actions else ""
            else:
                optimal_hint = ""
        except Exception as e:
            optimal_hint = ""
            if self.verbose:
                cprint.error(f"Path calc error: {e}")
        
        prompt = f"""You are a path planner for a DoorKey environment. Output the EXACT actions.

CURRENT STATE:
- Agent at {agent_pos} facing {agent_dir_str}
- Has key: {has_key}
- Door open: {door_open}
- Current target: {target_desc}

{phase_instruction}

{optimal_hint}

AVAILABLE ACTIONS:
- "forward": move 1 cell in facing direction
- "left"/"right": rotate the agent (don't move)
- "pickup": pick up object in front (ONLY when facing the key, use ONCE)
- "toggle": open/close door in front (ONLY when facing the door with key, use ONCE)

CRITICAL RULES:
- Use "pickup" ONLY ONCE when directly facing the key
- Use "toggle" ONLY ONCE when directly facing the closed door (after having the key)
- DO NOT use "pickup" if you already have the key
- DO NOT use "toggle" if the door is already open
- In front of agent: {front_type}
LIMIT: Suggest MAX {max_actions} actions.

{{"actions": [...]}}"""
        
        # --- BLOCCO CPRINT ORIGINALE REINTEGRATO ---
        if self.verbose:
            cprint.info("=== DoorKey State ===")
            cprint.info("Phase:", phase)
            cprint.info("Agent position:", agent_pos)
            cprint.info("Agent direction:", agent_dir_str)
            cprint.info("Has key:", has_key)
            cprint.info("Door open:", door_open)
            cprint.info("Key position:", key_pos)
            cprint.info("Door position:", door_pos)
            cprint.info("Goal position:", goal_pos)
            cprint.info("Current target:", target_desc)
            cprint.info("Front cell:", front_type)
            
            if phase == 'go_to_goal':
                cprint.success(">>> PHASE 3: A* Path provided as Hint <<<")
            
            if optimal_hint:
                cprint.info(optimal_hint)
            
            cprint.info("GRID:")
            for line in grid_ascii.split("\n"):
                cprint.info(line)
        # -------------------------------------------
        
        return prompt

    def suggest_actions(self, grid, agent_pos, agent_dir, goal_pos, max_actions=5):
        """Suggerisce azioni usando LLM"""
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        grid_ascii = build_ascii_grid(grid, agent_pos, goal_pos)
        prompt = self.build_prompt(grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid)
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.0,
                    "num_predict": 256,
                    "num_gpu": 0  # Forza uso CPU
                }
            )
            raw = response["response"]
            if self.verbose:
                cprint.highlight("Max actions richieste:", max_actions)
                cprint.highlight("--- RAW LLM RESPONSE ---")
                cprint.highlight(raw)
        except Exception as e:
            cprint.error("Errore Ollama:", e)
            return [], 0
        
        # Parse JSON response - cerca il pattern actions anche se c'è testo intorno
        try:
            # Cerca qualsiasi JSON con "actions" nella risposta
            json_match = re.search(r'\{\s*"actions"\s*:\s*\[[^\]]*\]\s*\}', raw, re.DOTALL)
            if json_match:
                resp_json = json.loads(json_match.group(0))
            else:
                # Prova a pulire e parsare
                clean = raw.strip()
                if clean.startswith("```"):
                    # Rimuovi code blocks
                    clean = re.sub(r'```\w*\n?', '', clean)
                resp_json = json.loads(clean)
            
            seq = resp_json.get("actions", [])
            if not isinstance(seq, list):
                seq = []
            seq = [str(t).lower().strip() for t in seq if str(t).lower().strip() in VALID_ACTIONS][:max_actions]
        except json.JSONDecodeError:
            # Fallback: estrai tutte le azioni dalla risposta
            all_actions = re.findall(r'["\'](forward|left|right)["\']', raw, re.IGNORECASE)
            if not all_actions:
                if self.verbose:
                    cprint.warn("Impossibile parsare la risposta LLM")
                return [], 0
            seq = [a.lower() for a in all_actions][:max_actions]
        
        if self.verbose:
            cprint.info("Sequenza parsata:", seq)
        
        # Validazione rigorosa: simula ogni azione
        safe = []
        sim_pos = [int(agent_pos[0]), int(agent_pos[1])]
        sim_dir = int(agent_dir)
        
        for i, action in enumerate(seq):
            # Verifica se l'azione è valida nello stato attuale
            if simulate_actions(sim_pos, sim_dir, [action], grid):
                safe.append(action)
                
                # Aggiorna stato simulato
                if action == "forward":
                    dx, dy = DIR2VEC[sim_dir]
                    sim_pos = [sim_pos[0] + dx, sim_pos[1] + dy]
                elif action == "left":
                    sim_dir = (sim_dir - 1) % 4
                elif action == "right":
                    sim_dir = (sim_dir + 1) % 4
            else:
                if self.verbose:
                    cprint.warn(f"Azione {i+1} '{action}' non valida, troncata sequenza")
                break
        
        if self.verbose:
            cprint.success(f"Azioni valide: {len(safe)}/{len(seq)}")
        
        # Log della chiamata: azioni suggerite dall'LLM vs azioni parsate valide
        self.log_call(len(seq), len(safe))
        
        # Restituisce (azioni_valide, 1 se c'è stata troncatura, 0 altrimenti)
        was_truncated = 1 if len(safe) < len(seq) else 0
        return safe, was_truncated

    def suggest_actions_doorkey(self, grid, agent_pos, agent_dir, goal_pos, base_env, max_actions=5):
        """Suggerisce azioni per DoorKey con rilevamento allucinazioni per pickup/toggle"""
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        grid_ascii = build_ascii_grid(grid, agent_pos, goal_pos)
        prompt = self.build_prompt_doorkey(grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid, base_env)
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.0,
                    "num_predict": 512,  # Più token per DoorKey
                    "num_gpu": 0
                }
            )
            raw = response["response"]
            if self.verbose:
                cprint.highlight("Max actions richieste:", max_actions)
                cprint.highlight("--- RAW LLM RESPONSE (DoorKey) ---")
                cprint.highlight(raw)
        except Exception as e:
            cprint.error("Errore Ollama:", e)
            return [], 0
        
        # Parse JSON response - estrai SOLO il primo JSON valido con actions
        try:
            seq = []
            
            # 1. Cerca prima il formato {"actions": [...]}
            json_match = re.search(r'\{\s*"actions"\s*:\s*\[[^\]]*\]\s*\}', raw, re.DOTALL)
            if json_match:
                resp_json = json.loads(json_match.group(0))
                seq = resp_json.get("actions", [])
                if self.verbose:
                    cprint.info("Trovato formato {actions: [...]}")
            else:
                # 2. Cerca un array standalone [...] con azioni valide
                # Pattern: array che inizia con [ e contiene stringhe tra virgolette
                array_match = re.search(r'\[\s*"[^"]*"(?:\s*,\s*"[^"]*")*\s*\]', raw)
                if array_match:
                    try:
                        seq = json.loads(array_match.group(0))
                        if self.verbose:
                            cprint.info("Trovato formato array standalone [...]")
                    except json.JSONDecodeError:
                        seq = []
                
                # 3. Se ancora vuoto, prova a estrarre azioni tra virgolette in sequenza
                if not seq:
                    # Cerca pattern tipo ["action1", "action2", ...]
                    actions_match = re.findall(r'"(forward|left|right|pickup|toggle)"', raw, re.IGNORECASE)
                    if actions_match:
                        # Prendi solo le prime occorrenze consecutive (probabilmente l'array)
                        seq = actions_match
                        if self.verbose:
                            cprint.info("Estratte azioni da pattern regex")
            
            if not isinstance(seq, list):
                seq = []
            
            # Filtra azioni valide per DoorKey
            seq = [str(t).lower().strip() for t in seq if str(t).lower().strip() in VALID_ACTIONS_DOORKEY][:max_actions]
            
            if not seq:
                if self.verbose:
                    cprint.warn("Impossibile parsare la risposta LLM per DoorKey")
                return [], 0
                
        except Exception as e:
            if self.verbose:
                cprint.warn(f"Errore parsing per DoorKey: {e}")
            return [], 0
        
        if self.verbose:
            cprint.info("Sequenza parsata DoorKey:", seq)
        
        # Validazione con rilevamento allucinazioni
        safe = []
        has_hallucination = False  # Flag binario: 1 se almeno un errore
        sim_pos = [int(agent_pos[0]), int(agent_pos[1])]
        sim_dir = int(agent_dir)
        
        # Stato corrente per validazione
        has_key = agent_has_key(base_env)
        door_open = is_door_open(grid)
        key_pos = find_key_position(grid)
        door_pos = find_door_position(grid)
        
        for i, action in enumerate(seq):
            is_hallucination = False
            
            # Verifica allucinazioni specifiche per pickup/toggle
            if action == "pickup":
                if self.pickup_executed:
                    # Pickup già eseguito - ALLUCINAZIONE
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'pickup' già eseguito in precedenza!")
                    has_hallucination = True
                    is_hallucination = True
                elif has_key:
                    # Ha già la chiave - ALLUCINAZIONE
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'pickup' non necessario, già ha la chiave!")
                    has_hallucination = True
                    is_hallucination = True
                elif not is_agent_in_front_of(sim_pos, sim_dir, key_pos):
                    # Non è di fronte alla chiave
                    if self.verbose:
                        cprint.warn(f"Azione 'pickup' non valida: non di fronte alla chiave")
                    has_hallucination = True
                    is_hallucination = True
                    
            elif action == "toggle":
                if self.toggle_executed:
                    # Toggle già eseguito - ALLUCINAZIONE
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'toggle' già eseguito in precedenza!")
                    has_hallucination = True
                    is_hallucination = True
                elif door_open:
                    # Porta già aperta - ALLUCINAZIONE
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'toggle' non necessario, porta già aperta!")
                    has_hallucination = True
                    is_hallucination = True
                elif not has_key and not self.pickup_executed:
                    # Non ha la chiave per aprire la porta
                    if self.verbose:
                        cprint.warn(f"Azione 'toggle' non valida: non ha la chiave")
                    has_hallucination = True
                    is_hallucination = True
                elif not is_agent_in_front_of(sim_pos, sim_dir, door_pos):
                    # Non è di fronte alla porta
                    if self.verbose:
                        cprint.warn(f"Azione 'toggle' non valida: non di fronte alla porta")
                    has_hallucination = True
                    is_hallucination = True
            
            if is_hallucination:
                continue  # Salta questa azione ma continua con le successive
            
            # Validazione movimento normale
            if action in ["forward", "left", "right"]:
                # Controlla specificamente se forward va su porta chiusa
                if action == "forward":
                    dx, dy = DIR2VEC[sim_dir]
                    next_x, next_y = sim_pos[0] + dx, sim_pos[1] + dy
                    if 0 <= next_x < grid.width and 0 <= next_y < grid.height:
                        cell = grid.get(next_x, next_y)
                        if cell and getattr(cell, "type", None) == "door" and not getattr(cell, "is_open", False):
                            if self.verbose:
                                cprint.error(f"HALLUCINATION: 'forward' su porta chiusa a ({next_x}, {next_y})!")
                            has_hallucination = True
                            continue  # Salta questa azione ma continua con le successive
                
                if simulate_actions(sim_pos, sim_dir, [action], grid, base_env, env_type="DoorKey"):
                    safe.append(action)
                    # Aggiorna stato simulato
                    if action == "forward":
                        dx, dy = DIR2VEC[sim_dir]
                        sim_pos = [sim_pos[0] + dx, sim_pos[1] + dy]
                    elif action == "left":
                        sim_dir = (sim_dir - 1) % 4
                    elif action == "right":
                        sim_dir = (sim_dir + 1) % 4
                else:
                    if self.verbose:
                        cprint.warn(f"Azione {i+1} '{action}' non valida (movimento bloccato)")
                    has_hallucination = True  # Anche movimento non valido conta come errore
                    continue  # Continua con le azioni successive invece di fermarsi
            elif action == "pickup":
                safe.append(action)
                self.pickup_executed = True
                has_key = True  # Aggiorna stato simulato
                if self.verbose:
                    cprint.success("Pickup valido aggiunto")
            elif action == "toggle":
                safe.append(action)
                self.toggle_executed = True
                door_open = True  # Aggiorna stato simulato
                if self.verbose:
                    cprint.success("Toggle valido aggiunto")
        
        if self.verbose:
            cprint.success(f"Azioni valide DoorKey: {len(safe)}/{len(seq)}")
            if has_hallucination:
                cprint.error(f"Allucinazione rilevata in questa risposta")
        
        # Log della chiamata: azioni suggerite dall'LLM vs azioni parsate valide
        self.log_call(len(seq), len(safe))
        
        # Ritorna 1 se c'è stata almeno un'allucinazione, 0 altrimenti
        return safe, 1 if has_hallucination else 0

# ---------------- Mid-cell finder ----------------
def find_mid_cell_and_path(grid, agent_pos, goal_pos):
    safe_cells=[(x,y) for x in range(grid.width) for y in range(grid.height)
                if grid.get(x,y) is None or getattr(grid.get(x,y),"type",None) not in ["wall","lava"]]
    key_mid_cells=[]
    for x,y in safe_cells:
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx,ny=x+dx,y+dy
            if 0<=nx<grid.width and 0<=ny<grid.height:
                c=grid.get(nx,ny)
                if c is not None and getattr(c,"type",None)=="lava":
                    key_mid_cells.append((x,y))
                    break
    mid_cell=None
    min_neighbors=9
    for x,y in key_mid_cells:
        neighbors=sum((x+dx,y+dy) in safe_cells for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)])
        if neighbors<min_neighbors:
            min_neighbors=neighbors
            mid_cell=(x,y)
    return mid_cell, []

# ---------------- Train ----------------
def train(env_wrapper, env_id="MiniGrid", episodes=5, use_llm_helper=True, render_env=False,
          max_steps_before_llm=4, max_llm_actions_factor=2.0, replay_frequency=4, verbose=False):

    example_state=flatten_obs(env_wrapper.get_state())
    agent=DQNAgent(len(example_state),env_wrapper.action_size)
    
    # Determina il tipo di ambiente
    env_type = getattr(env_wrapper, 'task_type', 'empty')
    helper=LLMHelper(verbose=verbose, env_type=env_type) if use_llm_helper else None
    
    # Inizializza logging se helper è attivo
    if helper:
        helper.init_logging(env_id)
    
    stats={"episode_rewards":[],"episode_moves":[],"llm_suggestions_used":[],"llm_hallucinations":[]}

    # --- SETUP LAYOUT MANAGER ---
    layout_mgr = None
    if render_env and LayoutManager:
        layout_mgr = LayoutManager(env_id)
    
    dir_names = ["Right (>)", "Down (v)", "Left (<)", "Up (^)"]

    for ep in range(episodes):
        state_raw=env_wrapper.reset()
        state=flatten_obs(state_raw)
        done=False
        total_reward=0
        moves=0
        steps_since_llm=0
        ep_sugg=0
        ep_hall=0
        step_counter=0
        
        # Reset contatori helper per nuovo episodio
        if helper:
            helper.reset_episode(ep + 1)

        # Update testo iniziale layout
        if layout_mgr:
            start_pos = (int(env_wrapper.base_env.agent_pos[0]), int(env_wrapper.base_env.agent_pos[1]))
            start_dir = dir_names[env_wrapper.base_env.agent_dir]
            layout_mgr.update_text([f"Inizio Episodio...", f"Start Pos: {start_pos} Dir: {start_dir}"], episode_num=ep+1)

        while not done:
            # --- RECUPERO DATI AGENTE (POSIZIONE E DIREZIONE) ---
            agent_pos_raw = env_wrapper.base_env.agent_pos
            current_pos = (int(agent_pos_raw[0]), int(agent_pos_raw[1]))
            current_dir_idx = env_wrapper.base_env.agent_dir
            current_dir_str = dir_names[current_dir_idx]
            goal_pos = env_wrapper.base_env.goal_pos
            # ----------------------------------------------------

            valid_actions=env_wrapper.get_valid_actions()
            use_llm=use_llm_helper and steps_since_llm>=max_steps_before_llm

            if use_llm:
                # --- PAUSA E RILEVAMENTO POSIZIONE (RICHIESTA UTENTE) ---
                if render_env:
                    if layout_mgr:
                        layout_mgr.update_text([f"Pausa 3s...", "Acquisizione posizione..."], episode_num=ep+1)
                        layout_mgr.render(env_wrapper.base_env.render())
                    time.sleep(3) # Pausa 1: Pre-Acquisizione (3 secondi)

                # Ri-acquisizione dati post-pausa (in caso di glitch)
                agent_pos_raw = env_wrapper.base_env.agent_pos
                current_pos = (int(agent_pos_raw[0]), int(agent_pos_raw[1]))
                agent_dir=env_wrapper.base_env.agent_dir
                goal_pos=env_wrapper.base_env.goal_pos
                grid = env_wrapper.base_env.grid
                
                current_dir_idx = agent_dir
                current_dir_str = dir_names[current_dir_idx]
                
                # FORZA MAX 5 AZIONI
                max_actions = 5
                
                if env_type == "DoorKey":
                    plan_actions, llm_hallucinations = helper.suggest_actions_doorkey(
                        grid, agent_pos_raw, agent_dir, goal_pos, env_wrapper.base_env, max_actions=max_actions
                    ) if helper else ([], 0)
                else:
                    plan_actions, llm_hallucinations = helper.suggest_actions(grid, agent_pos_raw, agent_dir, goal_pos, max_actions=max_actions) if helper else ([], 0)
                
                ep_hall += llm_hallucinations

                # --- MOSTRA PIANO ---
                if layout_mgr:
                    if plan_actions:
                        layout_mgr.update_text([
                            f"Step: {moves} | Goal: {goal_pos}",
                            f"Pos: {current_pos} | Dir: {current_dir_str}",
                            "--- PIANO LLM (Max 5) ---",
                            str(plan_actions),
                            "-------------------------",
                        ], episode_num=ep+1)
                        # Render forzato per mostrare il testo prima della pausa
                        if render_env: layout_mgr.render(env_wrapper.base_env.render())
                    else:
                        layout_mgr.update_text(["LLM Fallito / Nessun piano.", "Fallback agente."], episode_num=ep+1)
                        if render_env: layout_mgr.render(env_wrapper.base_env.render())

                # --- PAUSA LETTURA PIANO (RICHIESTA UTENTE) ---
                if render_env and plan_actions:
                    time.sleep(3) # Pausa 2: Post-Risposta / Pre-Esecuzione (3 secondi)

                if not plan_actions:
                    cprint.error("LLM non ha suggerito azioni valide.")
                    ep_hall+=1
                    action=agent.act(state, valid_actions)
                    next_state_raw,reward,done,info=env_wrapper.step(action)
                    next_state=flatten_obs(next_state_raw)
                    valid_mask=[1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                    next_valid_actions=env_wrapper.get_valid_actions()
                    next_valid_mask=[1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                    agent.remember(state,action,reward,next_state,done,valid_mask,next_valid_mask)
                    if step_counter%replay_frequency==0: agent.replay()
                    state=next_state
                    total_reward+=reward
                    moves+=1
                    step_counter+=1
                    steps_since_llm=0
                    
                    if render_env and layout_mgr:
                        layout_mgr.render(env_wrapper.base_env.render())
                else:
                    for s in plan_actions:
                        action_map={"left":0,"right":1,"forward":2,"pickup":3,"toggle":5}
                        if s not in action_map: continue
                        action=action_map[s]
                        
                        if s in ["left", "right", "forward"]:
                            ok=simulate_actions(env_wrapper.base_env.agent_pos, env_wrapper.base_env.agent_dir, [s], env_wrapper.base_env.grid, env_wrapper.base_env, env_type=env_type)
                        else: ok = True
                            
                        if not ok:
                            ep_hall+=1
                            action=agent.act(state, valid_actions)
                        else: ep_sugg+=1
                        
                        next_state_raw,reward,done,info=env_wrapper.step(action)
                        next_state=flatten_obs(next_state_raw)
                        valid_mask=[1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                        next_valid_actions=env_wrapper.get_valid_actions()
                        next_valid_mask=[1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                        agent.remember(state,action,reward,next_state,done,valid_mask,next_valid_mask)
                        if step_counter%replay_frequency==0: agent.replay()
                        state=next_state
                        total_reward+=reward
                        moves+=1
                        step_counter+=1
                        
                        # --- SYNC FORCED IN-LOOP ---
                        if layout_mgr:
                            # Posizione aggiornata post-step (cast a int per pulizia)
                            raw_p = env_wrapper.base_env.agent_pos
                            new_pos = (int(raw_p[0]), int(raw_p[1]))
                            new_dir = dir_names[env_wrapper.base_env.agent_dir]
                            
                            layout_mgr.update_text([
                                f"Step: {moves} | Goal: {goal_pos}",
                                f"Pos: {new_pos} | Dir: {new_dir}",
                                "--- PIANO LLM (Running) ---",
                                str(plan_actions),
                                f"> Action: {s}", 
                                "----------------------------",
                            ], episode_num=ep+1)
                            
                            if render_env: layout_mgr.render(env_wrapper.base_env.render())
                        
                        if done: break
                    steps_since_llm=0
            else:
                # AZIONE AGENTE (NO LLM)
                action=agent.act(state, valid_actions)
                next_state_raw,reward,done,info=env_wrapper.step(action)
                next_state=flatten_obs(next_state_raw)
                valid_mask=[1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                next_valid_actions=env_wrapper.get_valid_actions()
                next_valid_mask=[1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                agent.remember(state,action,reward,next_state,done,valid_mask,next_valid_mask)
                if step_counter%replay_frequency==0: agent.replay()
                state=next_state
                total_reward+=reward
                moves+=1
                step_counter+=1
                steps_since_llm+=1

                # Update testo layout (Posizione agente)
                if layout_mgr and moves % 2 == 0:
                    layout_mgr.update_text([
                        f"Step: {moves} (Agent)",
                        f"Pos: {current_pos}",
                        f"Dir: {current_dir_str}",
                        "---", "In attesa LLM..."
                    ], episode_num=ep+1)

                if render_env and layout_mgr:
                    layout_mgr.render(env_wrapper.base_env.render())

            if render_env and not layout_mgr and moves%5==0:
                env_wrapper.render()
                time.sleep(0.01)

        stats["episode_rewards"].append(total_reward)
        stats["episode_moves"].append(moves)
        stats["llm_suggestions_used"].append(ep_sugg)
        stats["llm_hallucinations"].append(ep_hall)
        cprint.success(f"Ep {ep+1}/{episodes}: reward={total_reward:.2f}, moves={moves}, llm_sugg={ep_sugg}, llm_hall={ep_hall}")

    total_rewards = sum(stats["episode_rewards"])
    total_moves = sum(stats["episode_moves"])
    total_llm_suggestions = sum(stats["llm_suggestions_used"])
    total_llm_hallucinations = sum(stats["llm_hallucinations"])
    avg_reward = total_rewards / episodes
    avg_moves = total_moves / episodes
    
    cprint.highlight("\n=== TRAINING SUMMARY ===")
    cprint.info(f"Total Episodes: {episodes}")
    cprint.info(f"Total Rewards: {total_rewards:.2f}")
    cprint.info(f"Average Reward per Episode: {avg_reward:.2f}")
    cprint.info(f"Total Moves: {total_moves}")
    cprint.info(f"Average Moves per Episode: {avg_moves:.2f}")
    cprint.info(f"Total LLM Suggestions Used: {total_llm_suggestions}")
    cprint.info(f"Total LLM Hallucinations: {total_llm_hallucinations}")
    cprint.info(f"Average Suggestions per Episode: {total_llm_suggestions/episodes:.2f}")
    cprint.info(f"Average Hallucinations per Episode: {total_llm_hallucinations/episodes:.2f}")
    cprint.highlight("========================\n")

    if layout_mgr:
        layout_mgr.close()
    
    # Chiudi il logging
    if helper:
        helper.close_logging()

    return stats

# ---------------- Plot ----------------
def plot_stats(stats):
    # Crea la cartella plots se non esiste
    plots_dir = ROOT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(8,4))
    plt.plot(stats["episode_rewards"])
    plt.title("Rewards per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(plots_dir / "rewards.png")
    plt.figure(figsize=(8,4))
    plt.plot(stats["episode_moves"])
    plt.title("Moves per episode")
    plt.xlabel("Episode")
    plt.ylabel("Moves")
    plt.tight_layout()
    plt.savefig(plots_dir / "moves.png")
    plt.figure(figsize=(8,4))
    plt.plot(stats["llm_suggestions_used"], label="sugg used")
    plt.plot(stats["llm_hallucinations"], label="hallucinations")
    plt.legend()
    plt.title("LLM suggestions / hallucinations")
    plt.tight_layout()
    plt.savefig(plots_dir / "llm_stats.png")

# ---------------- Main ----------------
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--llm",action="store_true",default=True)
    parser.add_argument("--episodes",type=int,default=5)
    parser.add_argument("--max-steps",type=int,default=10)
    parser.add_argument("--replay-freq",type=int,default=4)
    parser.add_argument("--verbose",action="store_true")
    args=parser.parse_args()

    env_options={
        "1":{"label":"Crossing","ids":["MiniGrid-LavaCrossingS9N1-v0","MiniGrid-LavaCrossingS9N3-v0","MiniGrid-LavaCrossingS11N5-v0"]},
        "2":{"label":"DoorKey","ids":["MiniGrid-DoorKey-5x5-v0","MiniGrid-DoorKey-8x8-v0","MiniGrid-DoorKey-16x16-v0"]},
        "3":{"label":"Empty","ids":["MiniGrid-Empty-5x5-v0","MiniGrid-Empty-8x8-v0","MiniGrid-Empty-16x16-v0"]},
    }

    while True:
        choice=input("Scegli l'ambiente (1=Crossing, 2=DoorKey, 3=Empty): ").strip()
        if choice in env_options: break
        cprint.error("Scelta non valida, inserisci 1,2 o 3.")
    while True:
        size=input("Scegli la dimensione della mappa (1=piccolo,2=medio,3=grande): ").strip()
        if size in ("1","2","3"): break
        cprint.error("Scelta non valida, inserisci 1,2 o 3.")

    env_id = env_options[choice]["ids"][int(size)-1]
    task_label = env_options[choice]["label"]
    
    # Se render è attivo, usiamo rgb_array per passarlo alla UI custom
    render_mode = "rgb_array" if args.render else None
    
    env = gym.make(env_id, render_mode=render_mode)
    env = FullyObsWrapper(env)
    env_wrapper = DynamicMiniGridWrapper(env, task_label)


    stats=train(env_wrapper, env_id=env_id, episodes=args.episodes, use_llm_helper=args.llm,
                render_env=args.render, max_steps_before_llm=args.max_steps, verbose=args.verbose)
    plot_stats(stats)