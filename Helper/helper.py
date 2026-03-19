import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path

# Add the root directory to the path to allow imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import time
import json
import heapq
import numpy as np
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
import re
import ollama
import gymnasium as gym
import csv
import random
from datetime import datetime
from minigrid.wrappers import FullyObsWrapper
from DQNAgent.agent import DQNAgent
from DQNAgent.enviroment import DynamicMiniGridWrapper
from Dataset.dataset_generator import path_to_interaction
import matplotlib.pyplot as plt

# Colored print helpers
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

# Import UI after cprint so color constants are available for error output
try:
    from UI.visual_game import LayoutManager
except ImportError as e:
    print(f"{cprint.FAIL}ATTENTION: Error importing UI: {e}. The graphical rendering will not work.{cprint.ENDC}")
    LayoutManager = None

# Action and direction configuration
VALID_ACTIONS = ["forward", "left", "right"]
VALID_ACTIONS_DOORKEY = ["forward", "left", "right", "pickup", "toggle"]
# MiniGrid coordinate system: 0=right, 1=down, 2=left, 3=up
DIR2VEC = {0: (1,0), 1: (0,1), 2: (-1,0), 3: (0,-1)}  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Environment curriculum configuration
ALL_ENVIRONMENTS = [
    {"id": "MiniGrid-Empty-5x5-v0", "type": "Empty", "size": "small"},
    {"id": "MiniGrid-Empty-8x8-v0", "type": "Empty", "size": "medium"},
    {"id": "MiniGrid-Empty-16x16-v0", "type": "Empty", "size": "large"},
    {"id": "MiniGrid-LavaCrossingS9N1-v0", "type": "Crossing", "size": "small"},
    {"id": "MiniGrid-LavaCrossingS9N3-v0", "type": "Crossing", "size": "medium"},
    {"id": "MiniGrid-LavaCrossingS11N5-v0", "type": "Crossing", "size": "large"},
    {"id": "MiniGrid-DoorKey-5x5-v0", "type": "DoorKey", "size": "small"},
    {"id": "MiniGrid-DoorKey-8x8-v0", "type": "DoorKey", "size": "medium"},
    {"id": "MiniGrid-DoorKey-16x16-v0", "type": "DoorKey", "size": "large"},
]

# Global counters for Crossing
lava_death_count = 0
crossing_timeout_count = 0
crossing_goal_count = 0
timeout_streak = 0

def get_environment_for_episode(episode_num):
    if episode_num < 100:
        return ALL_ENVIRONMENTS[0]  # Empty-5x5
    elif episode_num < 250:
        return ALL_ENVIRONMENTS[1]  # Empty-8x8
    elif episode_num < 550:
        return ALL_ENVIRONMENTS[2]  # Empty-16x16
    elif episode_num < 850:
        return ALL_ENVIRONMENTS[3]  # LavaCrossingS9N1
    elif episode_num < 1150:
        return ALL_ENVIRONMENTS[4]  # LavaCrossingS9N3
    elif episode_num < 1450:
        return ALL_ENVIRONMENTS[5]  # LavaCrossingS11N5
    elif episode_num < 1650:
        return ALL_ENVIRONMENTS[6]  # DoorKey-5x5
    elif episode_num < 1900:
        return ALL_ENVIRONMENTS[7]  # DoorKey-8x8
    elif episode_num < 2200:
        return ALL_ENVIRONMENTS[8]  # DoorKey-16x16
    else:
        # Mixed phase with random environment selection
        return random.choice(ALL_ENVIRONMENTS)

def configure_agent_for_map(agent, env_config, prev_env_config=None):
    env_id = env_config["id"]
    task_type = env_config["type"]
    prev_type = prev_env_config["type"] if prev_env_config else None

    cprint.info(f"\n>>> Configuration for: {env_id}")

    if task_type == "Crossing":
        if prev_type == "Empty":
            agent.epsilon = 0.55       # Navigation skill already learned
            cprint.info(">>> KNOWLEDGE TRANSFER from Empty: epsilon=0.55 (navigation transferred)")
        elif prev_type == "Crossing":
            agent.epsilon = 0.45       # Partial lava avoidance already learned
            cprint.info(">>> KNOWLEDGE TRANSFER from Crossing: epsilon=0.45 (lava skill transferred)")
        else:
            agent.epsilon = 1.0        # No prior knowledge
        
        agent.epsilon_decay = 0.999    # Slightly faster decay
        agent.epsilon_min = 0.15
        base_lr = 0.00025
        agent.gamma = 0.99

        global lava_death_count, crossing_timeout_count, crossing_goal_count
        lava_death_count = 0
        crossing_timeout_count = 0
        crossing_goal_count = 0
        cprint.info(f">>> Crossing config: epsilon={agent.epsilon:.2f}, decay=0.997, gamma=0.99, lr=0.00025")

    elif task_type == "DoorKey":
        if "16x16" in env_id:
            agent.epsilon = 1.0 if prev_type != "DoorKey" else 0.85
            agent.epsilon_decay = 0.998
            agent.epsilon_min = 0.15
            base_lr = 0.0002
            agent.gamma = 0.99
            agent.batch_size = 32
        elif "8x8" in env_id:
            agent.epsilon = 1.0 if prev_type != "DoorKey" else 0.80
            agent.epsilon_decay = 0.998
            agent.epsilon_min = 0.15
            base_lr = 0.0002
            agent.gamma = 0.99
            agent.batch_size = 32
        else:
            agent.epsilon = 1.0 if prev_type != "DoorKey" else 0.75
            agent.epsilon_decay = 0.997
            agent.epsilon_min = 0.20
            base_lr = 0.00025
            agent.gamma = 0.98
            agent.batch_size = 32

    else:  # Empty
        if "16x16" in env_id:
            agent.epsilon = 0.8
            agent.epsilon_decay = 0.992
            agent.epsilon_min = 0.05
            base_lr = 0.0002
            agent.gamma = 0.999
        elif "8x8" in env_id:
            agent.epsilon = 0.8
            agent.epsilon_decay = 0.996
            agent.epsilon_min = 0.05
            base_lr = 0.00025
            agent.gamma = 0.99
        else:
            agent.epsilon = 0.6
            agent.epsilon_decay = 0.995
            agent.epsilon_min = 0.10
            base_lr = 0.0003
            agent.gamma = 0.95

    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = base_lr

    cprint.info(f">>> epsilon={agent.epsilon:.2f}, decay={agent.epsilon_decay}, lr={base_lr}")

# Utility helpers
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
    pos = [int(agent_pos[0]), int(agent_pos[1])]
    dir_ = int(agent_dir)
    simulate_open_door = set()
    
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
                if cell_type == "door":
                    is_really_open = getattr(cell, "is_open", False)
                    if not is_really_open and (nx, ny) not in simulate_open_door:
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
                simulate_open_door.add((front_x, front_y))
            else:
                return False
        else:
            return False
    return True

# A* pathfinding helpers
def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def valid_grid_cell(grid, pos, ignore_obstacles=None):
    x, y = pos
    if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
        return False
    
    cell = grid.get(x, y)
    
    # Empty cell is valid
    if cell is None:
        return True
    
    # Blocked by hard obstacles
    cell_type = getattr(cell, "type", None)
    if cell_type in ["wall", "lava"]:
        return False
    
    # A key occupies the cell physically
    if cell_type == "key":
        return False
    
    # In DoorKey, a closed door blocks movement while an open door is passable
    if cell_type == "door":
        is_open = getattr(cell, "is_open", False)
        if not is_open:
            return False  # Closed door blocks movement
    
    return True

def get_turn_actions(current_dir, desired_dir):
    diff = (desired_dir - current_dir) % 4
    if diff == 0:
        return []
    if diff == 1:
        return ["right"]
    if diff == 2:
        return ["right", "right"]
    if diff == 3:
        return ["left"]
    return []

def astar_grid(grid, start_pos, start_dir, goal_pos, ignore_obstacles=None):
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
            ("left", (pos, (direction - 1) % 4)),
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
    tx, ty = target_pos
    neighbors = [
        ((tx - 1, ty), 0),  # left of target, facing right
        ((tx, ty - 1), 1),  # above target, facing down
        ((tx + 1, ty), 2),  # right of target, facing left
        ((tx, ty + 1), 3)   # below target, facing up
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

def build_ascii_grid(grid, agent_pos, goal_pos):
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
                        # Show door state: D=closed, O=open
                        is_open = getattr(c, "is_open", False)
                        row.append("O" if is_open else "D")
                    elif t == "key":
                        row.append("K")
                    else:
                        row.append(".")
        rows.append(" ".join(row))
    return "\n".join(rows)

# DoorKey-specific helpers
def find_key_position(grid):
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and getattr(cell, "type", None) == "key":
                return (x, y)
    return None

def find_door_position(grid):
    for x in range(grid.width):
        for y in range(grid.height):
            cell = grid.get(x, y)
            if cell and getattr(cell, "type", None) == "door":
                return (x, y)
    return None

def is_door_open(grid):
    door_pos = find_door_position(grid)
    if door_pos:
        cell = grid.get(door_pos[0], door_pos[1])
        if cell:
            return getattr(cell, "is_open", False)
    return False

def agent_has_key(base_env):
    carrying = getattr(base_env, "carrying", None)
    if carrying and getattr(carrying, "type", None) == "key":
        return True
    return False

def get_cell_in_front(agent_pos, agent_dir, grid):
    dx, dy = DIR2VEC[agent_dir]
    front_x = int(agent_pos[0]) + dx
    front_y = int(agent_pos[1]) + dy
    if 0 <= front_x < grid.width and 0 <= front_y < grid.height:
        return grid.get(front_x, front_y), (front_x, front_y)
    return None, None

def get_doorkey_phase(base_env, grid):
    has_key = agent_has_key(base_env)
    door_open = is_door_open(grid)
    
    if not has_key and find_key_position(grid) is not None:
        return 'get_key'
    elif has_key and not door_open:
        return 'open_door'
    else:
        return 'go_to_goal'

def is_agent_in_front_of(agent_pos, agent_dir, target_pos):
    # Ensure the agent is exactly one cell away and facing the target
    if target_pos is None:
        return False
    
    agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
    agent_dir = int(agent_dir)
    
    dx, dy = DIR2VEC[agent_dir]
    front_x = int(agent_pos[0]) + dx
    front_y = int(agent_pos[1]) + dy
    
    return (front_x, front_y) == target_pos

def get_optimal_path_length(grid, agent_pos, agent_dir, goal_pos, env_type="empty"):
    try:
        path, _ = astar_grid(grid, agent_pos, agent_dir, goal_pos)
        return len(path) if path else 0
    except:
        return 0

# LLM helper
class LLMHelper:
    def __init__(self, model_name="qwen3:1.7b", verbose=False, env_type="empty"):
        self.verbose=verbose
        self.model_name=model_name
        self.env_type=env_type
        
        # Track DoorKey actions to detect hallucinations
        self.pickup_executed = False
        self.toggle_executed = False
        
        # Track helper calls for logging
        self.call_number = 0
        self.csv_writer = None
        self.csv_file = None
    
    def reset_doorkey_state(self):
        self.pickup_executed = False
        self.toggle_executed = False
    
    def init_logging(self, base_name="multi_env", log_dir=None):
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir = ROOT_DIR / "Helper" / "Logs" / timestamp
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV log file
        csv_path = log_dir / "helper_calls.csv"
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow(["Environment", "Episode", "Call_number", "Suggested_actions", "Parsed_actions"])
        self.csv_file.flush()
        
        self.base_name = base_name
        self.current_episode = 0
        self.current_env_name = ""
        
        if self.verbose:
            cprint.success(f"Logging initialized to file: {csv_path}")
    
    def reset_episode(self, episode_num, env_name):
        self.call_number = 0
        self.current_episode = episode_num
        self.current_env_name = env_name
        self.reset_doorkey_state()
        
        if self.verbose:
            cprint.info(f"Episode {episode_num} ({env_name}): reset helper state and logging counters")
    
    def log_call(self, num_suggested, num_parsed):
        self.call_number += 1
        if self.csv_writer:
            self.csv_writer.writerow([
                self.current_env_name,
                self.current_episode,
                self.call_number,
                num_suggested,
                num_parsed
            ])
            self.csv_file.flush()
        
        if self.verbose:
            cprint.info(f"Logged call #{self.call_number}: {num_suggested} suggested, {num_parsed} parsed")
    
    def close_logging(self):
        if self.csv_file:
            self.csv_file.close()
        if self.verbose:
            cprint.success("Closed logging file")
    
    def get_cell_info(self, grid, x, y):
        if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
            return "wall"
        cell = grid.get(x, y)
        if cell is None:
            return "empty"
        cell_type = getattr(cell, "type", "unknown")
        
        # Include door state
        if cell_type == "door":
            is_open = getattr(cell, "is_open", False)
            return "door_open" if is_open else "door_closed"
        
        return cell_type
    
    def build_prompt(self, grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid):
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        # Directions: 0=right, 1=down, 2=left, 3=up
        dir_names = ["right", "down", "left", "up"]
        agent_dir_str = dir_names[agent_dir]
        
        # Read adjacent cells around the agent
        ax, ay = agent_pos
        directions_info = {
            "right": self.get_cell_info(grid, ax+1, ay),
            "down": self.get_cell_info(grid, ax, ay+1),
            "left": self.get_cell_info(grid, ax-1, ay),
            "up": self.get_cell_info(grid, ax, ay-1)
        }
        
        # Cell in front of the agent
        front_cell = directions_info[agent_dir_str]
        can_move_forward = front_cell in ["empty", "goal"]
        
        # Compute an A* path and provide it as a hint
        try:
            optimal_path, _ = astar_grid(grid, agent_pos, agent_dir, goal_pos)
            # Limit to 5 actions
            optimal_path = optimal_path[:5] if optimal_path else None
            optimal_hint = f"OPTIMAL PATH (use this!): {optimal_path}" if optimal_path else ""
        except:
            optimal_hint = ""
        
        prompt = f"""You are a path planner. Navigate from {agent_pos} to {goal_pos}.

STATE:
- Agent: {agent_pos} facing {agent_dir_str}
- Goal: {goal_pos}
- In front: {front_cell}

{optimal_hint}

ACTIONS: "forward" (move 1 cell), "left"/"right" (rotate only)

RULES:
- MAX {max_actions} actions to suggest in the response
- Avoid walls and lava
- "forward" moves in facing direction
- "left"/"right" only rotate

RESPOND WITH ONLY THIS JSON FORMAT (no text before or after):
{{"actions": ["action1", "action2", ...]}}

YOUR RESPONSE:"""
        
        # Verbose debugging output
        if self.verbose:
            cprint.info("=== Environment State ===")
            cprint.info("Agent position:", agent_pos)
            cprint.info("Agent direction:", agent_dir_str)
            cprint.info("Goal position:", goal_pos)
            cprint.info("Front cell:", front_cell, "(can move)" if can_move_forward else "(BLOCKED)")
            cprint.info("Max actions:", max_actions)
            cprint.info("GRID:")
            for line in grid_ascii.split("\n"):
                cprint.info(line)
        
        return prompt

        
    def build_prompt_doorkey(self, grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid, base_env):
        # Build the DoorKey prompt with key -> door -> goal phases
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        dir_names = ["right", "down", "left", "up"]
        agent_dir_str = dir_names[agent_dir]
        
        # Current DoorKey state
        phase = get_doorkey_phase(base_env, grid)
        has_key = agent_has_key(base_env)
        door_open = is_door_open(grid)
        key_pos = find_key_position(grid)
        door_pos = find_door_position(grid)
        
        # Cell in front of the agent
        front_cell, front_pos = get_cell_in_front(agent_pos, agent_dir, grid)
        front_type = self.get_cell_info(grid, front_pos[0], front_pos[1]) if front_pos else "wall"
        
        # Select the current target and required action phase
        if phase == 'get_key':
            target_pos = key_pos
            target_desc = f"Get key at {key_pos}"
            phase_instruction = "Navigate to key position and pick it up."
        elif phase == 'open_door':
            target_pos = door_pos
            target_desc = f"Open door at {door_pos}"
            phase_instruction = "Navigate to door and toggle it open."
        else:  # go_to_goal
            target_pos = goal_pos
            target_desc = f"Reach goal at {goal_pos}"
            phase_instruction = "Navigate to goal position."

        try:
            if phase == 'get_key' and key_pos:
                # Check first whether the agent is already in position
                if is_agent_in_front_of(agent_pos, agent_dir, key_pos):
                    required_actions = ["pickup"]
                    optimal_hint = f"""=== PICKUP READY ===
        You are ALREADY facing the key.

        MANDATORY RESPONSE (copy exactly):
        {{"actions": {required_actions}}}"""
                else:
                    # Path to key
                    path, end_state = path_to_interaction_grid(grid, agent_pos, agent_dir, key_pos)
                    if path:
                        # Full path to key plus pickup
                        if len(path) <= max_actions - 1:
                            required_actions = path + ["pickup"]
                        else:
                            required_actions = path[:max_actions]
                        
                        optimal_hint = f"""=== PATH TO KEY ===
        MANDATORY path (copy exactly):

        {{"actions": {required_actions}}}

        DO NOT create alternatives."""
                    else:
                        optimal_hint = "ERROR: No path to key found."
            

            elif phase == 'open_door' and door_pos:
                # Check first whether the agent is already in front of the door
                if is_agent_in_front_of(agent_pos, agent_dir, door_pos):
                    required_actions = ["toggle"]
                    optimal_hint = f"""=== TOGGLE READY ===
        You are ALREADY facing the door with the key.

        MANDATORY RESPONSE (copy exactly):
        {{"actions": {required_actions}}}"""
                else:
                    # Path to door
                    path, end_state = path_to_interaction_grid(grid, agent_pos, agent_dir, door_pos)
                    if path:
                        # Full path to door plus toggle
                        if len(path) <= max_actions - 1:
                            required_actions = path + ["toggle"]
                        else:
                            required_actions = path[:max_actions]
                        
                        optimal_hint = f"""=== PATH TO DOOR ===
        MANDATORY path (copy exactly):

        {{"actions": {required_actions}}}

        DO NOT create alternatives."""
                    else:
                        optimal_hint = "ERROR: No path to door found."
            else:  # go_to_goal
                path, _ = astar_grid(grid, agent_pos, agent_dir, goal_pos)
                if path:
                    required_actions = path[:max_actions]
                    
                    # Create a numbered path view
                    numbered_path = []
                    for i, action in enumerate(required_actions, 1):
                        numbered_path.append(f"Step {i}: '{action}'")
                    
                    optimal_hint = f"""=== PATH TO GOAL ===
        MANDATORY path - EXECUTE IN EXACT ORDER:

        {chr(10).join(numbered_path)}

        CRITICAL: Order is SEQUENTIAL and CANNOT be changed!

        MANDATORY RESPONSE (copy exactly):
        {{"actions": {required_actions}}}"""
                else:
                    optimal_hint = "ERROR: No path to goal found."
                    
        except Exception as e:
            optimal_hint = f"ERROR: Path calculation failed: {e}"
            if self.verbose:
                cprint.error(f"DoorKey path calculation error: {e}")

        prompt = f"""You are a path planner for a DoorKey environment.

        GRID VISUALIZATION:
        {grid_ascii}

        CURRENT STATE:
        - Agent: {agent_pos} facing {agent_dir_str}
        - Has key: {has_key} | Door open: {door_open}
        - Target: {target_desc}
        - In front: {front_type}

        {phase_instruction}

        {optimal_hint}

        ACTIONS: "forward", "left", "right", "pickup" (only if facing key), "toggle" (only if facing closed door with key)

        RULES:
        - MAX {max_actions} actions
        - "pickup" ONLY when facing key (not if already has key)
        - "toggle" ONLY when facing closed door with key (not if door open)
        - CRITICAL: Actions MUST be executed in the EXACT ORDER provided in the path above!
        - DO NOT reorder, swap, or modify the sequence!
        - Each action depends on the previous one being executed first!

        RESPOND WITH ONLY THIS JSON FORMAT (no text before or after):
        {{"actions": ["action1", "action2", ...]}}

        YOUR RESPONSE:"""

        
        # Verbose prompt logging block
        if self.verbose:
            prompt_for_log = prompt
            prompt_for_log = re.sub(r"[ \t]*MANDATORY RESPONSE \(copy exactly\):[^\n]*", "", prompt_for_log)
            prompt_for_log = re.sub(r"\n[ \t]*\{\"actions\":\s*\[[^\n]*\]\}\s*", "\n", prompt_for_log)
            cprint.info("=" * 60)
            cprint.info("GENERATED DOORKEY PROMPT:")
            cprint.info(f"Phase: {phase}")
            cprint.info(f"Has key: {has_key}, Door open: {door_open}")
            cprint.info(f"Key pos: {key_pos}, Door pos: {door_pos}")
            cprint.info(f"Target: {target_pos}")
            cprint.info(f"Agent: {agent_pos} facing {agent_dir_str}")
            cprint.info(f"Front: {front_type}")
            cprint.info("-" * 60)
            cprint.info(prompt_for_log)
            cprint.info("=" * 60)
        # End verbose prompt logging block
        
        return prompt
    
    def suggest_actions(self, grid, agent_pos, agent_dir, goal_pos, max_actions=5):

        # Suggest actions through the LLM and parse only action outputs
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        grid_ascii = build_ascii_grid(grid, agent_pos, goal_pos)
        prompt = self.build_prompt(grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid)
        
        # Force JSON-only output
        prompt = prompt.replace(
            "RESPOND WITH ONLY THIS JSON FORMAT (no text before or after):",
            "Output ONLY the JSON with the actions, no explanation.\n\n"
            "RESPOND WITH THIS JSON FORMAT:"
        )
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.15,
                    "num_predict": 25
                },
                format="json" 
            )
            raw = response.response
            context = getattr(response, 'context', [])
            thinking = getattr(response, 'thinking', '')
            
            # Remove optional <thinking> tags (models like Qwen)
            if "<thinking>" in raw or "</thinking>" in raw:
                raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
                raw = raw.strip()
                if self.verbose:
                    cprint.warn("Removed <thinking> tags from LLM response")
            
            if self.verbose:
                cprint.highlight("=== FULL OLLAMA RESPONSE ===")
                cprint.highlight("Response type:", type(response).__name__)
                cprint.highlight("Context length:", len(context) if context else 0)
                if thinking:
                    cprint.highlight("Thinking:", thinking)
                cprint.highlight("Max actions required:", max_actions)
                cprint.highlight("--- RAW LLM RESPONSE ---")
                cprint.highlight(f"Response length: {len(raw)} chars")
                cprint.highlight(raw if raw else "(EMPTY RESPONSE)")
                
        except Exception as e:
            cprint.error("Ollama error:", e)
            return [], 0
        
            
        # Parse JSON response with a robust multi-strategy approach
        try:
            seq = []
            
            # Strategy 1: Match {"actions": [...]} with tolerant formatting
            json_match = re.search(r'\{\s*["\']?actions["\']?\s*:\s*\[[^\]]*\]\s*\}', raw, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    resp_json = json.loads(json_match.group(0))
                    seq = resp_json.get("actions", [])
                    if self.verbose:
                        cprint.info(f"Found {{actions: [...]}} format - {len(seq)} actions")
                except json.JSONDecodeError as je:
                    if self.verbose:
                        cprint.warn(f"JSON parsing error: {je}")
            
            # Strategy 2: Match standalone arrays with valid actions
            if not seq:
                array_match = re.search(r'\[\s*["\'](' + '|'.join(VALID_ACTIONS) + r')["\'](?:\s*,\s*["\'](' + '|'.join(VALID_ACTIONS) + r')["\'])*\s*\]', raw, re.IGNORECASE)
                if array_match:
                    try:
                        seq = json.loads(array_match.group(0))
                        if self.verbose:
                            cprint.info(f"Found standalone array - {len(seq)} actions")
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Extract quoted actions in order
            if not seq:
                actions_match = re.findall(r'["\'](' + '|'.join(VALID_ACTIONS) + r')["\']', raw, re.IGNORECASE)
                if actions_match:
                    seq = actions_match[:max_actions]
                    if self.verbose:
                        cprint.info(f"Extracted actions from regex pattern - {len(seq)} actions")
            
            if not seq:
                valid_list = VALID_ACTIONS_DOORKEY if "doorkey" in self.env_type.lower() else VALID_ACTIONS
                # Keep only letters and spaces for free-text fallback parsing
                raw_clean = re.sub(r'[^a-zA-Z\s]', ' ', raw).lower()
                tokens = raw_clean.split()
                found_actions = [t for t in tokens if t in valid_list]
                
                # Accept if at least one valid action is found
                if found_actions:
                    seq = found_actions
                    if self.verbose:
                        cprint.warn("Used Strategy 4 (free text parsing).")
            
            if not isinstance(seq, list):
                seq = []
            
            # Final normalization and truncation
            valid_list = VALID_ACTIONS_DOORKEY if "doorkey" in self.env_type.lower() else VALID_ACTIONS
            seq = [str(t).lower().strip() for t in seq if str(t).lower().strip() in valid_list][:max_actions]
            
            if not seq:
                # No A* fallback here so the LLM must improve
                cprint.error("=" * 60)
                cprint.error("PARSING FAILED (LavaCrossing/Empty) - LLM response:")
                cprint.error(raw)
                cprint.error("=" * 60)
                cprint.error("No valid suggestion - LLM must improve")
                return [], 1  # Parsing failure counts as hallucination
        except Exception as e:
            cprint.error(f"Error parsing: {e}")
            cprint.error("Raw response:")
            cprint.error(raw)
            return [], 0
        
        if self.verbose:
            cprint.info("Parsed sequence:", seq)
        
        # Strict validation: simulate each action
        safe = []
        sim_pos = [int(agent_pos[0]), int(agent_pos[1])]
        sim_dir = int(agent_dir)
        
        for i, action in enumerate(seq):
            # Check if the action is valid in the current simulated state
            if simulate_actions(sim_pos, sim_dir, [action], grid):
                safe.append(action)
                # Update simulated state
                if action == "forward":
                    dx, dy = DIR2VEC[sim_dir]
                    sim_pos = [sim_pos[0] + dx, sim_pos[1] + dy]
                elif action == "left":
                    sim_dir = (sim_dir - 1) % 4
                elif action == "right":
                    sim_dir = (sim_dir + 1) % 4
            else:
                if self.verbose:
                    cprint.warn(f"Action {i+1} '{action}' not valid, truncated sequence")
                break
        
        if self.verbose:
            cprint.success(f"Valid actions: {len(safe)}/{len(seq)}")
        
        # Log suggested actions versus validated actions
        self.log_call(len(seq), len(safe))
        
        # Return (validated_actions, 1 if truncated, else 0)
        was_truncated = 1 if len(safe) < len(seq) else 0
        return safe, was_truncated
    
    def suggest_actions_doorkey(self, grid, agent_pos, agent_dir, goal_pos, base_env, max_actions=5):
        # Suggest actions for DoorKey and parse only valid actions
        agent_pos = (int(agent_pos[0]), int(agent_pos[1]))
        goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        agent_dir = int(agent_dir)
        
        grid_ascii = build_ascii_grid(grid, agent_pos, goal_pos)
        prompt = self.build_prompt_doorkey(grid_ascii, agent_pos, agent_dir, goal_pos, max_actions, grid, base_env)
        
        # Force JSON-only output

        prompt = prompt.replace(
            "RESPOND WITH ONLY THIS JSON FORMAT (no text before or after):",
            "Output ONLY the JSON with the actions, no explanation.\n\n"
            "RESPOND WITH THIS JSON FORMAT:"
        )
                
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "repeat_penalty": 1.15,
                    "num_predict": 25,
                },
                format="json"
            )
            raw = response.response
            context = getattr(response, 'context', [])
            thinking = getattr(response, 'thinking', '')
            
            if "<thinking>" in raw or "</thinking>" in raw:
                raw = re.sub(r'<thinking>.*?</thinking>', '', raw, flags=re.DOTALL)
                raw = raw.strip()
                if self.verbose:
                    cprint.warn("Removed <thinking> tags from LLM response (DoorKey)")
            
            if self.verbose:
                cprint.highlight("=== FULL OLLAMA RESPONSE (DoorKey) ===")
                cprint.highlight("Response type:", type(response).__name__)
                cprint.highlight("Context length:", len(context) if context else 0)
                if thinking:
                    cprint.highlight("Thinking:", thinking)
                cprint.highlight("Max actions requested:", max_actions)
                cprint.highlight("--- RAW LLM RESPONSE (DoorKey) ---")
                cprint.highlight(f"Response length: {len(raw)} chars")
                cprint.highlight(raw if raw else "(EMPTY RESPONSE)")
                
        except Exception as e:
            cprint.error("Ollama error:", e)
            return [], 0
        
        # Parsing logic remains unchanged
        
        # Parse JSON response with a robust extraction pipeline
        try:
            seq = []
            
            # Strategy 1: Match {"actions": [...]} with spaces/newlines
            json_match = re.search(r'\{\s*["\']?actions["\']?\s*:\s*\[[^\]]*\]\s*\}', raw, re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    resp_json = json.loads(json_match.group(0))
                    seq = resp_json.get("actions", [])
                    if self.verbose:
                        cprint.info(f"Found format {{actions: [...]}} - {len(seq)} actions")
                except json.JSONDecodeError as je:
                    if self.verbose:
                        cprint.warn(f"Malformed JSON in match: {je}")
            
            # Strategy 2: Match standalone arrays with valid actions
            if not seq:
                # More robust pattern for string arrays
                array_match = re.search(r'\[\s*["\'](' + '|'.join(VALID_ACTIONS_DOORKEY) + r')["\'](?:\s*,\s*["\'](' + '|'.join(VALID_ACTIONS_DOORKEY) + r')["\'])*\s*\]', raw, re.IGNORECASE)
                if array_match:
                    try:
                        seq = json.loads(array_match.group(0))
                        if self.verbose:
                            cprint.info(f"Found standalone array - {len(seq)} actions")
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Extract all quoted actions in order
            if not seq:
                actions_match = re.findall(r'["\'](' + '|'.join(VALID_ACTIONS_DOORKEY) + r')["\']', raw, re.IGNORECASE)
                if actions_match:
                    # Keep only the first N actions (likely the main array)
                    seq = actions_match[:max_actions]
                    if self.verbose:
                        cprint.info(f"Extracted actions from regex pattern - {len(seq)} actions")
            
            if not isinstance(seq, list):
                seq = []
            # Normalize common action aliases
            action_map = {
                "rotate_left": "left",
                "rotate_right": "right",
                "turn_left": "left",
                "turn_right": "right",
                # Add more aliases if needed
            }
            seq = [action_map.get(action, action) for action in seq]
            # Keep only valid DoorKey actions
            seq = [str(t).lower().strip() for t in seq if str(t).lower().strip() in VALID_ACTIONS_DOORKEY][:max_actions]
            
            if not seq:
                # No A* fallback here so the LLM must improve
                cprint.error("=" * 60)
                cprint.error("PARSING FAILED (DoorKey) - Full LLM response:")
                cprint.error(raw)
                cprint.error("=" * 60)
                cprint.error("No valid suggestion - LLM must improve")
                return [], 1  # Parsing failure counts as hallucination
                
        except Exception as e:
            if self.verbose:
                cprint.warn(f"Error parsing for DoorKey: {e}")
            return [], 0
        
        if self.verbose:
            cprint.info("Parsed sequence DoorKey:", seq)
        
        # Strict validation with hallucination detection
        safe = []
        has_hallucination = False  # Binary flag: 1 if at least one error occurs
        sim_pos = [int(agent_pos[0]), int(agent_pos[1])]
        sim_dir = int(agent_dir)
        
        # Current state used for validation
        has_key = agent_has_key(base_env)
        door_open = is_door_open(grid)
        key_pos = find_key_position(grid)
        door_pos = find_door_position(grid)
        
        for i, action in enumerate(seq):
            is_hallucination = False
            
            # Validate pickup/toggle preconditions
            if action == "pickup":
                # Pickup is valid only when the agent has no key and is facing the key
                if has_key:
                    # Already holding the key -> hallucination
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'pickup' but already has key!")
                    has_hallucination = True
                    is_hallucination = True
                elif not is_agent_in_front_of(sim_pos, sim_dir, key_pos):
                    # Not facing the key -> hallucination
                    if self.verbose:
                        # Compute effective distance for debugging
                        if key_pos:
                            dist = abs(sim_pos[0] - key_pos[0]) + abs(sim_pos[1] - key_pos[1])
                            cprint.error(f"HALLUCINATION: 'pickup' but not facing key! Distance: {dist}, Key at {key_pos}, Agent at {sim_pos} facing {sim_dir}")
                        else:
                            cprint.error(f"HALLUCINATION: 'pickup' but no key exists!")
                    has_hallucination = True
                    is_hallucination = True
                    
            elif action == "toggle":
                # Toggle is valid only when the agent has the key, door is closed, and agent faces the door
                if door_open:
                    # Door already open -> hallucination
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'toggle' but door already open!")
                    has_hallucination = True
                    is_hallucination = True
                elif not has_key and not self.pickup_executed:
                    # No key available -> hallucination
                    if self.verbose:
                        cprint.error(f"HALLUCINATION: 'toggle' but doesn't have key!")
                    has_hallucination = True
                    is_hallucination = True
                elif not is_agent_in_front_of(sim_pos, sim_dir, door_pos):
                    # Not facing the door -> hallucination
                    if self.verbose:
                        # Compute effective distance for debugging
                        if door_pos:
                            dist = abs(sim_pos[0] - door_pos[0]) + abs(sim_pos[1] - door_pos[1])
                            cprint.error(f"HALLUCINATION: 'toggle' but not facing door! Distance: {dist}, Door at {door_pos}, Agent at {sim_pos} facing {sim_dir}")
                        else:
                            cprint.error(f"HALLUCINATION: 'toggle' but no door exists!")
                    has_hallucination = True
                    is_hallucination = True
            
            if is_hallucination:
                continue  # Skip this action and continue
            
            # Validate movement actions
            if action in ["forward", "left", "right"]:
                # Check forward into closed doors and hard obstacles
                if action == "forward":
                    dx, dy = DIR2VEC[sim_dir]
                    next_x, next_y = sim_pos[0] + dx, sim_pos[1] + dy
                    if 0 <= next_x < grid.width and 0 <= next_y < grid.height:
                        cell = grid.get(next_x, next_y)
                        if cell:
                            cell_type = getattr(cell, "type", None)
                            # Closed door
                            if cell_type == "door" and not getattr(cell, "is_open", False):
                                if self.verbose:
                                    cprint.error(f"HALLUCINATION: 'forward' with DOOR CLOSE at ({next_x}, {next_y})!")
                                has_hallucination = True
                                continue
                            # Wall or lava
                            if cell_type in ["wall", "lava"]:
                                if self.verbose:
                                    cprint.error(f"HALLUCINATION: 'forward' on {cell_type.upper()} at ({next_x}, {next_y})!")
                                has_hallucination = True
                                continue
                            # Key occupies a physical cell
                            if cell_type == "key":
                                if self.verbose:
                                    cprint.error(f"HALLUCINATION: 'forward' on KEY (use pickup) at ({next_x}, {next_y})!")
                                has_hallucination = True
                                continue
                    else:
                        # Out of bounds
                        if self.verbose:
                            cprint.error(f"HALLUCINATION: 'forward' out of bounds towards ({next_x}, {next_y})!")
                        has_hallucination = True
                        continue
                
                # Final validation via simulate_actions
                if simulate_actions(sim_pos, sim_dir, [action], grid, base_env, env_type="DoorKey"):
                    safe.append(action)
                    # Update simulated state
                    if action == "forward":
                        dx, dy = DIR2VEC[sim_dir]
                        sim_pos = [sim_pos[0] + dx, sim_pos[1] + dy]
                    elif action == "left":
                        sim_dir = (sim_dir - 1) % 4
                    elif action == "right":
                        sim_dir = (sim_dir + 1) % 4
                else:
                    if self.verbose:
                        cprint.warn(f"Action {i+1} '{action}' not valid (movement blocked)")
                    has_hallucination = True
                    continue
                    
            elif action == "pickup":
                safe.append(action)
                self.pickup_executed = True
                has_key = True  # Update simulated state
                if self.verbose:
                    cprint.success("Added valid pickup")
                    
            elif action == "toggle":
                safe.append(action)
                self.toggle_executed = True
                door_open = True  # Update simulated state
                if self.verbose:
                    cprint.success("Added valid toggle")
        
        if self.verbose:
            cprint.success(f"Valid actions in DoorKey: {len(safe)}/{len(seq)}")
            if has_hallucination:
                cprint.error(f"Hallucination detected in this response")
        
        # Log helper call statistics
        self.log_call(len(seq), len(safe))
        
        # Return 1 if at least one hallucination occurred, else 0
        return safe, 1 if has_hallucination else 0

# Train with environment rotation
def train(episodes=2650,use_llm_helper=True,render_env=False,helper_episode_limit=2200,threshold_initial=1.0,threshold_decay=0.1,max_steps_before_llm=4,max_llm_actions_factor=2.0,replay_frequency=4,verbose=False,strategy="threshold",initial_max_moves=15):
    helper = LLMHelper(verbose=verbose) if use_llm_helper else None
    
    log_dir = ROOT_DIR / "Helper" / "Logs" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if helper:
        helper.init_logging("multi_env_rotation",log_dir=log_dir)
    
    
    stats = {
    "episode_rewards": [],
    "episode_moves": [],
    "llm_suggestions_used": [],
    "llm_hallucinations": [],
    "environment_used": [],
    "episode_success": [],
    "episode_termination": []
    }

    # Aggregate statistics per environment
    env_stats = {env["id"]: {
        "episodes": 0,
        "successes": 0,
        "total_reward": 0,
        "total_steps": 0,
        "rewards": [],
        "steps": [],
        "terminations": {"goal": 0, "timeout": 0, "death": 0}
    } for env in ALL_ENVIRONMENTS}

    # Track current environment state
    current_env_config = None
    prev_env_config = None
    env_wrapper = None
    agent = None
    layout_mgr = None
    
    dir_names = ["Right (>)", "Down (v)", "Left (<)", "Up (^)"]

    for ep in range(episodes):
        # Select environment for this episode
        scheduler = None
        env_config = get_environment_for_episode(ep)
        env_id = env_config["id"]
        env_type = env_config["type"]
        env_size = env_config["size"]
        env_name = f"{env_type}-{env_size}"
        # Reset buffer as in train_all_maps.py (curriculum only, not mixed phase)
        # Keep a soft reset instead of clearing everything
        if ep < 2200 and prev_env_config is not None:
            prev_type = prev_env_config["type"]
            current_type = env_config["type"]

            if current_type == "Crossing" and current_env_config["id"] != env_id:
                if agent is not None:
                    KEEP_SAMPLES = 3000
                    if len(agent.replay_buffer) > KEEP_SAMPLES:
                        remove_count = len(agent.replay_buffer) - KEEP_SAMPLES
                        for _ in range(remove_count):
                            agent.replay_buffer.popleft()
                        if hasattr(agent, 'priorities') and len(agent.priorities) > KEEP_SAMPLES:
                            for _ in range(remove_count):
                                if hasattr(agent.priorities, 'popleft'):
                                    agent.priorities.popleft()
                        cprint.warn(f">>> CROSSING SOFT RESET (ep {ep+1}): kept {KEEP_SAMPLES} samples from the previous buffer")
                    else:
                        cprint.warn(f">>> CROSSING: buffer too small ({len(agent.replay_buffer)}), no removal")

            elif current_type == "DoorKey" and prev_type != "DoorKey":
                if agent is not None:
                    # DoorKey differs enough to need a more aggressive, non-total reset
                    KEEP_SAMPLES = 1000
                    if len(agent.replay_buffer) > KEEP_SAMPLES:
                        remove_count = len(agent.replay_buffer) - KEEP_SAMPLES
                        for _ in range(remove_count):
                            agent.replay_buffer.popleft()
                    cprint.warn(f">>> DOORKEY PHASE START (ep {ep+1}): buffer reduced to {KEEP_SAMPLES} samples")


        # Change environment when required
        if current_env_config is None or current_env_config["id"] != env_id:
            cprint.highlight(f"\n{'='*60}")
            if ep >= 2200:
                cprint.highlight(f"EPISODE {ep+1}: SELECTED ENVIRONMENT -> {env_name}")
            else:
                cprint.highlight(f"EPISODE {ep+1}: NEW ENVIRONMENT -> {env_name}")
            cprint.highlight(f"{'='*60}\n")
            
            # Close the previous environment if present
            if layout_mgr:
                layout_mgr.close()
                layout_mgr = None
            
            # Create a new environment
            render_mode = "rgb_array" if render_env else None
            env = gym.make(env_id, render_mode=render_mode)
            env = FullyObsWrapper(env)
            env_wrapper = DynamicMiniGridWrapper(env, env_type)
            
            if ep >= 2200:  # Mixed phase
                env_wrapper.skip_seed_logic = True
                cprint.info(">>> MIXED PHASE: skip_seed_logic ENABLED (max variability)")
            else:  # Curriculum phase
                env_wrapper.skip_seed_logic = False
                cprint.info(">>> CURRICULUM PHASE: seed logic ACTIVE (consistency)")
            
            if agent is None:
                example_state = flatten_obs(env_wrapper.get_state())
                agent = DQNAgent(
                    len(example_state), 
                    env_wrapper.action_size,
                    dueling=False,      # Standard DQN, not dueling
                    prioritized=False,  # No prioritized replay
                    batch_size=64       # Batch size standard
                )
            
            # Update helper with current environment type
            if helper:
                helper.env_type = env_type
            
            # Set up the layout manager when rendering is enabled
            if render_env and LayoutManager:
                layout_mgr = LayoutManager(env_id)
            
            # Reconfigure the agent on environment change
            if agent is not None:
                configure_agent_for_map(agent, env_config, prev_env_config)
                if env_type == "Crossing":
                    scheduler = None
                else:
                    scheduler = ReduceLROnPlateau(agent.optimizer, mode='max', factor=0.5, patience=40)
            
            prev_env_config = current_env_config
            current_env_config = env_config
        
        
        # Reset episode
        state_raw = env_wrapper.reset()
        state = flatten_obs(state_raw)
        done = False
        total_reward = 0
        moves = 0
        steps_since_llm = 0  # Kept for compatibility but no longer used
        ep_sugg = 0
        ep_hall = 0
        step_counter = 0
        episode_loss = 0
        loss_count = 0
        
        # Initialize per-episode threshold.
        threshold = threshold_initial  # Reset threshold each episode
        
        # Reset helper counters for the new episode.
        if helper:
            helper.reset_episode(ep + 1, env_name)

        # Update initial layout text
        if layout_mgr:
            start_pos = (int(env_wrapper.base_env.agent_pos[0]), int(env_wrapper.base_env.agent_pos[1]))
            start_dir = dir_names[env_wrapper.base_env.agent_dir]
            layout_mgr.update_text([
                f"Env: {env_name}",
                f"Start Episode...",
                f"Start Pos: {start_pos} Dir: {start_dir}"
            ], episode_num=ep+1)

        while not done:
            # Read current agent position and direction
            agent_pos_raw = env_wrapper.base_env.agent_pos
            current_pos = (int(agent_pos_raw[0]), int(agent_pos_raw[1]))
            current_dir_idx = env_wrapper.base_env.agent_dir
            current_dir_str = dir_names[current_dir_idx]
            goal_pos = env_wrapper.base_env.goal_pos


            valid_actions = env_wrapper.get_valid_actions()
            
            if strategy == "threshold":
                p = np.random.rand()
                use_llm = (p > threshold and ep < helper_episode_limit and use_llm_helper and helper is not None)

            elif strategy == "initial":
                use_llm = (moves < initial_max_moves and ep < helper_episode_limit and use_llm_helper and helper is not None)

            elif strategy == "random":
                p = np.random.rand()
                use_llm = (steps_since_llm >= max_steps_before_llm and p > 0.5 and ep < helper_episode_limit and use_llm_helper and helper is not None)

            else:
                raise ValueError(f"Invalid strategy '{strategy}'. Use: 'threshold', 'initial', 'random'")
            
            
            if use_llm:
                # Pause and capture position (requested behavior)
                if render_env:
                    if layout_mgr:
                        layout_mgr.update_text([
                            f"Env: {env_name}",
                            f"Paused 3s...",
                            "Position requested..."
                        ], episode_num=ep+1)
                        layout_mgr.render(env_wrapper.base_env.render())
                    time.sleep(3)  # Pause 1: pre-capture (3 seconds)

                # Re-acquire state after pause (glitch-safe)
                agent_pos_raw = env_wrapper.base_env.agent_pos
                current_pos = (int(agent_pos_raw[0]), int(agent_pos_raw[1]))
                agent_dir = env_wrapper.base_env.agent_dir
                goal_pos = env_wrapper.base_env.goal_pos
                grid = env_wrapper.base_env.grid
                
                current_dir_idx = agent_dir
                current_dir_str = dir_names[current_dir_idx]
                
                # Force max 5 actions
                max_actions = 5
                
                if env_type == "DoorKey":
                    plan_actions, llm_hallucinations = helper.suggest_actions_doorkey(
                        grid, agent_pos_raw, agent_dir, goal_pos, env_wrapper.base_env, max_actions=max_actions
                    ) if helper else ([], 0)
                else:
                    plan_actions, llm_hallucinations = helper.suggest_actions(
                        grid, agent_pos_raw, agent_dir, goal_pos, max_actions=max_actions
                    ) if helper else ([], 0)
                
                ep_hall += llm_hallucinations

                # Show generated plan
                if layout_mgr:
                    if plan_actions:
                        layout_mgr.update_text([
                            f"Env: {env_name}",
                            f"Step: {moves} | Goal: {goal_pos}",
                            f"Pos: {current_pos} | Dir: {current_dir_str}",
                            "--- LLM PLAN (Max 5) ---",
                            str(plan_actions),
                            "-------------------------",
                        ], episode_num=ep+1)
                        # Force render so text is visible before the pause
                        if render_env:
                            layout_mgr.render(env_wrapper.base_env.render())
                    else:
                        layout_mgr.update_text([
                            f"Env: {env_name}",
                            "LLM Failed / No plan.",
                            "Fallback agent."
                        ], episode_num=ep+1)
                        if render_env:
                            layout_mgr.render(env_wrapper.base_env.render())

                # Pause for plan reading (requested behavior)
                if render_env and plan_actions:
                    time.sleep(3)  # Pause 2: post-response / pre-execution (3 seconds)

                if not plan_actions:
                    cprint.error("LLM did not suggest valid actions.")
                    ep_hall += 1
                    action = agent.act(state, valid_actions)
                    next_state_raw, reward, done, info = env_wrapper.step(action)
                    # Sync lava death reward with environment.py for stability
                    if env_type == "Crossing" and done and not info.get("goal_reached", False) and not info.get("timeout", False):
                        reward = -50.0
                    next_state = flatten_obs(next_state_raw)
                    valid_mask = [1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                    next_valid_actions = env_wrapper.get_valid_actions()
                    next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                    agent.remember(state, action, reward, next_state, done, valid_mask, next_valid_mask)
                    train_freq = 8 if env_type == "DoorKey" else 4
                    if step_counter % train_freq == 0:
                        loss = agent.replay(n_steps=1)
                        if loss is not None:
                            episode_loss += loss
                            loss_count += 1
                    state = next_state
                    total_reward += reward
                    moves += 1
                    step_counter += 1
                    steps_since_llm = 0  
                    
                    # Decrease threshold as in the baseline
                    threshold = max(0.0, threshold - threshold_decay)
                    
                    if render_env and layout_mgr:
                        layout_mgr.render(env_wrapper.base_env.render())
                else:
                    for s in plan_actions:
                        action_map = {"left": 0, "right": 1, "forward": 2, "pickup": 3, "toggle": 5}
                        if s not in action_map:
                            continue
                        action = action_map[s]
                        
                        if s in ["left", "right", "forward"]:
                            ok = simulate_actions(env_wrapper.base_env.agent_pos, env_wrapper.base_env.agent_dir, [s], env_wrapper.base_env.grid, env_wrapper.base_env, env_type=env_type)
                        else:
                            ok = True
                            
                        if not ok:
                            ep_hall += 1
                            action = agent.act(state, valid_actions)
                        else:
                            ep_sugg += 1
                        
                        next_state_raw, reward, done, info = env_wrapper.step(action)
                        # Sync lava death reward with environment.py for stability
                        if env_type == "Crossing" and done and not info.get("goal_reached", False) and not info.get("timeout", False):
                            reward = -50.0
                        next_state = flatten_obs(next_state_raw)
                        valid_mask = [1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                        next_valid_actions = env_wrapper.get_valid_actions()
                        next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                        agent.remember(state, action, reward, next_state, done, valid_mask, next_valid_mask)
                        # Train with adaptive frequency
                        train_freq = 8 if env_type == "DoorKey" else 4
                        if step_counter % train_freq == 0:
                            loss = agent.replay(n_steps=1)
                            if loss is not None:
                                episode_loss += loss
                                loss_count += 1
                        state = next_state
                        total_reward += reward
                        moves += 1
                        step_counter += 1
                        
                        # Decrease threshold as in the baseline
                        threshold = max(0.0, threshold - threshold_decay)
                        
                        # Forced in-loop render synchronization
                        if layout_mgr:
                            # Update post-step position
                            raw_p = env_wrapper.base_env.agent_pos
                            new_pos = (int(raw_p[0]), int(raw_p[1]))
                            new_dir = dir_names[env_wrapper.base_env.agent_dir]
                            
                            layout_mgr.update_text([
                                f"Env: {env_name}",
                                f"Step: {moves} | Goal: {goal_pos}",
                                f"Pos: {new_pos} | Dir: {new_dir}",
                                "--- LLM PLAN (Running) ---",
                                str(plan_actions),
                                f"> Action: {s}",
                                "----------------------------",
                            ], episode_num=ep+1)
                            
                            if render_env:
                                layout_mgr.render(env_wrapper.base_env.render())
                        
                        if done:
                            break
                    steps_since_llm = 0
            else:
                # Agent action (no LLM)
                action = agent.act(state, valid_actions)
                next_state_raw, reward, done, info = env_wrapper.step(action)
                # Sync lava death reward with environment.py for stability
                if env_type == "Crossing" and done and not info.get("goal_reached", False) and not info.get("timeout", False):
                    reward = -50.0
                next_state = flatten_obs(next_state_raw)
                valid_mask = [1 if a in valid_actions else 0 for a in range(env_wrapper.action_size)]
                next_valid_actions = env_wrapper.get_valid_actions()
                next_valid_mask = [1 if a in next_valid_actions else 0 for a in range(env_wrapper.action_size)]
                agent.remember(state, action, reward, next_state, done, valid_mask, next_valid_mask)
                if step_counter % replay_frequency == 0:
                    agent.replay()
                state = next_state
                total_reward += reward
                moves += 1
                step_counter += 1
                steps_since_llm += 1
                
                # Decrease threshold as in the baseline
                threshold = max(0.0, threshold - threshold_decay)

                # Update layout text with current agent position
                if layout_mgr and moves % 2 == 0:
                    layout_mgr.update_text([
                        f"Env: {env_name}",
                        f"Step: {moves} (Agent)",
                        f"Pos: {current_pos}",
                        f"Dir: {current_dir_str}",
                        f"Threshold: {threshold:.3f}",
                        "---", "Waiting for LLM..."
                    ], episode_num=ep+1)

                if render_env and layout_mgr:
                    layout_mgr.render(env_wrapper.base_env.render())

            if render_env and not layout_mgr and moves % 5 == 0:
                env_wrapper.render()
                time.sleep(0.01)

        stats["episode_rewards"].append(total_reward)
        # Post-episode logic (as in train_all_maps.py)
        global timeout_streak, lava_death_count, crossing_timeout_count, crossing_goal_count

        # 1. Anti-timeout epsilon boost.
        if info.get("timeout", False):
            timeout_streak += 1
        else:
            timeout_streak = 0

        if timeout_streak >= 15:
            # If stuck for 15 episodes, increase exploration
            agent.epsilon = min(agent.epsilon + 0.25, 1.0)
            timeout_streak = 0
            cprint.warn(f"    !!! TIMEOUT STREAK: Epsilon boosted to {agent.epsilon:.2f}")

        # 2. Crossing-specific counters (as in train_npc.py/train_all_maps.py)
        if env_type == "Crossing":
            if info.get("goal_reached", False):
                crossing_goal_count += 1
            if info.get("lava_death", False):
                lava_death_count += 1
                # Optional: speed up epsilon decay after lava death
                #agent.epsilon *= 0.99
            if info.get("timeout", False):
                crossing_timeout_count += 1

        # 3. Extra replay every 5 episodes (disabled for DoorKey)
        if (ep + 1) % 5 == 0 and env_type != "DoorKey":
            agent.replay(n_steps=1)

        # 4. Periodic DoorKey buffer cleanup (anti-overfitting)
        if env_type == "DoorKey" and (ep + 1) % 100 == 0:
            if len(agent.replay_buffer) > 5000:
                remove_count = len(agent.replay_buffer) - 5000
                for _ in range(remove_count):
                    agent.replay_buffer.popleft()
                if hasattr(agent, 'priorities') and len(agent.priorities) > 0:
                    for _ in range(min(remove_count, len(agent.priorities))):
                        agent.priorities.popleft() if hasattr(agent.priorities, 'popleft') else agent.priorities.pop(0)
                cprint.info(f"    >>> Buffer cleanup: removed {remove_count} old samples (anti-overfitting)")

            # 5. Update scheduler when available
        if scheduler is not None:
            scheduler.step(total_reward)

            # 6. Print average loss when available
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        if (ep + 1) % 10 == 0 and loss_count > 0:
            cprint.info(f"    Avg Loss (ep {ep+1}): {avg_loss:.4f}")
        stats["episode_moves"].append(moves)
        stats["llm_suggestions_used"].append(ep_sugg)
        stats["llm_hallucinations"].append(ep_hall)
        stats["environment_used"].append(env_name)
        # Determine success and termination type
        success = 1 if info.get("goal_reached", False) else 0
        if info.get("goal_reached", False):
            termination = "goal"
        elif info.get("timeout", False):
            termination = "timeout"
        elif info.get("lava_death", False) or (env_type == "Crossing" and done and not info.get("goal_reached", False)):
            termination = "death"
        else:
            termination = "timeout"

        stats["episode_success"].append(success)
        stats["episode_termination"].append(termination)

        # Update per-environment statistics
        env_stats[env_id]["episodes"] += 1
        env_stats[env_id]["successes"] += success
        env_stats[env_id]["total_reward"] += total_reward
        env_stats[env_id]["total_steps"] += moves
        env_stats[env_id]["rewards"].append(total_reward)
        env_stats[env_id]["steps"].append(moves)
        env_stats[env_id]["terminations"][termination] += 1
        
        # Completed episode summary
        mode_str = "RANDOM" if ep >= helper_episode_limit else f"Batch {ep//50 + 1}"
        helper_str = f"[Helper: {ep_sugg}s/{ep_hall}h]" if ep < helper_episode_limit else "[No Helper]"
        cprint.success(f"Ep {ep+1}/{episodes} [{mode_str}] [{env_name}] {helper_str}: "
                      f"reward={total_reward:.2f}, moves={moves}, "
                      f"llm_sugg={ep_sugg}, llm_hall={ep_hall}, "
                      f"epsilon={agent.epsilon:.3f}")

    total_rewards = sum(stats["episode_rewards"])
    total_moves = sum(stats["episode_moves"])
    total_llm_suggestions = sum(stats["llm_suggestions_used"])
    total_llm_hallucinations = sum(stats["llm_hallucinations"])
    avg_reward = total_rewards / episodes
    avg_moves = total_moves / episodes
    
    cprint.highlight("\n=== TRAINING SUMMARY ===")
    cprint.info(f"Total Episodes: {episodes}")
    cprint.info(f"Helper used up to episode: {helper_episode_limit}")
    cprint.info(f"Threshold config: initial={threshold_initial}, decay={threshold_decay}")
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
    
    # Close logging
    if helper:
        helper.close_logging()

    return stats, env_stats, log_dir

# Plot helpers
def print_environment_stats(env_stats):
    # Print detailed statistics for each environment
    cprint.highlight("\n" + "="*80)
    cprint.highlight("DETAILED STATISTICS PER ENVIRONMENT")
    cprint.highlight("="*80)
    
    for env_id, stats in env_stats.items():
        if stats["episodes"] == 0:
            continue
            
        env_config = next((e for e in ALL_ENVIRONMENTS if e["id"] == env_id), None)
        if not env_config:
            continue
            
        env_type = env_config["type"]
        env_size = env_config["size"]
        
        success_rate = (stats["successes"] / stats["episodes"]) * 100
        avg_reward = stats["total_reward"] / stats["episodes"]
        avg_steps = stats["total_steps"] / stats["episodes"]
        
        cprint.info(f"\n{env_type}-{env_size} ({env_id}):")
        cprint.info(f"  Episodes: {stats['episodes']}")
        cprint.info(f"  Success Rate: {success_rate:.1f}% ({stats['successes']}/{stats['episodes']})")
        cprint.info(f"  Avg Reward: {avg_reward:.2f}")
        cprint.info(f"  Avg Steps: {avg_steps:.1f}")
        
        # Termination statistics
        term_stats = stats["terminations"]
        cprint.info(f"  Terminations:")
        cprint.info(f"    - Goal: {term_stats['goal']} ({term_stats['goal']/stats['episodes']*100:.1f}%)")
        cprint.info(f"    - Timeout: {term_stats['timeout']} ({term_stats['timeout']/stats['episodes']*100:.1f}%)")
        cprint.info(f"    - Death: {term_stats['death']} ({term_stats['death']/stats['episodes']*100:.1f}%)")
        
        if stats["rewards"]:
            cprint.info(f"  Reward Range: [{min(stats['rewards']):.1f}, {max(stats['rewards']):.1f}]")
        if stats["steps"]:
            cprint.info(f"  Steps Range: [{min(stats['steps'])}, {max(stats['steps'])}]")
    
    cprint.highlight("="*80 + "\n")


def plot_stats(stats, log_dir):
    plots_dir = log_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Time-series plots
    
    plt.figure(figsize=(12, 5))
    plt.plot(stats["episode_rewards"], linewidth=1.5)
    plt.title("Rewards per Episode", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.axvline(x=2200, color='red', linestyle='--', linewidth=2, label='Helper stops (ep 2200)')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "rewards_over_time.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(12, 5))
    plt.plot(stats["episode_moves"], linewidth=1.5, color='orange')
    plt.title("Steps per Episode", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Steps", fontsize=12)
    plt.axvline(x=2200, color='red', linestyle='--', linewidth=2, label='Helper stops (ep 2200)')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "steps_over_time.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(12, 5))
    plt.plot(stats["llm_suggestions_used"], label="LLM Suggestions Used", linewidth=1.5)
    plt.plot(stats["llm_hallucinations"], label="LLM Hallucinations", linewidth=1.5)
    plt.title("LLM Suggestions & Hallucinations", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.axvline(x=2200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Helper stops')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "llm_stats_over_time.png", dpi=150)
    plt.close()
    
    # Success rate rolling average
    plt.figure(figsize=(12, 5))
    window = 50
    success_smooth = np.convolve(stats["episode_success"], np.ones(window)/window, mode='valid')
    plt.plot(success_smooth, linewidth=2, color='green')
    plt.title(f"Success Rate (rolling average, window={window})", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate", fontsize=12)
    plt.axvline(x=2200, color='red', linestyle='--', linewidth=2, label='Helper stops (ep 2200)')
    for i in range(100, 900, 100):
        plt.axvline(x=i, color='gray', linestyle=':', alpha=0.3)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "success_rate_rolling.png", dpi=150)
    plt.close()
    
    # Steps distribution
    plt.figure(figsize=(10, 5))
    plt.hist(stats["episode_moves"], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    plt.title("Distribution of Steps per Episode", fontsize=14, fontweight='bold')
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.axvline(x=np.mean(stats["episode_moves"]), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(stats["episode_moves"]):.1f}')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plots_dir / "steps_distribution.png", dpi=150)
    plt.close()
    
    # Bar plots by task type
    
    # Aggregate data by task type
    task_types = {"Empty": [], "Crossing": [], "DoorKey": []}
    task_steps = {"Empty": [], "Crossing": [], "DoorKey": []}
    task_rewards = {"Empty": [], "Crossing": [], "DoorKey": []}
    
    for i, env_name in enumerate(stats["environment_used"]):
        task_type = env_name.split("-")[0]
        if task_type in task_types:
            task_types[task_type].append(stats["episode_success"][i])
            task_steps[task_type].append(stats["episode_moves"][i])
            task_rewards[task_type].append(stats["episode_rewards"][i])
    
    # Compute aggregate metrics
    task_success_rates = {}
    task_avg_steps = {}
    task_avg_rewards = {}
    
    for task in task_types.keys():
        if task_types[task]:
            task_success_rates[task] = (sum(task_types[task]) / len(task_types[task])) * 100
            task_avg_steps[task] = sum(task_steps[task]) / len(task_steps[task])
            task_avg_rewards[task] = sum(task_rewards[task]) / len(task_rewards[task])
    
    # Consistent plot colors
    colors = {'Empty': '#5A9BD4', 'Crossing': '#E74C3C', 'DoorKey': '#52B788'}
    
    # Bar plot 1: success rate.
    plt.figure(figsize=(10, 6))
    tasks = list(task_success_rates.keys())
    rates = [task_success_rates[t] for t in tasks]
    bars = plt.bar(tasks, rates, color=[colors[t] for t in tasks], 
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{rate:.1f}%',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.title("Success Rate per Task Type", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Task Type", fontsize=13)
    plt.ylabel("Success Rate (%)", fontsize=13)
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(plots_dir / "success_by_task_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot 2: average steps
    plt.figure(figsize=(10, 6))
    tasks = list(task_avg_steps.keys())
    steps = [task_avg_steps[t] for t in tasks]
    bars = plt.bar(tasks, steps, color=[colors[t] for t in tasks], 
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    
    for bar, step in zip(bars, steps):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(steps)*0.02,
                 f'{step:.1f}',
                 ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.title("Average Steps per Task Type", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Task Type", fontsize=13)
    plt.ylabel("Average Steps", fontsize=13)
    plt.ylim(0, max(steps) * 1.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(plots_dir / "steps_by_task_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot 3: average reward
    plt.figure(figsize=(10, 6))
    tasks = list(task_avg_rewards.keys())
    rewards = [task_avg_rewards[t] for t in tasks]
    bars = plt.bar(tasks, rewards, color=[colors[t] for t in tasks], 
                   edgecolor='black', linewidth=1.5, alpha=0.9)
    
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        # Handle negative rewards
        y_pos = height + abs(max(rewards) - min(rewards))*0.02 if height >= 0 else height - abs(max(rewards) - min(rewards))*0.05
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{reward:.2f}',
                 ha='center', va='bottom' if height >= 0 else 'top', 
                 fontsize=14, fontweight='bold')
    
    plt.title("Average Reward per Task Type", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Task Type", fontsize=13)
    plt.ylabel("Average Reward", fontsize=13)
    # Set dynamic y-limits to include all values
    y_range = max(rewards) - min(rewards)
    plt.ylim(min(rewards) - y_range*0.15, max(rewards) + y_range*0.15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(plots_dir / "rewards_by_task_type.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    cprint.success(f"\nPlots saved in: {plots_dir}")
    cprint.info("Generated plots:")
    cprint.info("  - rewards_over_time.png")
    cprint.info("  - steps_over_time.png")
    cprint.info("  - llm_stats_over_time.png")
    cprint.info("  - success_rate_rolling.png")
    cprint.info("  - steps_distribution.png")
    cprint.info("  - success_by_task_type.png")
    cprint.info("  - steps_by_task_type.png")
    cprint.info("  - rewards_by_task_type.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--llm", action="store_true", default=True)
    parser.add_argument("--episodes", type=int, default=2650, help="Total episodes")
    parser.add_argument("--helper-limit", type=int, default=2200, help="Use helper up to this episode")
    parser.add_argument("--threshold", type=float, default=1.0, help="Initial threshold value")
    parser.add_argument("--decay", type=float, default=0.1, help="Threshold decay per step")
    parser.add_argument("--max-steps", type=int, default=10, help="(Not used anymore, kept for compatibility)")
    parser.add_argument("--replay-freq", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="threshold",
                    choices=["threshold", "initial", "random"],
                    help="LLM strategy: 'threshold' (default), 'initial' (first N moves only), 'random' (every N steps)")
    parser.add_argument("--initial-max-moves", type=int, default=15,
                    help="Initial moves per episode where LLM is used (only for 'initial' strategy)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cprint.highlight("\n" + "="*60)
    cprint.highlight("SIMPLE HELPER - Threshold-based (like baseline)")
    cprint.highlight("="*60)
    cprint.info("Configuration:")
    cprint.info(f"- CURRICULUM SCHEDULE:")
    cprint.info(f"  * Episodes 0-99: Empty-5x5 (100 ep)")
    cprint.info(f"  * Episodes 100-249: Empty-8x8 (150 ep)")
    cprint.info(f"  * Episodes 250-549: Empty-16x16 (300 ep)")
    cprint.info(f"  * Episodes 550-849: LavaCrossingS9N1 (300 ep)")
    cprint.info(f"  * Episodes 850-1149: LavaCrossingS9N3 (300 ep)")
    cprint.info(f"  * Episodes 1150-1449: LavaCrossingS11N5 (300 ep)")
    cprint.info(f"  * Episodes 1450-1649: DoorKey-5x5 (200 ep)")
    cprint.info(f"  * Episodes 1650-1899: DoorKey-8x8 (250 ep)")
    cprint.info(f"  * Episodes 1900-2199: DoorKey-16x16 (300 ep)")
    cprint.info(f"- MIXED PHASE: Episodes 2200+ (random selection)")
    cprint.info(f"- Total episodes: {args.episodes}")
    cprint.info(f"- Helper episode limit: {args.helper_limit}")
    cprint.info(f"- Threshold: initial={args.threshold}, decay={args.decay}")
    cprint.info(f"- LLM Helper: {'Enabled' if args.llm else 'Disabled'}")
    cprint.info(f"- Render: {'Enabled' if args.render else 'Disabled'}")
    cprint.highlight("="*60 + "\n")

    stats, env_stats, log_dir = train(
        episodes=args.episodes,
        use_llm_helper=args.llm,
        render_env=args.render,
        verbose=args.verbose,
        strategy=args.strategy,
        initial_max_moves=args.initial_max_moves,
    )

    # Print detailed statistics
    print_environment_stats(env_stats)

    # Global success rate
    overall_success_rate = (sum(stats["episode_success"]) / args.episodes) * 100
    cprint.highlight("=== GLOBAL SUCCESS RATE ===")
    cprint.info(f"Overall Success: {overall_success_rate:.1f}% ({sum(stats['episode_success'])}/{args.episodes})")
    cprint.highlight("===========================\n")

    plot_stats(stats, log_dir)