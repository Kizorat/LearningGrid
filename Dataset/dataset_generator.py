import csv
import random
import heapq
import json
import itertools
import os

# Map configurations for different MiniGrid environments
ENV_CONFIGS = {
    # Empty maps
    "MiniGrid-Empty-5x5-v0": {"size": 5, "type": "empty"},
    "MiniGrid-Empty-8x8-v0": {"size": 8, "type": "empty"},
    "MiniGrid-Empty-16x16-v0": {"size": 16, "type": "empty"},

    # LavaCrossing maps
    "MiniGrid-LavaCrossingS9N1-v0": {"size": 9, "type": "lava", "crossings": 1},
    "MiniGrid-LavaCrossingS9N3-v0": {"size": 9, "type": "lava", "crossings": 3},
    "MiniGrid-LavaCrossingS11N5-v0": {"size": 11, "type": "lava", "crossings": 5},

    # DoorKey maps
    "MiniGrid-DoorKey-5x5-v0": {"size": 5, "type": "doorkey"},
    "MiniGrid-DoorKey-8x8-v0": {"size": 8, "type": "doorkey"},
    "MiniGrid-DoorKey-16x16-v0": {"size": 16, "type": "doorkey"},
}

# Direction mappings
DIRECTIONS = ["East", "South", "West", "North"]
DIR_DELTA = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

# Base actions
BASE_ACTIONS = ["left", "right", "forward"]
# Extra actions for DoorKey
DOORKEY_ACTIONS = ["pickup", "toggle"]

# Grid Building Functions
def build_grid_empty(size):
    grid = [[" " for _ in range(size)] for _ in range(size)]
    for i in range(size):
        grid[0][i] = "#"
        grid[size-1][i] = "#"
        grid[i][0] = "#"
        grid[i][size-1] = "#"
    return grid

def build_grid_crossing(size, num_crossings, obstacle_char="L"):
    grid = build_grid_empty(size)
    height, width = size, size
    
    rivers_v_candidates = [i for i in range(2, width - 2, 2)]
    rivers_h_candidates = [j for j in range(2, height - 2, 2)]
    
    all_rivers = []
    for x in rivers_v_candidates:
        all_rivers.append((0, x)) 
    for y in rivers_h_candidates:
        all_rivers.append((1, y)) 
        
    random.shuffle(all_rivers)
    selected_rivers = all_rivers[:num_crossings]
    
    rivers_v = sorted([pos for (d, pos) in selected_rivers if d == 0])
    rivers_h = sorted([pos for (d, pos) in selected_rivers if d == 1])
    
    for x in rivers_v:
        for y in range(1, height - 1):
            grid[y][x] = obstacle_char
            
    for y in rivers_h:
        for x in range(1, width - 1):
            grid[y][x] = obstacle_char

    path_moves = ['cross_col'] * len(rivers_v) + ['cross_row'] * len(rivers_h)
    random.shuffle(path_moves)
    
    gap_info = []
    limits_v = [0] + rivers_v + [width - 1]
    limits_h = [0] + rivers_h + [height - 1]
    room_i = 0
    room_j = 0
    
    for move in path_moves:
        if move == 'cross_col':
            river_x = limits_v[room_i + 1]
            min_y = limits_h[room_j] + 1
            max_y = limits_h[room_j + 1]
            gap_y = random.choice(range(min_y, max_y))
            grid[gap_y][river_x] = " " 
            gap_info.append(f"Vertical river at x={river_x} has gap at y={gap_y}")
            room_i += 1
            
        elif move == 'cross_row':
            river_y = limits_h[room_j + 1]
            min_x = limits_v[room_i] + 1
            max_x = limits_v[room_i + 1]
            gap_x = random.choice(range(min_x, max_x))
            grid[river_y][gap_x] = " " 
            gap_info.append(f"Horizontal river at y={river_y} has gap at x={gap_x}")
            room_j += 1
            
    return grid, gap_info

def build_grid_doorkey(size):
    grid = build_grid_empty(size)
    split_col = random.randint(2, size - 3)
    for y in range(1, size - 1):
        grid[y][split_col] = "W" 
        
    door_y = random.randint(1, size - 2)
    grid[door_y][split_col] = "D"
    door_pos = (split_col, door_y)
    
    key_x = random.randint(1, split_col - 1)
    key_y = random.randint(1, size - 2)
    key_pos = (key_x, key_y)
    grid[key_y][key_x] = "K" 

    return grid, split_col, door_pos, key_pos

# Pathfinding Logic
def heuristic(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def valid(grid, pos, ignore_obstacles=None):
    if ignore_obstacles is None:
        ignore_obstacles = []
        
    x, y = pos
    if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):
        return False
    
    cell = grid[y][x]
    
    if cell in ["#", "L", "W"]:
        return False
    
    if cell == "K" and "K" not in ignore_obstacles:
        return False
    if cell == "D" and "D" not in ignore_obstacles:
        return False
        
    return True

def get_turn_actions(current_dir, desired_dir):
    diff = (desired_dir - current_dir) % 4
    if diff == 0: return []
    if diff == 1: return ["right"]
    if diff == 2: return ["right", "right"] 
    if diff == 3: return ["left"]
    return []

def astar(grid, start_pos, start_dir, goal_pos, ignore_obstacles=None):
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
            ("forward", ((x + DIR_DELTA[direction][0], y + DIR_DELTA[direction][1]), direction)),
        ]

        for action, new_state in successors:
            new_pos, new_dir = new_state

            if action == "forward":
                if not valid(grid, new_pos, ignore_obstacles):
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

def path_to_interaction(grid, start_pos, start_dir, target_pos, ignore_obstacles=None):
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
        if valid(grid, n_pos, ignore_obstacles):
            path, end_state = astar(grid, start_pos, start_dir, n_pos, ignore_obstacles)
            
            if end_state:
                curr_dir = end_state[1]
                turn_actions = get_turn_actions(curr_dir, req_dir)
                full_path = path + turn_actions
                
                if len(full_path) < min_len:
                    min_len = len(full_path)
                    best_path = full_path
                    best_state = (n_pos, req_dir) 

    return best_path, best_state

# Scenario Generation
def generate_scenario(env_name, config):
    size = config["size"]
    env_type = config["type"]

    while True:
        key_pos = None
        door_pos = None
        lava_gaps = []

        if env_type == "empty":
            grid = build_grid_empty(size)
        elif env_type == "lava":
            num_crossings = config.get("crossings", 1)
            grid, lava_gaps = build_grid_crossing(size, num_crossings)
        elif env_type == "doorkey":
            grid, split_col, door_pos, key_pos = build_grid_doorkey(size)

        if env_type == "doorkey":
            agent_x = random.randint(1, split_col - 1)
            agent_y = random.randint(1, size - 2)
            goal_x = random.randint(split_col + 1, size - 2)
            goal_y = random.randint(1, size - 2)
        else:
            agent_x = random.randint(1, size-2)
            agent_y = random.randint(1, size-2)
            goal_x = size-2
            goal_y = size-2

            if env_type == "lava":
                 agent_x, agent_y = 1, 1

        agent_pos = (agent_x, agent_y)
        goal_pos = (goal_x, goal_y)
        agent_dir = random.randint(0, 3)

        if grid[agent_y][agent_x] != " ":
            continue
        if grid[goal_y][goal_x] != " ":
            continue
        if agent_pos == goal_pos:
            continue

        grid[goal_y][goal_x] = "G"
        
        return {
            "env_name": env_name,
            "grid": grid,
            "agent_pos": agent_pos,
            "agent_dir": agent_dir,
            "goal_pos": goal_pos,
            "key_pos": key_pos,
            "door_pos": door_pos,
            "lava_gaps": lava_gaps,
            "type": env_type
        }

def generate_instructions(scenario):
    grid = scenario["grid"]
    agent_pos = scenario["agent_pos"]
    agent_dir = scenario["agent_dir"]
    goal_pos = scenario["goal_pos"]
    env_type = scenario["type"]

    if agent_pos == goal_pos:
        return "Goal reached."

    full_path = []

    if env_type == "doorkey":
        key_pos = scenario["key_pos"]
        door_pos = scenario["door_pos"]
        
        path1, state1 = path_to_interaction(grid, agent_pos, agent_dir, key_pos, ignore_obstacles=[])
        if path1 is None: return "No valid path to reach key."
        full_path.extend(path1)
        full_path.append("pickup")
        
        curr_pos, curr_dir = state1
        path2, state2 = path_to_interaction(grid, curr_pos, curr_dir, door_pos, ignore_obstacles=["K"])
        if path2 is None: return "No valid path from key to door."
        full_path.extend(path2)
        full_path.append("toggle")
        
        curr_pos, curr_dir = state2
        path3, state3 = astar(grid, curr_pos, curr_dir, goal_pos, ignore_obstacles=["K", "D"])
        if path3 is None: return "No valid path from door to goal."
        full_path.extend(path3)

    else:
        path, _ = astar(grid, agent_pos, agent_dir, goal_pos)
        if not path:
            return "No valid path to goal."
        full_path.extend(path)

    return str(full_path[:5])

# Main Logic
def generate_minigrid_dataset(rows_per_map=50):
    dataset = []

    for map_name, config in ENV_CONFIGS.items():
        for _ in range(rows_per_map):
            scenario = generate_scenario(map_name, config)
            instructions = generate_instructions(scenario)

            if instructions.startswith("No valid path"):
                continue

            dir_str = DIRECTIONS[scenario["agent_dir"]]
            
            extra_info_val = ""
            possible_acts = BASE_ACTIONS

            if config['type'] == 'doorkey':
                # DoorKey Logic
                is_door_open = random.choice([True, False])

                if is_door_open:
                    has_key = True 
                    door_info = "Door already open."
                else:
                    has_key = random.choice([True, False])
                    door_info = f"Door at {scenario['door_pos']}."

                if has_key:
                    key_info = "Has Key."
                else:
                    key_info = f"Key at {scenario['key_pos']}."
                
                # Merge info
                extra_info_val = f"{key_info} {door_info}"
                possible_acts = BASE_ACTIONS + DOORKEY_ACTIONS

            elif config['type'] == 'lava':
                # LavaCrossing Logic
                if scenario["lava_gaps"]:
                    extra_info_val = "; ".join(scenario["lava_gaps"])
                else:
                    extra_info_val = "N/A"

            elif config['type'] == 'empty':
                # Empty Logic
                extra_info_val = "No obstacles"

            # Random actions selection
            selected_actions = random.choices(possible_acts, k=random.randint(1, 5))

            data = {
                "actions": selected_actions,
            }

            sequence = json.dumps(data)
            parsed_data = json.loads(sequence)
            response = parsed_data["actions"]

            # Construction of data rows
            data_row = {
                "Environment": scenario['env_name'],
                "Agent_Position": str(scenario['agent_pos']),
                "Agent_Direction": dir_str,
                "Goal_Position": str(scenario['goal_pos']),
                "Extra_Info": extra_info_val, # Colonna unificata
                "response": response,
                "instructions": instructions
            }

            dataset.append(data_row)

    return dataset

def save_to_csv(dataset, filename="Dataset/reviewer_dataset_offline.csv"):
    if not dataset:
        return
    keys = dataset[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dataset)

if __name__ == "__main__":
    if not os.path.exists("Dataset"):
        os.makedirs("Dataset")
        
    dataset = generate_minigrid_dataset(rows_per_map=1000)
    save_to_csv(dataset)
    print("Dataset generated successfully.")