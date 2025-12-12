import csv
import random
import heapq
import math

# ================================================================
# CONFIGURAZIONI DELLE MAPPE
# ================================================================
ENV_CONFIGS = {
    # Empty maps
    "MiniGrid-Empty-5x5-v0": {"size": 5, "type": "empty"},
    "MiniGrid-Empty-8x8-v0": {"size": 8, "type": "empty"},
    "MiniGrid-Empty-16x16-v0": {"size": 16, "type": "empty"},

    # LavaCrossing maps
    "MiniGrid-LavaCrossingS9N1-v0": {"size": 9, "type": "lava"},
    "MiniGrid-LavaCrossingS9N3-v0": {"size": 9, "type": "lava"},
    "MiniGrid-LavaCrossingS11N5-v0": {"size": 11, "type": "lava"},

    # DoorKey maps
    "MiniGrid-DoorKey-5x5-v0": {"size": 5, "type": "doorkey"},
    "MiniGrid-DoorKey-8x8-v0": {"size": 8, "type": "doorkey"},
    "MiniGrid-DoorKey-16x16-v0": {"size": 16, "type": "doorkey"},
}

# ================================================================
# DIREZIONI
# ================================================================
DIRECTIONS = ["East", "South", "West", "North"]
DIR_DELTA = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

# ================================================================
# COSTRUZIONE DELLE MAPPE
# ================================================================
def build_grid_empty(size):
    grid = [[" " for _ in range(size)] for _ in range(size)]
    for i in range(size):
        grid[0][i] = "#"
        grid[size-1][i] = "#"
        grid[i][0] = "#"
        grid[i][size-1] = "#"
    return grid

def build_grid_lava(size, n_gaps):
    grid = build_grid_empty(size)

    row_lava = size // 2
    for x in range(1, size-1):
        grid[row_lava][x] = "L"

    gaps = random.sample(range(1, size-1), n_gaps)
    for g in gaps:
        grid[row_lava][g] = " "

    return grid, (row_lava, gaps)

def build_grid_doorkey(size):
    grid = build_grid_empty(size)
    return grid

# ================================================================
# A* PESATO SU POSIZIONE + DIREZIONE
# ================================================================
def rotation_cost(d1, d2):
    diff = abs(d1 - d2) % 4
    return min(diff, 4 - diff)

def heuristic(pos, dir, goal):
    x, y = pos
    gx, gy = goal
    man = abs(x - gx) + abs(y - gy)
    if abs(gx - x) > abs(gy - y):
        desired = 0 if gx > x else 2
    else:
        desired = 1 if gy > y else 3
    rot = rotation_cost(dir, desired)
    return man + 1.5 * rot

def valid(grid, pos):
    x, y = pos
    if y < 0 or y >= len(grid):
        return False
    if x < 0 or x >= len(grid[0]):
        return False
    if grid[y][x] in ["#", "L"]:
        return False
    return True

def astar(grid, start_pos, start_dir, goal_pos):
    start = (start_pos, start_dir)
    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {start: None}
    action_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, (pos, dir) = heapq.heappop(frontier)

        if pos == goal_pos:
            break

        x, y = pos

        successors = [
            ("left",  (pos, (dir - 1) % 4)),
            ("right", (pos, (dir + 1) % 4)),
            ("forward", ((x + DIR_DELTA[dir][0], y + DIR_DELTA[dir][1]), dir)),
        ]

        for action, new_state in successors:
            new_pos, new_dir = new_state

            if action == "forward" and not valid(grid, new_pos):
                continue

            new_cost = cost_so_far[(pos, dir)] + 1

            if (new_pos, new_dir) not in cost_so_far or new_cost < cost_so_far[(new_pos, new_dir)]:
                cost_so_far[(new_pos, new_dir)] = new_cost
                priority = new_cost + heuristic(new_pos, new_dir, goal_pos)
                heapq.heappush(frontier, (priority, (new_pos, new_dir)))
                came_from[(new_pos, new_dir)] = (pos, dir)
                action_from[(new_pos, new_dir)] = action

    goal_states = [s for s in cost_so_far if s[0] == goal_pos]
    if not goal_states:
        return []

    end = min(goal_states, key=lambda s: cost_so_far[s])
    actions = []

    cur = end
    while action_from[cur] is not None:
        actions.append(action_from[cur])
        cur = came_from[cur]

    actions.reverse()
    return actions

# ================================================================
# GENERAZIONE SCENARI
# ================================================================
def generate_scenario(env_name, config):
    size = config["size"]
    env_type = config["type"]

    if env_type == "empty":
        grid = build_grid_empty(size)

    elif env_type == "lava":
        mapname = env_name
        n = int(mapname.split("N")[-1].split("-")[0])
        grid, lava_info = build_grid_lava(size, n)

    elif env_type == "doorkey":
        grid = build_grid_doorkey(size)

    agent_pos = (random.randint(1, size-2), random.randint(1, size-2))
    agent_dir = random.randint(0, 3)
    goal_pos = (size-2, size-2)
    grid[goal_pos[1]][goal_pos[0]] = "G"

    return {
        "env_name": env_name,
        "grid": grid,
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "goal_pos": goal_pos,
        "type": env_type
    }

# ================================================================
# GENERAZIONE DELLE ISTRUZIONI OTTIMALI
# ================================================================
def generate_instructions(scenario):
    grid = scenario["grid"]
    agent_pos = scenario["agent_pos"]
    agent_dir = scenario["agent_dir"]
    goal_pos = scenario["goal_pos"]

    path = astar(grid, agent_pos, agent_dir, goal_pos)

    if not path:
        return "No valid path to goal."

    return "Optimal action sequence: [" + ", ".join(path) + "]."

# ================================================================
# DATASET
# ================================================================
def generate_minigrid_dataset(rows_per_map=50):
    dataset = []

    for map_name, config in ENV_CONFIGS.items():
        for _ in range(rows_per_map):
            scenario = generate_scenario(map_name, config)
            instructions = generate_instructions(scenario)

            dir_str = DIRECTIONS[scenario["agent_dir"]]

            prompt = (
                f"Environment: {scenario['env_name']}. "
                f"Agent at {scenario['agent_pos']} facing {dir_str}. "
                f"Goal at {scenario['goal_pos']}. "
                f"Map size: {config['size']}x{config['size']}."
            )

            dataset.append({
                "prompt": prompt,
                "response": "The next action to take is [unknown].",
                "instructions": instructions
            })

    return dataset

def save_to_csv(dataset, filename="minigrid_dataset.csv"):
    keys = dataset[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(dataset)

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    dataset = generate_minigrid_dataset(rows_per_map=100)
    save_to_csv(dataset)
    print("Dataset generated successfully.")
