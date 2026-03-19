import csv
import random
import heapq
import os
import ast

# Environment configuration
ENV_CONFIGS = {
    "MiniGrid-Empty-5x5-v0": {"size": 5, "type": "empty"},
    "MiniGrid-Empty-8x8-v0": {"size": 8, "type": "empty"},
    "MiniGrid-Empty-16x16-v0": {"size": 16, "type": "empty"},

    "MiniGrid-LavaCrossingS9N1-v0": {"size": 9, "type": "lava", "crossings": 1},
    "MiniGrid-LavaCrossingS9N3-v0": {"size": 9, "type": "lava", "crossings": 3},
    "MiniGrid-LavaCrossingS11N5-v0": {"size": 11, "type": "lava", "crossings": 5},

    "MiniGrid-DoorKey-5x5-v0": {"size": 5, "type": "doorkey"},
    "MiniGrid-DoorKey-8x8-v0": {"size": 8, "type": "doorkey"},
    "MiniGrid-DoorKey-16x16-v0": {"size": 16, "type": "doorkey"},
}

DIRECTIONS = ["East", "South", "West", "North"]

DIR_DELTA = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}

BASE_ACTIONS = ["left", "right", "forward"]
DOORKEY_ACTIONS = ["pickup", "toggle"]


# Grid builders
def build_grid_empty(size: int):
    grid = [[" " for _ in range(size)] for _ in range(size)]

    for i in range(size):
        grid[0][i] = "#"
        grid[size - 1][i] = "#"
        grid[i][0] = "#"
        grid[i][size - 1] = "#"

    return grid


def build_grid_crossing(size: int, num_crossings: int, obstacle_char="L"):
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

    path_moves = ["cross_col"] * len(rivers_v) + ["cross_row"] * len(rivers_h)
    random.shuffle(path_moves)

    gap_info = []
    limits_v = [0] + rivers_v + [width - 1]
    limits_h = [0] + rivers_h + [height - 1]
    room_i = 0
    room_j = 0

    for move in path_moves:
        if move == "cross_col":
            river_x = limits_v[room_i + 1]
            min_y = limits_h[room_j] + 1
            max_y = limits_h[room_j + 1]
            gap_y = random.choice(range(min_y, max_y))

            grid[gap_y][river_x] = " "
            gap_info.append(
                f"Vertical lava at column {river_x} and you can cross it through the bridge at ({river_x}, {gap_y})"
            )
            room_i += 1

        elif move == "cross_row":
            river_y = limits_h[room_j + 1]
            min_x = limits_v[room_i] + 1
            max_x = limits_v[room_i + 1]
            gap_x = random.choice(range(min_x, max_x))

            grid[river_y][gap_x] = " "
            gap_info.append(
                f"Horizontal lava at row {river_y} and you can cross it through the bridge at ({gap_x}, {river_y})"
            )
            room_j += 1

    return grid, gap_info


def build_grid_doorkey(size: int):
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


# Pathfinding helpers
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

    if diff == 0:
        return []
    if diff == 1:
        return ["right"]
    if diff == 2:
        return ["right", "right"]
    if diff == 3:
        return ["left"]

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
            ("left", (pos, (direction - 1) % 4)),
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
        ((tx, ty + 1), 3),
    ]

    best_path = None
    best_state = None
    min_len = float("inf")

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


# Scenario generation
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
            goal_x = size - 2
            goal_y = size - 2
        else:
            goal_x = size - 2
            goal_y = size - 2

            if env_type == "lava":
                legal_positions = []
                for y in range(1, size - 1):
                    for x in range(1, size - 1):
                        if grid[y][x] == " ":
                            test_dir = 0
                            path, _ = astar(grid, (x, y), test_dir, (goal_x, goal_y))
                            if path is not None:
                                legal_positions.append((x, y))

                if not legal_positions:
                    continue

                agent_x, agent_y = random.choice(legal_positions)
            else:
                agent_x = random.randint(1, size - 2)
                agent_y = random.randint(1, size - 2)

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
            "type": env_type,
        }


def generate_instructions(scenario, has_key=True, is_door_open=True):
    grid = scenario["grid"]
    agent_pos = scenario["agent_pos"]
    agent_dir = scenario["agent_dir"]
    goal_pos = scenario["goal_pos"]
    env_type = scenario["type"]

    if agent_pos == goal_pos:
        return str([])

    full_path = []

    if env_type == "doorkey":
        key_pos = scenario["key_pos"]
        door_pos = scenario["door_pos"]

        if not has_key and not is_door_open:
            path1, _ = path_to_interaction(grid, agent_pos, agent_dir, key_pos, ignore_obstacles=[])
            if path1 is None:
                return None
            full_path.extend(path1)
            full_path.append("pickup")
            return str(full_path[:5])

        if not is_door_open:
            path2, _ = path_to_interaction(grid, agent_pos, agent_dir, door_pos, ignore_obstacles=["K"])
            if path2 is None:
                return None
            full_path.extend(path2)
            full_path.append("toggle")
            return str(full_path[:5])

        path3, _ = astar(grid, agent_pos, agent_dir, goal_pos, ignore_obstacles=["K", "D"])
        if path3 is None:
            return None
        full_path.extend(path3)

    else:
        path, _ = astar(grid, agent_pos, agent_dir, goal_pos)
        if path is None:
            return None
        full_path.extend(path)

    return str(full_path[:5])


# Helper suggestion generation
# Generate plausible but imperfect helper suggestions instead of pure random noise
def generate_helper_suggestion(instructions_str, possible_acts, mistake_type=None):
    try:
        correct_actions = ast.literal_eval(instructions_str)
    except (ValueError, SyntaxError):
        correct_actions = []

    if not correct_actions:
        # No valid reference sequence available
        return str(["forward"] * random.randint(1, 3))

    if mistake_type is None:
        # Realistic error distribution
        mistake_type = random.choices(
            ['correct', 'truncated', 'wrong_dir', 'extra_forward', 'wrong_action', 'ignore_obstacle'],
            weights=[0.30, 0.20, 0.15, 0.15, 0.10, 0.10],
            k=1
        )[0]

    if mistake_type == 'correct':
        return str(correct_actions)

    elif mistake_type == 'truncated':
        # Remove 1-2 actions from the end
        cut = random.randint(1, min(2, len(correct_actions)))
        truncated = correct_actions[:-cut] if len(correct_actions) > cut else correct_actions[:1]
        return str(truncated)

    elif mistake_type == 'wrong_dir':
        # Flip left/right actions
        flipped = []
        for a in correct_actions:
            if a == 'left':
                flipped.append('right')
            elif a == 'right':
                flipped.append('left')
            else:
                flipped.append(a)
        return str(flipped)

    elif mistake_type == 'extra_forward':
        # Insert one extra forward at a random position
        if len(correct_actions) < 5:
            pos = random.randint(0, len(correct_actions))
            suggestion = correct_actions[:pos] + ['forward'] + correct_actions[pos:]
            return str(suggestion[:5])
        return str(correct_actions)

    elif mistake_type == 'wrong_action':
        # Replace one action with a plausible alternative
        suggestion = list(correct_actions)
        idx = random.randint(0, len(suggestion) - 1)
        alternatives = [a for a in possible_acts if a != suggestion[idx]]
        if alternatives:
            suggestion[idx] = random.choice(alternatives)
        return str(suggestion)

    elif mistake_type == 'ignore_obstacle':
        # Greedy suggestion that ignores obstacles and turning
        n = random.randint(1, 5)
        return str(['forward'] * n)

    # Fallback to the reference sequence
    return str(correct_actions)


# Dataset generation
def generate_minigrid_dataset(rows_per_map=200):
    dataset = []

    for map_name, config in ENV_CONFIGS.items():
        for _ in range(rows_per_map):
            scenario = generate_scenario(map_name, config)

            has_key = True
            is_door_open = True

            if config["type"] == "doorkey":
                is_door_open = random.choice([True, False])
                if is_door_open:
                    has_key = True
                else:
                    has_key = random.choice([True, False])

            instructions = generate_instructions(scenario, has_key, is_door_open)
            if instructions is None:
                continue

            dir_str = DIRECTIONS[scenario["agent_dir"]]

            possible_acts = BASE_ACTIONS
            extra_info_val = ""

            if config["type"] == "doorkey":
                if is_door_open:
                    door_info = "Door already open, go to goal."
                else:
                    door_info = f"Door closed at {scenario['door_pos']}."

                if has_key:
                    key_info = "Has Key."
                else:
                    key_info = f"Key at {scenario['key_pos']}. Pick it up."

                extra_info_val = f"{key_info} {door_info}"

                if not has_key:
                    possible_acts = BASE_ACTIONS + ["pickup"]
                elif not is_door_open:
                    possible_acts = BASE_ACTIONS + ["toggle"]
                else:
                    possible_acts = BASE_ACTIONS

            elif config["type"] == "lava":
                possible_acts = BASE_ACTIONS
                extra_info_val = "; ".join(scenario["lava_gaps"]) if scenario["lava_gaps"] else "N/A"

            elif config["type"] == "empty":
                possible_acts = BASE_ACTIONS
                extra_info_val = "No obstacles."

            # Generate a plausible helper suggestion with structured mistakes
            response = generate_helper_suggestion(instructions, possible_acts)

            prompt = (
                f"Environment: {scenario['env_name']}. "
                f"Agent Position: {scenario['agent_pos']}. "
                f"Agent Direction: {dir_str}. "
                f"Goal Position: {scenario['goal_pos']}. "
                f"Extra Info: {extra_info_val}. "
                f"Available Actions: {', '.join(possible_acts)}"
            )

            dataset.append({
                "prompt": prompt,
                "response": response,         # Structured helper suggestion
                "instructions": instructions  # Optimal ground-truth sequence
            })

    return dataset


def save_to_csv(dataset, filename="Dataset/reviewer_dataset_offline.csv"):
    if not dataset:
        raise ValueError("Dataset is empty, nothing to save.")

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    keys = dataset[0].keys()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=keys,
            quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(dataset)


if __name__ == "__main__":
    dataset = generate_minigrid_dataset(rows_per_map=500)
    save_to_csv(dataset, filename="Dataset/reviewer_dataset_offline.csv")

    print(f"Dataset generated successfully: {len(dataset)} rows")

    # Suggestion distribution statistics
    correct_count = sum(1 for d in dataset if d['response'] == d['instructions'])
    print(f"Correct suggestions: {correct_count}/{len(dataset)} ({100*correct_count/len(dataset):.1f}%)")
    print(f"Incorrect suggestions (to be corrected): {len(dataset)-correct_count}/{len(dataset)} ({100*(len(dataset)-correct_count)/len(dataset):.1f}%)")