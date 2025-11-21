# enviroment.py
import numpy as np

class DynamicMiniGridWrapper:
    BASIC_ACTIONS = [0, 1, 2]
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6

    def __init__(self, env, task_type):
        self.env = env
        self.base_env = env.unwrapped
        self.task_type = task_type

        self.obs, _ = self.env.reset()
        self.state = self._obs_to_state(self.obs)
        self.action_size = 7
        self.state_size = len(self.state)

        # DoorKey
        self.door_open = False

        #Per reward shaping 
        self.prev_dist_to_goal = self._compute_distance_to_goal()

    def _obs_to_state(self, obs):
        return (obs.reshape(-1) * 0.1).astype(np.float32)

    def get_state(self):
        return self.state.reshape(1, -1)

    def _get_front_cell_type(self):
        agent_pos = self.base_env.agent_pos
        agent_dir = self.base_env.agent_dir
        grid = self.base_env.grid

        DIR2VEC = {0:(0,-1), 1:(1,0), 2:(0,1), 3:(-1,0)}
        dx, dy = DIR2VEC[agent_dir]
        front_x, front_y = agent_pos[0]+dx, agent_pos[1]+dy
        if not (0 <= front_x < grid.width and 0 <= front_y < grid.height):
            return "wall"
        cell = grid.get(front_x, front_y)
        if cell is None:
            return "empty"
        return cell.type

    def get_valid_actions(self):
        valid = self.BASIC_ACTIONS.copy()
        front = self._get_front_cell_type()

        if self.task_type == "DoorKey":
            if front == "key":
                valid.append(self.PICKUP)
            if front == "door" and not self.door_open:
                valid.append(self.TOGGLE)

        return valid

    def reset(self):
        self.obs, _ = self.env.reset()
        self.state = self._obs_to_state(self.obs)
        self.door_open = False
        self.prev_dist_to_goal = self._compute_distance_to_goal()
        return self.get_state()

    def _compute_distance_to_goal(self):
        # Utilizzo dell'algoritmo di Manhattan per incentivare l'agente al goal
        agent_pos = getattr(self.base_env, "agent_pos", None)
        goal_pos = getattr(self.base_env, "goal_pos", None)
        if agent_pos is not None and goal_pos is not None:
            return abs(agent_pos[0]-goal_pos[0]) + abs(agent_pos[1]-goal_pos[1])
        return None

    def step(self, action):
        valid = self.get_valid_actions()
        if action not in valid:
            return self.get_state(), -1.0, False, {"invalid_action": True}

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.state = self._obs_to_state(obs)
        done = terminated or truncated
        info = info or {}

        # 1)Calcolo della distanza al goal
        dist_to_goal = self._compute_distance_to_goal()
        if dist_to_goal is not None and self.prev_dist_to_goal is not None:
            delta = self.prev_dist_to_goal - dist_to_goal
            reward += 0.1 * delta 
        self.prev_dist_to_goal = dist_to_goal



        if self.task_type == "Crossing":
            agent_pos = self.base_env.agent_pos
            grid = self.base_env.grid
            cell = grid.get(*agent_pos)
            if cell and cell.type == "lava":
                reward -= 10.0
                done = True
                info["lava_penalty"] = True




        if self.task_type == "DoorKey":
            front = self._get_front_cell_type()
            if action == self.DROP:
                reward -= 5.0
                info["dropped_key_penalty"] = True
            elif action == self.TOGGLE:
                if front == "door" and not self.door_open:
                    reward += 20.0
                    self.door_open = True
                    info["door_opened"] = True
                elif front == "door" and self.door_open:
                    reward -= 2.0
                    info["toggle_redundant"] = True
                else:
                    reward -= 3.0
                    info["toggle_miss"] = True
            elif action == self.PICKUP and front == "key":
                reward += 5.0
                info["key_picked"] = True

        return self.get_state(), float(reward), bool(done), info

    def render(self):
        return self.env.render()
