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

        front = self._get_front_cell_type()


        # 1)Calcolo della distanza al goal
        # Shaping: reward for reducing Manhattan distance to goal (small), small time penalty
        dist_to_goal = self._compute_distance_to_goal()
        if dist_to_goal is not None and self.prev_dist_to_goal is not None:
            delta = self.prev_dist_to_goal - dist_to_goal
            reward += 0.2 * delta  # encourage getting closer
        self.prev_dist_to_goal = dist_to_goal

        reward -= 0.01  # small per-step time penalty to encourage efficiency

        if action == [0,1]:
            reward -= 0.2

        # Crossing task: heavy penalty for stepping on lava
        if self.task_type == "Crossing":
            agent_pos = self.base_env.agent_pos
            grid = self.base_env.grid
            cell = grid.get(*agent_pos)
            if cell and cell.type == "lava":
                reward -= 10.0
                done = True
                info["lava_penalty"] = True

        # Safer DoorKey shaping: smaller magnitudes, one-shot rewards, and guards to avoid repeated large bonuses
        # Use attributes with getattr so this patch doesn't require edits to __init__/reset elsewhere.
        if self.task_type == "DoorKey" and not done:
            carrying = getattr(self.base_env, "carrying", None)
            carry_type = getattr(carrying, "type", None) if carrying is not None else None

            # one-shot bookkeeping flags
            picked_flag = getattr(self, "_key_picked_rewarded", False)
            door_unlocked_flag = getattr(self, "_door_unlocked_rewarded", False)

            if action == self.PICKUP:
                # pickup a key in front when not carrying => modest positive reward (only once)
                if front == "key" and carrying is None and not picked_flag:
                    reward += 5.0
                    info["key_picked"] = True
                    self._key_picked_rewarded = True

            elif action == self.DROP:
                # discourage pointless drops but keep penalty small
                if carrying is None:
                    reward -= 1.0
                    info["drop_fail"] = True
                else:
                    reward -= 0.5
                    info["dropped_key"] = True
                    try:
                        self.base_env.carrying = None
                    except Exception:
                        pass
                    # if they drop the key after having picked it, allow future pickup reward again
                    self._key_picked_rewarded = False

            elif action == self.TOGGLE:
                if front == "door":
                    if not self.door_open:
                        if carry_type == "key" and not door_unlocked_flag:
                            # correct sequence: picked key -> toggle door => moderate reward (one-shot)
                            reward += 15.0
                            self.door_open = True
                            info["door_unlocked_with_key"] = True
                            self._door_unlocked_rewarded = True
                            # consume the key if env exposes carrying
                            try:
                                self.base_env.carrying = None
                            except Exception:
                                pass
                    else:
                        # trying to open without key or already rewarded is penalized mildly
                        reward -= 2.5
                        info["toggle_without_key"] = True
                else:
                    # redundant toggle on an already open door
                    reward -= 1.0
                    info["toggle_redundant"] = True
            else:
                # toggle where there is no door
                reward -= 1.5
                info["toggle_miss"] = True

        # Clip reward to avoid extreme values
        reward = float(np.clip(reward, -100.0, 100.0))

        return self.get_state(), float(reward), bool(done), info

    def render(self):
        return self.env.render()
