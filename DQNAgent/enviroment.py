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
        #Controllo che non spinna
        self.spin_count = 0
        self.last_was_rotation = False

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

        DIR2VEC = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
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
        # Controllo azione valida
        valid = self.get_valid_actions()
        if action not in valid:
            return self.get_state(), -1.0, False, {"invalid_action": True}

        # Esegui l'azione nell'environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.state = self._obs_to_state(obs)
        done = terminated or truncated
        info = info or {}

        # Controllo frontale e oggetto trasportato
        front_type = self._get_front_cell_type()
        carrying = getattr(self.base_env, "carrying", None)

        # Penalità per tentare di andare contro un muro
        if action == 2 and front_type == "wall":
            reward -= 0.2
            info["bumped_wall"] = True


        # Debug
        #print("DEBUG STEP")
        #print("Azione eseguita:", action)
        #print("Agente davanti a:", front_type)
        #print("Carrying:", carrying)

        # Reward shaping: distanza al goal
        dist_to_goal = self._compute_distance_to_goal()
        if dist_to_goal is not None and self.prev_dist_to_goal is not None:
            delta = self.prev_dist_to_goal - dist_to_goal
            reward += 0.2 * delta
        self.prev_dist_to_goal = dist_to_goal

        # Reward goal +1 solo se episodio terminato con successo
        if terminated:
            print("Goal raggiunto!")
            reward += 1.0
            info["goal_reached"] = True

        # Piccola penalità per step
        reward -= 0.01
        # Anti-spinning: penalità per rotazioni eccessive
        if action in [0, 1]:  # 0=left, 1=right
            if self.last_was_rotation:
                self.spin_count += 1
            else:
                self.spin_count = 1
                self.last_was_rotation = True

            # penalità crescente dopo la terza rotazione consecutiva
            if self.spin_count > 3:
                penalty = min(0.05 * self.spin_count, 0.5)
                reward -= penalty
                info["spin_penalty"] = penalty

        else:
            # reset se l’agente fa avanti o altro
            self.spin_count = 0
            self.last_was_rotation = False


        # Crossing task: lava
        if self.task_type == "Crossing":
            agent_pos = self.base_env.agent_pos
            grid = self.base_env.grid
            cell = grid.get(*agent_pos)
            if cell and cell.type == "lava":
                reward -= 11
                done = True
                info["lava_penalty"] = True

        # DoorKey task
        if self.task_type == "DoorKey" and not done:
            carry_type = getattr(carrying, "type", None) if carrying else None
            picked_flag = getattr(self, "_key_picked_rewarded", False)
            door_unlocked_flag = getattr(self, "_door_unlocked_rewarded", False)

            # PICKUP solo frontale se non trasporta nulla
            if action == self.PICKUP:
                if front_type == "key" and carrying is None:
                    print("Raccolgo la chiave!")
                    reward += 0.5 if not picked_flag else 0.0
                    info["key_picked"] = True
                    self._key_picked_rewarded = True
                else:
                    print("Pickup fallito")
                    reward -= 0.1
                    info["pickup_fail"] = True

            # DROP
            elif action == self.DROP:
                if carrying is None:
                    reward -= 0.1
                    info["drop_fail"] = True
                else:
                    reward -= 0.05
                    info["dropped_key"] = True
                    try:
                        self.base_env.carrying = None
                    except Exception:
                        pass
                    self._key_picked_rewarded = False

            # TOGGLE porta
            elif action == self.TOGGLE:
                if front_type == "door":
                    if not self.door_open:
                        if carry_type == "key" and not door_unlocked_flag:
                            print("Apro la porta!")
                            reward += 0.5
                            info["door_unlocked_with_key"] = True
                            self._door_unlocked_rewarded = True
                            self.door_open = True
                            try:
                                self.base_env.carrying = None
                            except Exception:
                                pass
                        else:
                            reward -= 0.2
                            info["toggle_without_key"] = True
                    else:
                        reward -= 0.1
                        info["toggle_redundant"] = True
                else:
                    reward -= 0.15
                    info["toggle_miss"] = True

        # Clip reward
        reward = float(np.clip(reward, -100.0, 100.0))

        return self.get_state(), float(reward), bool(done), info



    def render(self):
        return self.env.render()
