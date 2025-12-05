import numpy as np

class DynamicMiniGridWrapper:
    BASIC_ACTIONS = [0, 1]  # left, right, forward
    FORWARD = 2
    PICKUP = 3
    DROP = 4
    TOGGLE = 5
    DONE = 6

    def __init__(self, env, task_type):
        self.env = env
        self.base_env = env.unwrapped
        self.task_type = task_type

        # Imposta goal di default se non esiste
        if not hasattr(self.base_env, "goal_pos") or self.base_env.goal_pos is None:
            self.base_env.goal_pos = (self.base_env.width-2, self.base_env.height-2)
            print(f"DEBUG: goal_pos impostato a {self.base_env.goal_pos}")

        self.obs, _ = self.env.reset()
        self.state = self._obs_to_state(self.obs)
        self.action_size = 7
        self.state_size = len(self.state)

        self.best_dist_to_goal = float('inf')

        # Controllo anti-spinning
        self.spin_count = 0
        self.last_was_rotation = False

        # DoorKey
        self.door_open = False
        self._door_unlocked_rewarded = False

        # Per reward shaping
        self.prev_dist_to_goal = self._compute_distance_to_goal()
        self.prev_agent_pos = getattr(self.base_env, "agent_pos", None)

    def _obs_to_state(self, obs):
        if isinstance(obs, dict):
            if 'image' in obs:
                img = obs['image']
                return (img.reshape(-1) * 0.1).astype(np.float32)
            else:
                state_parts = []
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        state_parts.append(value.reshape(-1))
                    elif isinstance(value, (int, float)):
                        state_parts.append(np.array([value]))
                return (np.concatenate(state_parts) * 0.1).astype(np.float32)
        else:
            return (obs.reshape(-1) * 0.1).astype(np.float32)

    def get_state(self):
        return self.state.reshape(1, -1)

    def _get_front_cell_type(self):
        agent_pos = self.base_env.agent_pos
        agent_dir = self.base_env.agent_dir
        grid = self.base_env.grid

        DIR2VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
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
        valid.append(self.FORWARD)
        if self.task_type == "DoorKey":
            if front == "key":
                valid.append(self.PICKUP)
            if front == "door" and not self.door_open:
                valid.append(self.TOGGLE)
        return valid

    def reset(self):
        self.obs, _ = self.env.reset()
        if not hasattr(self.base_env, "goal_pos") or self.base_env.goal_pos is None:
            self.base_env.goal_pos = (self.base_env.width-2, self.base_env.height-2)
            print(f"DEBUG: goal_pos impostato a {self.base_env.goal_pos}")
        self.state = self._obs_to_state(self.obs)
        self.door_open = False
        self._door_unlocked_rewarded = False
        self.best_dist_to_goal = float('inf')
        self.prev_dist_to_goal = self._compute_distance_to_goal()
        self.prev_agent_pos = getattr(self.base_env, "agent_pos", None)
        return self.get_state()

    def _compute_distance_to_goal(self):
        agent_pos = getattr(self.base_env, "agent_pos", None)
        goal_pos = getattr(self.base_env, "goal_pos", None)
        if agent_pos is not None and goal_pos is not None:
            return abs(agent_pos[0]-goal_pos[0]) + abs(agent_pos[1]-goal_pos[1])
        return None

    def get_safe_actions_towards_goal(self):
            """
            Restituisce una lista di azioni 'left', 'right', 'forward' sicure 
            verso il goal evitando lava.
            """
            from heapq import heappush, heappop

            DIR2VEC = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)} # 0=east,1=south,2=west,3=north
            start = tuple(self.base_env.agent_pos)
            goal = self.base_env.goal_pos
            grid = self.base_env.grid

            def cell_cost(x, y):
                cell = grid.get(x, y)
                if cell is None:
                    return 1
                if cell.type == "lava":
                    return float('inf')
                return 1

            open_set = []
            heappush(open_set, (0 + abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, start, []))
            visited = set()

            while open_set:
                f, g, current, path = heappop(open_set)
                if current in visited:
                    continue
                visited.add(current)
                path = path + [current]
                if current == goal:
                    actions = []
                    dir_ = self.base_env.agent_dir
                    for i in range(len(path)-1):
                        cur, nxt = path[i], path[i+1]
                        dx, dy = nxt[0]-cur[0], nxt[1]-cur[1]
                        for d, vec in DIR2VEC.items():
                            if vec == (dx, dy):
                                target_dir = d
                                break
                        delta = (target_dir - dir_) % 4
                        if delta == 1:
                            actions.append("right"); dir_=(dir_+1)%4
                        elif delta == 2:
                            actions.append("right"); actions.append("right"); dir_=(dir_+2)%4
                        elif delta == 3:
                            actions.append("left"); dir_=(dir_-1)%4
                        actions.append("forward")
                    return actions
                x, y = current
                for dx, dy in DIR2VEC.values():
                    nx, ny = x+dx, y+dy
                    if 0<=nx<grid.width and 0<=ny<grid.height and (nx, ny) not in visited:
                        cost = cell_cost(nx, ny)
                        if cost < float('inf'):
                            g_new = g+cost
                            f_new = g_new + abs(nx-goal[0]) + abs(ny-goal[1])
                            heappush(open_set, (f_new, g_new, (nx, ny), path))
            return []



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

        # Incentivare a muoversi in avanti
        if action == 2 and front_type not in ["wall", "door", "lava", "key"]:
            reward += 0.01


        # Debug
        #print("DEBUG STEP")
        #print("Azione eseguita:", action)
        #print("Agente davanti a:", front_type)
        #print("Carrying:", carrying)

        # Reward shaping: distanza al goal
        dist_to_goal = self._compute_distance_to_goal()
        # Premia SOLO se ottieni una distanza migliore della migliore mai raggiunta
        if dist_to_goal is not None and dist_to_goal < self.best_dist_to_goal:
            # aggiorna il record
            self.best_dist_to_goal = dist_to_goal

            # shaping: più ti avvicini la prima volta, più reward
            improvement = self.best_dist_to_goal - dist_to_goal
            reward += 0.02 * improvement


        # Reward goal +1 solo se episodio terminato con successo
        if terminated:
            print("Goal raggiunto!")
            reward += 1.0
            info["goal_reached"] = True

        # Piccola penalità per step
        reward -= 0.01
        # Anti-spinning: penalità per rotazioni eccessive
        if action == 0 and action == 1:  # 0=left, 1=right
            if self.last_was_rotation:
                self.spin_count += 1
            else:
                self.spin_count = 1
                self.last_was_rotation = True

            # penalità crescente dopo la terza rotazione consecutiva
            if self.spin_count > 2:
                penalty = min(0.01 * self.spin_count, 0.05)
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
                if front_type == "key" or carry_type == "key":
                    print("Raccolgo la chiave!")
                    reward += 1
                    info["key_picked"] = True
                    self._key_picked_rewarded = True
                else:
                    print("Pickup fallito")
                    reward -= 0.01
                    info["pickup_fail"] = True

            # DROP
            elif action == self.DROP:
                if carrying is None:
                    reward -= 0.01
                    info["drop_fail"] = True
                else:
                    reward -= 0.01
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
                            reward += 1
                            info["door_unlocked_with_key"] = True
                            self._door_unlocked_rewarded = True
                            self.door_open = True
                            try:
                                self.base_env.carrying = None
                            except Exception:
                                pass
                        else:
                            print("Porta bloccata, mi serve una chiave!")
                            reward -= 0.01
                            info["toggle_without_key"] = True
                    else:
                        print("Chiudo la porta!")
                        reward -= 0.1
                        info["toggle_redundant"] = True
                else:
                    print("Non c'è nessuna porta da aprire/chiudere davanti a me!")
                    reward -= 0.01
                    info["toggle_miss"] = True

        # Clip reward
        reward = float(np.clip(reward, -100.0, 100.0))

        return self.get_state(), float(reward), bool(done), info

    def render(self):
        return self.env.render()
