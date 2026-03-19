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

        # A* cached per episodio 
        self.episode_astar_cell_path = None
        self.astar_replans_used = 0
        self.last_astar_suggested_action = None

        # DoorKey variabili di stato
        self.door_open = False
        self._door_unlocked_rewarded = False
        self.key_pos = None  # Posizione della chiave
        self.door_pos = None  # Posizione della porta
        self.phase_1_key_reached = False  # Fase 1: chiave raccolta
        self.phase_2_door_opened = False  # Fase 2: porta aperta
        self.phase_3_goal_reached = False  # Fase 3: goal raggiunto

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
        
        # Controllo anti-wall-bumping
        self.wall_bump_count = 0

        # Per reward shaping
        self.prev_dist_to_goal = self._compute_distance_to_goal()
        self.prev_agent_pos = getattr(self.base_env, "agent_pos", None)
        
        # Tracking per lava proximity
        self.lava_proximity_penalty = {}
        self.steps_without_lava = 0
        
        # Step counter per episodio
        self.current_step = 0
        
        # Calcola dimensione griglia per adattare max_steps
        grid_size = getattr(self.base_env, 'width', 5) 
        
        # DoorKey necessita più step per le 3 fasi 
        if task_type == "Crossing":
            self.max_steps = 300
        elif task_type == "DoorKey":
            # Max steps proporzionale alla dimensione della mappa
            if grid_size <= 5:
                self.max_steps = 300
            elif grid_size <= 8:
                self.max_steps = 600  # 8x8 richiede più tempo
            else:
                self.max_steps = 600  # 16x16 richiede molto più tempo
            print(f"DEBUG: DoorKey grid_size={grid_size}, max_steps={self.max_steps}")
        else:
            # Default per altri task
            if grid_size <= 8:
                self.max_steps = 300 
            else:
                self.max_steps = 600  # 16x16 richiede più tempo
        
        # Flag per evitare reward multiple per safe passage
        self.safe_passage_rewarded = False
        
        # Tracking A* hint reward per cella (una sola volta per cella)
        self.astar_reward_given_cells = set()
        
        # Tracking goal proximity (blocca reward distanza se gira intorno al goal)
        self.near_goal_distance = None  # Memorizza distanza quando entra in prossimità goal
        
        # Tracking reward totale episodio (per correzione finale se fallisce)
        self.episode_reward_sum = 0.0
        
        # A* cached per episodio (lista di celle dal start al goal)
        self.episode_astar_cell_path = None
        
        # Curriculum con seed fissi per Crossing e DoorKey
        # Funzionamento: N episodi per seed × 10 seed diversi
        # Per mappe grandi: più episodi per seed
        self.episode_count = 0
        
        # Flag per disabilitare completamente la logica dei seed (usato in fase Mixed)
        self.skip_seed_logic = False
        
        # Episodi per seed dipende dalla dimensione mappa
        grid_size = getattr(self.base_env, 'width', 5)
        if task_type == "DoorKey" and grid_size > 5:
            self.episodes_per_seed = 20  # Più episodi per mappe grandi
        else:
            self.episodes_per_seed = 10  # Standard per mappe piccole
            
        self.fixed_seed_cycles = 10   # Numero di seed diversi da provare
        self.current_seed = 42       # Seed iniziale (42 per Crossing, 1000 per DoorKey)
        self.use_fixed_seed = True   # Inizia con seed fissi

    def _obs_to_state(self, obs):
        """Converte l'osservazione in stato, con PADDING FISSO per tutte le mappe."""
        state_parts = []
        
        # --- MODIFICA QUI: Dimensione fissa per supportare curriculum learning ---
        # Usiamo 20x20 che copre comodamente anche la mappa 16x16 (che ha bordi)
        FIXED_TARGET_SIZE = 20 
        
        if isinstance(obs, dict):
            if 'image' in obs:
                img = obs['image']
                if img.ndim == 3:
                    h, w, c = img.shape
                    
                    # Usa SEMPRE una dimensione fissa, non dipendente da h o w correnti
                    target = FIXED_TARGET_SIZE 
                    
                    # Se l'immagine è più piccola del target, fai padding
                    # Se è più grande (improbabile se target=20), viene tagliata o gestita
                    padded = np.zeros((target, target, c), dtype=img.dtype)
                    
                    # Copia l'immagine nell'angolo in alto a sinistra
                    # Usa min() per evitare errori se l'immagine fosse > 20
                    safe_h = min(h, target)
                    safe_w = min(w, target)
                    padded[:safe_h, :safe_w, :] = img[:safe_h, :safe_w, :]
                    img = padded
                    
                # Normalizza l'immagine
                state_parts.append((img.reshape(-1) * 0.1).astype(np.float32))
            else:
                # Gestione fallback per osservazioni non-immagine
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        state_parts.append((value.reshape(-1) * 0.1).astype(np.float32))
                    elif isinstance(value, (int, float)):
                        state_parts.append(np.array([value], dtype=np.float32))
        else:
            state_parts.append((obs.reshape(-1) * 0.1).astype(np.float32))
        
        # Aggiungi informazioni contestuali
        context_info = self._get_context_features()
        state_parts.append(context_info)
        
        # Aggiungi A* hint
        astar_hint = self._get_astar_hint_features()
        state_parts.append(astar_hint)
        
        return np.concatenate(state_parts).astype(np.float32)

    def get_state(self):
        return self.state.reshape(1, -1)
    
    def _get_context_features(self):
        """Estrae features contestuali. 
        FIX: Usa SEMPRE la dimensione massima (16) per compatibilità tra task."""
        
        # Dimensione fissa 16 (il massimo richiesto da DoorKey)
        # Indici 0-9: Standard / Crossing
        # Indici 10-15: Specifici DoorKey (saranno 0 per gli altri task)
        features = np.zeros(16, dtype=np.float32)
        
        agent_pos = self.base_env.agent_pos
        grid = self.base_env.grid
        goal_pos = getattr(self.base_env, "goal_pos", None)
        
        # Feature 0-1: Direzione verso il goal 
        if goal_pos:
            dx = goal_pos[0] - agent_pos[0]
            dy = goal_pos[1] - agent_pos[1]
            dist_to_goal = np.sqrt(dx**2 + dy**2)
            if dist_to_goal > 0:
                features[0] = dx / (dist_to_goal + 1e-6)
                features[1] = dy / (dist_to_goal + 1e-6)
        
        # Features 2-9: Celle circostanti pericolose 
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for idx, (dx, dy) in enumerate(directions):
            x, y = agent_pos[0] + dx, agent_pos[1] + dy
            if 0 <= x < grid.width and 0 <= y < grid.height:
                cell = grid.get(x, y)
                if self.task_type == "Crossing":
                    if cell and cell.type == "lava":
                        features[idx + 2] = 1.0
        
        # Features extra per DoorKey (10-15): posizioni chiave, porta, goal
        # Se non siamo in DoorKey, questi rimarranno a 0.0, mantenendo la dimensione input costante
        if self.task_type == "DoorKey":
            # Features 10-11: Direzione verso la chiave
            if not self.phase_1_key_reached and self.key_pos:
                dx = self.key_pos[0] - agent_pos[0]
                dy = self.key_pos[1] - agent_pos[1]
                dist_to_key = np.sqrt(dx**2 + dy**2)
                if dist_to_key > 0:
                    features[10] = dx / (dist_to_key + 1e-6)
                    features[11] = dy / (dist_to_key + 1e-6)
            
            # Features 12-13: Direzione verso la porta
            if self.phase_1_key_reached and not self.phase_2_door_opened and self.door_pos:
                dx = self.door_pos[0] - agent_pos[0]
                dy = self.door_pos[1] - agent_pos[1]
                dist_to_door = np.sqrt(dx**2 + dy**2)
                if dist_to_door > 0:
                    features[12] = dx / (dist_to_door + 1e-6)
                    features[13] = dy / (dist_to_door + 1e-6)
            
            # Features 14-15: Stato fasi
            features[14] = 1.0 if self.phase_1_key_reached else 0.0
            features[15] = 1.0 if self.phase_2_door_opened else 0.0
        
        return features
    
    def _get_astar_hint_features(self):
        """Restituisce one-hot encoding dell'azione consigliata da A* + info sul path.
        FIX: Usa SEMPRE dimensione 6 per evitare crash della rete neurale durante il curriculum."""
        
        # --- MODIFICA CRUCIALE: Dimensione fissa 6 per TUTTI i task ---
        # Crossing userà solo gli indici 0-2 (lasciando 3-5 a zero)
        # DoorKey userà gli indici 0-5
        hint = np.zeros(6, dtype=np.float32)
        
        # --- LOGICA ORIGINALE CROSSING (INTATTA) ---
        if self.task_type == "Crossing":
            # Se l'agente è fuori dal path cached, prova un replan
            self._ensure_astar_path_contains_agent()
            suggested = self._get_cached_astar_suggested_action()
            if suggested is not None and 0 <= suggested <= 2:
                hint[int(suggested)] = 1.0
        
        # --- LOGICA ORIGINALE DOORKEY (INTATTA) ---
        elif self.task_type == "DoorKey":
            # A* calcolata solo dopo apertura porta
            if self.phase_2_door_opened:
                # Calcola A* path UNA SOLA VOLTA dopo apertura porta
                if hasattr(self, 'doorkey_astar_path') and self.doorkey_astar_path:
                    agent_pos = tuple(self.base_env.agent_pos)
                    
                    # Trova prossima cella sul path
                    try:
                        idx = self.doorkey_astar_path.index(agent_pos)
                        if idx < len(self.doorkey_astar_path) - 1:
                            next_cell = self.doorkey_astar_path[idx + 1]
                            # Calcola azione suggerita
                            suggested = self._path_step_to_action(agent_pos, next_cell)
                            if suggested is not None and 0 <= suggested <= 2:
                                hint[int(suggested)] = 1.0
                            
                            # Features 3-5: direzione verso prossima cella del path
                            dx = next_cell[0] - agent_pos[0]
                            dy = next_cell[1] - agent_pos[1]
                            hint[3] = float(dx)  # -1, 0, o 1
                            hint[4] = float(dy)  # -1, 0, o 1
                            hint[5] = 1.0  # Flag: siamo sul path
                    except ValueError:
                        # Agente non sul path, calcola direzione verso goal
                        goal_pos = getattr(self.base_env, "goal_pos", None)
                        if goal_pos:
                            dx = goal_pos[0] - agent_pos[0]
                            dy = goal_pos[1] - agent_pos[1]
                            dist = np.sqrt(dx**2 + dy**2)
                            if dist > 0:
                                hint[3] = dx / dist
                                hint[4] = dy / dist
                            hint[5] = 0.0  # Flag: NON siamo sul path
        
        return hint
    
    def _get_lava_proximity_features(self):
        """Ritorna features relative alla vicinanza alla lava (per Crossing)"""
        return self._get_context_features()[2:]
    
    def _detect_key_and_door_positions(self):
        """Rileva e memorizza le posizioni della chiave e della porta nella griglia."""
        grid = self.base_env.grid
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell:
                    if cell.type == "key":
                        self.key_pos = (x, y)
                    elif cell.type == "door":
                        self.door_pos = (x, y)
    
    def _get_doorkey_astar_action(self):
        """Calcola azione suggerita da A* per DoorKey basata sulla fase."""
        agent_pos = tuple(self.base_env.agent_pos)
        
        # Determina target in base alla fase
        target = None
        if not self.phase_1_key_reached and self.key_pos:
            # Fase 1: vai verso cella adiacente alla chiave
            target = self._find_adjacent_cell(self.key_pos)
        elif self.phase_1_key_reached and not self.phase_2_door_opened and self.door_pos:
            # Fase 2: vai verso cella davanti alla porta
            target = self._find_adjacent_cell(self.door_pos)
        elif self.phase_2_door_opened:
            # Fase 3: vai verso il goal
            target = getattr(self.base_env, "goal_pos", None)
        
        if not target or target == agent_pos:
            return None
        
        # Calcola path A* verso target
        path = self._compute_astar_to_target(target)
        if not path or len(path) < 2:
            return None
        
        # Converti prossimo step in azione
        return self._path_step_to_action(path[0], path[1])
    
    def _find_adjacent_cell(self, target_pos):
        """Trova cella adiacente libera vicino al target."""
        grid = self.base_env.grid
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        for dx, dy in directions:
            adj_x, adj_y = target_pos[0] + dx, target_pos[1] + dy
            if 0 <= adj_x < grid.width and 0 <= adj_y < grid.height:
                cell = grid.get(adj_x, adj_y)
                if cell is None:
                    return (adj_x, adj_y)
        return target_pos
    
    def _compute_astar_to_target(self, target):
        """Calcola path A* verso target specifico."""
        from heapq import heappush, heappop
        
        start = tuple(self.base_env.agent_pos)
        grid = self.base_env.grid
        
        def cell_cost(x, y):
            cell = grid.get(x, y)
            if cell is None:
                return 1
            if cell.type in ["wall"]:
                return float('inf')
            return 1
        
        open_set = []
        heappush(open_set, (abs(start[0]-target[0]) + abs(start[1]-target[1]), 0, start, []))
        visited = set()
        
        while open_set:
            f, g, current, path = heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]
            if current == target:
                return path
            
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height and (nx, ny) not in visited:
                    cost = cell_cost(nx, ny)
                    if cost < float('inf'):
                        g_new = g + cost
                        f_new = g_new + abs(nx - target[0]) + abs(ny - target[1])
                        heappush(open_set, (f_new, g_new, (nx, ny), path))
        return []
    
    def _compute_astar_distance_to_target(self, target, avoid_lava=False):
        """Calcola la distanza A* (lunghezza path) verso un target.
        
        Args:
            target: Tupla (x, y) del target
            avoid_lava: Se True, evita celle lava (per Crossing)
        
        Returns:
            Lunghezza del path A* (numero di passi), o float('inf') se non raggiungibile
        """
        from heapq import heappush, heappop
        
        start = tuple(self.base_env.agent_pos)
        if start == target:
            return 0
            
        grid = self.base_env.grid
        
        def cell_cost(x, y):
            cell = grid.get(x, y)
            if cell is None:
                return 1
            cell_type = getattr(cell, 'type', None)
            if cell_type == "wall":
                return float('inf')
            if avoid_lava and cell_type == "lava":
                return float('inf')
            # Per DoorKey: porte chiuse sono attraversabili solo se abbiamo la chiave
            if cell_type == "door":
                if self.door_open:
                    return 1  # Porta aperta
                else:
                    # Porta chiusa: considerala raggiungibile ma non attraversabile
                    return float('inf')
            return 1
        
        open_set = []
        heappush(open_set, (abs(start[0]-target[0]) + abs(start[1]-target[1]), 0, start))
        visited = set()
        dist = {start: 0}
        
        while open_set:
            f, g, current = heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            
            if current == target:
                return g  # Ritorna la distanza effettiva
            
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height and (nx, ny) not in visited:
                    cost = cell_cost(nx, ny)
                    if cost < float('inf'):
                        g_new = g + cost
                        if (nx, ny) not in dist or g_new < dist[(nx, ny)]:
                            dist[(nx, ny)] = g_new
                            f_new = g_new + abs(nx - target[0]) + abs(ny - target[1])
                            heappush(open_set, (f_new, g_new, (nx, ny)))
        
        return float('inf')  # Target non raggiungibile
    
    def _get_current_subgoal_astar_distance(self):
        """Calcola la distanza A* verso il subgoal corrente per DoorKey.
        
        Fase 1: distanza A* verso la chiave
        Fase 2: distanza A* verso la porta
        Fase 3: distanza A* verso il goal
        
        Returns:
            Distanza A* verso il subgoal corrente, o None se non calcolabile
        """
        if self.task_type != "DoorKey":
            return None
        
        # Fase 1: vai verso la chiave
        if not self.phase_1_key_reached:
            if self.key_pos:
                return self._compute_astar_distance_to_target(self.key_pos, avoid_lava=False)
            return None
        
        # Fase 2: vai verso la porta
        if not self.phase_2_door_opened:
            if self.door_pos:
                return self._compute_astar_distance_to_target(self.door_pos, avoid_lava=False)
            return None
        
        # Fase 3: vai verso il goal
        goal_pos = getattr(self.base_env, "goal_pos", None)
        if goal_pos:
            return self._compute_astar_distance_to_target(tuple(goal_pos), avoid_lava=False)
        return None
    
    def _path_step_to_action(self, current_pos, next_pos):
        """Converte step del path in azione."""
        DIR2VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]
        
        target_dir = None
        for d, vec in DIR2VEC.items():
            if vec == (dx, dy):
                target_dir = d
                break
        
        if target_dir is None:
            return None
        
        cur_dir = self.base_env.agent_dir
        delta = (target_dir - cur_dir) % 4
        
        if delta == 0:
            return 2  # forward
        elif delta == 1:
            return 1  # right
        elif delta == 3:
            return 0  # left
        else:
            return 1  # 180°: right

    def _get_cached_astar_suggested_action(self):
        """Ritorna l'azione suggerita (0=left,1=right,2=forward) usando il path A* dell'episodio.
        None se non definito o non applicabile dall'attuale cella."""
        if self.task_type != "Crossing" or not self.episode_astar_cell_path:
            return None
        path = self.episode_astar_cell_path
        pos = tuple(self.base_env.agent_pos)
        try:
            idx = path.index(pos)
        except ValueError:
            return None
        if idx >= len(path) - 1:
            return None
        next_pos = path[idx + 1]
        dx = next_pos[0] - pos[0]
        dy = next_pos[1] - pos[1]

        DIR2VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        target_dir = None
        for d, vec in DIR2VEC.items():
            if vec == (dx, dy):
                target_dir = d
                break
        if target_dir is None:
            return None
        cur_dir = self.base_env.agent_dir
        delta = (target_dir - cur_dir) % 4
        if delta == 0:
            return self.FORWARD  # 2
        elif delta == 1:
            return 1  # right
        elif delta == 3:
            return 0  # left
        else:
            # 180 gradi: scegliamo right come primo passo
            return 1

    def _ensure_astar_path_contains_agent(self):
        """Se l'agente è uscito dal path cached o il path è vuoto, ricalcola A*.
        Replans illimitati per garantire sempre un suggerimento valido."""
        if self.task_type != "Crossing":
            return
        pos = tuple(self.base_env.agent_pos)
        
        # Se path è vuoto, consenti replans infiniti
        if not self.episode_astar_cell_path:
            if self.astar_replans_used < 10:  # Limita solo per debug
                pass  #print(f" A* replan (empty path, attempt {self.astar_replans_used + 1})")
            self.episode_astar_cell_path = self._compute_astar_cell_path()
            self.astar_replans_used += 1
            if self.episode_astar_cell_path and self.astar_replans_used < 10:
                pass  #print(f"New path: {len(self.episode_astar_cell_path)} cells")
            return
        
        # Se agente esce dal path, replan illimitato
        if pos not in self.episode_astar_cell_path:
            if self.astar_replans_used < 10: 
                pass  #print(f" A* replan #{self.astar_replans_used + 1} (agent off path at {pos})")
            self.episode_astar_cell_path = self._compute_astar_cell_path()
            self.astar_replans_used += 1

    def _compute_astar_cell_path(self):
        """Calcola una volta per episodio il path di celle sicuro, ignorando lava."""
        from heapq import heappush, heappop
        DIR2VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        start = tuple(self.base_env.agent_pos)
        goal = getattr(self.base_env, "goal_pos", None)
        if goal is None:
            return []
        grid = self.base_env.grid

        def cell_cost(x, y):
            cell = grid.get(x, y)
            if cell is None:
                return 1
            if getattr(cell, 'type', None) == "lava":
                return float('inf')
            return 1

        open_set = []
        heappush(open_set, (abs(start[0]-goal[0]) + abs(start[1]-goal[1]), 0, start, []))
        visited = set()

        while open_set:
            f, g, current, path = heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]
            if current == tuple(goal):
                return path
            x, y = current
            for dx, dy in DIR2VEC.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height and (nx, ny) not in visited:
                    cost = cell_cost(nx, ny)
                    if cost < float('inf'):
                        g_new = g + cost
                        f_new = g_new + abs(nx - goal[0]) + abs(ny - goal[1])
                        heappush(open_set, (f_new, g_new, (nx, ny), path))
        return []

    def _get_front_cell_type(self):
        """Restituisce il tipo di cella davanti all'agente."""
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

    def get_state(self):
        return self.state.reshape(1, -1)

    def get_valid_actions(self):
        valid = self.BASIC_ACTIONS.copy()  # [0, 1] = left, right (always valid)
        front = self._get_front_cell_type()
        
        # FORWARD (2): only valid if not facing wall or closed door
        if front not in ["wall"]:
            # Check if it's a closed door (can't walk through)
            if front == "door":
                # Can only walk through if door is open
                if self.door_open:
                    valid.append(self.FORWARD)
            else:
                # Empty, goal, key, lava (agent can step on these)
                valid.append(self.FORWARD)
        
        if self.task_type == "DoorKey":
            # Check if agent is carrying the key
            carrying = getattr(self.base_env, "carrying", None)
            has_key = carrying is not None and getattr(carrying, "type", None) == "key"
            
            # PICKUP (3): only valid if key is in front AND agent doesn't have it yet
            if not has_key and not self.phase_1_key_reached:
                valid.append(self.PICKUP)
            
            # TOGGLE (5): only valid if door is in front AND agent HAS the key AND door is closed
            if has_key and not self.door_open:
                valid.append(self.TOGGLE)
        
        return valid

    def reset(self):
        # Gestione seed per Crossing e DoorKey curriculum
        # Skip completo della logica seed se siamo in fase Mixed
        if self.skip_seed_logic:
            self.obs, _ = self.env.reset()
        elif self.task_type in ["Crossing", "DoorKey"]:
            self.episode_count += 1
            
            # Calcola quale ciclo siamo (ogni episodes_per_seed episodi cambia seed)
            cycle = (self.episode_count - 1) // self.episodes_per_seed
            
            if cycle < self.fixed_seed_cycles:
                # Usa seed fisso per questo ciclo
                self.use_fixed_seed = True
                # Seed diversi per task diversi
                base_seed = 42 if self.task_type == "Crossing" else 1000
                self.current_seed = base_seed + cycle * 100
                if (self.episode_count - 1) % self.episodes_per_seed == 0:
                    print(f"Seed fisso {self.current_seed} per i prossimi {self.episodes_per_seed} episodi (ciclo {cycle+1}/{self.fixed_seed_cycles})")
                self.obs, _ = self.env.reset(seed=self.current_seed)
            else:
                # Dopo fixed_seed_cycles, usa seed casuali
                if self.use_fixed_seed:
                    total_fixed_episodes = self.fixed_seed_cycles * self.episodes_per_seed
                    print(f"Completati {total_fixed_episodes} episodi con {self.fixed_seed_cycles} seed fissi!")
                    print(f"   Ora training con seed casuali per generalizzazione")
                    self.use_fixed_seed = False
                self.obs, _ = self.env.reset()
        else:
            self.obs, _ = self.env.reset()
            
        if not hasattr(self.base_env, "goal_pos") or self.base_env.goal_pos is None:
            self.base_env.goal_pos = (self.base_env.width-2, self.base_env.height-2)
            print(f"DEBUG: goal_pos impostato a {self.base_env.goal_pos}")
        self.state = self._obs_to_state(self.obs)
        self.door_open = False
        self._door_unlocked_rewarded = False
        self.best_dist_to_goal = float('inf')
        
        # DoorKey: rileva posizioni e reset fasi
        if self.task_type == "DoorKey":
            self.key_pos = None
            self.door_pos = None
            self.phase_1_key_reached = False
            self.phase_2_door_opened = False
            self.phase_3_goal_reached = False
            self._key_pickup_rewarded = False  # Flag per reward chiave una sola volta
            self._door_open_rewarded = False   # Flag per reward porta una sola volta
            self.doorkey_astar_path = None     # A* path calcolato SOLO dopo apertura porta
            self._toggle_on_open_door_count = 0  # Contatore TOGGLE su porta già aperta
            self._phase1_rewarded_cells = set()  # Celle già premiate in fase 1
            self._phase2_rewarded_cells = set()  # Celle già premiate in fase 2
            self._near_key_rewarded = False      # Bonus adiacente chiave (una volta)
            self._near_door_rewarded = False     # Bonus adiacente porta (una volta)
            self._forward_blocked_count = 0      # Contatore forward contro ostacoli
            # Reset distanze per reward shaping continuo
            if hasattr(self, '_prev_dist_to_key'):
                delattr(self, '_prev_dist_to_key')
            if hasattr(self, '_best_dist_to_key'):
                delattr(self, '_best_dist_to_key')
            if hasattr(self, '_prev_dist_to_door'):
                delattr(self, '_prev_dist_to_door')
            if hasattr(self, '_best_dist_to_door'):
                delattr(self, '_best_dist_to_door')
            if hasattr(self, '_prev_dist_to_goal_phase3'):
                delattr(self, '_prev_dist_to_goal_phase3')
            self._detect_key_and_door_positions()
        
        # Precalcolo A* path per l'episodio (prima di inizializzare prev_dist_to_goal)
        if self.task_type == "Crossing":
            self.astar_replans_used = 0
            self.episode_astar_cell_path = self._compute_astar_cell_path()
            # Per Crossing: inizializza prev_dist_to_goal con distanza A*
            goal_pos = getattr(self.base_env, "goal_pos", None)
            if goal_pos:
                astar_dist = self._compute_astar_distance_to_target(tuple(goal_pos), avoid_lava=True)
                if astar_dist < float('inf'):
                    self.prev_dist_to_goal = astar_dist
                else:
                    self.prev_dist_to_goal = self._compute_distance_to_goal()
            else:
                self.prev_dist_to_goal = self._compute_distance_to_goal()
        else:
            self.episode_astar_cell_path = None
            self.prev_dist_to_goal = self._compute_distance_to_goal()
        
        self.prev_agent_pos = getattr(self.base_env, "agent_pos", None)
        
        # Reset contatori
        self.current_step = 0
        self.safe_passage_rewarded = False
        self.steps_without_lava = 0
        
        # Reset A* hint reward tracking
        self.astar_reward_given_cells = set()
        self.near_goal_distance = None
        
        # Reset tracking reward episodio
        self.episode_reward_sum = 0.0
        
        # Reset milestone flags
        if hasattr(self, '_reached_dist_3'):
            delattr(self, '_reached_dist_3')
        if hasattr(self, '_reached_dist_2'):
            delattr(self, '_reached_dist_2')
        if hasattr(self, '_reached_dist_1'):
            delattr(self, '_reached_dist_1')
        
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

        # Incrementa step counter
        self.current_step += 1
        
        # SALVA front_type PRIMA di eseguire l'azione, in questo modo DoorKey può usarlo dopo
        front_type_before = self._get_front_cell_type()
        
        # se azione è FORWARD, controlla se bloccato
        forward_blocked = False
        if action == self.FORWARD:
            if front_type_before == "wall":
                forward_blocked = True
        
        # Controllo 2: Prossimità goal 
        dist_to_goal = self._compute_distance_to_goal()
        near_goal = dist_to_goal is not None and dist_to_goal <= 2
        
        # A* guidance hint per Crossing
        astar_suggested_action = None
        if self.task_type == "Crossing":
            self._ensure_astar_path_contains_agent()
            astar_suggested_action = self._get_cached_astar_suggested_action()
            self.last_astar_suggested_action = astar_suggested_action
            # Debug iniziale (non invasivo)
            if self.current_step < 3:
                print(f"    Step {self.current_step}: A* suggests action {astar_suggested_action} (path exists: {bool(self.episode_astar_cell_path)}, size: {len(self.episode_astar_cell_path) if self.episode_astar_cell_path else 0})")

        # Esegui l'azione nell'environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.state = self._obs_to_state(obs)
        done = terminated or truncated
        info = info or {}
        reward = -0.001 # Penalità base per step
        # Penalità forward contro muro, lava, o ostacolo
        # Penalità forward contro muro, lava, o ostacolo
        if forward_blocked:
            self.wall_bump_count += 1
            penalty = 0.05 
            
            reward = -penalty
            info["forward_blocked"] = True
            info["wall_bump_count"] = self.wall_bump_count
            info["wall_penalty"] = penalty
            
            # Se anche vicino al goal, penalità leggermente superiore ma non letale
            if near_goal:
                reward = -penalty * 2.0 
                info["forward_blocked_near_goal"] = True
            
            # Traccia reward e return immediato
            self.episode_reward_sum += reward
            # Replan mirato: se davanti è muro/lava, ricalcola path per evitare deadlock
            if self.task_type == "Crossing":
                self._replan_astar(info, reason="front_blocked")
            return self.get_state(), float(reward), False, info

        # Controllo lava immediato, solo per Crossing
        if self.task_type == "Crossing":
            agent_pos = self.base_env.agent_pos
            grid = self.base_env.grid
            cell = grid.get(*agent_pos)
            
            if cell and cell.type == "lava":
                # MORTE LAVA: penalità ridotta per favorire apprendimento
                print("LAVA! Agente morto!")
                info["lava_penalty"] = True
                info["lava_death"] = True
                return self.get_state(), -30.0, True, info

        # Controllo timeout episodio
        if self.current_step >= self.max_steps and not terminated:
            print("Timeout! Episodio troppo lungo")
            info["timeout"] = True
            # Penalità CONSISTENTE: basata su distanza dal goal per tutti i task
            dist_to_goal = self._compute_distance_to_goal() or 10
            # Penalità base -10, scalata in base a quanto lontano dal goal
            timeout_penalty = -10.0 - (dist_to_goal * 0.5)
            timeout_penalty = max(timeout_penalty, -25.0)  # Cap a -25
            return self.get_state(), timeout_penalty, True, info

        # Controllo frontale DOPO azione, eventuali stati speciali
        # Per DoorKey usiamo front_type_before salvato prima dell'azione
        front_type = self._get_front_cell_type()
        carrying = getattr(self.base_env, "carrying", None)

        # Inizializza reward base per step
        if self.task_type == "DoorKey":
            reward = -0.001  # Penalità minima per DoorKey
        else:
            reward = -0.001

        # Crossing task: Lava avoidance bonuses e penalità
        if self.task_type == "Crossing":
            # Premia leggermente la sopravvivenza lontano dalla lava
            self.steps_without_lava += 1
            if self.steps_without_lava > 5:
                reward += 0.005

            # Penalità per prossimità lava, bonus per passaggi sicuri
            lava_proximity = self._compute_lava_proximity()
            safe_cells_nearby = self._count_safe_cells_near_lava()

            # Bonus passaggio sicuro, solo una volta per episodio
            if lava_proximity < 3 and safe_cells_nearby > 0 and not self.safe_passage_rewarded:
                reward += 0.3
                self.safe_passage_rewarded = True
                info["found_safe_passage"] = True
            elif lava_proximity == 1:
                reward -= 0.1  # Penalità ridotta per essere adiacente alla lava
            elif lava_proximity == 2:
                reward -= 0.02  # Penalità leggera a distanza 2

            # A* guidance bonus: premia se l'agente segue il suggerimento ma solo una volta per cella
            agent_pos = tuple(self.base_env.agent_pos)
            if astar_suggested_action is not None and action == astar_suggested_action:
                if agent_pos not in self.astar_reward_given_cells:
                    reward += 1.5
                    self.astar_reward_given_cells.add(agent_pos)
                    info["astar_hint_followed"] = True
            elif astar_suggested_action is not None and action != astar_suggested_action:
                # Penalità ridotta per non seguire A*
                if lava_proximity <= 2:
                    reward -= 0.15  # Penalità moderata: vicino lava e non segue A*
                else:
                    reward -= 0.05  # Penalità minima: lontano da lava
                info["astar_hint_ignored"] = True

            info["lava_proximity"] = lava_proximity
            info["safe_cells_nearby"] = safe_cells_nearby

        # [MODIFICA TURBO] POTENTIAL-BASED REWARD per Empty e Crossing
        # Evita reward loops: reward = gamma * phi(s') - phi(s)
        # dove phi(s) = -distanza (più vicino = potenziale più alto)
        # 
        # IMPORTANTE per Crossing: usa distanza A* (che evita lava) invece di Manhattan
        # Questo permette all'agente di allontanarsi dal goal se necessario per aggirare la lava
        if self.task_type == "Empty":
            # Empty: usa distanza Manhattan semplice
            dist_to_goal = self._compute_distance_to_goal()
            
            if dist_to_goal is not None and self.prev_dist_to_goal is not None:
                gamma_pb = 0.99
                phi_prev = -self.prev_dist_to_goal
                phi_curr = -dist_to_goal
                potential_reward = gamma_pb * phi_curr - phi_prev
                reward += potential_reward * 0.1
            
            self.prev_dist_to_goal = dist_to_goal
            
        elif self.task_type == "Crossing":
            # Crossing: usa distanza A* che evita la lava
            # Questo premia l'agente per seguire il percorso sicuro, anche se si allontana dal goal
            goal_pos = getattr(self.base_env, "goal_pos", None)
            if goal_pos:
                astar_dist = self._compute_astar_distance_to_target(tuple(goal_pos), avoid_lava=True)
                
                # Se il path esiste, usa la distanza A*
                if astar_dist < float('inf'):
                    if self.prev_dist_to_goal is not None:
                        gamma_pb = 0.99
                        phi_prev = -self.prev_dist_to_goal
                        phi_curr = -astar_dist
                        potential_reward = gamma_pb * phi_curr - phi_prev
                        reward += potential_reward * 0.1
                        
                        # Debug per verificare che stia usando A*
                        if abs(potential_reward) > 0.01:
                            info["astar_distance_used"] = True
                    
                    self.prev_dist_to_goal = astar_dist
                else:
                    # Fallback a distanza Manhattan se A* fallisce
                    dist_to_goal = self._compute_distance_to_goal()
                    if dist_to_goal is not None and self.prev_dist_to_goal is not None:
                        gamma_pb = 0.99
                        phi_prev = -self.prev_dist_to_goal
                        phi_curr = -dist_to_goal
                        potential_reward = gamma_pb * phi_curr - phi_prev
                        reward += potential_reward * 0.1
                    self.prev_dist_to_goal = dist_to_goal
            
            # Reset near_goal_distance se non serve più per logiche complesse
            self.near_goal_distance = None

        # Goal raggiunto: controlla se l'agente è sulla posizione del goal
        goal_pos = getattr(self.base_env, "goal_pos", None)
        agent_pos = self.base_env.agent_pos
        
        if goal_pos is not None and tuple(agent_pos) == tuple(goal_pos):
            # print("Goal raggiunto!") # Commentato per pulire il log
            
            # Bonus efficienza
            efficiency_bonus = max(0.0, (self.max_steps - self.current_step) / self.max_steps) * 50.0
            
            # SUPER BONUS basato sulla dimensione
            grid_size = getattr(self.base_env, 'width', 5)
            if grid_size > 10: # Per le 16x16
                base_reward = 600.0 # Abbastanza alto da coprire 2000 passi di penalità
            elif grid_size > 6: # Per le 8x8
                # Crossing 9x9/11x11 beneficia di reward più alto
                if self.task_type == "Crossing":
                    base_reward = 400.0  # Aumentato per bilanciare penalità lava
                else:
                    base_reward = 300.0
            else:
                # Crossing piccolo anche aumentato
                if self.task_type == "Crossing":
                    base_reward = 150.0
                else:
                    base_reward = 100.0
                
            reward += base_reward + efficiency_bonus
            info["goal_reached"] = True
            info["efficiency_bonus"] = efficiency_bonus

        # Anti-spinning: penalità per rotazioni eccessive, specialmente vicino al goal
        if action == 0 or action == 1:  # 0=left, 1=right
            if self.last_was_rotation:
                self.spin_count += 1
            else:
                self.spin_count = 1
                self.last_was_rotation = True

            # Penalità progressiva crescente per ogni rotazione consecutiva
            if self.task_type == "DoorKey":
                # Penalità spinning per DoorKey più lieve
                penalty = 0.1 * self.spin_count  # 0.1, 0.2, 0.3...
                penalty = min(penalty, 0.5)  # Cap a -0.5 per DoorKey
            else:
                penalty = 0.3 * self.spin_count
                penalty = min(penalty, 2.0)  # Cap a -2.0 per altri
            reward -= penalty
            info["spin_penalty"] = penalty
            info["spin_count"] = self.spin_count

            # Penalità raddoppiata se vicino al goal
            if near_goal:
                reward -= penalty  # Raddoppia effettivamente
                info["spin_near_goal"] = True
            # Replan mirato: se la rotazione non segue il suggerimento A*, ricalcola
            if self.task_type == "Crossing" and self.last_astar_suggested_action is not None:
                if action != self.last_astar_suggested_action:
                    self._replan_astar(info, reason="rotation_away_from_hint")
            return self.get_state(), float(reward), False, info

        else:
            # reset se l’agente fa avanti o altro
            self.spin_count = 0
            self.last_was_rotation = False
            # Reset anche wall bump se fa azioni diverse
            self.wall_bump_count = 0
        # DoorKey task - Reward shaping per 3 fasi
        if self.task_type == "DoorKey" and not done:
            carry_type = getattr(carrying, "type", None) if carrying else None
            agent_pos = tuple(self.base_env.agent_pos)
            
            # Calcola dimensione griglia per reward scaling
            grid_size = getattr(self.base_env, 'width', 5)
            is_large_map = grid_size > 5
            
            # Fase 1: raccogliere chiave
            if not self.phase_1_key_reached:
                # Reward shaping: distanza A* verso la chiave (considera muri)
                if self.key_pos:
                    # USA DISTANZA A* invece di Manhattan per considerare i muri
                    dist_to_key = self._compute_astar_distance_to_target(self.key_pos, avoid_lava=False)
                    
                    # Se A* fallisce, fallback a Manhattan
                    if dist_to_key == float('inf'):
                        dist_to_key = abs(agent_pos[0] - self.key_pos[0]) + abs(agent_pos[1] - self.key_pos[1])
                    
                    # POTENTIAL-BASED: reward per avvicinamento alla chiave (usa A*)
                    if hasattr(self, '_prev_dist_to_key') and self._prev_dist_to_key is not None:
                        gamma_pb = 0.99
                        phi_prev = -self._prev_dist_to_key
                        phi_curr = -dist_to_key
                        potential_reward = gamma_pb * phi_curr - phi_prev
                        # Scala: 0.15 per allineare con Empty/Crossing (~0.1)
                        reward += potential_reward * 0.15
                        if potential_reward > 0:
                            info["approaching_key"] = True
                            info["using_astar_distance"] = True
                        elif potential_reward < 0:
                            info["moving_away_from_key"] = True
                    self._prev_dist_to_key = dist_to_key
                    
                    # Se adiacente alla chiave (dist=1), bonus una sola volta (NORMALIZZATO)
                    if dist_to_key == 1 and not self._near_key_rewarded:
                        reward += 0.5  # Ridotto da 3.0/2.0
                        self._near_key_rewarded = True
                        info["near_key"] = True
                        print(f"Fase 1: Adiacente alla chiave!")
                    
                    # Bonus milestone per nuova distanza minima (NORMALIZZATO)
                    if agent_pos not in self._phase1_rewarded_cells:
                        if hasattr(self, '_best_dist_to_key'):
                            if dist_to_key < self._best_dist_to_key:
                                reward += 0.2  # Ridotto da 1.0/0.6
                                self._phase1_rewarded_cells.add(agent_pos)
                                self._best_dist_to_key = dist_to_key
                        else:
                            self._best_dist_to_key = dist_to_key
                
                # FORWARD contro ostacolo: penalità progressiva
                if action == self.FORWARD:
                    if front_type_before in ("wall", "key", "door"):
                        self._forward_blocked_count += 1
                        penalty = 0.1 * self._forward_blocked_count  # 0.1, 0.2, 0.3...
                        penalty = min(penalty, 0.5)  # Cap
                        reward -= penalty
                        info["forward_blocked"] = True
                        info["forward_blocked_count"] = self._forward_blocked_count
                        # Suggerimento: davanti chiave devi fare PICKUP
                        if front_type_before == "key":
                            # Penalità moderata per non fare PICKUP quando dovresti
                            reward -= 0.3
                            info["hint_pickup"] = True
                            info["missed_pickup_opportunity"] = True
                    else:
                        # Reset contatore se forward va a buon fine
                        self._forward_blocked_count = 0
                
                # Penalità per rotazioni quando sei davanti alla chiave 
                if front_type_before == "key" and action in [0, 1]:  # left, right
                    reward -= 0.3  # Ridotto da 2.0
                    info["rotating_at_key"] = True
                
                # PICKUP chiave
                if action == self.PICKUP:
                    if front_type_before == "key" and not self._key_pickup_rewarded:
                        print(" FASE 1: Chiave raccolta!")
                        # Reward NORMALIZZATA: allineata con goal reward (~5x approaching bonus)
                        reward += 5.0
                        self.phase_1_key_reached = True
                        self._key_pickup_rewarded = True
                        info["key_picked"] = True
                        info["phase_1_complete"] = True
                    elif front_type_before != "key":
                        reward -= 0.1
                        info["pickup_fail_no_key"] = True
                    # Se già raccolta, nessun reward extra
            
            # Fase 2: aprire la porta
            elif self.phase_1_key_reached and not self.phase_2_door_opened:
                # Reward shaping: distanza A* verso la porta (considera muri)
                if self.door_pos:
                    # USA DISTANZA A* invece di Manhattan per considerare i muri
                    dist_to_door = self._compute_astar_distance_to_target(self.door_pos, avoid_lava=False)
                    
                    # Se A* fallisce, fallback a Manhattan
                    if dist_to_door == float('inf'):
                        dist_to_door = abs(agent_pos[0] - self.door_pos[0]) + abs(agent_pos[1] - self.door_pos[1])
                    
                    # POTENTIAL-BASED: reward per avvicinamento alla porta (usa A*)
                    if hasattr(self, '_prev_dist_to_door') and self._prev_dist_to_door is not None:
                        gamma_pb = 0.99
                        phi_prev = -self._prev_dist_to_door
                        phi_curr = -dist_to_door
                        potential_reward = gamma_pb * phi_curr - phi_prev
                        # Scala: 0.15 per allineare con Empty/Crossing (~0.1)
                        reward += potential_reward * 0.15
                        if potential_reward > 0:
                            info["approaching_door"] = True
                            info["using_astar_distance"] = True
                        elif potential_reward < 0:
                            info["moving_away_from_door"] = True
                    self._prev_dist_to_door = dist_to_door
                    
                    # Se adiacente alla porta con chiave, bonus UNA SOLA VOLTA (NORMALIZZATO)
                    if dist_to_door == 1 and carry_type == "key" and not self._near_door_rewarded:
                        reward += 0.5  # Ridotto da 3.0/2.0
                        self._near_door_rewarded = True
                        info["near_door_with_key"] = True
                        print(f"Fase 2: Adiacente alla porta con chiave!")
                    
                    # Bonus milestone per nuova distanza minima (NORMALIZZATO)
                    if agent_pos not in self._phase2_rewarded_cells:
                        if hasattr(self, '_best_dist_to_door'):
                            if dist_to_door < self._best_dist_to_door:
                                reward += 0.2  # Ridotto da 1.0/0.6
                                self._phase2_rewarded_cells.add(agent_pos)
                                self._best_dist_to_door = dist_to_door
                        else:
                            self._best_dist_to_door = dist_to_door
                
                # FORWARD contro porta chiusa: penalità PROGRESSIVA
                if action == self.FORWARD:
                    if front_type_before in ("wall", "door"):
                        self._forward_blocked_count += 1
                        penalty = 0.1 * self._forward_blocked_count  # 0.1, 0.2, 0.3...
                        penalty = min(penalty, 0.5)  # Cap
                        reward -= penalty
                        info["forward_blocked"] = True
                        info["forward_blocked_count"] = self._forward_blocked_count
                        # Suggerimento: davanti porta devi fare TOGGLE
                        if front_type_before == "door":
                            # Penalità moderata per non fare TOGGLE quando dovresti
                            reward -= 0.3
                            info["hint_toggle"] = True
                            info["missed_toggle_opportunity"] = True
                    else:
                        self._forward_blocked_count = 0
                
                # Penalità per rotazioni quando sei davanti alla porta con chiave
                if front_type_before == "door" and carry_type == "key" and action in [0, 1]:  # left, right
                    reward -= 0.3  # Ridotto da 2.0
                    info["rotating_at_door"] = True
                
                # TOGGLE porta
                if action == self.TOGGLE:
                    if front_type_before == "door" and carry_type == "key" and not self._door_open_rewarded:
                        print(" Fase 2: Porta aperta!")
                        # Reward NORMALIZZATA: allineata con goal reward (~5x approaching bonus)
                        reward += 5.0
                        self.phase_2_door_opened = True
                        self.door_open = True
                        self._door_open_rewarded = True
                        info["door_opened"] = True
                        info["phase_2_complete"] = True
                        # CALCOLA A* PATH UNA SOLA VOLTA - dal punto corrente al goal
                        goal_pos = getattr(self.base_env, "goal_pos", None)
                        if goal_pos:
                            self.doorkey_astar_path = self._compute_astar_to_target(goal_pos)
                            if self.doorkey_astar_path:
                                print(f" A* path calcolato: {len(self.doorkey_astar_path)} celle verso goal")
                    elif front_type_before != "door":
                        reward -= 0.2
                        info["toggle_fail_no_door"] = True
                    elif front_type_before == "door" and not carry_type:
                        reward -= 0.2
                        info["toggle_without_key"] = True
                
                # Penalità per DROP chiave prima di aprire porta
                if action == self.DROP and carry_type == "key":
                    reward -= 0.5
                    info["dropped_key_before_door"] = True
            
            # Fase 3: raggiungere il goal (con reward distanza + A* path)
            elif self.phase_2_door_opened:
                # POTENTIAL-BASED reward shaping: distanza A* verso il goal (considera muri e porta)
                goal_pos = getattr(self.base_env, "goal_pos", None)
                if goal_pos:
                    # USA DISTANZA A* invece di Manhattan per considerare il layout della mappa
                    dist_to_goal = self._compute_astar_distance_to_target(tuple(goal_pos), avoid_lava=False)
                    
                    # Se A* fallisce, fallback a Manhattan
                    if dist_to_goal == float('inf'):
                        dist_to_goal = abs(agent_pos[0] - goal_pos[0]) + abs(agent_pos[1] - goal_pos[1])
                    
                    # POTENTIAL-BASED: reward per avvicinamento al goal (usa A*)
                    if hasattr(self, '_prev_dist_to_goal_phase3') and self._prev_dist_to_goal_phase3 is not None:
                        gamma_pb = 0.99
                        phi_prev = -self._prev_dist_to_goal_phase3
                        phi_curr = -dist_to_goal
                        potential_reward = gamma_pb * phi_curr - phi_prev
                        # Scala: 0.15 per allineare con Empty/Crossing (~0.1)
                        reward += potential_reward * 0.15
                        if potential_reward > 0:
                            info["approaching_goal"] = True
                            info["using_astar_distance"] = True
                        elif potential_reward < 0:
                            info["moving_away_from_goal"] = True
                    self._prev_dist_to_goal_phase3 = dist_to_goal
                
                # Reward BONUS se l'agente è su una cella del path A* (NORMALIZZATO)
                if self.doorkey_astar_path and agent_pos in self.doorkey_astar_path:
                    if agent_pos not in self.astar_reward_given_cells:
                        reward += 0.3  # Ridotto da 2.0
                        self.astar_reward_given_cells.add(agent_pos)
                        info["on_astar_path"] = True
                
                # Penalità ridotte per azioni inutili in fase 3
                if action == self.PICKUP:
                    reward -= 0.1
                    info["pickup_in_phase3"] = True
                elif action == self.TOGGLE:
                    if front_type == "door":
                        # TOGGLE su porta già aperta - penalità progressiva ma moderata
                        self._toggle_on_open_door_count += 1
                        penalty = 0.3 * self._toggle_on_open_door_count  # 0.3, 0.6, 0.9...
                        penalty = min(penalty, 1.0)  # Cap
                        reward -= penalty
                        info["toggle_door_already_open"] = True
                        info["toggle_door_count"] = self._toggle_on_open_door_count
                    elif front_type is None or front_type == "empty":
                        # TOGGLE a vuoto - penalità piccola
                        reward -= 0.1
                        info["toggle_empty"] = True
                    else:
                        # TOGGLE su altro oggetto - penalità minore
                        reward -= 0.1
                        info["toggle_in_phase3"] = True

        # Traccia reward totale episodio
        self.episode_reward_sum += reward
        
        # Replan post-azione se necessario: nessun hint, fronte pericoloso
        if self.task_type == "Crossing":
            if self.last_astar_suggested_action is None:
                self._replan_astar(info, reason="hint_none")
            front_now = self._get_front_cell_type()
            if front_now in ("wall", "lava"):
                self._replan_astar(info, reason=f"front_{front_now}")
        
        # Tracking fase fallita per DoorKey (senza penalità)
        if self.task_type == "DoorKey" and done and not info.get("goal_reached", False):
            # Solo tracking, nessuna penalità finale
            if not self.phase_1_key_reached:
                info["failed_phase"] = 1
            elif not self.phase_2_door_opened:
                info["failed_phase"] = 2
            else:
                info["failed_phase"] = 3
        
        # Correzione per Crossing (non DoorKey)
        if self.task_type != "DoorKey" and done and not info.get("goal_reached", False):
            correction = -30.0 - self.episode_reward_sum
            reward += correction
            info["failed_episode_correction"] = correction
        
        # Clip reward, meno restrittivo per evitare segnali troppo negativi
        reward = float(np.clip(reward, -50.0, 150.0))

        return self.get_state(), float(reward), bool(done), info
    
    def _compute_lava_proximity(self):
        """Calcola la distanza minima alla lava più vicina."""
        agent_pos = self.base_env.agent_pos
        grid = self.base_env.grid
        
        min_dist = float('inf')
        
        # Cerca in un raggio di 5 celle
        for x in range(max(0, agent_pos[0]-5), min(grid.width, agent_pos[0]+6)):
            for y in range(max(0, agent_pos[1]-5), min(grid.height, agent_pos[1]+6)):
                cell = grid.get(x, y)
                if cell and cell.type == "lava":
                    dist = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
                    min_dist = min(min_dist, dist)
        
        return min_dist if min_dist != float('inf') else 10
    
    def _count_safe_cells_near_lava(self):
        """Conta quante celle sicure (non-lava) ci sono vicino all'agente E vicino alla lava.
        Questo identifica i passaggi sicuri attraverso la lava."""
        agent_pos = self.base_env.agent_pos
        grid = self.base_env.grid
        
        safe_count = 0
        
        # Controlla le 8 direzioni intorno all'agente
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            x, y = agent_pos[0] + dx, agent_pos[1] + dy
            
            if 0 <= x < grid.width and 0 <= y < grid.height:
                cell = grid.get(x, y)
                
                # Se la cella è sicura
                if cell is None or (cell.type not in ["lava", "wall"]):
                    # Controlla se questa cella sicura ha lava nelle vicinanze
                    has_lava_nearby = False
                    for dx2, dy2 in directions:
                        x2, y2 = x + dx2, y + dy2
                        if 0 <= x2 < grid.width and 0 <= y2 < grid.height:
                            cell2 = grid.get(x2, y2)
                            if cell2 and cell2.type == "lava":
                                has_lava_nearby = True
                                break
                    
                    # Se è una cella sicura vicino alla lava , contala
                    if has_lava_nearby:
                        safe_count += 1
        
        return safe_count

    def render(self):
        return self.env.render()

    # Metodi A* per Crossing task
    def _replan_astar(self, info: dict, reason: str = ""):
        try:
            self.episode_astar_cell_path = self._compute_astar_cell_path()
            self.astar_replans_used += 1
            if info is not None:
                info_key = "astar_replan_reason"
                # conserva solo la prima ragione se non presente
                if info_key not in info:
                    info[info_key] = reason
        except Exception:
            pass

