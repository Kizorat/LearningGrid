import os
import re
import sys
import argparse
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
from pathlib import Path
from datetime import datetime
from collections import deque
from transformers import T5Tokenizer, T5ForConditionalGeneration
from minigrid.wrappers import FullyObsWrapper
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from DQNAgent.agent import DQNAgent
from DQNAgent.enviroment import DynamicMiniGridWrapper
from DQNAgent.train_npc import make_env, detect_task_type
from DQNAgent.train_all_maps import run_training_suite
from Helper.helper import LLMHelper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

DEFAULT_EPISODES = 450  # Episodes to run after the helper curriculum phase
DEFAULT_HELPER_EPISODES = 2200
DEFAULT_INITIAL_STEPS = 15
DEFAULT_THRESHOLD_INITIAL = 1.0
DEFAULT_THRESHOLD_DECAY = 0.1
DEFAULT_MODEL_NAME = "qwen3:1.7b"
DEFAULT_BATCH_SIZE = 64

ALL_ENVIRONMENTS = [
    "MiniGrid-Empty-5x5-v0",
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-Empty-16x16-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-LavaCrossingS9N3-v0",
    "MiniGrid-LavaCrossingS11N5-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-DoorKey-16x16-v0",
]

# Curriculum schedule — each env runs for the specified number of episodes
# Mirrors train_all_maps.py: knowledge transfers across envs via the same agent
MAP_SCHEDULE = [
    # Phase 1: Empty (basic navigation)
    {"id": "MiniGrid-Empty-5x5-v0",        "episodes": 100},
    {"id": "MiniGrid-Empty-8x8-v0",        "episodes": 150},
    {"id": "MiniGrid-Empty-16x16-v0",      "episodes": 300},
    # Phase 2: Lava Crossing (obstacle avoidance)
    {"id": "MiniGrid-LavaCrossingS9N1-v0", "episodes": 300},
    {"id": "MiniGrid-LavaCrossingS9N3-v0", "episodes": 300},
    {"id": "MiniGrid-LavaCrossingS11N5-v0","episodes": 300},
    # Phase 3: DoorKey (object manipulation)
    {"id": "MiniGrid-DoorKey-5x5-v0",      "episodes": 200},
    {"id": "MiniGrid-DoorKey-8x8-v0",      "episodes": 250},
    {"id": "MiniGrid-DoorKey-16x16-v0",    "episodes": 300},
]


def build_training_schedule(post_helper_episodes: int) -> list:
    """Build full HeRoN schedule: fixed helper curriculum + random post-helper phase.

    The helper curriculum is MAP_SCHEDULE (2200 episodes by default). After that,
    ``post_helper_episodes`` episodes are added by sampling environments uniformly
    at random from ALL_ENVIRONMENTS.
    """
    schedule = [dict(entry) for entry in MAP_SCHEDULE]
    for _ in range(post_helper_episodes):
        schedule.append({"id": random.choice(ALL_ENVIRONMENTS), "episodes": 1})
    return schedule

REVIEWER_MODEL_DIR = ROOT_DIR / "ReviewerPPO" / "Model"
PLOTS_DIR = Path(__file__).parent / "Plots"
RESULTS_DIR = Path(__file__).parent / "Results"

# Maps action names to MiniGrid action IDs and back
_ACTION_NAME_TO_ID = {'forward': 2, 'left': 0, 'right': 1, 'pickup': 3, 'toggle': 5}
_ACTION_ID_TO_NAME = {v: k for k, v in _ACTION_NAME_TO_ID.items()}
_DIR_NAMES = ["East", "South", "West", "North"]
_DIR_VECTORS = [(1, 0), (0, 1), (-1, 0), (0, -1)]


# T5 Reviewer LLM proposes actions, reviewer provides feedback, and helper revises the proposal based on feedback
class T5Reviewer:
    def __init__(self, model_path: Path, verbose: bool = False):
        self.verbose = verbose
        self.model_path = model_path
        print(f"Loading Reviewer T5 from {model_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(str(model_path))
        self.model = T5ForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(DEVICE)
        self.model.eval()
        if torch.cuda.is_available():
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Reviewer T5 loaded ({n_params:.1f}M params)")

    # Returns a dict with common state fields extracted from the wrapper
    def _extract_env_state(self, env_wrapper):
        base_env = env_wrapper.env.unwrapped
        env_name = getattr(env_wrapper, 'env_id', 'MiniGrid-Empty-5x5-v0')
        is_doorkey = 'doorkey' in env_name.lower()
        grid = base_env.grid

        agent_pos = tuple(int(x) for x in base_env.agent_pos)
        agent_dir = int(base_env.agent_dir)
        goal_pos = getattr(base_env, 'goal_pos', (base_env.width - 2, base_env.height - 2))

        carrying = getattr(base_env, 'carrying', None)
        has_key = carrying is not None and getattr(carrying, 'type', None) == 'key'

        door_open = False
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell and getattr(cell, 'type', None) == 'door':
                    door_open = getattr(cell, 'is_open', False)

        return {
            'env_name': env_name,
            'is_doorkey': is_doorkey,
            'grid': grid,
            'agent_pos': agent_pos,
            'agent_dir': agent_dir,
            'goal_pos': goal_pos,
            'has_key': has_key,
            'door_open': door_open,
        }

    # Builds the prompt sent to the T5 model, optionally including the Helper's suggested actions for context
    def _build_prompt(self, state: dict, helper_suggestion: str = None) -> str:
        env_name = state['env_name']
        agent_pos = state['agent_pos']
        agent_dir = state['agent_dir']
        goal_pos = state['goal_pos']
        is_doorkey = state['is_doorkey']
        has_key = state['has_key']
        door_open = state['door_open']

        agent_dir_str = _DIR_NAMES[agent_dir]

        if is_doorkey:
            actions_desc = (
                '"forward" (move 1 cell), "left"/"right" (rotate), '
                '"pickup" (take key), "toggle" (open door)'
            )
            available = "0,1,2,3,5"
            rules = (
                'RULES:\n'
                '- Avoid walls and lava\n'
                '- "forward" moves in facing direction\n'
                '- "left"/"right" only rotate\n'
                '- "pickup" to take the key when in front of it\n'
                '- "toggle" to open door when facing it (requires key)'
            )
            extra_info = f"Has key: {has_key}. Door open: {door_open}."
        else:
            actions_desc = '"forward" (move 1 cell), "left"/"right" (rotate only)'
            available = "0,1,2"
            rules = (
                'RULES:\n'
                '- Avoid walls and lava\n'
                '- "forward" moves in facing direction\n'
                '- "left"/"right" only rotate'
            )
            extra_info = "No obstacles."

        prompt = (
            f"You are a path planner. Environment: {env_name}. "
            f"Agent Position: {agent_pos}. Agent Direction: {agent_dir_str}. "
            f"Goal Position: {goal_pos}. Extra Info: {extra_info} "
            f"Available Actions: {available}.\n\n"
            f"ACTIONS: {actions_desc}\n\n{rules}"
        )

        if helper_suggestion:
            prompt += f"\n\nHelper suggestion: {helper_suggestion}"

        prompt += "\n\nRESPOND WITH ONLY THE ACTION LIST:"
        return prompt

    # Parses a comma-separated action string into a list of action IDs
    def _parse_actions(self, response: str) -> list:
        response = (
            response.strip().lower()
            .replace('[', '').replace(']', '')
            .replace('"', '').replace("'", '')
        )
        parts = [p.strip() for p in response.split(',') if p.strip()]
        actions = []
        for part in parts:
            for name, aid in _ACTION_NAME_TO_ID.items():
                if name in part:
                    actions.append(aid)
                    break
        return actions

    # Generates a list of action IDs based on the prompt describing the current state and optionally the Helper's suggestion
    def _generate(self, prompt: str) -> list:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self._parse_actions(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Simulates the proposed action sequence against the current grid state to identify invalid actions and generate feedback
    def _validate_actions(self, action_ids: list, agent_pos, agent_dir: int, grid) -> tuple:
        pos = list(agent_pos)
        direction = agent_dir
        valid = 0
        errors = []

        for i, aid in enumerate(action_ids):
            if aid == 0:  # left
                direction = (direction - 1) % 4
                valid += 1
            elif aid == 1:  # right
                direction = (direction + 1) % 4
                valid += 1
            elif aid == 2:  # forward
                dx, dy = _DIR_VECTORS[direction]
                nx, ny = pos[0] + dx, pos[1] + dy
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    errors.append(f"out_of_bounds@{i}")
                    continue
                cell = grid.get(nx, ny)
                ct = getattr(cell, 'type', None) if cell else None
                if ct == 'wall':
                    errors.append(f"wall@{i}")
                    continue
                elif ct == 'lava':
                    errors.append(f"lava@{i}")
                    continue
                elif ct == 'door' and not getattr(cell, 'is_open', False):
                    errors.append(f"closed_door@{i}")
                    continue
                pos = [nx, ny]
                valid += 1
            elif aid == 3:  # pickup
                dx, dy = _DIR_VECTORS[direction]
                fx, fy = pos[0] + dx, pos[1] + dy
                fc = grid.get(fx, fy) if 0 <= fx < grid.width and 0 <= fy < grid.height else None
                if getattr(fc, 'type', None) == 'key':
                    valid += 1
                else:
                    errors.append(f"pickup_no_key@{i}")
            elif aid == 5:  # toggle
                dx, dy = _DIR_VECTORS[direction]
                fx, fy = pos[0] + dx, pos[1] + dy
                fc = grid.get(fx, fy) if 0 <= fx < grid.width and 0 <= fy < grid.height else None
                if getattr(fc, 'type', None) == 'door':
                    valid += 1
                else:
                    errors.append(f"toggle_no_door@{i}")
            else:
                errors.append(f"unknown@{i}")

        return valid, len(action_ids), errors

    # Generate feedback on the Helper's proposed actions
    def generate_feedback(self, env_wrapper, helper_actions: list) -> str:
        try:
            state = self._extract_env_state(env_wrapper)
            helper_ids = [
                _ACTION_NAME_TO_ID[a.lower()]
                for a in helper_actions
                if a.lower() in _ACTION_NAME_TO_ID
            ]
            _, _, errors = self._validate_actions(
                helper_ids, state['agent_pos'], state['agent_dir'], state['grid']
            )

            if not errors:
                return ""

            lines = ["Reviewer feedback:"]
            for err in errors:
                parts = err.split("@")
                reason = parts[0].replace('_', ' ')
                step = int(parts[1]) if len(parts) > 1 else 0
                action_name = _ACTION_ID_TO_NAME.get(helper_ids[step], "?") if step < len(helper_ids) else "?"
                lines.append(f"  - Step {step}: '{action_name}' is invalid ({reason})")

            # Append the Reviewer's corrected sequence as a hint for the Helper
            helper_action_str = ", ".join(helper_actions)
            prompt = self._build_prompt(state, helper_suggestion=helper_action_str)
            corrected_ids = self._generate(prompt)
            corrected_names = [_ACTION_ID_TO_NAME.get(a, str(a)) for a in corrected_ids]
            lines.append(f"  Suggested correction: {corrected_names}")

            return "\n".join(lines)

        except Exception as e:
            if self.verbose:
                print(f"Reviewer feedback error: {e}")
            return ""

    # Optional method to generate a standalone action suggestion when no Helper output is available
    def suggest_actions(self, env_wrapper, max_actions: int = 5) -> list:
        try:
            state = self._extract_env_state(env_wrapper)
            prompt = self._build_prompt(state)
            return self._generate(prompt)[:max_actions]
        except Exception as e:
            if self.verbose:
                print(f"Reviewer suggest error: {e}")
            return []


# LLM Helper: revised action call
def _ask_llm_revised(env_wrapper, helper_actions: list, feedback: str,
                     model_name: str, task_type: str, verbose: bool = False) -> list:
    if not feedback:
        return helper_actions

    try:
        base_env = env_wrapper.env.unwrapped
        agent_pos = tuple(int(x) for x in base_env.agent_pos)
        agent_dir = int(base_env.agent_dir)
        goal_pos = getattr(base_env, 'goal_pos', (base_env.width - 2, base_env.height - 2))
        env_name = getattr(env_wrapper, 'env_id', 'MiniGrid-Empty-5x5-v0')
        is_doorkey = task_type.lower() == 'doorkey'

        valid_list = list(_ACTION_NAME_TO_ID.keys()) if is_doorkey else ['forward', 'left', 'right']
        dir_str = _DIR_NAMES[agent_dir]

        prompt = (
            f"Environment: {env_name}. Agent at {agent_pos} facing {dir_str}. Goal at {goal_pos}.\n\n"
            f"Your previous suggestion had the following issues:\n{feedback}\n\n"
            f"Please provide a corrected action sequence using only: {valid_list}.\n"
            f"Output ONLY a JSON object: {{\"actions\": [\"action1\", \"action2\", ...]}}"
        )

        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={"temperature": 0.0, "top_p": 1.0, "num_predict": 30},
            format="json",
        )
        raw = re.sub(r'<thinking>.*?</thinking>', '', response.response, flags=re.DOTALL).strip()

        m = re.search(r'\{\s*["\']?actions["\']?\s*:\s*\[[^\]]*\]\s*\}', raw, re.DOTALL | re.IGNORECASE)
        if m:
            seq = json.loads(m.group(0)).get("actions", [])
        else:
            seq = re.findall(r'["\'](' + '|'.join(valid_list) + r')["\']', raw, re.IGNORECASE)

        seq = [s.lower().strip() for s in seq if s.lower().strip() in valid_list]
        return seq if seq else helper_actions

    except Exception as e:
        if verbose:
            print(f"Revised LLM call error: {e}")
        return helper_actions


# Call strategies for deciding when to invoke the Helper+Reviewer pipeline
class CallStrategy:
    def should_call(self, episode: int, step: int) -> bool:
        raise NotImplementedError

    # Optional method for strategies that need to update internal state each step, e.g. for decay
    def update(self):
        """Called once per environment step, regardless of whether the helper was used."""
        pass

    # Optional method for strategies that want to reset internal state at the start of each episode
    def reset_episode(self):
        """Called at the start of each episode. Does NOT reset decay state."""
        pass

# RandomStrategy implements a fixed 50% chance to call the helper on each step, but only for the first helper_episodes episodes
class RandomStrategy(CallStrategy):
    def __init__(self, helper_episodes: int = DEFAULT_HELPER_EPISODES):
        self.helper_episodes = helper_episodes

    def should_call(self, episode: int, step: int) -> bool:
        return episode < self.helper_episodes and bool(np.random.choice([True, False]))

# InitialStrategy calls the helper on every step but only for the first initial_steps of each episode, which is a more local decay
class InitialStrategy(CallStrategy):
    def __init__(self, initial_steps: int = DEFAULT_INITIAL_STEPS,
                 helper_episodes: int = DEFAULT_HELPER_EPISODES):
        self.initial_steps = initial_steps
        self.helper_episodes = helper_episodes

    def should_call(self, episode: int, step: int) -> bool:
        return step < self.initial_steps and episode < self.helper_episodes

# The FinalStrategy implements a decaying threshold for calling the helper, which is a more gradual and global decay
class FinalStrategy(CallStrategy):
    def __init__(self, threshold_initial: float = DEFAULT_THRESHOLD_INITIAL,
                 threshold_decay: float = DEFAULT_THRESHOLD_DECAY,
                 helper_episodes: int = DEFAULT_HELPER_EPISODES):
        self.threshold_initial = threshold_initial
        self.threshold_decay = threshold_decay
        self.helper_episodes = helper_episodes
        self.threshold = threshold_initial

    def should_call(self, episode: int, step: int) -> bool:
        return episode < self.helper_episodes and float(np.random.rand()) > self.threshold

    def update(self):
        self.threshold = max(0.0, self.threshold - self.threshold_decay)

    # reset_episode intentionally left as a no-op so the threshold accumulates
    # across episode boundaries rather than resetting each episode

# Factory function to create strategy instances based on name
def get_strategy(name: str, **kwargs) -> CallStrategy:
    strategies = {'random': RandomStrategy, 'initial': InitialStrategy, 'final': FinalStrategy}
    if name not in strategies:
        raise ValueError(f"Unknown strategy '{name}'. Valid options: {list(strategies.keys())}")
    return strategies[name](**kwargs)


# Utilities
def flatten_obs(obs):
    if isinstance(obs, dict) and 'image' in obs:
        return obs['image'].flatten().astype(np.float32)
    if isinstance(obs, dict):
        arrays = []
        for v in obs.values():
            if isinstance(v, np.ndarray):
                arrays.append(v.flatten())
            elif isinstance(v, (int, float)):
                arrays.append(np.array([v]))
        return np.concatenate(arrays).astype(np.float32) if arrays else np.array([], dtype=np.float32)
    return np.array(obs, dtype=np.float32).ravel()

# Returns the most recently created valid Reviewer model folder, or None
def find_latest_reviewer_model() -> Path:
    if not REVIEWER_MODEL_DIR.exists():
        return None
    skip_prefixes = ('ppo_', 'continued_', 'reward_', '.')
    folders = [
        f for f in REVIEWER_MODEL_DIR.iterdir()
        if f.is_dir() and not any(f.name.startswith(p) for p in skip_prefixes)
    ]
    if not folders:
        return None
    for folder in sorted(folders, reverse=True):
        if (folder / "config.json").exists():
            return folder
    return None


# Configure DQN hyperparameters for the given environment — mirrors helper.py exactly
def configure_agent_for_map(agent, env_id: str, prev_env_id: str = None) -> float:
    """Set epsilon, decay, lr, gamma and batch_size for the target environment.

    Parameters mirror those used in Helper/helper.py ``configure_agent_for_map``.
    Returns the base learning-rate that was applied so callers can use it if needed.
    """
    def _task_type(eid):
        if eid is None:
            return None
        eid_l = eid.lower()
        if 'crossing' in eid_l:
            return 'Crossing'
        elif 'doorkey' in eid_l:
            return 'DoorKey'
        return 'Empty'

    task_type = _task_type(env_id)
    prev_type = _task_type(prev_env_id)

    print(f"\n>>> Configuring DQN for: {env_id}")

    if task_type == 'Crossing':
        if prev_type == 'Empty':
            agent.epsilon = 0.55
            print(">>> KNOWLEDGE TRANSFER from Empty: epsilon=0.55 (navigation transferred)")
        elif prev_type == 'Crossing':
            agent.epsilon = 0.45
            print(">>> KNOWLEDGE TRANSFER from Crossing: epsilon=0.45 (lava skill transferred)")
        else:
            agent.epsilon = 1.0
        agent.epsilon_decay = 0.999
        agent.epsilon_min = 0.15
        base_lr = 0.00025
        agent.gamma = 0.99
        print(f">>> Crossing config: epsilon={agent.epsilon:.2f}, decay=0.999, gamma=0.99, lr={base_lr}")

    elif task_type == 'DoorKey':
        if '16x16' in env_id:
            agent.epsilon = 1.0 if prev_type != 'DoorKey' else 0.85
            agent.epsilon_decay = 0.998
            agent.epsilon_min = 0.15
            base_lr = 0.0002
            agent.gamma = 0.99
            agent.batch_size = 32
        elif '8x8' in env_id:
            agent.epsilon = 1.0 if prev_type != 'DoorKey' else 0.80
            agent.epsilon_decay = 0.998
            agent.epsilon_min = 0.15
            base_lr = 0.0002
            agent.gamma = 0.99
            agent.batch_size = 32
        else:
            agent.epsilon = 1.0 if prev_type != 'DoorKey' else 0.75
            agent.epsilon_decay = 0.997
            agent.epsilon_min = 0.20
            base_lr = 0.00025
            agent.gamma = 0.98
            agent.batch_size = 32

    else:  # Empty
        if '16x16' in env_id:
            agent.epsilon = 0.8
            agent.epsilon_decay = 0.992
            agent.epsilon_min = 0.05
            base_lr = 0.0002
            agent.gamma = 0.999
        elif '8x8' in env_id:
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

    print(f">>> epsilon={agent.epsilon:.2f}, decay={agent.epsilon_decay}, lr={base_lr}")
    return base_lr


# Reviewer training — fine-tuning then PPO
def train_reviewer(skip_if_model_exists: bool = True) -> Path:
    existing = find_latest_reviewer_model()
    if skip_if_model_exists and existing is not None:
        print(f"Reviewer model found at {existing}. Skipping re-training.")
        return existing

    print("\n=== Step 1/2: Reviewer fine-tuning (supervised) ===")
    from ReviewerPPO.fine_tuning import main as fine_tune_main
    fine_tune_main()

    ft_model_path = find_latest_reviewer_model()
    print(f"Fine-tuning complete. Model: {ft_model_path}")

    print("\n=== Step 2/2: Reviewer PPO alignment training ===")
    from ReviewerPPO.ppo_training import main as ppo_main
    ppo_main(finetuned_model_path=ft_model_path)

    ppo_model_path = find_latest_reviewer_model()
    print(f"PPO training complete. Model: {ppo_model_path}")
    return ppo_model_path


# Training loop: curriculum schedule
# Each entry in map_schedule runs for its prescribed number of episodes
# A single DQN agent is shared across all environments (knowledge transfer)
def train_heron(
    map_schedule: list = None,
    strategy_name: str = "final",
    reviewer_path: Path = None,
    dqn_model_path: Path = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    helper_episodes: int = DEFAULT_HELPER_EPISODES,
    initial_steps: int = DEFAULT_INITIAL_STEPS,
    threshold_initial: float = DEFAULT_THRESHOLD_INITIAL,
    threshold_decay: float = DEFAULT_THRESHOLD_DECAY,
    model_name: str = DEFAULT_MODEL_NAME,
    render: bool = False,
    verbose: bool = False,
):
    if map_schedule is None:
        map_schedule = MAP_SCHEDULE

    total_episodes = sum(entry["episodes"] for entry in map_schedule)
    print(f"HeRoN | curriculum={len(map_schedule)} envs | total_eps={total_episodes} | "
          f"strategy={strategy_name} | device={DEVICE}")

    layout_manager = None
    LayoutManagerClass = None
    if render:
        try:
            from UI.visual_game import LayoutManager as _LayoutManager
            LayoutManagerClass = _LayoutManager
        except Exception as e:
            print(f"Render UI disabled: could not load LayoutManager ({e})")
            render = False

    plots_dir = PLOTS_DIR / DATESTAMP
    results_dir = RESULTS_DIR / DATESTAMP
    model_dir = results_dir / "checkpoints"
    helper_log_dir = ROOT_DIR / "Helper" / "Logs" / DATESTAMP
    plots_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    helper_log_dir.mkdir(parents=True, exist_ok=True)

    if reviewer_path is None:
        reviewer_path = find_latest_reviewer_model()

    reviewer = T5Reviewer(reviewer_path, verbose=verbose) if reviewer_path else None
    if reviewer is None:
        print("No Reviewer model found. Running DQN-only mode.")

    # Build agent once using the first env's state/action sizes
    first_env_id = map_schedule[0]["id"]
    first_task = detect_task_type(first_env_id)
    init_env = DynamicMiniGridWrapper(make_env(first_env_id, render=False), first_task)
    init_obs = init_env.reset()
    state_size = len(flatten_obs(init_obs[0] if isinstance(init_obs, tuple) else init_obs))
    action_size = init_env.action_size

    agent = DQNAgent(state_size, action_size, dueling=False, prioritized=False, batch_size=64)
    agent.batch_size = batch_size

    if dqn_model_path is not None and dqn_model_path.exists():
        print(f"Loading DQN from {dqn_model_path}...")
        agent.load(str(dqn_model_path))
    else:
        print("Training DQN from scratch.")

    # Strategy (episode counter is global across all envs)
    strategy_kwargs = {'helper_episodes': helper_episodes}
    if strategy_name == 'initial':
        strategy_kwargs['initial_steps'] = initial_steps
    elif strategy_name == 'final':
        strategy_kwargs['threshold_initial'] = threshold_initial
        strategy_kwargs['threshold_decay'] = threshold_decay
    strategy = get_strategy(strategy_name, **strategy_kwargs)

    # Keep a single helper logger for the full curriculum so helper_calls.csv
    # contains calls from all environments in this run.
    llm_helper = LLMHelper(model_name=model_name, verbose=verbose, env_type="empty")
    llm_helper.init_logging(base_name="heron_curriculum", log_dir=helper_log_dir)

    # Global accumulators
    rewards_per_episode = []
    moves_per_episode = []
    success_rate = []
    helper_calls_per_episode = []
    hallucinations_per_episode = []
    env_labels_per_episode = []   # env short-name for each episode (for plots)
    episode_task_types = []       # Empty/Crossing/DoorKey per episode
    episode_successes = []        # bool success per episode
    episode_timeouts = []         # bool timeout per episode
    episode_deaths = []           # bool death per episode
    total_successes = 0
    total_hallucinations = 0
    global_ep = 0                 # absolute episode counter across all envs
    helper_successes = 0          # successes in helper phase only
    mixed_successes = 0           # successes in post-helper mixed phase only

    # Boundary markers: (global_ep_start, env_id)
    env_boundaries = []

    print("\nStarting curriculum training...\n")

    prev_env_id = None

    if render and LayoutManagerClass is not None and map_schedule:
        layout_manager = LayoutManagerClass(map_schedule[0]["id"])

    for sched_idx, entry in enumerate(map_schedule):
        env_id = entry["id"]
        n_episodes = entry["episodes"]
        task_type = detect_task_type(env_id)
        env_short = env_id.replace("MiniGrid-", "").replace("-v0", "")

        print(f"\n{'='*60}")
        print(f"[{sched_idx+1}/{len(map_schedule)}] {env_id}  ({n_episodes} episodes)")
        print(f"{'='*60}")

        env_boundaries.append((global_ep, env_short))

        # Buffer management on env transition (mirrors train_all_maps.py)
        if prev_env_id is not None:
            prev_task = detect_task_type(prev_env_id)
            if task_type == 'crossing' and prev_task == 'empty':
                KEEP = 3000
                if len(agent.replay_buffer) > KEEP:
                    for _ in range(len(agent.replay_buffer) - KEEP):
                        agent.replay_buffer.popleft()
                    print(f">>> Crossing soft-reset: kept {KEEP} samples from buffer")
            elif task_type == 'doorkey' and prev_task != 'doorkey':
                KEEP = 1000
                if len(agent.replay_buffer) > KEEP:
                    for _ in range(len(agent.replay_buffer) - KEEP):
                        agent.replay_buffer.popleft()
                    print(f">>> DoorKey phase start: buffer trimmed to {KEEP} samples")

        # Per env DQN hyperparameters + scheduler
        configure_agent_for_map(agent, env_id, prev_env_id)
        scheduler = (
            ReduceLROnPlateau(agent.optimizer, mode='max', factor=0.5, patience=40)
            if task_type.lower() != 'crossing' else None
        )

        # Environment + helper
        env_gym = make_env(env_id, render=False)
        env = DynamicMiniGridWrapper(env_gym, task_type)
        env.env_id = env_id

        if render and layout_manager is not None:
            layout_manager.set_env(env_id)

        llm_helper.env_type = task_type
        action_queue = deque()

        env_rewards = []   # per-env reward list for scheduler
        env_successes = 0
        timeout_streak = 0

        # Episode loop for this environment
        for local_ep in range(n_episodes):
            obs = env.reset()
            state = flatten_obs(obs[0] if isinstance(obs, tuple) else obs)
            done = False
            total_reward = 0.0
            moves = 0
            helper_calls = 0
            episode_hallucinations = 0
            step_counter = 0
            max_moves = 500
            last_helper_actions = []
            last_reviewer_feedback = ""
            last_revised_actions = []

            strategy.reset_episode()
            action_queue.clear()
            llm_helper.reset_episode(global_ep, env_id)

            while not done and moves < max_moves:
                valid_actions = env.get_valid_actions()
                valid_mask = [1 if a in valid_actions else 0 for a in range(env.action_size)]
                action = None

                # HeRoN 3-step pipeline: Helper -> Reviewer feedback -> Helper revised
                if reviewer is not None and strategy.should_call(global_ep, moves):
                    if not action_queue:
                        try:
                            base_env = env.env.unwrapped
                            agent_pos = tuple(int(x) for x in base_env.agent_pos)
                            agent_dir = int(base_env.agent_dir)
                            goal_pos = getattr(base_env, 'goal_pos',
                                               (base_env.width - 2, base_env.height - 2))
                            grid = base_env.grid

                            if task_type.lower() == 'doorkey':
                                helper_actions, hallucination_flag = llm_helper.suggest_actions_doorkey(
                                    grid, agent_pos, agent_dir, goal_pos, base_env, max_actions=5
                                )
                            else:
                                helper_actions, hallucination_flag = llm_helper.suggest_actions(
                                    grid, agent_pos, agent_dir, goal_pos, max_actions=5
                                )

                            last_helper_actions = list(helper_actions)

                            if hallucination_flag:
                                episode_hallucinations += 1

                            feedback = reviewer.generate_feedback(env, helper_actions) if helper_actions else ""
                            last_reviewer_feedback = feedback
                            revised_actions = _ask_llm_revised(
                                env, helper_actions, feedback, model_name, task_type, verbose
                            )
                            last_revised_actions = list(revised_actions)

                            suggested_ids = [
                                _ACTION_NAME_TO_ID[a.lower()]
                                for a in revised_actions
                                if a.lower() in _ACTION_NAME_TO_ID
                            ]

                            if not suggested_ids:
                                suggested_ids = reviewer.suggest_actions(env)
                                episode_hallucinations += 1

                            action_queue.extend(suggested_ids)
                            helper_calls += 1

                        except Exception as e:
                            if verbose:
                                print(f"HeRoN pipeline error: {e}")
                            action_queue.extend(reviewer.suggest_actions(env))

                    while action_queue:
                        candidate = action_queue.popleft()
                        if candidate in valid_actions:
                            action = candidate
                            break

                if action is None:
                    action = agent.act(state, valid_actions)

                next_obs, reward, done, info = env.step(action)
                if (task_type.lower() == 'crossing' and done
                        and not info.get('goal_reached', False)
                        and not info.get('timeout', False)):
                    reward = -50.0

                next_state = flatten_obs(next_obs)
                next_valid_actions = env.get_valid_actions()
                next_valid_mask = [1 if a in next_valid_actions else 0
                                   for a in range(env.action_size)]

                agent.remember(state, action, reward, next_state, done, valid_mask, next_valid_mask)
                state = next_state
                total_reward += reward
                moves += 1
                step_counter += 1

                strategy.update()

                if render and layout_manager is not None:
                    frame = None
                    try:
                        base_env = env.env.unwrapped
                        if hasattr(base_env, 'get_frame'):
                            frame = base_env.get_frame(highlight=True, tile_size=32)
                    except Exception:
                        frame = None

                    status_lines = [
                        f"Env: {env_id}",
                        f"Task: {task_type.capitalize()}",
                        f"Episode: {local_ep + 1}/{n_episodes}",
                        f"Step: {moves}/{max_moves}",
                        f"Reward: {total_reward:.2f}",
                        f"Epsilon: {agent.epsilon:.3f}",
                        f"Helper calls: {helper_calls}",
                    ]

                    if last_helper_actions:
                        status_lines.append(f"Helper: {last_helper_actions}")
                    if last_reviewer_feedback:
                        status_lines.append(
                            f"Reviewer: {last_reviewer_feedback.replace(chr(10), ' | ')[:220]}"
                        )
                    if last_revised_actions:
                        status_lines.append(f"Revised: {last_revised_actions}")

                    layout_manager.update_text(status_lines, episode_num=global_ep + 1)
                    if not layout_manager.render(frame):
                        render = False

                train_freq = 8 if task_type.lower() == 'doorkey' else 4
                if step_counter % train_freq == 0 and len(agent.replay_buffer) > batch_size:
                    agent.replay()

                if done and (info.get('goal_reached', False) or reward > 0):
                    total_successes += 1
                    env_successes += 1
                    if global_ep < helper_episodes:
                        helper_successes += 1
                    else:
                        mixed_successes += 1

            # Post-episode bookkeeping
            rewards_per_episode.append(total_reward)
            moves_per_episode.append(moves)
            env_sr = env_successes / (local_ep + 1)
            global_sr = total_successes / (global_ep + 1)
            success_rate.append(global_sr)
            helper_calls_per_episode.append(helper_calls)
            hallucinations_per_episode.append(episode_hallucinations)
            env_labels_per_episode.append(env_short)
            episode_task_types.append(task_type.capitalize())
            episode_successes.append(bool(info.get('goal_reached', False) or total_reward > 0))
            episode_timeouts.append(bool(info.get('timeout', False)))
            episode_deaths.append(bool(info.get('lava_death', False)))
            total_hallucinations += episode_hallucinations
            env_rewards.append(total_reward)

            # 1. Timeout-streak epsilon boost
            if info.get('timeout', False):
                timeout_streak += 1
            else:
                timeout_streak = 0
            if timeout_streak >= 15:
                agent.epsilon = min(agent.epsilon + 0.25, 1.0)
                timeout_streak = 0
                print(f"    !!! TIMEOUT STREAK: Epsilon boosted to {agent.epsilon:.2f}")

            # 2. Extra replay pass every 5 episodes (not for DoorKey)
            if (local_ep + 1) % 5 == 0 and task_type.lower() != 'doorkey':
                agent.replay()

            # 3. DoorKey buffer cleanup every 100 episodes
            if task_type.lower() == 'doorkey' and (local_ep + 1) % 100 == 0:
                if len(agent.replay_buffer) > 5000:
                    remove_count = len(agent.replay_buffer) - 5000
                    for _ in range(remove_count):
                        agent.replay_buffer.popleft()
                    print(f"    >>> Buffer cleanup: removed {remove_count} old samples")

            # 4. LR scheduler step
            if scheduler is not None:
                scheduler.step(total_reward)

            print(
                f"[{env_short}] Ep {local_ep+1}/{n_episodes} (global {global_ep+1}) | "
                f"R: {total_reward:.2f} | Steps: {moves} | "
                f"SR: {env_sr*100:.1f}% | Calls: {helper_calls} | "
                f"e: {agent.epsilon:.3f}"
            )

            global_ep += 1

        # Save per-env checkpoint
        env_name_clean = env_id.replace("MiniGrid-", "").replace("-v0", "")
        ckpt_path = model_dir / f"model_{env_name_clean}.pth"
        agent.save(str(ckpt_path))
        print(f">>> Checkpoint saved: {ckpt_path}")

        prev_env_id = env_id

    # Save final master model
    master_path = model_dir / "master_model_heron.pth"
    agent.save(str(master_path))
    print(f"\n>>> Master model saved: {master_path}")

    # Summary
    helper_episode_count = min(helper_episodes, total_episodes)
    mixed_episode_count = max(0, total_episodes - helper_episode_count)
    helper_success_rate_avg = (helper_successes / helper_episode_count) if helper_episode_count > 0 else 0.0
    mixed_success_rate_avg = (mixed_successes / mixed_episode_count) if mixed_episode_count > 0 else 0.0
    total_success_rate_avg = (total_successes / total_episodes) if total_episodes > 0 else 0.0

    print(f"\nCurriculum training complete.")
    print(f"Avg Reward: {np.mean(rewards_per_episode):.2f}")
    print(f"Avg Moves:  {np.mean(moves_per_episode):.2f}")
    print(f"Helper Avg SR ({helper_episode_count} eps): {helper_success_rate_avg*100:.1f}%")
    print(f"Mixed Avg SR ({mixed_episode_count} eps):  {mixed_success_rate_avg*100:.1f}%")
    print(f"Total Avg SR ({total_episodes} eps):       {total_success_rate_avg*100:.1f}%")
    print(f"Final Cumulative SR: {success_rate[-1]*100:.1f}%")
    print(f"Total Helper Calls: {sum(helper_calls_per_episode)}")
    print(f"Total Hallucinations: {total_hallucinations}")

    metrics = {
        'strategy': strategy_name,
        'map_schedule': [e["id"] for e in map_schedule],
        'total_episodes': total_episodes,
        'helper_episodes': helper_episodes,
        'avg_reward': float(np.mean(rewards_per_episode)),
        'avg_moves': float(np.mean(moves_per_episode)),
        'final_success_rate': float(success_rate[-1]),
        'helper_success_rate_avg': float(helper_success_rate_avg),
        'mixed_success_rate_avg': float(mixed_success_rate_avg),
        'total_success_rate_avg': float(total_success_rate_avg),
        'helper_successes': int(helper_successes),
        'mixed_successes': int(mixed_successes),
        'total_successes': int(total_successes),
        'total_helper_calls': int(sum(helper_calls_per_episode)),
        'total_hallucinations': int(total_hallucinations),
        'rewards': rewards_per_episode,
        'moves': moves_per_episode,
        'success_rate': success_rate,
        'episode_task_types': episode_task_types,
        'episode_successes': episode_successes,
        'episode_timeouts': episode_timeouts,
        'episode_deaths': episode_deaths,
    }
    with open(results_dir / f"metrics_{strategy_name}_curriculum.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    plot_results(
        rewards_per_episode, moves_per_episode, success_rate,
        helper_calls_per_episode, strategy_name,
        "Curriculum", plots_dir,
        env_boundaries=env_boundaries,
        task_types=episode_task_types,
        episode_successes=episode_successes,
        episode_timeouts=episode_timeouts,
        episode_deaths=episode_deaths,
    )
    llm_helper.close_logging()
    if layout_manager is not None:
        layout_manager.close()
    print(f"Results saved to {results_dir} | Plots saved to {plots_dir}")
    return metrics

# Visualization
def plot_results(rewards, moves, success_rate, helper_calls, strategy_name, env_id,
                 output_dir, env_boundaries=None, task_types=None,
                 episode_successes=None, episode_timeouts=None, episode_deaths=None):
    """
    env_boundaries: list of (x_position, label) tuples marking environment transitions.
    """
    env_name_safe = env_id.replace('-', '_').replace(':', '_')
    n = len(rewards)
    if n == 0:
        return

    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # Fallbacks keep plotting robust when optional data is not passed
    if task_types is None:
        task_types = ["Unknown"] * n
    if episode_successes is None:
        episode_successes = [False] * n
    if episode_timeouts is None:
        episode_timeouts = [False] * n
    if episode_deaths is None:
        episode_deaths = [False] * n

    # 1. Success Rate per Task
    fig, ax = plt.subplots()
    task_success = {}
    for task, ok in zip(task_types, episode_successes):
        if task not in task_success:
            task_success[task] = {'success': 0, 'total': 0}
        task_success[task]['total'] += 1
        if ok:
            task_success[task]['success'] += 1

    tasks = list(task_success.keys())
    success_rates = [
        (task_success[t]['success'] / task_success[t]['total']) * 100 if task_success[t]['total'] > 0 else 0.0
        for t in tasks
    ]
    bars = ax.bar(tasks, success_rates, color=['#2ecc71', '#3498db', '#e74c3c'][:len(tasks)])
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_title('HeRoN: Success Rate per Task', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h, f'{h:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f'success_rate_per_task_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()

    # 2. Steps per Episode trend
    fig, ax = plt.subplots()
    episodes = list(range(1, n + 1))
    ax.plot(episodes, moves, alpha=0.6, linewidth=0.8, color='#3498db')
    window = min(10, len(moves) // 5)
    if window > 1:
        ma = np.convolve(moves, np.ones(window) / window, mode='valid')
        ax.plot(episodes[window - 1:], ma, linewidth=2, color='#e74c3c', label=f'Moving Avg (window={window})')
        ax.legend()
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps to Completion', fontsize=12)
    ax.set_title('HeRoN: Steps per Episode', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'steps_per_episode_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()

    # 3. Reward Distribution per Task
    fig, ax = plt.subplots()
    task_rewards = {}
    for task, r in zip(task_types, rewards):
        task_rewards.setdefault(task, []).append(r)
    tasks = list(task_rewards.keys())
    rewards_data = [task_rewards[t] for t in tasks]
    bp = ax.boxplot(rewards_data, labels=tasks, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#3498db', '#e74c3c'][:len(tasks)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_title('HeRoN: Reward Distribution per Task', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'reward_distribution_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()

    # 4. Helper Usage + Success Rate Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.hist(helper_calls, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(helper_calls), color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(helper_calls):.2f}')
    ax1.set_xlabel('Helper Calls per Episode', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Helper Usage Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    sr_percent = [x * 100 for x in success_rate]
    ax2.hist(sr_percent, bins=20, color='#16a085', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(sr_percent), color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(sr_percent):.1f}%')
    ax2.set_xlabel('Cumulative Success Rate (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Success Rate Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'helper_success_usage_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()

    # 5. Episode outcomes (success/timeout/death/other-failure)
    fig, ax = plt.subplots()
    success_count = int(sum(1 for x in episode_successes if x))
    timeout_count = int(sum(1 for x in episode_timeouts if x))
    death_count = int(sum(1 for x in episode_deaths if x))
    other_fail = max(0, n - success_count - timeout_count - death_count)

    categories = ['Success', 'Death', 'Timeout']
    counts = [success_count, death_count, other_fail]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=categories,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title('HeRoN: Episode Outcomes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'outcome_distribution_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()

    # 6. Avg Steps per Task
    fig, ax = plt.subplots()
    task_steps = {}
    for task, step_count in zip(task_types, moves):
        task_steps.setdefault(task, []).append(step_count)
    tasks = list(task_steps.keys())
    avg_steps = [np.mean(task_steps[t]) for t in tasks]
    std_steps = [np.std(task_steps[t]) for t in tasks]
    bars = ax.bar(tasks, avg_steps, yerr=std_steps, capsize=5,
                  color=['#2ecc71', '#3498db', '#e74c3c'][:len(tasks)], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Average Steps', fontsize=12)
    ax.set_xlabel('Task Type', fontsize=12)
    ax.set_title('HeRoN: Average Steps per Task', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, avg, std in zip(bars, avg_steps, std_steps):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h, f'{avg:.1f}+/-{std:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / f'avg_steps_per_task_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()

    # 7. Helper usage vs outcome
    fig, ax = plt.subplots()
    success_helper = [hc for hc, ok in zip(helper_calls, episode_successes) if ok]
    fail_helper = [hc for hc, ok in zip(helper_calls, episode_successes) if not ok]
    data_to_plot = [success_helper, fail_helper]
    labels = ['Success', 'Failure']
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Helper Calls', fontsize=12)
    ax.set_title('HeRoN: Helper Usage vs Episode Outcome', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / f'helper_vs_success_{strategy_name}_{env_name_safe}.png', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='HeRoN - DQN + T5 Reviewer Training')
    parser.add_argument('--env', type=str, default=None,
                        help='Single MiniGrid env ID (default: all environments)')
    parser.add_argument('--episodes', type=int, default=DEFAULT_EPISODES)
    parser.add_argument('--strategy', type=str, default='final',
                        choices=['random', 'initial', 'final'],
                        help='Helper call strategy')
    parser.add_argument('--reviewer', type=str, default=None,
                        help='Path to a pre-trained T5 Reviewer model')
    parser.add_argument('--train-reviewer', action='store_true',
                        help='Run fine-tuning + PPO training for the Reviewer before RL training')
    parser.add_argument('--force-retrain-reviewer', action='store_true',
                        help='Re-train the Reviewer even if a model already exists')
    parser.add_argument('--dqnagent', type=str, default=None,
                        help='Path to a pre-trained DQN model')
    parser.add_argument('--helper-episodes', type=int, default=DEFAULT_HELPER_EPISODES)
    parser.add_argument('--initial-steps', type=int, default=DEFAULT_INITIAL_STEPS)
    parser.add_argument('--threshold-initial', type=float, default=DEFAULT_THRESHOLD_INITIAL)
    parser.add_argument('--threshold-decay', type=float, default=DEFAULT_THRESHOLD_DECAY)
    parser.add_argument('--model-name', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Reviewer training: fine-tuning first, then PPO alignment
    if args.train_reviewer:
        reviewer_path = train_reviewer(skip_if_model_exists=not args.force_retrain_reviewer)
    elif args.reviewer:
        reviewer_path = Path(args.reviewer)
    else:
        reviewer_path = find_latest_reviewer_model()

    # DQN agent setup
    if args.dqnagent:
        dqn_model_path = Path(args.dqnagent)
    else:
        default_dqn = ROOT_DIR / "DQNAgent" / "Model" / "CurriculumAgent" / "master_model_universal.pth"
        dqn_model_path = default_dqn if default_dqn.exists() else None
        if dqn_model_path is None:
            print("No pre-trained DQN found. Running curriculum training first...")
            run_training_suite(run_mixed=False, mixed_episodes=0)
            model_base = ROOT_DIR / "DQNAgent" / "Model"
            folders = sorted(
                [f for f in model_base.iterdir()
                 if f.is_dir() and not any(f.name.startswith(p) for p in ('CurriculumAgent', '.', '__'))],
                reverse=True,
            )
            if folders:
                for candidate in [
                    folders[0] / "master_model_universal.pth",
                    folders[0] / "master_model_curriculum.pth",
                ]:
                    if candidate.exists():
                        dqn_model_path = candidate
                        print(f"Using DQN model: {dqn_model_path}")
                        break

    # Build map schedule
    if args.env:
        # Single-env run: honour --episodes; build a one-entry schedule
        map_schedule = [{"id": args.env, "episodes": args.episodes}]
    else:
        # Full run: helper curriculum + random post-helper phase
        map_schedule = build_training_schedule(args.episodes)

    metrics = train_heron(
        map_schedule=map_schedule,
        strategy_name=args.strategy,
        reviewer_path=reviewer_path,
        dqn_model_path=dqn_model_path,
        batch_size=args.batch_size,
        helper_episodes=args.helper_episodes,
        initial_steps=args.initial_steps,
        threshold_initial=args.threshold_initial,
        threshold_decay=args.threshold_decay,
        model_name=args.model_name,
        render=args.render,
        verbose=args.verbose,
    )

    print(f"\nFinal success rate : {metrics.get('final_success_rate', 0) * 100:.1f}%")
    print(f"Helper avg SR      : {metrics.get('helper_success_rate_avg', 0) * 100:.1f}%")
    print(f"Mixed avg SR       : {metrics.get('mixed_success_rate_avg', 0) * 100:.1f}%")
    print(f"Total avg SR       : {metrics.get('total_success_rate_avg', 0) * 100:.1f}%")
    print(f"Average reward     : {metrics.get('avg_reward', 0):.2f}")
    print(f"Average moves      : {metrics.get('avg_moves', 0):.1f}")

if __name__ == "__main__":
    main()