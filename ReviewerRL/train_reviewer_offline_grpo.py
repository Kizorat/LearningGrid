import sys
import os
import ast
import re
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from minigrid.core.world_object import Goal, Wall, Lava, Key, Door
from minigrid.wrappers import FullyObsWrapper
import matplotlib.pyplot as plt
import json
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict
import random

# Adding root directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)


from Helper.simple_helper import flatten_obs
from ReviewerRL.reward_model import SequenceRewardModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Paths
DATASET_DIR = os.path.join("ReviewerRL/Dataset_GRPO", current_time)
PLOTS_DIR = os.path.join(DATASET_DIR, "Plots")
MODEL_DIR = os.path.join(DATASET_DIR, "Model")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class cprint:
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    INFO = '\033[94m'
    HEADER = '\033[95m'
    WARNING = '\033[93m'
    CYAN = '\033[96m'

# Mapping actions based on DynamicMiniGridWrapper
ACTION_MAP = {
    "left": 0,     
    "right": 1,     
    "forward": 2,   
    "pickup": 3,    
    "toggle": 5     
}

REVERSE_ACTION_MAP = {v: k for k, v in ACTION_MAP.items()}

DIR_MAP = {"East": 0, "South": 1, "West": 2, "North": 3}
DIR_SYMBOLS = {0: '>', 1: 'v', 2: '<', 3: '^'}

# Maximum sequence length for GRPO
MAX_SEQUENCE_LENGTH = 5


class MinigridOfflinePreferenceDataset(Dataset):
    def __init__(self, csv_path, limit_rows=None, seed=42):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")
            
        print(f"Loading CSV from: {csv_path}...")
        df = pd.read_csv(csv_path)
        if limit_rows:
            df = df.head(limit_rows)
        self.data = df
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.env_cache = {}
        
        # Calculate maximum dimension
        print("Calculating maximum input dimension (based on 16x16)...")
        try:
            dummy_env = FullyObsWrapper(gym.make("MiniGrid-Empty-16x16-v0", render_mode=None))
            obs, _ = dummy_env.reset()
            obs_wrapped = dummy_env.observation(obs)
            self.max_flat_size = flatten_obs(obs_wrapped).shape[0]
            dummy_env.close()
        except Exception as e:
            print(f"Warning: failed to calculate dimension ({e}), using default 768.")
            self.max_flat_size = 768 
            
        print(f"Tensor size set to: {self.max_flat_size} (with Zero-Padding)")
        print(f"Dataset loaded: {len(self.data)} rows.")
        print(f"Loading preference pairs from 'instructions' and 'response' columns...")

    def __len__(self):
        return len(self.data)

    def _parse_tuple(self, val):
        try:
            return ast.literal_eval(val)
        except:
            return (0, 0)

    def _get_action_sequence(self, instruction_str):
        try:
            match = re.search(r'\[(.*?)\]', instruction_str)
            if match:
                actions_list = match.group(1).replace("'", "").replace('"', '').split(',')
                sequence = []
                for action_str in actions_list[:MAX_SEQUENCE_LENGTH]:
                    action_str = action_str.strip().lower()
                    if action_str in ACTION_MAP:
                        sequence.append(ACTION_MAP[action_str])
                return sequence if sequence else [2]
        except Exception:
            pass
        return [2]

    def _reconstruct_env(self, row, env):
        # Access the unwrapped environment
        base_env = env.unwrapped
        grid = base_env.grid
        
        # Clear the grid while keeping the outer walls
        for i in range(1, grid.width - 1):
            for j in range(1, grid.height - 1):
                grid.set(i, j, None)
        
        # Positions
        agent_pos = self._parse_tuple(row['Agent_Position'])
        goal_pos = self._parse_tuple(row['Goal_Position'])
        
        # Agent direction
        dir_str = row.get('Agent_Direction', 'East')
        agent_dir = DIR_MAP.get(dir_str, 0)
        
        # Safety Check
        if goal_pos[0] >= grid.width or goal_pos[1] >= grid.height:
            goal_pos = (grid.width - 2, grid.height - 2)

        base_env.agent_pos = agent_pos
        base_env.agent_dir = agent_dir
        grid.set(*goal_pos, Goal())

        # Parsing Obstacles from Extra_Info
        info = str(row.get('Extra_Info', ''))
        
        # Vertical lava
        if "Vertical river" in info:
            matches = re.findall(r"Vertical river at x=(\d+) has gap at y=(\d+)", info)
            for x_str, gap_y_str in matches:
                x, gap_y = int(x_str), int(gap_y_str)
                if x < grid.width - 1:
                    for y in range(1, grid.height - 1):
                        if y != gap_y: 
                            grid.set(x, y, Lava())
        
        # Horizontal lava
        if "Horizontal river" in info:
            matches = re.findall(r"Horizontal river at y=(\d+) has gap at x=(\d+)", info)
            for y_str, gap_x_str in matches:
                y, gap_x = int(y_str), int(gap_x_str)
                if y < grid.height - 1:
                    for x in range(1, grid.width - 1):
                        if x != gap_x: 
                            grid.set(x, y, Lava())

        # Key
        if "Key at" in info:
            key_match = re.search(r"Key at \((\d+), (\d+)\)", info)
            if key_match:
                kx, ky = int(key_match.group(1)), int(key_match.group(2))
                if kx < grid.width and ky < grid.height:
                    grid.set(kx, ky, Key('yellow'))

        # Door
        if "Door at" in info:
            door_match = re.search(r"Door at \((\d+), (\d+)\)", info)
            if door_match:
                dx, dy = int(door_match.group(1)), int(door_match.group(2))
                if dx < grid.width and dy < grid.height:
                    door = Door('yellow', is_locked=True)
                    grid.set(dx, dy, door)
        
        return env

    def _get_env(self, env_name):
        if env_name not in self.env_cache:
            try:
                self.env_cache[env_name] = FullyObsWrapper(
                    gym.make(env_name, render_mode=None)
                )
            except Exception as e:
                print(f"Warning: Failed to create {env_name}, using Empty-8x8 ({e})")
                self.env_cache[env_name] = FullyObsWrapper(
                    gym.make("MiniGrid-Empty-8x8-v0", render_mode=None)
                )
        return self.env_cache[env_name]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get environment and state
        env_name = row['Environment']
        env = self._get_env(env_name)
        env.reset()
        env = self._reconstruct_env(row, env)
        
        # Get observation from the base environment
        obs = env.unwrapped.gen_obs()
        flattened = flatten_obs(obs)
        
        # Pad or truncate to max_flat_size
        if flattened.shape[0] < self.max_flat_size:
            flattened = np.pad(flattened, (0, self.max_flat_size - flattened.shape[0]))
        elif flattened.shape[0] > self.max_flat_size:
            flattened = flattened[:self.max_flat_size]
        
        state_tensor = torch.tensor(flattened, dtype=torch.float32)
        
        # Get good sequence from 'instructions' column (correct path)
        instructions_str = row.get('instructions', '[]')
        good_seq = self._get_action_sequence(instructions_str)
        
        # Get bad sequence from 'response' column (LLM's potentially incorrect response)
        response_str = row.get('response', '[]')
        bad_seq = self._get_action_sequence(response_str)
        

        padding_value = 6
        good_seq_padded = good_seq + [padding_value] * (MAX_SEQUENCE_LENGTH - len(good_seq))
        bad_seq_padded = bad_seq + [padding_value] * (MAX_SEQUENCE_LENGTH - len(bad_seq))
        
        good_seq_tensor = torch.tensor(good_seq_padded, dtype=torch.long)
        bad_seq_tensor = torch.tensor(bad_seq_padded, dtype=torch.long)
        
        return state_tensor, good_seq_tensor, bad_seq_tensor


class GRPOOfflineTrainer:
    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
        self.model.to(device)
        self.action_size = 7  

    def _to_one_hot(self, action_indices, num_classes=None):
        if num_classes is None:
            num_classes = self.action_size
        # action_indices: (batch, seq_len) or (seq_len,)
        # Returns one-hot encoded tensor of shape (batch, seq_len, num_classes) or (seq_len, num_classes)
        return torch.nn.functional.one_hot(action_indices, num_classes=num_classes).float()

    def compute_bradley_terry_loss(self, state, good_seq, bad_seq):
        # Convert action indices to one-hot encoding
        good_seq_onehot = self._to_one_hot(good_seq) 
        bad_seq_onehot = self._to_one_hot(bad_seq)    
        
        # Get sequence scores
        good_output = self.model(state.unsqueeze(0), good_seq_onehot.unsqueeze(0))
        bad_output = self.model(state.unsqueeze(0), bad_seq_onehot.unsqueeze(0))
        
        # Use sequence_score as overall reward
        reward_good = good_output['sequence_score'].squeeze()
        reward_bad = bad_output['sequence_score'].squeeze()
        
        # Bradley-Terry: P(good > bad) = sigmoid(reward_good - reward_bad)
        logit_diff = reward_good - reward_bad
        loss = -torch.log(torch.sigmoid(logit_diff) + 1e-8)
        
        return loss

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, good_seqs, bad_seqs in dataloader:
            states = states.to(self.device)
            good_seqs = good_seqs.to(self.device)
            bad_seqs = bad_seqs.to(self.device)
            
            optimizer.zero_grad()
            
            # Compute loss for each sample in batch
            batch_losses = []
            for i in range(states.size(0)):
                loss = self.compute_bradley_terry_loss(
                    states[i], good_seqs[i], bad_seqs[i]
                )
                batch_losses.append(loss)
            
            # Average loss over batch
            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for states, good_seqs, bad_seqs in dataloader:
                states = states.to(self.device)
                good_seqs = good_seqs.to(self.device)
                bad_seqs = bad_seqs.to(self.device)
                
                for i in range(states.size(0)):
                    # Convert action indices to one-hot encoding
                    good_seq_onehot = self._to_one_hot(good_seqs[i])
                    bad_seq_onehot = self._to_one_hot(bad_seqs[i])
                    
                    good_output = self.model(
                        states[i].unsqueeze(0), 
                        good_seq_onehot.unsqueeze(0)
                    )
                    bad_output = self.model(
                        states[i].unsqueeze(0), 
                        bad_seq_onehot.unsqueeze(0)
                    )
                    
                    reward_good = good_output['sequence_score'].item()
                    reward_bad = bad_output['sequence_score'].item()
                    
                    # Check if model prefers good over bad
                    if reward_good > reward_bad:
                        correct += 1
                    total += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return accuracy


def save_loss_plot(losses, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses, color='#e74c3c', linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

def save_confusion_matrix_plot(model, dataloader, device, save_path):
    print(f"\n{cprint.INFO}Generating Confusion Matrix...{cprint.ENDC}")
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for states, good_seqs, bad_seqs in dataloader:
            states = states.to(device)
            good_seqs = good_seqs.to(device)
            bad_seqs = bad_seqs.to(device)
            
            for i in range(states.size(0)):
                # Prepare one-hot encoding
                good_onehot = torch.nn.functional.one_hot(good_seqs[i], num_classes=7).float().unsqueeze(0)
                bad_onehot = torch.nn.functional.one_hot(bad_seqs[i], num_classes=7).float().unsqueeze(0)
                state_in = states[i].unsqueeze(0)
                
                # Calculate Rewards
                r_good = model(state_in, good_onehot)['sequence_score'].item()
                r_bad = model(state_in, bad_onehot)['sequence_score'].item()
                
                # Case 1: Input (Good, Bad). Truth: Good > Bad (Label 1)
                y_true.append(1)
                # Predict 1 if Reward Good > Reward Bad, else 0
                y_pred.append(1 if r_good > r_bad else 0)
                
                # Case 2: Input (Bad, Good). Truth: Bad < Good (Label 0)
                y_true.append(0)
                # Predict 1 if Reward Bad > Reward Good (error), 0 if correct
                y_pred.append(1 if r_bad > r_good else 0)

    # Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bad > Good", "Good > Bad"])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap='Blues', ax=ax)
    plt.title("Preference Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"{cprint.OKGREEN}  Saved: {save_path}{cprint.ENDC}")





def save_accuracy_plot(accuracies, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(accuracies, color='#2ecc71', linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Preference Accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def create_analysis_csv(dataset, model, device, save_path):
    print(f"\n{cprint.INFO}Creating analysis CSV...{cprint.ENDC}")
    
    model.eval()
    records = []
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            if idx % 100 == 0:
                print(f"\rProcessing {idx}/{len(dataset)}", end="")
            
            state, good_seq, bad_seq = dataset[idx]
            state = state.to(device)
            good_seq = good_seq.to(device)
            bad_seq = bad_seq.to(device)
            
            # Convert action indices to one-hot
            good_seq_onehot = torch.nn.functional.one_hot(good_seq, num_classes=7).float()
            bad_seq_onehot = torch.nn.functional.one_hot(bad_seq, num_classes=7).float()
            
            # Get model outputs
            good_output = model(state.unsqueeze(0), good_seq_onehot.unsqueeze(0))
            bad_output = model(state.unsqueeze(0), bad_seq_onehot.unsqueeze(0))
            
            # Extract values
            reward_good = good_output['sequence_score'].item()
            reward_bad = bad_output['sequence_score'].item()
            coherence_good = good_output['coherence'].item()
            coherence_bad = bad_output['coherence'].item()
            
            # Convert sequences to action names
            good_actions = [REVERSE_ACTION_MAP.get(a.item(), 'unknown') 
                           for a in good_seq if a.item() != 6]
            bad_actions = [REVERSE_ACTION_MAP.get(a.item(), 'unknown') 
                          for a in bad_seq if a.item() != 6]
            
            # Get environment name from dataset
            env_name = dataset.data.iloc[idx]['Environment']
            
            # Create record in GRPO format
            record = {
                'episode': idx,
                'env_name': env_name,
                'chosen_sequence': str(good_actions),
                'rejected_sequence': str(bad_actions),
                'chosen_reward': reward_good,
                'rejected_reward': reward_bad,
                'reward_margin': reward_good - reward_bad,
                'chosen_coherence': coherence_good,
                'rejected_coherence': coherence_bad,
                'sequence_length': len(good_actions)
            }
            records.append(record)
    
    print(f"\rProcessing {len(dataset)}/{len(dataset)} - Done!")
    
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"{cprint.OKGREEN}Analysis CSV saved: {save_path}{cprint.ENDC}")
    
    return df


def evaluate_on_environments(model, device, episodes_per_env=5):
    print(f"{cprint.HEADER}EVALUATING ON TEST ENVIRONMENTS{cprint.ENDC}")
   
    
    env_configs = {
        "Empty": ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-8x8-v0", "MiniGrid-Empty-16x16-v0"],
        "Crossing": ["MiniGrid-LavaCrossingS9N1-v0", "MiniGrid-LavaCrossingS9N3-v0", "MiniGrid-LavaCrossingS11N5-v0"],
        "DoorKey": ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-8x8-v0", "MiniGrid-DoorKey-16x16-v0"]
    }
    
    # Import helper
    from Helper.simple_helper import LLMHelper, flatten_obs
    from DQNAgent.enviroment import DynamicMiniGridWrapper
    
    helper = LLMHelper(verbose=False, env_type="evaluation")
    
    all_results = []
    
    for task_type, env_ids in env_configs.items():
        print(f"\n{cprint.CYAN}----Testing {task_type} Environments----{cprint.ENDC}")
        
        for env_id in env_ids:
            print(f"\n{cprint.INFO}Environment: {env_id}{cprint.ENDC}")
            
            env = gym.make(env_id)
            env = FullyObsWrapper(env)
            wrapped_env = DynamicMiniGridWrapper(env, task_type)
            
            env_results = {
                'env_id': env_id,
                'task_type': task_type,
                'episodes_completed': 0,
                'avg_sequence_reward': 0.0,
                'avg_coherence': 0.0,
                'max_reward': -float('inf'),
                'min_reward': float('inf')
            }
            
            total_reward = 0.0
            total_coherence = 0.0
            num_evaluations = 0
            
            for ep in range(episodes_per_env):
                wrapped_env.skip_seed_logic = True
                state_raw = wrapped_env.reset()
                state = flatten_obs(state_raw)
                
                # Pad state to 768
                if len(state) < 768:
                    state = np.pad(state, (0, 768 - len(state)))
                elif len(state) > 768:
                    state = state[:768]
                
                state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
                
                # Get suggested action sequence from helper
                base_env = wrapped_env.base_env
                agent_pos = tuple(base_env.agent_pos)
                agent_dir = base_env.agent_dir
                goal_pos = tuple(base_env.goal_pos) if hasattr(base_env, 'goal_pos') else None
                grid = base_env.grid
                
                if task_type == "DoorKey":
                    actions, _ = helper.suggest_actions_doorkey(
                        grid, agent_pos, agent_dir, goal_pos, base_env, max_actions=MAX_SEQUENCE_LENGTH
                    )
                else:
                    actions, _ = helper.suggest_actions(
                        grid, agent_pos, agent_dir, goal_pos, max_actions=MAX_SEQUENCE_LENGTH
                    )
                
                if not actions:
                    continue
                
                # Convert actions to indices
                sequence = []
                for action_str in actions[:MAX_SEQUENCE_LENGTH]:
                    action_idx = ACTION_MAP.get(action_str.lower(), 2)
                    sequence.append(action_idx)
                
                # Pad sequence
                sequence_padded = sequence + [0] * (MAX_SEQUENCE_LENGTH - len(sequence))
                sequence_tensor = torch.tensor(sequence_padded, dtype=torch.long).to(device)
                
                # Convert to one-hot encoding
                sequence_onehot = torch.nn.functional.one_hot(sequence_tensor, num_classes=7).float()
                
                # Evaluate with model
                model.eval()
                with torch.no_grad():
                    output = model(state_tensor.unsqueeze(0), sequence_onehot.unsqueeze(0))
                    sequence_reward = output['sequence_score'].item()
                    coherence = output['coherence'].item()
                
                total_reward += sequence_reward
                total_coherence += coherence
                num_evaluations += 1
                
                env_results['max_reward'] = max(env_results['max_reward'], sequence_reward)
                env_results['min_reward'] = min(env_results['min_reward'], sequence_reward)
                
                print(f"  Episode {ep+1}/{episodes_per_env}: "
                      f"Reward={sequence_reward:.3f}, Coherence={coherence:.3f}")
            
            env_results['episodes_completed'] = episodes_per_env
            env_results['avg_sequence_reward'] = total_reward / num_evaluations if num_evaluations > 0 else 0.0
            env_results['avg_coherence'] = total_coherence / num_evaluations if num_evaluations > 0 else 0.0
            
            all_results.append(env_results)
            
            print(f"{cprint.OKGREEN} Avg Reward: {env_results['avg_sequence_reward']:.3f}, "
                  f"Avg Coherence: {env_results['avg_coherence']:.3f}{cprint.ENDC}")
            
            env.close()
    
    # Summary
    print(f"{cprint.HEADER}EVALUATION SUMMARY{cprint.ENDC}")
    
    for result in all_results:
        print(f"{cprint.CYAN}{result['env_id']}{cprint.ENDC}")
        print(f"  Avg Reward: {result['avg_sequence_reward']:.4f}")
        print(f"  Avg Coherence: {result['avg_coherence']:.4f}")
        print(f"  Reward Range: [{result['min_reward']:.3f}, {result['max_reward']:.3f}]")
        print()
    
    return all_results


def train_offline_grpo():
    csv_path = os.path.join(root_dir, "Dataset/reviewer_dataset_offline.csv")
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-5
    ROWS_LIMIT = 9000
    WEIGHT_DECAY = 1e-3
    GAMMA = 0.7

    print(f"{cprint.HEADER}OFFLINE REVIEWER TRAINING WITH GRPO{cprint.ENDC}")
    print(f"\nDataset: {csv_path}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LR}")
    print(f"Max Sequence Length: {MAX_SEQUENCE_LENGTH}")

    # Load dataset
    dataset = MinigridOfflinePreferenceDataset(csv_path, limit_rows=ROWS_LIMIT)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Dimensions
    state_size = dataset.max_flat_size
    action_size = 7
    
    print(f"\nInput Size: {state_size}")
    print(f"Action Size: {action_size}")
    print(f"Training Samples: {train_size}")
    print(f"Validation Samples: {val_size}")
    
    # Model setup
    model = SequenceRewardModel(
        state_size=state_size,
        action_size=action_size
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=GAMMA)
    
    trainer = GRPOOfflineTrainer(model, device=DEVICE)
    
    # Tracking
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # Training loop
    print(f"\n{cprint.HEADER}STARTING TRAINING LOOP{cprint.ENDC}\n")
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer)
        
        # Validate
        val_acc = trainer.evaluate(val_loader)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Track metrics
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        duration = time.time() - start_time
        
        # Print progress
        color = cprint.OKGREEN if val_acc > best_val_acc else cprint.INFO
        print(
            f"{color}Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | "
            f"Time: {duration:.1f}s{cprint.ENDC}"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_DIR, "reviewer_offline_grpo_best.pth")
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': {
                    'state_size': state_size,
                    'action_size': action_size,
                    'learning_rate': LR,
                    'batch_size': BATCH_SIZE,
                    'max_sequence_length': MAX_SEQUENCE_LENGTH
                },
                'epoch': epoch + 1,
                'best_val_accuracy': best_val_acc,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }
            torch.save(checkpoint, save_path)
    
    # Save final model
    final_save_path = os.path.join(MODEL_DIR, "reviewer_offline_grpo_final.pth")
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'state_size': state_size,
            'action_size': action_size,
            'learning_rate': LR,
            'batch_size': BATCH_SIZE,
            'max_sequence_length': MAX_SEQUENCE_LENGTH
        },
        'epochs': EPOCHS,
        'final_val_accuracy': val_accuracies[-1],
        'best_val_accuracy': best_val_acc,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'timestamp': time.time()
    }
    torch.save(checkpoint, final_save_path)
    
    print(f"\n{cprint.OKGREEN}Final model saved: {final_save_path}{cprint.ENDC}")
    print(f"{cprint.OKGREEN}Best model saved: {save_path}{cprint.ENDC}")
    print(f"{cprint.OKGREEN}Best validation accuracy: {best_val_acc:.2f}%{cprint.ENDC}")
    
    # Generate training plots
    print(f"{cprint.HEADER}GENERATING TRAINING PLOTS{cprint.ENDC}")
    
    save_loss_plot(
        train_losses,
        "Bradley-Terry Loss Over Epochs",
        os.path.join(PLOTS_DIR, "training_loss.png")
    )
    
    save_accuracy_plot(
        val_accuracies,
        "Preference Accuracy Over Epochs",
        os.path.join(PLOTS_DIR, "validation_accuracy.png")
    )
    
    save_confusion_matrix_plot(
        model, 
        val_loader, 
        DEVICE, 
        os.path.join(PLOTS_DIR, "confusion_matrix.png")
    )


    # Save best validation accuracy to text file for easy reference
    txt_path = os.path.join(DATASET_DIR, "best_val_accuracy.txt")
    with open(txt_path, "w") as f:
        f.write(f"{best_val_acc:.4f}")
    print(f"{cprint.OKGREEN} Saved value: {txt_path}{cprint.ENDC}")


    
    # Create analysis CSV
    analysis_csv_path = os.path.join(DATASET_DIR, "reviewer_dataset_offline_grpo.csv")
    create_analysis_csv(dataset, model, DEVICE, analysis_csv_path)
    
    # Run automatic analysis
    print(f"\n{cprint.HEADER}RUNNING AUTOMATIC ANALYSIS{cprint.ENDC}\n")
    
    try:
        from ReviewerRL.train_reviewer_grpo_analysis import run_grpo_post_training_analysis
        run_grpo_post_training_analysis(
            csv_path=analysis_csv_path,
            plot_dir=PLOTS_DIR,
            force_degenerate=True
        )
    except Exception as e:
        print(f"{cprint.WARNING}Warning: Could not run automatic analysis: {e}{cprint.ENDC}")
    
    print(f"\n{cprint.HEADER}TRAINING COMPLETED{cprint.ENDC}\n")
    print(f"Model: {final_save_path}")
    print(f"Best Accuracy: {best_val_acc:.2f}%")
    print(f"Plots: {PLOTS_DIR}")
    print(f"CSV: {analysis_csv_path}")
    
    return model, best_val_acc


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train Reviewer Offline with GRPO")
    parser.add_argument("--skip-eval", action="store_true", 
                       help="Skip evaluation on test environments")
    parser.add_argument("--eval-episodes", type=int, default=5,
                       help="Episodes per environment for evaluation")
    args = parser.parse_args()
    
    # Run training
    trained_model, best_acc = train_offline_grpo()
    
    if not args.skip_eval:
        print(f"\n{cprint.INFO}Starting evaluation on test environments...{cprint.ENDC}")
        eval_results = evaluate_on_environments(
            trained_model, 
            DEVICE, 
            episodes_per_env=args.eval_episodes
        )
        
        # Save evaluation results
        eval_csv_path = os.path.join(DATASET_DIR, "evaluation_results.csv")
        eval_df = pd.DataFrame(eval_results)
        eval_df.to_csv(eval_csv_path, index=False)
        print(f"{cprint.OKGREEN}Evaluation results saved: {eval_csv_path}{cprint.ENDC}")
    else:
        print(f"{cprint.WARNING}Skipping evaluation (--skip-eval flag set){cprint.ENDC}")
