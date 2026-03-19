import os
import warnings
import logging

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from trl import AutoModelForSeq2SeqLMWithValueHead
from datasets import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import copy
import json


ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "Dataset" / "reviewer_dataset_offline.csv"
OUTPUT_DIR = Path(__file__).parent / "Model"
PLOTS_DIR = Path(__file__).parent / "Plots"
DATESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


LEARNING_RATE = 3e-6
NUM_TRAINING_EPOCHS = 10
GRADIENT_ACCUMULATION = 8
MAX_INPUT_LENGTH = 192
MAX_NEW_TOKENS = 48

PPO_BATCH_SIZE = 1
PPO_EPOCHS = 1
PPO_CLIP_EPSILON = 0.15
VALUE_COEF = 0.05
ENTROPY_COEF = 0.06
MIN_ENTROPY_COEF = 0.03

RUNNING_REWARD_ALPHA = 0.1


TEMPERATURE = 0.5
TOP_K = 30
TOP_P = 0.9

LR_WARMUP_STEPS = 25


SAMPLE_WEIGHTS = {
    'empty': 0.3,
    'lava': 2.0,
    'doorkey_goal': 1.0,
    'doorkey_pickup': 4.0,
    'doorkey_toggle': 4.0,
}


EPOCH_BOUNDARY_CLIP_STEPS = 0
EPOCH_BOUNDARY_CLIP_EPSILON = 0.15


COLLAPSE_THRESHOLD = 5.0
MAX_ROLLBACKS = 3
LR_ROLLBACK_FACTOR = 0.5


EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MIN_DELTA = 0.1

KL_COEF = 0.05
SUPERVISED_WEIGHT = 0.45
USE_GRADIENT_CHECKPOINTING = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()
print(f"Using device: {device}")
# This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")



# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def build_prompt(prompt_text: str) -> str:
    return (
        f"You are a path planner. {prompt_text}\n\n"
        f"ACTIONS: \"forward\" (move 1 cell), \"left\"/\"right\" (rotate only), "
        f"\"pickup\" (pick up key), \"toggle\" (open door)\n\n"
        f"RULES:\n"
        f"- Avoid walls and lava\n"
        f"- \"forward\" moves in facing direction\n"
        f"- \"left\"/\"right\" only rotate\n\n"
        f"- \"pickup\" picks up a key if is in front of the cell and distance is 1\n"
        f"- \"toggle\" opens a door if is in front of the cell and distance is 1\n\n"
        f"RESPOND WITH ONLY THE ACTION LIST:"
    )



# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def classify_sample(prompt: str, instructions: str) -> str:
    prompt_low = prompt.lower()
    instr_low  = instructions.lower()

    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if 'empty' in prompt_low:
        return 'empty'
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    elif 'lava' in prompt_low:
        return 'lava'
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    elif 'doorkey' in prompt_low:
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if 'pickup' in instr_low:
            return 'doorkey_pickup'
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        elif 'toggle' in instr_low:
            return 'doorkey_toggle'
        else:
            return 'doorkey_goal'
    return 'lava'


# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def load_weighted_dataset(csv_path):
    print(f"Caricamento dataset da {csv_path}...")
    df = pd.read_csv(csv_path)

    data = []
    weights_per_sample = []
    stats = {k: 0 for k in SAMPLE_WEIGHTS}

    for _, row in df.iterrows():
        prompt = str(row['prompt'])
        instructions = str(row['instructions'])
        category = classify_sample(prompt, instructions)

        input_text = build_prompt(
            prompt + " Helper suggestion: " + str(row['response'])
        )
        data.append({
            'query': input_text,
            'ideal_actions': instructions.strip(),
            'category': category,
        })
        weights_per_sample.append(SAMPLE_WEIGHTS[category])
        stats[category] += 1

    print(f"\n  Distribuzione dataset ({len(data)} totale):")
    total_w = sum(n * SAMPLE_WEIGHTS[k] for k, n in stats.items())
    for cat, n in stats.items():
        natural_pct = n / len(data) * 100
        sampled_pct  = (n * SAMPLE_WEIGHTS[cat] / total_w) * 100
        print(f" {cat:20s}: {n:4d} esempi  "
              f"naturale={natural_pct:5.1f}%  "
              f"campionatoâ‰ˆ{sampled_pct:5.1f}%  "
              f"(peso={SAMPLE_WEIGHTS[cat]})")

    dataset = Dataset.from_pandas(pd.DataFrame(data))
    return dataset, weights_per_sample, stats


# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def parse_action_sequence(text: str) -> list:
    text = text.strip().lower()
    text = text.replace('[','').replace(']','').replace('"','').replace("'",'')
    actions = [a.strip() for a in text.split(',') if a.strip()]

    normalized = []
    for a in actions:
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if   'forward' in a or 'move' in a:
            normalized.append('forward')
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        elif 'left' in a:
            normalized.append('left')
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        elif 'right' in a:
            normalized.append('right')
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        elif 'pickup' in a or 'pick' in a:
            normalized.append('pickup')
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        elif 'toggle' in a or 'open' in a:
            normalized.append('toggle')
    return normalized


SPECIAL_ACTIONS = {'pickup', 'toggle'}

# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def calculate_reward(generated_text: str, ideal_actions_str: str) -> torch.Tensor:
    try:
        generated = parse_action_sequence(generated_text)
        ideal = parse_action_sequence(ideal_actions_str)

        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if not generated:
            return torch.tensor([-10.0])

        n_ideal = len(ideal)
        n_gen = len(generated)
        min_len = min(n_gen, n_ideal)

        correct_count = 0
        prefix_streak = 0
        action_reward = 0.0

        for i in range(min_len):
            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if generated[i] == ideal[i]:
                action_reward += 8.0
                correct_count += 1
                # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                if i == prefix_streak:
                    prefix_streak += 1
            else:
                action_reward -= 2.0

        accuracy = correct_count / n_ideal if n_ideal > 0 else 0.0
        proportional_reward = (accuracy - 0.5) * 20.0

        prefix_bonus = prefix_streak * 2.0

        len_diff = n_gen - n_ideal
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if len_diff > 0:
            length_penalty = -3.0 * len_diff
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        elif len_diff < 0:
            length_penalty = -4.0 * abs(len_diff)
        else:
            length_penalty = 3.0

        terminal_bonus = 0.0
        ideal_terminal = ideal[-1] if ideal else None
        gen_terminal = generated[-1] if generated else None

        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if ideal_terminal in SPECIAL_ACTIONS:
            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if gen_terminal == ideal_terminal:
                terminal_bonus = 25.0
            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            elif gen_terminal in SPECIAL_ACTIONS:
                terminal_bonus = -14.0
            else:
                terminal_bonus = -8.0

        perfect_bonus = 35.0 if (generated == ideal) else 0.0

        reward = (action_reward
                  + proportional_reward
                  + prefix_bonus
                  + length_penalty
                  + terminal_bonus
                  + perfect_bonus)

        return torch.tensor([reward])

    except Exception as e:
        return torch.tensor([0.0])


# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def tokenize_query(tokenizer, input_text: str) -> dict:
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=True
    )
    return {
        'input_ids':      enc.input_ids.to(device),
        'attention_mask': enc.attention_mask.to(device),
    }


# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def compute_all_in_one(model, tokenized: dict, generated_ids: torch.Tensor,
                       no_grad: bool = False, pad_token_id: int = 0,
                       ideal_ids: torch.Tensor = None):
    gen_labels = generated_ids.clone()
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if gen_labels.dim() == 1:
        gen_labels = gen_labels.unsqueeze(0)
    gen_labels = gen_labels.to(device)

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
            outputs = model.pretrained_model(
                input_ids = tokenized['input_ids'],
                attention_mask = tokenized['attention_mask'],
                labels = gen_labels.masked_fill(
                                           gen_labels == pad_token_id, -100),
                output_hidden_states = True,
                return_dict = True,
            )

    logits = outputs.logits.float().clamp(-50.0, 50.0)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = gen_labels[:, 1:].contiguous()

    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if shift_labels.shape[1] == 0:
        log_prob = torch.tensor(0.0, device=device,
                                requires_grad=not no_grad)
    else:
        lp_all = F.log_softmax(shift_logits, dim=-1)
        token_lp = lp_all.gather(
            2, shift_labels.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)
        mask = (shift_labels != pad_token_id) & (shift_labels != -100)
        valid_lp = token_lp * mask.float()
        n_valid = mask.float().sum().clamp(min=1)
        log_prob = valid_lp.sum() / n_valid
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if torch.isnan(log_prob) or torch.isinf(log_prob):
            log_prob = torch.tensor(0.0, device=device)

    probs = F.softmax(logits, dim=-1).clamp(min=1e-9)
    entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if torch.isnan(entropy) or torch.isinf(entropy):
        entropy = torch.tensor(0.0, device=device)

    try:
        dec_hidden = outputs.decoder_hidden_states[-1].float().clamp(-50.0, 50.0)
        value = model.v_head(dec_hidden[:, -1, :]).squeeze()
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if torch.isnan(value) or torch.isinf(value):
            value = torch.tensor(0.0, device=device)
        else:
            value = value.clamp(-50.0, 50.0)
    except Exception:
        value = torch.tensor(0.0, device=device)

    supervised_ce = torch.tensor(0.0, device=device)
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if ideal_ids is not None and not no_grad:
        sup_labels = ideal_ids.clone().to(device)
        sup_labels = sup_labels.masked_fill(sup_labels == pad_token_id, -100)
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if (sup_labels != -100).sum() > 0:
            with torch.enable_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=torch.cuda.is_available()):
                    sup_out = model.pretrained_model(
                        input_ids = tokenized['input_ids'],
                        attention_mask = tokenized['attention_mask'],
                        labels = sup_labels,
                        return_dict = True,
                    )
            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if not (torch.isnan(sup_out.loss) or torch.isinf(sup_out.loss)):
                supervised_ce = sup_out.loss

    return log_prob, value, entropy, supervised_ce


class RunningMeanStd:
    # This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
    def __init__(self, alpha: float = RUNNING_REWARD_ALPHA):
        self.alpha = alpha
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    # This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
    def update(self, x: float):
        self.count += 1
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if self.count == 1:
            self.mean = x
        else:
            old_mean = self.mean
            self.mean = (1 - self.alpha) * self.mean + self.alpha * x
            self.var = (1 - self.alpha) * self.var  + self.alpha * (x - old_mean) ** 2

    # This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
    def normalize(self, x: float) -> float:
        std = max(self.var ** 0.5, 1e-8)
        return (x - self.mean) / std


# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def plot_metrics(metrics_history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(metrics_history) + 1))

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    axes[0,0].plot(epochs, [m['avg_reward'] for m in metrics_history], 'b-o', linewidth=2)
    axes[0,0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Avg Reward'); axes[0,0].set_xlabel('Epoch'); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(epochs, [m['token_accuracy'] for m in metrics_history], 'g-s', linewidth=2)
    axes[0,1].set_title('Token Accuracy (%)'); axes[0,1].set_xlabel('Epoch'); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(epochs, [m['acc_empty'] for m in metrics_history], 'c-^', label='Empty', linewidth=2)
    axes[0,2].plot(epochs, [m['acc_lava'] for m in metrics_history], 'r-s', label='Lava', linewidth=2)
    axes[0,2].plot(epochs, [m['acc_doorkey_goal'] for m in metrics_history], 'm-o', label='DK goal', linewidth=2)
    axes[0,2].plot(epochs, [m['acc_pickup'] for m in metrics_history], 'y-d', label='pickup', linewidth=2)
    axes[0,2].plot(epochs, [m['acc_toggle'] for m in metrics_history], 'k-p', label='toggle', linewidth=2)
    axes[0,2].set_title('Accuracy per categoria'); axes[0,2].set_xlabel('Epoch')
    axes[0,2].legend(fontsize=8); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(epochs, [m['avg_policy_loss'] for m in metrics_history], 'r-^', linewidth=2)
    axes[1,0].set_title('Policy Loss'); axes[1,0].set_xlabel('Epoch'); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, [m['avg_value_loss'] for m in metrics_history], 'm-d', linewidth=2)
    axes[1,1].set_title('Value Loss'); axes[1,1].set_xlabel('Epoch'); axes[1,1].grid(True, alpha=0.3)

    axes[1,2].plot(epochs, [m['rollback_epoch'] for m in metrics_history], 'k-x', linewidth=2, markersize=8)
    axes[1,2].set_title('Rollback events'); axes[1,2].set_xlabel('Epoch'); axes[1,2].grid(True, alpha=0.3)

    axes[2,0].plot(epochs, [m['precision'] for m in metrics_history], 'b-o', label='Precision', linewidth=2)
    axes[2,0].plot(epochs, [m['recall'] for m in metrics_history], 'r-s', label='Recall', linewidth=2)
    axes[2,0].plot(epochs, [m['f1_score']  for m in metrics_history], 'g-^', label='F1 Score', linewidth=2)
    axes[2,0].plot(epochs, [m['token_accuracy'] for m in metrics_history], 'k--d', label='Token Accuracy', linewidth=1.5, alpha=0.6)
    axes[2,0].set_title('Precision / Recall / F1 / Token Accuracy')
    axes[2,0].set_xlabel('Epoch'); axes[2,0].set_ylabel('%')
    axes[2,0].legend(fontsize=8); axes[2,0].grid(True, alpha=0.3)

    axes[2,1].plot(epochs, [m['success_rate'] for m in metrics_history], 'c-o', linewidth=2)
    axes[2,1].set_title('Success Rate (%)')
    axes[2,1].set_xlabel('Epoch'); axes[2,1].set_ylabel('%')
    axes[2,1].grid(True, alpha=0.3)

    axes[2,2].plot(epochs, [m['avg_entropy'] for m in metrics_history], 'm-s', linewidth=2)
    axes[2,2].set_title('Entropy')
    axes[2,2].set_xlabel('Epoch'); axes[2,2].set_ylabel('Entropy')
    axes[2,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'ppo_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    single_plots = [
        {
            'filename': 'ppo_avg_reward.png',
            'title': 'Avg Reward',
            'values': [m['avg_reward'] for m in metrics_history],
            'style': 'b-o',
            'xlabel': 'Epoch',
            'ylabel': 'Reward',
            'axhline_zero': True,
        },
        {
            'filename': 'ppo_sequence_accuracy.png',
            'title': 'Token Accuracy (%)',
            'values': [m['token_accuracy'] for m in metrics_history],
            'style': 'g-s',
            'xlabel': 'Epoch',
            'ylabel': 'Accuracy (%)',
            'axhline_zero': False,
        },
        {
            'filename': 'ppo_policy_loss.png',
            'title': 'Policy Loss',
            'values': [m['avg_policy_loss'] for m in metrics_history],
            'style': 'r-^',
            'xlabel': 'Epoch',
            'ylabel': 'Loss',
            'axhline_zero': False,
        },
        {
            'filename': 'ppo_value_loss.png',
            'title': 'Value Loss',
            'values': [m['avg_value_loss'] for m in metrics_history],
            'style': 'm-d',
            'xlabel': 'Epoch',
            'ylabel': 'Loss',
            'axhline_zero': False,
        },
        {
            'filename': 'ppo_rollback_events.png',
            'title': 'Rollback events',
            'values': [m['rollback_epoch'] for m in metrics_history],
            'style': 'k-x',
            'xlabel': 'Epoch',
            'ylabel': 'Events',
            'axhline_zero': False,
        },
        {
            'filename': 'ppo_success_rate.png',
            'title': 'Success Rate (%)',
            'values': [m['success_rate'] for m in metrics_history],
            'style': 'c-o',
            'xlabel': 'Epoch',
            'ylabel': 'Success Rate (%)',
            'axhline_zero': False,
        },
        {
            'filename': 'ppo_entropy.png',
            'title': 'Entropy',
            'values': [m['avg_entropy'] for m in metrics_history],
            'style': 'm-s',
            'xlabel': 'Epoch',
            'ylabel': 'Entropy',
            'axhline_zero': False,
        },
    ]

    for cfg in single_plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, cfg['values'], cfg['style'], linewidth=2, markersize=6)
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if cfg['axhline_zero']:
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(cfg['title'])
        ax.set_xlabel(cfg['xlabel'])
        ax.set_ylabel(cfg['ylabel'])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / cfg['filename'], dpi=150, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [m['acc_empty'] for m in metrics_history], 'c-^', label='Empty', linewidth=2)
    ax.plot(epochs, [m['acc_lava'] for m in metrics_history], 'r-s', label='Lava', linewidth=2)
    ax.plot(epochs, [m['acc_doorkey_goal'] for m in metrics_history], 'm-o', label='DK goal', linewidth=2)
    ax.plot(epochs, [m['acc_pickup'] for m in metrics_history], 'y-d', label='pickup', linewidth=2)
    ax.plot(epochs, [m['acc_toggle'] for m in metrics_history], 'k-p', label='toggle', linewidth=2)
    ax.set_title('Accuracy per categoria')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'ppo_accuracy_per_categoria.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [m['precision'] for m in metrics_history], 'b-o', label='Precision', linewidth=2)
    ax.plot(epochs, [m['recall']    for m in metrics_history], 'r-s', label='Recall',    linewidth=2)
    ax.plot(epochs, [m['f1_score']  for m in metrics_history], 'g-^', label='F1 Score',  linewidth=2)
    ax.plot(epochs, [m['token_accuracy'] for m in metrics_history], 'k--d', label='Token Accuracy', linewidth=1.5, alpha=0.6)
    ax.set_title('Precision / Recall / F1 / Token Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'ppo_precision_recall_f1.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    bar_cfg = [
        ('Precision (%)', [m['precision'] for m in metrics_history],'#1f77b4'),
        ('Recall (%)', [m['recall'] for m in metrics_history], '#d62728'),
        ('F1 Score (%)', [m['f1_score'] for m in metrics_history],'#2ca02c'),
        ('Token Accuracy (%)', [m['token_accuracy'] for m in metrics_history], '#9467bd'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, (title, values, color) in zip(axes, bar_cfg):
        ax.bar(epochs, values, color=color, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value (%)')
        ax.set_xticks(epochs)
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('PPO Metrics Bar Overview', fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(output_dir / 'ppo_metrics_bar_overview.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Grafici salvati in {output_dir}")



# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def find_latest_finetuned_model():
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if not OUTPUT_DIR.exists():
        return None
    folders = [
        f for f in OUTPUT_DIR.iterdir()
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if f.is_dir()
        and not f.name.startswith('.')
        and not f.name.startswith('ppo_')
        and (f / "config.json").exists()
    ]
    folders.sort(reverse=True)
    return folders[0] if folders else None


# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def build_model_and_optimizer(model_path, lr, vram_gb):
    model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
        str(model_path), dtype=torch.bfloat16
    ).to(device)
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if USE_GRADIENT_CHECKPOINTING:
        try:
            model.pretrained_model.gradient_checkpointing_enable()
        except Exception:
            pass
    else:
        try:
            model.pretrained_model.gradient_checkpointing_disable()
        except Exception:
            pass
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    return model, optimizer



# This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
def main(finetuned_model_path=None, num_epochs=None):
    epochs = num_epochs if num_epochs is not None else NUM_TRAINING_EPOCHS

    print("=" * 60)
    print("MINIGRID REVIEWER â€” PPO â€” WEIGHTED UNIFIED DATASET")
    print("=" * 60)
    print(f"Datestamp:   {DATESTAMP}")
    print(f"Epoche:      {epochs}  |  LR: {LEARNING_RATE}")
    print(f"ENTROPY:     {ENTROPY_COEF}  |  CLIP_Îµ: {PPO_CLIP_EPSILON}")
    print(f"PPO_EPOCHS:  {PPO_EPOCHS}  |  Warmup steps: {LR_WARMUP_STEPS}")
    print(f"Epoch boundary warmup: {EPOCH_BOUNDARY_CLIP_STEPS} step, Îµ={EPOCH_BOUNDARY_CLIP_EPSILON}")
    print(f"Collapse detection: drop>{COLLAPSE_THRESHOLD}% â†’ rollback (max {MAX_ROLLBACKS})\n")

    output_model_dir = OUTPUT_DIR / f"ppo_{DATESTAMP}"
    plots_output_dir = PLOTS_DIR  / f"ppo_{DATESTAMP}"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if finetuned_model_path is None:
        finetuned_model_path = find_latest_finetuned_model()
    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
    if finetuned_model_path is None:
        raise FileNotFoundError(
            f"Modello fine-tuned non trovato in {OUTPUT_DIR}\n"
            "Esegui prima fine_tune.py"
        )
    print(f"Modello base: {finetuned_model_path}\n")

    dataset, sample_weights, ds_stats = load_weighted_dataset(DATASET_PATH)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(dataset),
        replacement = True
    )

    # This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
    def collate_fn(batch):
        return {
            'input':        [item['query']         for item in batch],
            'instructions': [item['ideal_actions']  for item in batch],
            'category':     [item['category']       for item in batch],
        }

    dataloader = DataLoader(
        dataset,
        batch_size  = PPO_BATCH_SIZE,
        sampler     = sampler,
        collate_fn  = collate_fn
    )

    print(f"\nCaricamento modello da {finetuned_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(finetuned_model_path), legacy=False, use_fast=False
    )
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
               # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
               if torch.cuda.is_available() else 99)
    model, optimizer = build_model_and_optimizer(
        finetuned_model_path, LEARNING_RATE, vram_gb
    )
    print(f"Parametri: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("Reference model salvato per KL penalty")

    total_steps = epochs * len(dataloader) // GRADIENT_ACCUMULATION
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = LR_WARMUP_STEPS,
        num_training_steps = total_steps,
    )

    exploration_generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_new_tokens": 3,
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    exploitation_generation_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_new_tokens": 3,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    metrics_history = []
    best_success_rate = 0.0
    epochs_no_improve = 0
    total_rollbacks = 0
    reward_rms = RunningMeanStd()
    global_step = 0

    print("\n" + "=" * 60)
    print("INIZIO PPO TRAINING â€” dataset unificato con weighted sampling")
    print("=" * 60)

    for epoch in range(epochs):
        entropy_coef_current = max(MIN_ENTROPY_COEF, ENTROPY_COEF * (0.95 ** epoch))
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'=' * 60}")
        print(f"EPOCA {epoch+1}/{epochs} | LR: {current_lr:.2e}"
              + (" [warmup in corso...]" if global_step < LR_WARMUP_STEPS else ""))
        print(f"Entropy coef corrente: {entropy_coef_current:.4f}")

        use_exploitation_decode = (
            epoch >= max(2, int(epochs * 0.4))
            or epochs_no_improve >= 2
        )
        generation_kwargs = (
            exploitation_generation_kwargs
            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if use_exploitation_decode
            else exploration_generation_kwargs
        )
        decode_mode = "exploit-greedy" if use_exploitation_decode else "explore-sampling"
        print(f"Decode mode: {decode_mode}")
        print(f"{'=' * 60}")

        total_reward = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        correct_sequences = 0
        num_batches = 0
        step_rewards = []
        epoch_boundary_steps = 0

        cat_correct = {k: 0 for k in SAMPLE_WEIGHTS}
        cat_total = {k: 0 for k in SAMPLE_WEIGHTS}

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_token_correct = 0
        total_token_slots   = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):

            epoch_boundary_steps += 1
            current_clip_eps = PPO_CLIP_EPSILON

            queries = batch['input']
            ideal_actions = batch['instructions']
            categories  = batch['category']

            model.eval()
            batch_generated_ids = []
            batch_old_log_probs = []
            batch_values = []
            batch_responses = []
            batch_tokenized = []

            with torch.no_grad():
                for q_idx, query in enumerate(queries):
                    tokenized = tokenize_query(tokenizer, query)
                    batch_tokenized.append(tokenized)

                    gen_out = model.generate(**tokenized, **generation_kwargs)

                    ideal_list = parse_action_sequence(ideal_actions[q_idx])
                    r_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                    r_list = parse_action_sequence(r_text)[:len(ideal_list)]
                    response_text_enc = ", ".join(r_list)
                    generated_ids     = gen_out[0]
                    batch_responses.append(response_text_enc)

                    reenc = tokenizer(
                        response_text_enc,
                        return_tensors="pt",
                        max_length=MAX_NEW_TOKENS,
                        truncation=True,
                    ).to(device)
                    generated_ids = reenc.input_ids[0]

                    batch_generated_ids.append(generated_ids)

                    _gen_list_check = parse_action_sequence(response_text_enc)
                    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                    if len(_gen_list_check) == 0:
                        batch_old_log_probs.append(torch.tensor(0.0, device=device))
                        batch_values.append(torch.tensor(0.0, device=device))
                        continue

                    old_lp, old_val, _, _ = compute_all_in_one(
                        model, tokenized, generated_ids.unsqueeze(0),
                        no_grad=True, pad_token_id=tokenizer.pad_token_id
                    )
                    batch_old_log_probs.append(old_lp.detach())
                    batch_values.append(old_val.detach())

            batch_rewards = []
            raw_reward_vals = []

            for i, (response_text, ideal, cat) in enumerate(zip(batch_responses, ideal_actions, categories)):
                reward = calculate_reward(response_text, ideal).to(device).float()
                r_val  = reward.item()
                batch_rewards.append(reward)
                raw_reward_vals.append(r_val)
                total_reward += r_val
                step_rewards.append(r_val)

                gen_parsed = parse_action_sequence(response_text)
                ideal_parsed = parse_action_sequence(ideal)
                cat_total[cat] += 1

                is_perfect = (gen_parsed == ideal_parsed) and len(ideal_parsed) > 0
                match_val = 1.0 if is_perfect else 0.0

                correct_sequences += match_val
                cat_correct[cat] += match_val

                _min_len = min(len(gen_parsed), len(ideal_parsed))
                _tp = sum(1 for _j in range(_min_len) if gen_parsed[_j] == ideal_parsed[_j])
                _fp = sum(1 for _j in range(_min_len) if gen_parsed[_j] != ideal_parsed[_j]) + max(0, len(gen_parsed) - len(ideal_parsed))
                _fn = sum(1 for _j in range(_min_len) if gen_parsed[_j] != ideal_parsed[_j]) + max(0, len(ideal_parsed) - len(gen_parsed))
                total_tp += _tp
                total_fp += _fp
                total_fn += _fn
                total_token_correct += _tp
                total_token_slots   += max(len(gen_parsed), len(ideal_parsed))

            for r_val in raw_reward_vals:
                reward_rms.update(r_val)

            batch_rewards_norm = [
                torch.tensor([reward_rms.normalize(r.item())], device=device)
                for r in batch_rewards
            ]

            epoch_decay = max(0.6, 1.0 - (epoch / max(epochs - 1, 1)) * 0.4)
            supervised_w_base = SUPERVISED_WEIGHT * epoch_decay

            raw_advantages = []
            for reward, old_value in zip(batch_rewards_norm, batch_values):
                raw_advantages.append((reward.squeeze() - old_value).detach())

            adv_tensor = torch.stack(raw_advantages)
            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if adv_tensor.numel() > 1:
                adv_mean = adv_tensor.mean()
                adv_std  = adv_tensor.std().clamp(min=1e-8)
                norm_advantages = [(a - adv_mean) / adv_std for a in raw_advantages]
            else:
                norm_advantages = raw_advantages

            model.train()
            policy_losses = []
            value_losses  = []
            entropy_values = []

            with torch.no_grad():
                cached_ref_log_probs = []
                for i_ref, (tok_ref, gen_ref) in enumerate(
                    zip(batch_tokenized, batch_generated_ids)
                ):
                    r_lp, _, _, _ = compute_all_in_one(
                        ref_model, tok_ref, gen_ref.unsqueeze(0),
                        no_grad=True, pad_token_id=tokenizer.pad_token_id
                    )
                    cached_ref_log_probs.append(r_lp.detach())

            for _ppo_ep in range(PPO_EPOCHS):
                for i, (tokenized, generated_ids, reward, old_log_prob, advantage) in enumerate(
                    zip(batch_tokenized, batch_generated_ids, batch_rewards_norm,
                        batch_old_log_probs, norm_advantages)
                ):
                    ideal_ids_enc = tokenizer(
                        ideal_actions[i],
                        return_tensors="pt",
                        max_length=MAX_NEW_TOKENS,
                        truncation=True,
                        padding=True,
                    ).to(device).input_ids

                    new_log_prob, current_value, entropy, supervised_ce = compute_all_in_one(
                        model, tokenized, generated_ids.unsqueeze(0),
                        no_grad=False, pad_token_id=tokenizer.pad_token_id,
                        ideal_ids=ideal_ids_enc
                    )

                    ref_log_prob = cached_ref_log_probs[i]
                    kl_penalty = (new_log_prob - ref_log_prob).clamp(min=0)

                    log_ratio = (new_log_prob - old_log_prob.detach()).clamp(-1.5, 1.5)
                    ratio  = torch.exp(log_ratio)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - current_clip_eps,
                        1.0 + current_clip_eps
                    ) * advantage

                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = VALUE_COEF * (reward.squeeze() - current_value).pow(2)
                    entropy_loss = -entropy_coef_current * entropy
                    kl_loss =  KL_COEF * kl_penalty
                    _cat_boost = {'lava': 2.0, 'doorkey_pickup': 2.2,
                                  'doorkey_toggle': 2.0, 'doorkey_goal': 1.3,
                                  'empty': 0.6}.get(categories[i] if i < len(categories) else '', 1.0)
                    sup_loss     =  supervised_w_base * _cat_boost * supervised_ce

                    # This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
                    def _safe(t, name):
                        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                        if torch.isnan(t) or torch.isinf(t):
                            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                            if (batch_idx + 1) % 50 == 0:
                                print(f"  [DBG] {name}=NaN, sostituito con 0")
                            return torch.tensor(0.0, device=device)
                        return t
                    policy_loss = _safe(policy_loss, "policy_loss")
                    value_loss = _safe(value_loss, "value_loss")
                    entropy_loss = _safe(entropy_loss, "entropy_loss")
                    kl_loss = _safe(kl_loss, "kl_loss")
                    sup_loss = _safe(sup_loss, "sup_loss")

                    loss = (policy_loss + value_loss + entropy_loss + kl_loss + sup_loss) / (
                        PPO_BATCH_SIZE * GRADIENT_ACCUMULATION
                    )

                    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  [WARN] Loss ancora NaN dopo guard â€” skip")
                        del loss
                        continue

                    loss.backward()

                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_values.append(entropy.item())

                    del new_log_prob, current_value, ratio, ref_log_prob
                    del surr1, surr2, policy_loss, value_loss, entropy, entropy_loss
                    del kl_penalty, kl_loss, sup_loss, supervised_ce, loss

            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if (batch_idx + 1) % GRADIENT_ACCUMULATION == 0 or \
               (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                if torch.cuda.is_available() and global_step % 200 == 0:
                    torch.cuda.empty_cache()

            num_batches += 1
            total_policy_loss += sum(policy_losses) / max(len(policy_losses), 1)
            total_value_loss += sum(value_losses)  / max(len(value_losses),  1)
            total_entropy += sum(entropy_values) / max(len(entropy_values), 1)

            # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
            if (batch_idx + 1) % 10 == 0:
                avg_r = total_reward / max(num_batches * PPO_BATCH_SIZE, 1)
                avg_pl = total_policy_loss / max(num_batches, 1)
                avg_vl = total_value_loss  / max(num_batches, 1)
                acc = correct_sequences / max(num_batches * PPO_BATCH_SIZE, 1) * 100
                _gen_s = parse_action_sequence(batch_responses[0])
                _ide_s = parse_action_sequence(ideal_actions[0])
                _prefix = 0
                for _g, _i in zip(_gen_s, _ide_s):
                    # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                    if _g == _i:
                        _prefix += 1
                    else:
                        break
                _is_perfect = (_gen_s == _ide_s) and len(_ide_s) > 0
                s_acc = 100.0 if _is_perfect else 0.0

                print(f"\n  Step {batch_idx+1}/{len(dataloader)}"
                      + (f" [boundary Îµ={current_clip_eps:.2f}]"
                         # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
                         if epoch_boundary_steps <= EPOCH_BOUNDARY_CLIP_STEPS else ""))
                print(f"  Reward:{avg_r:7.2f} | PLoss:{avg_pl:.4f} | "
                      f"VLoss:{avg_vl:.4f} | Acc:{acc:5.1f}% | StepAcc:{s_acc:.0f}%")
                print(f"  LR:{optimizer.param_groups[0]['lr']:.2e} | "
                      f"Îµ:{current_clip_eps:.3f} | "
                      f"RMS(Î¼={reward_rms.mean:.1f}, Ïƒ={reward_rms.var**0.5:.1f})")
                print(f"Cat: {categories[0]:20s}")
                print(f"Gen: {batch_responses[0]}")
                print(f"Ideal: {ideal_actions[0]}")

        nb = max(num_batches, 1)
        ne = max(num_batches * PPO_BATCH_SIZE, 1)
        avg_reward = total_reward / ne
        avg_policy_loss = total_policy_loss / nb
        avg_value_loss = total_value_loss / nb
        avg_entropy = total_entropy / nb
        success_rate = correct_sequences / ne * 100
        accuracy = total_token_correct / max(total_token_slots, 1) * 100

        precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0.0
        f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # This function handles a specific training pipeline step, keeping data flow consistent, improving readability, and supporting robust PPO execution during optimization and evaluation
        def cat_acc(cat):
            return cat_correct[cat] / cat_total[cat] * 100 if cat_total[cat] > 0 else 0.0

        epoch_metrics = {
            'epoch':             epoch + 1,
            'avg_reward':        float(avg_reward),
            'avg_policy_loss':   float(avg_policy_loss),
            'avg_value_loss':    float(avg_value_loss),
            'avg_entropy':       float(avg_entropy),
            'token_accuracy':    float(accuracy),
            'success_rate':      float(success_rate),
            'precision':         float(precision),
            'recall':            float(recall),
            'f1_score':          float(f1_score),
            'acc_empty':         cat_acc('empty'),
            'acc_lava':          cat_acc('lava'),
            'acc_doorkey_goal':  cat_acc('doorkey_goal'),
            'acc_pickup':        cat_acc('doorkey_pickup'),
            'acc_toggle':        cat_acc('doorkey_toggle'),
            'correct_sequences': int(correct_sequences),
            'rollback_epoch':    0,
            'step_rewards':      step_rewards,
        }
        metrics_history.append(epoch_metrics)

        print(f"\n{'=' * 60}")
        print(f"Fine Epoca {epoch+1} â€” Reward:{avg_reward:.2f} | TokenAcc:{accuracy:.1f}%")
        print(f"  PolicyLoss:{avg_policy_loss:.4f} | ValueLoss:{avg_value_loss:.4f} | Entropy:{avg_entropy:.4f}")
        print(f"  Precision:{precision:.1f}% | Recall:{recall:.1f}% | F1:{f1_score:.1f}%")
        print(f"  Success Rate:{success_rate:.1f}%")
        print(f"  Per categoria:")
        for cat in SAMPLE_WEIGHTS:
            print(f"    {cat:20s}: {cat_acc(cat):5.1f}%  "
                  f"({cat_correct[cat]}/{cat_total[cat]})")

        best_dir = output_model_dir / "best_checkpoint"
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if success_rate > best_success_rate + EARLY_STOPPING_MIN_DELTA:
            best_success_rate = success_rate
            epochs_no_improve = 0
            print(f" Nuovo best success rate: {best_success_rate:.1f}%")
            best_dir.mkdir(exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

        did_rollback  = False
        collapse_drop = best_success_rate - success_rate
        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if (collapse_drop >= COLLAPSE_THRESHOLD
                and total_rollbacks < MAX_ROLLBACKS
                and best_dir.exists()):

            total_rollbacks += 1
            new_lr = optimizer.param_groups[0]['lr'] * LR_ROLLBACK_FACTOR
            print(f"\n COLLAPSE! success_rate={success_rate:.1f}% "
                f"(drop={collapse_drop:.1f}% da best={best_success_rate:.1f}%)")
            print(f"  Rollback #{total_rollbacks}/{MAX_ROLLBACKS} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.2e} â†’ {new_lr:.2e}")

            model, optimizer = build_model_and_optimizer(best_dir, new_lr, vram_gb)
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
            print("Reference model salvato per KL penalty")

            remaining_steps = max(epochs - epoch - 1, 1) * len(dataloader) // GRADIENT_ACCUMULATION
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps   = min(LR_WARMUP_STEPS, remaining_steps // 4),
                num_training_steps = remaining_steps,
            )
            epochs_no_improve = 0
            epoch_metrics['rollback_epoch'] = 1
            did_rollback = True

        print(f"{'=' * 60}")

        # This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping dopo {EARLY_STOPPING_PATIENCE} epoche")
            break

    plot_metrics(metrics_history, plots_output_dir)
    model.save_pretrained(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))

    with open(output_model_dir / "ppo_metrics.json", "w") as f:
        json.dump({
            'training_method': 'PPO â€” Weighted Unified Dataset (improved)',
            'datestamp':       DATESTAMP,
            'finetuned_model': str(finetuned_model_path),
            'epochs_trained':  len(metrics_history),
            'dataset_stats':   ds_stats,
            'sample_weights':  SAMPLE_WEIGHTS,
            'entropy_coef':    ENTROPY_COEF,
            'min_entropy_coef': MIN_ENTROPY_COEF,
            'ppo_epochs':      PPO_EPOCHS,
            'learning_rate':   LEARNING_RATE,
            'temperature':     TEMPERATURE,
            'total_rollbacks': total_rollbacks,
            'best_token_accuracy': max((m.get('token_accuracy', 0.0) for m in metrics_history), default=0.0),
            'best_success_rate': float(best_success_rate),
            'metrics_history': [
                {k: v for k, v in m.items() if k != 'step_rewards'}
                for m in metrics_history
            ]
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("PPO TRAINING COMPLETATO!")
    print(f"Modello:         {output_model_dir}")
    print(f"Best checkpoint: {best_dir}")
    print(f"Best SuccessRate:{best_success_rate:.1f}%")
    print(f"Rollback totali: {total_rollbacks}")


# This conditional checks runtime state and applies the correct branch to preserve training stability, prevent invalid values, and keep outputs aligned with expected behavior
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO â€” Weighted Unified Dataset (improved)')
    parser.add_argument('--model',  type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    main(
        finetuned_model_path=Path(args.model) if args.model else None,
        num_epochs=args.epochs,
    )

