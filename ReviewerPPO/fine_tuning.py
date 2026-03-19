import os
import warnings
import logging


os.environ["TF_ENABLE_ONEDNN_OPTS"]= "0"  
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  
warnings.filterwarnings("ignore")                   


logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)    


import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict, load_from_disk
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import hashlib
import json
import platform

import torch._dynamo
torch._dynamo.config.suppress_errors = True


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


ROOT_DIR = Path(__file__).parent.parent
DATASET_PATH = ROOT_DIR / "Dataset" / "reviewer_dataset_offline.csv"
OUTPUT_DIR = Path(__file__).parent / "Model"
PLOTS_DIR = Path(__file__).parent / "Plots"
CACHE_DIR = Path(__file__).parent / "cache"

DATESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_NAME = "google/flan-t5-base"


MAX_INPUT_LENGTH  = 192   
MAX_TARGET_LENGTH = 32    


BATCH_SIZE = 2   
GRADIENT_ACCUMULATION = 8   
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
WARMUP_STEPS = 50
LOGGING_STEPS = 20
EVAL_STEPS = LOGGING_STEPS
SAVE_STEPS = 100


DATALOADER_NUM_WORKERS = 4 if platform.system() != "Windows" else 0
PIN_MEMORY = True
USE_DYNAMIC_PADDING = True
USE_DATASET_CACHE = True
USE_TORCH_COMPILE = False  


USE_FP16 = False
USE_BF16 = True
USE_GRADIENT_CHECKPOINTING = True


CLASS_WEIGHTS = {
    'forward': 0.5,
    'left': 2.0,
    'right': 2.0,
    'pickup': 3.0,
    'toggle': 4.0
}

ACTION_LABELS = ['forward', 'left', 'right', 'pickup', 'toggle']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Print runtime device info
if __name__ == "__main__" or "pytest" in __import__("sys").modules:
    print(f"Using device: {device}")
    # Show GPU details if present
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
        # Warn for low VRAM setups
        if vram < 5:
            print("< 5GB VRAM â€” configurazione 4GB attiva (gradient_checkpointing ON)")



# Build prompt from input text and fixed action constraints
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



class WeightedTrainer(Trainer):
    def __init__(self, *args, action_weights=None, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_weights    = action_weights or {}
        self._custom_tokenizer = tokenizer

        self.token_weights = {}
        for action, weight in self.action_weights.items():
            token_ids = self._custom_tokenizer.encode(action, add_special_tokens=False)
            for tid in token_ids:
                self.token_weights[tid] = weight
    # Override compute_loss to apply token-level weights based on the action tokens
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        logits_f32 = logits.float()
        flat_labels = labels.view(-1)

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits_f32.view(-1, logits_f32.size(-1)), flat_labels)
        # Create a weight tensor where each token's loss is multiplied by its corresponding weight
        weights = torch.ones_like(loss)
        for token_id, weight in self.token_weights.items():
            weights = torch.where(
                flat_labels == token_id,
                torch.full_like(weights, weight),
                weights
            )

        
        valid_mask = (flat_labels != -100).float()
        weighted_loss = (loss * weights * valid_mask).sum() / valid_mask.sum().clamp(min=1)

        return (weighted_loss, outputs) if return_outputs else weighted_loss


class FirstStepEvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Trigger eval at first step
        if state.global_step == 1:
            control.should_evaluate = True
        return control




# Augment dataset by duplicating low-rotation action samples
def augment_dataset(df: pd.DataFrame, augmentation_factor: float = 0.3) -> pd.DataFrame:
    print(f"Data augmentation: fattore {augmentation_factor}x")
    # Parsing of the 'instructions' column to extract action sequences robustly
    def safe_parse(text):
        try:
            result = ast.literal_eval(str(text))
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    parsed = df['instructions'].apply(safe_parse)

    def rotation_ratio(actions):
        # Empty actions get max ratio
        if not actions:
            return 1.0
        return sum(1 for a in actions if a in ['left', 'right']) / len(actions)

    ratios = parsed.apply(rotation_ratio)
    low_rotation_mask = ratios < 0.3
    rows_to_duplicate = df[low_rotation_mask]
    n_copies = max(1, int(augmentation_factor))
    augmented_df = pd.concat([df] + [rows_to_duplicate] * n_copies, ignore_index=True)

    print(f"Dataset: {len(df)} â†’ {len(augmented_df)} esempi "
          f"({len(rows_to_duplicate)} righe duplicate {n_copies}x)")
    return augmented_df



# Compute a cache hash from dataset and tokenization settings
def get_dataset_hash(csv_path, augmentation, model_name, max_input_len, max_target_len):
    h = {
        'csv_mtime':       csv_path.stat().st_mtime,
        'csv_size':        csv_path.stat().st_size,
        'augmentation':    augmentation,
        'model':           model_name,
        'max_input':       max_input_len,
        'max_target':      max_target_len,
        'dynamic_padding': USE_DYNAMIC_PADDING,
        'class_weights':   CLASS_WEIGHTS,
        'prompt_version':  'v2_structured',
    }
    return hashlib.md5(json.dumps(h, sort_keys=True).encode()).hexdigest()[:12]


# Load tokenized dataset from cache if available
def get_cached_dataset(cache_hash):
    # this function tries to load a tokenized dataset from disk using the provided cache hash
    cache_path = CACHE_DIR / f"tokenized_{cache_hash}"
    # Load cache only if folder exists
    if cache_path.exists():
        try:
            print(f"Caricamento dalla cache: {cache_hash}")
            ds = load_from_disk(str(cache_path))
            print(f"Cache: {len(ds['train'])} train, {len(ds['test'])} eval")
            return ds
        except Exception as e:
            print(f"Errore cache: {e}")
    return None


# Save tokenized dataset to cache for future runs
def save_dataset_to_cache(dataset, cache_hash):
    cache_path = CACHE_DIR / f"tokenized_{cache_hash}"
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_path))
        print(f"Dataset salvato in cache: {cache_hash}")
    except Exception as e:
        print(f"Errore salvataggio cache: {e}")


# Remove old cache folders and keep only recent ones
def clean_old_caches(keep_latest=3):
    # Skip if cache folder is missing
    if not CACHE_DIR.exists():
        return
    import shutil
    caches = sorted(CACHE_DIR.glob("tokenized_*"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    for old in caches[keep_latest:]:
        try:
            shutil.rmtree(old)
            print(f"Rimossa cache: {old.name}")
        except Exception:
            pass



# Load CSV, optionally augment it, then build training fields
def load_and_prepare_dataset(csv_path, apply_augmentation=True):
    print(f"Caricamento dataset da {csv_path}...")
    df = pd.read_csv(csv_path)

    # Apply optional data augmentation
    if apply_augmentation:
        df = augment_dataset(df, augmentation_factor=0.3)

    print(f"Dataset totale: {len(df)} esempi")

    df['input'] = df.apply(
        lambda row: build_prompt(
            str(row['prompt']) + " Helper suggestion: " + str(row['response'])
        ),
        axis=1
    )
    df['target'] = df['instructions'].astype(str).str.strip()

    return Dataset.from_pandas(df[['input', 'target']].reset_index(drop=True))


# Tokenize inputs and targets with task-specific padding and truncation
def preprocess_function(examples, tokenizer):
    padding_strategy = 'do_not_pad' if USE_DYNAMIC_PADDING else 'max_length'
    # Tokenizzed input and target separately to apply different max_length and padding
    model_inputs = tokenizer(
        examples['input'],
        max_length=MAX_INPUT_LENGTH,
        padding=padding_strategy,
        truncation=True,
    )
    # Tokenizzed target with special handling: replace pad_token_id with -100 for labels, and apply different max_length
    labels = tokenizer(
        examples['target'],
        max_length=MAX_TARGET_LENGTH,
        padding=padding_strategy,
        truncation=True,
    )


    labels_ids = [
        [l if l != tokenizer.pad_token_id else -100 for l in ex]
        for ex in labels['input_ids']
    ]
    model_inputs['labels'] = labels_ids
    return model_inputs


# Normalize predicted action text into canonical action tokens
def parse_action_sequence(text: str) -> list:
    text = str(text).strip().lower()
    text = text.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
    actions = [a.strip() for a in text.split(',') if a.strip()]

    normalized = []
    for a in actions:
        # Map variants to valid labels
        if 'forward' in a or 'move' in a:
            normalized.append('forward')
        elif 'left' in a:
            normalized.append('left')
        elif 'right' in a:
            normalized.append('right')
        elif 'pickup' in a or 'pick' in a:
            normalized.append('pickup')
        elif 'toggle' in a or 'open' in a:
            normalized.append('toggle')
    return normalized


# Evaluate model outputs and compute token-level confusion metrics
def compute_eval_confusion_metrics(model, tokenizer, tokenized_eval, batch_size=8):
    model.eval()
    eval_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )
    eval_loader = torch.utils.data.DataLoader(
        tokenized_eval,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collator,
    )

    action_to_idx = {a: i for i, a in enumerate(ACTION_LABELS)}
    cm = np.zeros((len(ACTION_LABELS), len(ACTION_LABELS)), dtype=np.int64)

    token_correct = 0
    token_total = 0
    exact_match_count = 0
    sequence_total = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            pred_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TARGET_LENGTH,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            labels = batch['labels'].clone()
            labels[labels == -100] = tokenizer.pad_token_id

            for i in range(pred_ids.size(0)):
                pred_text = tokenizer.decode(pred_ids[i], skip_special_tokens=True)
                true_text = tokenizer.decode(labels[i], skip_special_tokens=True)

                pred_actions = parse_action_sequence(pred_text)
                true_actions = parse_action_sequence(true_text)

                # Count exact sequence matches
                if pred_actions == true_actions:
                    exact_match_count += 1
                sequence_total += 1

                max_len = max(len(pred_actions), len(true_actions))
                # Skip empty pair sequences
                if max_len == 0:
                    continue

                for pos in range(max_len):
                    pred_a = pred_actions[pos] if pos < len(pred_actions) else None
                    true_a = true_actions[pos] if pos < len(true_actions) else None

                    # Count token-wise matches
                    if pred_a == true_a:
                        token_correct += 1
                    token_total += 1

                    # Update confusion matrix cells
                    if pred_a in action_to_idx and true_a in action_to_idx:
                        cm[action_to_idx[true_a], action_to_idx[pred_a]] += 1

    def _safe_div(a, b):
        return a / b if b > 0 else 0.0

    per_class = {}
    tp_sum = fp_sum = fn_sum = tn_sum = 0
    total_cm = int(cm.sum())

    for idx, action in enumerate(ACTION_LABELS):
        tp_i = int(cm[idx, idx])
        fn_i = int(cm[idx, :].sum() - tp_i)
        fp_i = int(cm[:, idx].sum() - tp_i)
        tn_i = int(total_cm - tp_i - fn_i - fp_i)

        precision_i = _safe_div(tp_i, tp_i + fp_i)
        recall_i = _safe_div(tp_i, tp_i + fn_i)
        f1_i = _safe_div(2 * precision_i * recall_i, precision_i + recall_i)

        per_class[action] = {
            'precision': float(precision_i),
            'recall': float(recall_i),
            'f1': float(f1_i),
            'support': int(cm[idx, :].sum()),
        }

        tp_sum += tp_i
        fp_sum += fp_i
        fn_sum += fn_i
        tn_sum += tn_i

    precision = _safe_div(tp_sum, tp_sum + fp_sum)
    recall = _safe_div(tp_sum, tp_sum + fn_sum)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(token_correct, token_total)
    exact_match_accuracy = _safe_div(exact_match_count, sequence_total)

    macro_precision = float(np.mean([per_class[a]['precision'] for a in ACTION_LABELS]))
    macro_recall = float(np.mean([per_class[a]['recall'] for a in ACTION_LABELS]))
    macro_f1 = float(np.mean([per_class[a]['f1'] for a in ACTION_LABELS]))

    return {
        'tp': int(tp_sum),
        'fp': int(fp_sum),
        'tn': int(tn_sum),
        'fn': int(fn_sum),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'exact_match_accuracy': float(exact_match_accuracy),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': per_class,
        'labels': ACTION_LABELS,
        'confusion_matrix': cm.tolist(),
    }


# Plot multiclass 5x5 confusion matrix with counts and percentages
def plot_multiclass_confusion_heatmap(confusion_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = confusion_metrics.get('labels', ACTION_LABELS)
    matrix = np.array(confusion_metrics.get('confusion_matrix', []), dtype=np.float64)

    # Abort when matrix is missing
    if matrix.size == 0:
        print("Matrice di confusione multiclass non disponibile.")
        return

    total = matrix.sum() if matrix.sum() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted action')
    ax.set_ylabel('True action')
    ax.set_title('Confusion Heatmap 5x5 (Token-level multiclass)')

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = int(matrix[r, c])
            pct = matrix[r, c] / total * 100.0
            ax.text(c, r, f"{value}\n({pct:.1f}%)", ha='center', va='center', color='black', fontsize=8)

    plt.tight_layout()
    out_path = output_dir / "confusion_heatmap_multiclass.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap multiclass salvata in {out_path}")


# Plot binary one-vs-rest confusion matrix and save image
def plot_confusion_heatmap(confusion_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    tp = confusion_metrics['tp']
    fp = confusion_metrics['fp']
    tn = confusion_metrics['tn']
    fn = confusion_metrics['fn']

    matrix = np.array([
        [tp, fn],
        [fp, tn],
    ], dtype=np.float64)

    total = matrix.sum() if matrix.sum() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Positive', 'Pred Negative'])
    ax.set_yticklabels(['True Positive', 'True Negative'])
    ax.set_title('Confusion Heatmap (Token-level one-vs-rest)')

    for r in range(2):
        for c in range(2):
            value = int(matrix[r, c])
            pct = matrix[r, c] / total * 100.0
            ax.text(c, r, f"{value}\n({pct:.1f}%)", ha='center', va='center', color='black')

    plt.tight_layout()
    out_path = output_dir / "confusion_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Heatmap salvata in {out_path}")


# Plot final percentage metrics for accuracy, F1, precision, and recall
def plot_final_percentage_metrics(confusion_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    metrics_vals = [
        confusion_metrics['accuracy'] * 100.0,
        confusion_metrics['f1'] * 100.0,
        confusion_metrics['precision'] * 100.0,
        confusion_metrics['recall'] * 100.0,
    ]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics_names, metrics_vals, color=colors, alpha=0.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Final Metrics (%)')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars, metrics_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, min(val + 1.5, 99.0),
                f"{val:.1f}%", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "final_metrics_percent.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Barplot metriche finali salvato in {out_path}")



# Plot training and evaluation loss trends over optimization steps
def plot_training_metrics(trainer, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    log_history = trainer.state.log_history
    train_logs = [l for l in log_history if 'loss' in l and 'eval_loss' not in l]
    eval_logs = [l for l in log_history if 'eval_loss' in l]

    # Stop if no logged losses
    if not train_logs and not eval_logs:
        print("Nessun log disponibile per il plot.")
        return

    plt.figure(figsize=(10, 6))

    # Train loss
    # Plot train loss when available
    if train_logs:
        train_steps = [l.get('step', i) for i, l in enumerate(train_logs)]
        train_loss = [l['loss'] for l in train_logs]
        plt.plot(train_steps, train_loss, label="Train Loss", linewidth=2)

    # Eval loss
    # Plot eval loss when available
    if eval_logs:
        eval_steps = [l.get('step', i) for i, l in enumerate(eval_logs)]
        eval_loss = [l['eval_loss'] for l in eval_logs]
        plt.plot(eval_steps, eval_loss, label="Eval Loss", linewidth=2, marker="o")

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = output_dir / "loss_train_eval.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot unificato salvato in {out_path}")




# Run full fine-tuning pipeline, evaluation, plots, and artifact export
def main():
    print("=" * 60)
    print("REVIEWER FINE-TUNING â€” FLAN-T5 â€” RTX 3050 4GB")
    print("=" * 60)
    print(f"Batch: {BATCH_SIZE} Ã— {GRADIENT_ACCUMULATION} = {BATCH_SIZE*GRADIENT_ACCUMULATION} effettivo")
    print(f"Seq lengths: input={MAX_INPUT_LENGTH}, target={MAX_TARGET_LENGTH}")
    print(f"Precision: bf16={USE_BF16} (fp16 disabilitato: NaN su T5+gc) | Grad Checkpointing: {USE_GRADIENT_CHECKPOINTING}")
    print(f"Workers: {DATALOADER_NUM_WORKERS} | Pin Memory: {PIN_MEMORY}")
    print(f"Datestamp: {DATESTAMP}\n")

    output_model_dir = OUTPUT_DIR / f"{DATESTAMP}_weighted"
    plots_output_dir = PLOTS_DIR  / f"balanced_{DATESTAMP}"
    output_model_dir.mkdir(parents=True, exist_ok=True)
    plots_output_dir.mkdir(parents=True, exist_ok=True)

    # Prune old tokenization caches
    if USE_DATASET_CACHE:
        clean_old_caches(keep_latest=3)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    cache_hash = get_dataset_hash(
        DATASET_PATH, augmentation=True,
        model_name=MODEL_NAME,
        max_input_len=MAX_INPUT_LENGTH,
        max_target_len=MAX_TARGET_LENGTH
    )

    tokenized_dataset = get_cached_dataset(cache_hash) if USE_DATASET_CACHE else None
    # Reuse cached tokenized data if found
    if tokenized_dataset:
        print("Dataset caricato dalla cache\n")
    else:
        print("Processamento dataset...")
        dataset = load_and_prepare_dataset(DATASET_PATH, apply_augmentation=True)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        print(f"Train: {len(dataset['train'])} | Val: {len(dataset['test'])}")

        n_proc = min(4, os.cpu_count() or 1)
        print(f"Tokenizzazione con {n_proc} processi...")
        tokenize_fn = lambda x: preprocess_function(x, tokenizer)

        tokenized_train = dataset['train'].map(
            tokenize_fn, batched=True, num_proc=n_proc,
            remove_columns=dataset['train'].column_names, desc="Tokenizing train"
        )
        tokenized_eval = dataset['test'].map(
            tokenize_fn, batched=True, num_proc=n_proc,
            remove_columns=dataset['test'].column_names, desc="Tokenizing eval"
        )
        tokenized_dataset = DatasetDict({'train': tokenized_train, 'test': tokenized_eval})
        # Store tokenized splits in cache
        if USE_DATASET_CACHE:
            save_dataset_to_cache(tokenized_dataset, cache_hash)
        print("Dataset tokenizzato\n")

    tokenized_train = tokenized_dataset['train']
    tokenized_eval  = tokenized_dataset['test']
    print(f"Pronto: {len(tokenized_train)} train | {len(tokenized_eval)} eval\n")


    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    ).to(device)


    # Toggle gradient checkpointing mode
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing abilitato\n")
    else:
        model.gradient_checkpointing_disable()
        print("Gradient checkpointing disabilitato (modalita' speed)\n")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if USE_DYNAMIC_PADDING else None,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=str(output_model_dir),
        overwrite_output_dir=True,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,   
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",

        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,

        logging_dir=str(output_model_dir / "logs"),
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        report_to="none",

        fp16=USE_FP16,
        bf16=USE_BF16,
        optim="adamw_torch_fused",

        group_by_length=True,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=PIN_MEMORY,

        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        max_grad_norm=1.0,

        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        auto_find_batch_size=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        action_weights=CLASS_WEIGHTS,
        callbacks=[FirstStepEvalCallback()],
    )

    print("Inizio fine-tuning...\n")
    try:
        trainer.train()
        print("\n Training completato!")
    except RuntimeError as e:
        # Handle CUDA out-of-memory errors
        if "out of memory" in str(e).lower():
            print("\n OOM! Prova: BATCH_SIZE=1 oppure MAX_INPUT_LENGTH=64")
            raise
        raise

    plot_training_metrics(trainer, plots_output_dir)
    trainer.save_model(str(output_model_dir))
    tokenizer.save_pretrained(str(output_model_dir))

    metrics = trainer.evaluate()
    print(f"\nValidation Loss: {metrics['eval_loss']:.4f}")

    confusion_metrics = compute_eval_confusion_metrics(
        trainer.model,
        tokenizer,
        tokenized_eval,
        batch_size=max(1, BATCH_SIZE)
    )
    plot_confusion_heatmap(confusion_metrics, plots_output_dir)
    plot_final_percentage_metrics(confusion_metrics, plots_output_dir)

    with open(output_model_dir / "metrics.txt", "w") as f:
        f.write(f"Datestamp: {DATESTAMP}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Hardware: RTX 3050 4GB + 12700H + 16GB RAM\n")
        f.write(f"Batch: {BATCH_SIZE} Ã— {GRADIENT_ACCUMULATION} = {BATCH_SIZE*GRADIENT_ACCUMULATION}\n")
        f.write(f"MAX_INPUT_LENGTH: {MAX_INPUT_LENGTH}\n")
        f.write(f"MAX_TARGET_LENGTH: {MAX_TARGET_LENGTH}\n")
        f.write(f"Precision: fp16\n")
        f.write(f"Gradient Checkpointing: ON\n")
        f.write(f"Class Weights: {CLASS_WEIGHTS}\n")
        f.write(f"Final Eval Loss: {metrics['eval_loss']:.4f}\n")
        f.write(f"TP: {confusion_metrics['tp']}\n")
        f.write(f"FP: {confusion_metrics['fp']}\n")
        f.write(f"TN: {confusion_metrics['tn']}\n")
        f.write(f"FN: {confusion_metrics['fn']}\n")
        f.write(f"Token Accuracy: {confusion_metrics['accuracy']*100:.2f}%\n")
        f.write(f"Exact Sequence Accuracy: {confusion_metrics['exact_match_accuracy']*100:.2f}%\n")
        f.write(f"Micro Precision: {confusion_metrics['precision']*100:.2f}%\n")
        f.write(f"Micro Recall: {confusion_metrics['recall']*100:.2f}%\n")
        f.write(f"Micro F1: {confusion_metrics['f1']*100:.2f}%\n")
        f.write(f"Macro Precision: {confusion_metrics['macro_precision']*100:.2f}%\n")
        f.write(f"Macro Recall: {confusion_metrics['macro_recall']*100:.2f}%\n")
        f.write(f"Macro F1: {confusion_metrics['macro_f1']*100:.2f}%\n")
        f.write("\nPer-class metrics:\n")
        for action in ACTION_LABELS:
            m = confusion_metrics['per_class'][action]
            f.write(
                f"- {action}: P={m['precision']*100:.2f}% "
                f"R={m['recall']*100:.2f}% "
                f"F1={m['f1']*100:.2f}% "
                f"Support={m['support']}\n"
            )

    print("\nFINE-TUNING COMPLETATO!")
    print(f"Modello: {output_model_dir}")


# Script entry point
if __name__ == "__main__":
    main()
