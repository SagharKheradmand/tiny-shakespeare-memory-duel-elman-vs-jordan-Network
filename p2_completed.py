r"""
Tiny Shakespeare Memory Duel - Completed p2.py
Assignment 2: Elman Network vs Jordan Network for character-level next-character prediction.

What this script does:
1) Uses only a configurable fraction of Tiny Shakespeare. Default: 30%.
2) Implements Elman Network: hidden-state feedback.
3) Implements Jordan Network: output-distribution feedback.
4) Trains both models under the same settings.
5) Reports train/validation loss, character accuracy, and perplexity.
6) Generates text from both models using the same seed text.
7) Tests temperatures 0.4, 0.8, and 1.2.
8) Saves plots, CSV history, best models, generated samples, and a report summary.

Example PowerShell run on your system:
& "c:\Users\Administrator\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  "C:\Users\Administrator\Downloads\p2_completed.py" `
  --data_path "C:\Users\Administrator\Downloads\tiny-shakespeare(1).txt"

Fast test run:
& "c:\Users\Administrator\AppData\Local\Python\pythoncore-3.14-64\python.exe" `
  "C:\Users\Administrator\Downloads\p2_completed.py" `
  --data_path "C:\Users\Administrator\Downloads\tiny-shakespeare(1).txt" `
  --epochs 3 --hidden_size 128 --steps_per_epoch 100 --batch_size 64
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# Configuration
# ============================================================


@dataclass
class Config:
    data_path: str = "tiny-shakespeare.txt"
    results_dir: str = "results_memory_duel_p2_completed"

    data_fraction: float = 0.30
    train_ratio: float = 0.90

    seed_text: str = "ROMEO:\n"
    temperatures: Tuple[float, ...] = (0.4, 0.8, 1.2)

    embedding_dim: int = 64
    hidden_size: int = 128
    seq_len: int = 100
    batch_size: int = 64
    epochs: int = 15
    steps_per_epoch: int = 100
    eval_iters: int = 30
    learning_rate: float = 0.001
    generate_length: int = 500
    grad_clip: float = 1.0

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Tiny Shakespeare Memory Duel: Elman vs Jordan RNN"
    )
    parser.add_argument("--data_path", type=str, default=None, help="Path to tiny-shakespeare txt file")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory for outputs")
    parser.add_argument("--data_fraction", type=float, default=None, help="Fraction of dataset to use, e.g. 0.30")
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio")
    parser.add_argument("--seed_text", type=str, default=None, help="Seed text for generation")
    parser.add_argument("--temperatures", type=float, nargs="+", default=None, help="Temperatures, e.g. 0.4 0.8 1.2")
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--eval_iters", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--generate_length", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])

    args = parser.parse_args()
    cfg = Config()

    for key, value in vars(args).items():
        if value is not None:
            if key == "temperatures":
                setattr(cfg, key, tuple(value))
            else:
                setattr(cfg, key, value)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        cfg.device = "cpu"

    return cfg


# ============================================================
# Reproducibility and paths
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_path(path_str: str) -> Path:
    """Find the dataset robustly.

    The original p2.py only searched for tiny-shakespeare.txt. This version also
    checks names commonly produced by browsers, including tiny-shakespeare(1).txt.
    """
    candidate = Path(path_str).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate

    script_dir = Path(__file__).resolve().parent
    common_names = [
        path_str,
        "tiny-shakespeare.txt",
        "tiny-shakespeare(1).txt",
        "tiny-shakespeare (1).txt",
        "tiny_shakespeare.txt",
        "input.txt",
    ]

    for name in common_names:
        p = script_dir / name
        if p.exists() and p.is_file():
            return p

    # Last resort: search near the script for a Shakespeare txt file.
    matches = sorted(script_dir.glob("*shakespeare*.txt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Dataset not found: {path_str}\n"
        "Put the Tiny Shakespeare .txt file beside this script or pass its full path with --data_path.\n"
        "Example: --data_path \"C:\\Users\\Administrator\\Downloads\\tiny-shakespeare(1).txt\""
    )


# ============================================================
# Dataset helpers
# ============================================================


def load_text(path: str, fraction: float) -> Tuple[str, Path]:
    if not (0 < fraction <= 1):
        raise ValueError("data_fraction must be between 0 and 1.")

    data_path = resolve_data_path(path)
    text = data_path.read_text(encoding="utf-8")
    cut = max(2, int(len(text) * fraction))
    return text[:cut], data_path


def build_vocab(text: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(indices: torch.Tensor | List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in indices)


def train_val_split(data: torch.Tensor, train_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be between 0 and 1.")
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError("Data is too short for the selected seq_len. Lower --seq_len or use more data.")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in starts])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in starts])
    return x.to(device), y.to(device)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean().item()


def safe_perplexity(loss: float) -> float:
    # Prevent overflow if loss becomes unexpectedly huge.
    return float(math.exp(min(loss, 50.0)))


# ============================================================
# Elman Network
# ============================================================


class ElmanNetwork(nn.Module):
    """
    Elman Network: hidden-state feedback.

    h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
    y_t = W_hy h_t
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.xh = nn.Linear(embedding_dim, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.hy = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        batch_size, seq_len = x.shape

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        emb = self.embedding(x)
        logits_list = []

        for t in range(seq_len):
            x_t = emb[:, t, :]
            hidden = torch.tanh(self.xh(x_t) + self.hh(hidden))
            logits = self.hy(hidden)
            logits_list.append(logits.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)
        return logits, hidden


# ============================================================
# Jordan Network
# ============================================================


class JordanNetwork(nn.Module):
    """
    Jordan Network: output-distribution feedback.

    h_t = tanh(W_xh x_t + W_yh y_{t-1} + b)
    y_t = W_hy h_t

    Important improvement over the original p2.py:
    we do NOT detach prev_output during training. This keeps the Jordan feedback
    path differentiable and makes the comparison more faithful.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.xh = nn.Linear(embedding_dim, hidden_size)
        self.yh = nn.Linear(vocab_size, hidden_size)
        self.hy = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, prev_output: Optional[torch.Tensor] = None):
        batch_size, seq_len = x.shape

        if prev_output is None:
            # Uniform distribution is a neutral initial output feedback.
            prev_output = torch.full(
                (batch_size, self.vocab_size),
                fill_value=1.0 / self.vocab_size,
                device=x.device,
            )

        emb = self.embedding(x)
        logits_list = []

        for t in range(seq_len):
            x_t = emb[:, t, :]
            hidden = torch.tanh(self.xh(x_t) + self.yh(prev_output))
            logits = self.hy(hidden)
            prev_output = torch.softmax(logits, dim=-1)
            logits_list.append(logits.unsqueeze(1))

        logits = torch.cat(logits_list, dim=1)
        return logits, prev_output


# ============================================================
# Evaluation
# ============================================================


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: torch.Tensor,
    criterion: nn.Module,
    batch_size: int,
    seq_len: int,
    device: str,
    eval_iters: int,
) -> Tuple[float, float, float]:
    model.eval()

    total_loss = 0.0
    total_acc = 0.0

    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        acc = calculate_accuracy(logits, y)

        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / eval_iters
    avg_acc = total_acc / eval_iters
    ppl = safe_perplexity(avg_loss)

    model.train()
    return avg_loss, avg_acc, ppl


# ============================================================
# Text generation
# ============================================================


@torch.no_grad()
def generate_text(
    model: nn.Module,
    seed_text: str,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    length: int,
    temperature: float,
    device: str,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    model.eval()

    # Keep only characters known by the vocabulary.
    filtered_seed = "".join(ch for ch in seed_text if ch in stoi)
    if not filtered_seed:
        filtered_seed = next(iter(stoi.keys()))

    generated = filtered_seed
    input_ids = torch.tensor([[stoi[ch] for ch in filtered_seed]], dtype=torch.long, device=device)

    logits, state = model(input_ids)
    current_id = input_ids[:, -1:]

    for _ in range(length):
        logits, state = model(current_id, state)
        last_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_char = itos[int(next_id.item())]

        generated += next_char
        current_id = next_id

    model.train()
    return generated


# ============================================================
# Training
# ============================================================


def train_model(
    model_name: str,
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    vocab_size: int,
    cfg: Config,
    results_dir: Path,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    print(f"\n{'=' * 80}")
    print(f"Training {model_name} Network")
    print(f"{'=' * 80}")

    model = model.to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "train_ppl": [],
        "val_loss": [],
        "val_acc": [],
        "val_ppl": [],
        "epoch_time_sec": [],
    }

    best_val_loss = float("inf")
    best_path = results_dir / f"best_{model_name.lower()}_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        start_time = time.time()
        model.train()

        total_train_loss = 0.0
        total_train_acc = 0.0

        for step in range(cfg.steps_per_epoch):
            x, y = get_batch(train_data, cfg.batch_size, cfg.seq_len, cfg.device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            acc = calculate_accuracy(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc

        train_loss = total_train_loss / cfg.steps_per_epoch
        train_acc = total_train_acc / cfg.steps_per_epoch
        train_ppl = safe_perplexity(train_loss)

        val_loss, val_acc, val_ppl = evaluate(
            model=model,
            data=val_data,
            criterion=criterion,
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            device=cfg.device,
            eval_iters=cfg.eval_iters,
        )

        epoch_time = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ppl"].append(val_ppl)
        history["epoch_time_sec"].append(epoch_time)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            best_note = " | best saved"
        else:
            best_note = ""

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train PPL: {train_ppl:.2f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val PPL: {val_ppl:.2f} | "
            f"Time: {epoch_time:.1f}s{best_note}"
        )

    # Load best weights before returning, so generation uses best validation model.
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=cfg.device))

    final_path = results_dir / f"final_{model_name.lower()}_model.pt"
    torch.save(model.state_dict(), final_path)

    return model, history


# ============================================================
# Saving plots and reports
# ============================================================


def save_history_csv(histories: Dict[str, Dict[str, List[float]]], results_dir: Path) -> None:
    output_path = results_dir / "training_history.csv"
    fieldnames = ["model", "epoch", "train_loss", "train_acc", "train_ppl", "val_loss", "val_acc", "val_ppl", "epoch_time_sec"]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_name, history in histories.items():
            num_epochs = len(history["train_loss"])
            for idx in range(num_epochs):
                writer.writerow(
                    {
                        "model": model_name,
                        "epoch": idx + 1,
                        "train_loss": history["train_loss"][idx],
                        "train_acc": history["train_acc"][idx],
                        "train_ppl": history["train_ppl"][idx],
                        "val_loss": history["val_loss"][idx],
                        "val_acc": history["val_acc"][idx],
                        "val_ppl": history["val_ppl"][idx],
                        "epoch_time_sec": history["epoch_time_sec"][idx],
                    }
                )


def plot_results(histories: Dict[str, Dict[str, List[float]]], results_dir: Path) -> None:
    metrics = [
        ("train_loss", "Training Loss"),
        ("val_loss", "Validation Loss"),
        ("train_acc", "Training Character Accuracy"),
        ("val_acc", "Validation Character Accuracy"),
        ("train_ppl", "Training Perplexity"),
        ("val_ppl", "Validation Perplexity"),
    ]

    for metric, title in metrics:
        plt.figure(figsize=(8, 5))
        for model_name, history in histories.items():
            plt.plot(range(1, len(history[metric]) + 1), history[metric], marker="o", label=model_name)
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"{metric}.png", dpi=200)
        plt.close()


def save_generated_samples(
    trained_models: Dict[str, nn.Module],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    cfg: Config,
    results_dir: Path,
) -> Dict[str, Dict[float, str]]:
    samples: Dict[str, Dict[float, str]] = {}
    samples_dir = results_dir / "generated_samples"
    samples_dir.mkdir(exist_ok=True)

    for model_name, model in trained_models.items():
        samples[model_name] = {}
        for temp in cfg.temperatures:
            sample = generate_text(
                model=model,
                seed_text=cfg.seed_text,
                stoi=stoi,
                itos=itos,
                length=cfg.generate_length,
                temperature=temp,
                device=cfg.device,
            )
            samples[model_name][temp] = sample
            output_file = samples_dir / f"{model_name.lower()}_temperature_{temp}.txt"
            output_file.write_text(sample, encoding="utf-8")

    return samples


def final_metrics_table(histories: Dict[str, Dict[str, List[float]]]) -> str:
    lines = []
    lines.append("| Model | Train Loss | Train Acc | Train PPL | Val Loss | Val Acc | Val PPL | Avg Epoch Time |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model_name, h in histories.items():
        avg_time = sum(h["epoch_time_sec"]) / len(h["epoch_time_sec"])
        lines.append(
            f"| {model_name} | "
            f"{h['train_loss'][-1]:.4f} | {h['train_acc'][-1]:.4f} | {h['train_ppl'][-1]:.2f} | "
            f"{h['val_loss'][-1]:.4f} | {h['val_acc'][-1]:.4f} | {h['val_ppl'][-1]:.2f} | {avg_time:.1f}s |"
        )
    return "\n".join(lines)


def write_report(
    cfg: Config,
    data_path: Path,
    text_length: int,
    vocab_size: int,
    train_size: int,
    val_size: int,
    histories: Dict[str, Dict[str, List[float]]],
    samples: Dict[str, Dict[float, str]],
    results_dir: Path,
) -> None:
    report_path = results_dir / "report_summary.md"

    lines: List[str] = []
    lines.append("# Tiny Shakespeare Memory Duel - Elman vs Jordan\n")
    lines.append("\n## Dataset and Experimental Setup\n")
    lines.append(f"- Dataset path: `{data_path}`\n")
    lines.append(f"- Dataset fraction used: **{cfg.data_fraction * 100:.0f}%**\n")
    lines.append(f"- Characters used: {text_length:,}\n")
    lines.append(f"- Vocabulary size: {vocab_size}\n")
    lines.append(f"- Train characters: {train_size:,}\n")
    lines.append(f"- Validation characters: {val_size:,}\n")
    lines.append(f"- Sequence length: {cfg.seq_len}\n")
    lines.append(f"- Batch size: {cfg.batch_size}\n")
    lines.append(f"- Epochs: {cfg.epochs}\n")
    lines.append(f"- Steps per epoch: {cfg.steps_per_epoch}\n")
    lines.append(f"- Optimizer: Adam, learning rate = {cfg.learning_rate}\n")
    lines.append(f"- Seed text: `{cfg.seed_text!r}`\n")
    lines.append(f"- Temperatures: {list(cfg.temperatures)}\n")
    lines.append(f"- Device: {cfg.device}\n")

    lines.append("\n## Architecture Summary\n")
    lines.append("- **Elman Network:** uses feedback from the previous hidden state.\n")
    lines.append("- **Jordan Network:** uses feedback from the previous output probability distribution.\n")
    lines.append("- Both models use the same embedding size, hidden size, optimizer, sequence length, batch size, and train/validation split.\n")

    lines.append("\n## Final Metrics\n")
    lines.append(final_metrics_table(histories))
    lines.append("\n")

    lines.append("\n## Convergence, Stability, and Validation Performance\n")
    elman = histories.get("Elman")
    jordan = histories.get("Jordan")
    if elman and jordan:
        better_val = "Elman" if elman["val_loss"][-1] < jordan["val_loss"][-1] else "Jordan"
        better_acc = "Elman" if elman["val_acc"][-1] > jordan["val_acc"][-1] else "Jordan"
        lines.append(f"- Lower final validation loss: **{better_val}**.\n")
        lines.append(f"- Higher final validation character accuracy: **{better_acc}**.\n")
        lines.append("- Compare the saved validation-loss, validation-accuracy, and validation-perplexity plots for convergence speed and stability.\n")
        lines.append("- A smoother validation curve indicates more stable learning; lower PPL indicates better next-character probability estimates.\n")

    lines.append("\n## Temperature Effect on Generation\n")
    lines.append("- Temperature 0.4 usually produces more conservative and repetitive text.\n")
    lines.append("- Temperature 0.8 usually balances coherence and diversity.\n")
    lines.append("- Temperature 1.2 usually increases variety but can introduce more noise and spelling/structure errors.\n")

    lines.append("\n## Generated Text Samples\n")
    for model_name, model_samples in samples.items():
        for temp, sample in model_samples.items():
            preview = sample[:1000]
            lines.append(f"\n### {model_name}, temperature={temp}\n")
            lines.append("```text\n")
            lines.append(preview)
            if len(sample) > len(preview):
                lines.append("\n...\n")
            lines.append("\n```\n")

    lines.append("\n## Saved Output Files\n")
    lines.append("- `training_history.csv`: numeric history for both models.\n")
    lines.append("- `train_loss.png`, `val_loss.png`, `train_acc.png`, `val_acc.png`, `train_ppl.png`, `val_ppl.png`: comparison plots.\n")
    lines.append("- `generated_samples/`: generated samples for every model and temperature.\n")
    lines.append("- `best_elman_model.pt`, `best_jordan_model.pt`: best models according to validation loss.\n")

    report_path.write_text("".join(lines), encoding="utf-8")


def save_config(cfg: Config, results_dir: Path) -> None:
    config_path = results_dir / "config.json"
    data = asdict(cfg)
    data["temperatures"] = list(cfg.temperatures)
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# Main
# ============================================================


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Tiny Shakespeare Memory Duel - Completed p2.py")
    print("=" * 80)
    print(f"Device: {cfg.device}")
    print(f"Dataset fraction: {cfg.data_fraction:.2f} = {cfg.data_fraction * 100:.0f}%")

    text, actual_data_path = load_text(cfg.data_path, cfg.data_fraction)
    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)

    data = encode(text, stoi)
    train_data, val_data = train_val_split(data, cfg.train_ratio)

    print(f"Dataset path: {actual_data_path}")
    print(f"Characters used: {len(text):,}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train characters: {len(train_data):,} | Validation characters: {len(val_data):,}")

    save_config(cfg, results_dir)

    elman_model = ElmanNetwork(vocab_size, cfg.embedding_dim, cfg.hidden_size)
    jordan_model = JordanNetwork(vocab_size, cfg.embedding_dim, cfg.hidden_size)

    trained_models: Dict[str, nn.Module] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}

    trained_models["Elman"], histories["Elman"] = train_model(
        model_name="Elman",
        model=elman_model,
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        cfg=cfg,
        results_dir=results_dir,
    )

    trained_models["Jordan"], histories["Jordan"] = train_model(
        model_name="Jordan",
        model=jordan_model,
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        cfg=cfg,
        results_dir=results_dir,
    )

    save_history_csv(histories, results_dir)
    plot_results(histories, results_dir)

    print("\nGenerating text samples...")
    samples = save_generated_samples(trained_models, stoi, itos, cfg, results_dir)

    write_report(
        cfg=cfg,
        data_path=actual_data_path,
        text_length=len(text),
        vocab_size=vocab_size,
        train_size=len(train_data),
        val_size=len(val_data),
        histories=histories,
        samples=samples,
        results_dir=results_dir,
    )

    print("\nTraining finished successfully.")
    print(f"Results saved to: {results_dir.resolve()}")
    print("Important files:")
    print(f"- {results_dir / 'report_summary.md'}")
    print(f"- {results_dir / 'training_history.csv'}")
    print(f"- {results_dir / 'generated_samples'}")
    print(f"- {results_dir / 'best_elman_model.pt'}")
    print(f"- {results_dir / 'best_jordan_model.pt'}")


if __name__ == "__main__":
    main()
