"""
Обучение, валидация, инференс с TTA и калибровка температуры.

Симметричная TTA: для правильной модели p(x) + p(rot180(x)) = 1, поэтому
    p_final = 0.5 * (p(x) + 1 - p(rot180(x)))
усредняет ошибки модели по двум ориентациям и гарантирует симметрию предсказаний.

Температура T подбирается по реальной валидации минимизацией NLL; она напрямую
улучшает Brier score, не меняя ранжирование.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from data import OrientationDataset, make_windows, rot180
from model import build_model, count_params


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init(worker_id: int):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


# ----------------------------------------------------------------------------- metrics

def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(
        brier=float(np.mean((p - y) ** 2)),
        score=float(1 - np.mean((p - y) ** 2)),
        logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        acc=float(np.mean((p > 0.5) == (y > 0.5))),
    )


# ----------------------------------------------------------------------------- train

@dataclass
class TrainConfig:
    model: str = "tiny"
    width: float = 1.0
    H: int = 32
    W: int = 256
    epochs: int = 8
    batch_size: int = 256
    lr: float = 3e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    amp: bool = True
    out_dir: str = "runs/tiny"
    init_from: str | None = None  # чекпоинт для дообучения (self-training)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (логиты, метки) без TTA — для мониторинга по эпохам."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def train(cfg: TrainConfig, train_paths: list[Path], val_sets: dict[str, list[Path]], select_on: str = "real"):
    """Обучает модель, каждую эпоху считает метрики на всех val_sets,
    сохраняет лучший чекпоинт по Brier на val_sets[select_on]."""
    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    g = torch.Generator().manual_seed(cfg.seed)
    train_ds = OrientationDataset(train_paths, train=True, H=cfg.H, W=cfg.W, seed=cfg.seed)
    train_dl = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                          pin_memory=True, drop_last=True, generator=g, worker_init_fn=_worker_init,
                          persistent_workers=cfg.num_workers > 0)
    val_dls = {k: DataLoader(OrientationDataset(v, train=False, H=cfg.H, W=cfg.W, seed=cfg.seed),
                             cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
               for k, v in val_sets.items()}

    model = build_model(cfg.model, **({"width": cfg.width} if cfg.model == "tiny" else {})).to(device)
    if cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location=device)["state_dict"])
    print(f"model={cfg.model} params={count_params(model):,} train={len(train_ds):,} "
          + " ".join(f"val_{k}={len(v):,}" for k, v in val_sets.items()))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    history, best = [], float("inf")
    for ep in range(cfg.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        row = {"epoch": ep + 1, "train_loss": tot / n, "time": time.time() - t0}
        for k, dl in val_dls.items():
            lg, yy = evaluate(model, dl, device)
            m = metrics(1 / (1 + np.exp(-lg)), yy)
            row.update({f"{k}_brier": m["brier"], f"{k}_acc": m["acc"]})
        history.append(row)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if row[f"{select_on}_brier"] < best:
            best = row[f"{select_on}_brier"]
            torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict()}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    return model, history


def load_model(path: str | Path, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location=device)
    cfg = TrainConfig(**ck["cfg"])
    model = build_model(cfg.model, **({"width": cfg.width} if cfg.model == "tiny" else {})).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


# ----------------------------------------------------------------------------- inference

@torch.no_grad()
def predict_logits(model, paths: list[Path], H: int, W: int, max_windows: int = 3,
                   batch_size: int = 512, device: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой картинки: логиты по окнам для исходной и повёрнутой версии.
    Возвращает (logit_orig, logit_rot) — усреднённые по окнам, форма [N] каждая.
    Полный симметричный TTA собирается снаружи: p = 0.5*(σ(l_o) + 1 - σ(l_r))."""
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    lo, lr = np.zeros(len(paths)), np.zeros(len(paths))
    buf, owner = [], []

    def flush():
        x = torch.stack(buf).to(device)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            a = model(x).float().cpu().numpy()
            b = model(rot180(x)).float().cpu().numpy()
        for j, i in enumerate(owner):
            lo[i] += a[j]
            lr[i] += b[j]
        buf.clear(); owner.clear()

    counts = np.zeros(len(paths))
    for i, p in enumerate(paths):
        ws = make_windows(p, H, W, max_windows)
        counts[i] = len(ws)
        buf.extend(ws); owner.extend([i] * len(ws))
        if len(buf) >= batch_size:
            flush()
    if buf:
        flush()
    return lo / counts, lr / counts


def combine(lo: np.ndarray, lr: np.ndarray, T: float = 1.0, tta: bool = True) -> np.ndarray:
    """Симметричная TTA + температура. tta=False -> только исходная ориентация."""
    p_o = 1 / (1 + np.exp(-lo / T))
    if not tta:
        return p_o
    p_r = 1 / (1 + np.exp(-lr / T))
    return 0.5 * (p_o + (1 - p_r))


def fit_temperature(lo: np.ndarray, lr: np.ndarray, y: np.ndarray, grid=np.linspace(0.3, 5, 95)) -> float:
    """Температура, минимизирующая NLL симметричных предсказаний на валидации."""
    best_T, best_nll = 1.0, float("inf")
    for T in grid:
        p = np.clip(combine(lo, lr, T), 1e-6, 1 - 1e-6)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_T, best_nll = float(T), nll
    return best_T


def val_labels(n: int, seed: int) -> np.ndarray:
    """Те же детерминированные метки, что и в OrientationDataset(train=False)."""
    return np.array([(i * 2654435761 + seed) % 2 == 1 for i in range(n)], dtype=float)


@torch.no_grad()
def predict_val(model, paths: list[Path], H: int, W: int, seed: int):
    """Логиты (orig, rot) на валидации С УЧЁТОМ детерминированных переворотов.
    Переворот картинки просто меняет местами роли lo и lr."""
    lo, lr = predict_logits(model, paths, H, W)
    y = val_labels(len(paths), seed)
    flip = y > 0.5
    lo2 = np.where(flip, lr, lo)
    lr2 = np.where(flip, lo, lr)
    return lo2, lr2, y
