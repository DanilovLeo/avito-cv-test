"""
Инференс на тесте, запись submission.csv и построение псевдоразмеченного набора
для self-training на тестовых данных.

Self-training: модель уверена на части теста (p близко к 0 или 1). Кропы с p <= 1-thr
берём как есть, кропы с p >= thr физически поворачиваем на 180° (становятся
"правильными") и сохраняем в data/test_pseudo/. Дальше они используются как
обычные "правильные" картинки — датасет сам их переворачивает при обучении.
Это адаптация к домену теста (кириллица, сканы, фото упаковок) без ручной разметки.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from train import predict_logits, combine


def predict_test(model, test_dir: str | Path, H: int, W: int, T: float = 1.0, tta: bool = True):
    """Возвращает DataFrame image_id / p_180 (image_id без расширения, как в sample_submission)."""
    paths = sorted(Path(test_dir).glob("*.png"))
    lo, lr = predict_logits(model, paths, H, W)
    p = combine(lo, lr, T=T, tta=tta)
    return pd.DataFrame({"image_id": [p_.stem for p_ in paths], "p_180": p}), lo, lr


def write_submission(df: pd.DataFrame, sample_csv: str | Path, out_csv: str | Path):
    sample = pd.read_csv(sample_csv)
    df = sample[["image_id"]].merge(df, on="image_id", how="left")
    assert df.p_180.notna().all(), "не для всех image_id есть предсказание"
    df["p_180"] = df.p_180.clip(0, 1)
    df.to_csv(out_csv, index=False)
    return df


def build_pseudo_set(df: pd.DataFrame, test_dir: str | Path, out_dir: str | Path, thr: float = 0.9) -> list[Path]:
    """df: image_id / p_180 на тесте. Сохраняет уверенные кропы в правильной ориентации."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for image_id, p in zip(df.image_id, df.p_180):
        if p <= 1 - thr:
            img = Image.open(Path(test_dir) / f"{image_id}.png")
        elif p >= thr:
            img = Image.open(Path(test_dir) / f"{image_id}.png").rotate(180)
        else:
            continue
        dst = out / f"{image_id}.png"
        img.convert("RGB").save(dst)
        kept.append(dst)
    print(f"pseudo set: {len(kept)} of {len(df)} (thr={thr}); "
          f"upright={int((df.p_180 <= 1 - thr).sum())} rotated={int((df.p_180 >= thr).sum())}")
    return kept
