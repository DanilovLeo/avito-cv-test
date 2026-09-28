"""
Данные для классификации ориентации кропа текста (0° / 180°).

Ключевая идея: размеченных данных нет, но метка получается бесплатно —
любой кроп в правильной ориентации, повёрнутый на 180°, это пример класса 1.
Поэтому датасет получает только "правильные" картинки и сам переворачивает
половину из них.

Препроцессинг:
  * grayscale (цвет не несёт информации об ориентации);
  * resize до высоты H с сохранением пропорций;
  * ширина фиксируется = W: короткие кропы дополняются справа, у длинных берётся
    окно шириной W (случайное при обучении, несколько равномерных на инференсе —
    ориентация текста локальна, поэтому любое окно несёт ту же метку);
  * нормировка в [-1, 1].

Поворот на 180° = отражение по обеим осям, применяется к уже готовому тензору,
так что train / val / TTA используют ровно одну и ту же операцию.
"""
from __future__ import annotations

import math
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(d: str | Path) -> list[Path]:
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXTS)


def load_gray(path: str | Path) -> np.ndarray:
    # PIL вместо cv2.imread: у части кропов TextOCR битые ICC-профили, и libpng
    # через OpenCV печатает предупреждения на каждый файл; на совсем битых
    # профилях PIL падает — тогда читаем через OpenCV
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.asarray(Image.open(path).convert("L"))
        except ValueError:
            return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    w = max(8, int(round(iw * h / ih)))
    interp = cv2.INTER_AREA if ih > h else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)


def pad_or_window(img: np.ndarray, W: int, rng: random.Random | None = None, max_windows: int = 3) -> list[np.ndarray]:
    """Приводит картинку высоты H к ширине W.
    rng задан  -> одно случайное окно (обучение);
    rng = None -> список из <= max_windows равномерных окон (инференс)."""
    h, w = img.shape
    if w <= W:
        pad_val = int(np.median(np.concatenate([img[:, -2:].ravel(), img[:, :2].ravel()])))
        out = np.full((h, W), pad_val, dtype=img.dtype)
        # при обучении случайно сдвигаем текст, чтобы модель не привязывалась к левому краю
        x0 = rng.randint(0, W - w) if rng is not None else 0
        out[:, x0:x0 + w] = img
        return [out]
    if rng is not None:
        x0 = rng.randint(0, w - W)
        return [img[:, x0:x0 + W]]
    k = min(max_windows, math.ceil(w / W))
    starts = np.linspace(0, w - W, k).astype(int)
    return [img[:, s:s + W] for s in starts]


def to_tensor(img: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0).sub_(0.5).div_(0.5)
    return x.unsqueeze(0)  # [1, H, W]


def rot180(x: torch.Tensor) -> torch.Tensor:
    """Поворот на 180° = flip по H и W. Работает и для батча [N,1,H,W], и для [1,H,W]."""
    return torch.flip(x, dims=(-2, -1))


# ----------------------------------------------------------------------------- augmentation

def augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Лёгкие аугментации поверх исходного кропа (до resize).
    Синтетика уже сильно деградирована генератором, реальные кропы — нет."""
    h, w = img.shape
    # случайная обрезка/дополнение по вертикали: детектор режет боксы неточно
    if rng.random() < 0.5:
        top = int(h * rng.uniform(-0.12, 0.15))
        bot = int(h * rng.uniform(-0.12, 0.15))
        if top < 0 or bot < 0:
            img = cv2.copyMakeBorder(img, max(0, -top), max(0, -bot), 0, 0, cv2.BORDER_REPLICATE)
        img = img[max(0, top): img.shape[0] - max(0, bot)]
        if img.shape[0] < 6:
            return img
    if rng.random() < 0.3:  # мелкие кропы: сначала уменьшить, потом увеличить
        f = rng.uniform(0.4, 0.9)
        small = cv2.resize(img, (max(4, int(w * f)), max(4, int(img.shape[0] * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, img.shape[0]), interpolation=rng.choice([cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    img = img.astype(np.float32)
    if rng.random() < 0.5:  # яркость / контраст
        img = (img - 128) * rng.uniform(0.6, 1.3) + 128 + rng.uniform(-30, 30)
    if rng.random() < 0.4:
        img = img + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 12), img.shape)
    if rng.random() < 0.1:
        img = 255 - img
    return np.clip(img, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- dataset

class OrientationDataset(Dataset):
    """paths: картинки в ПРАВИЛЬНОЙ ориентации.
    train=True : случайные аугментации, случайный переворот (p=0.5), случайное окно.
    train=False: без аугментаций; переворот детерминирован seed-ом, так что
                 валидационные метки одинаковы между запусками."""

    def __init__(self, paths: list[Path], train: bool, H: int = 32, W: int = 256, seed: int = 0):
        self.paths = list(paths)
        self.train = train
        self.H, self.W = H, W
        self.seed = seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i: int):
        img = load_gray(self.paths[i])
        if self.train:
            # rng независим от порядка воркеров: зависит от seed, индекса и "эпохи" (torch seed воркера)
            rng = random.Random(hash((self.seed, i, torch.initial_seed())) & 0xFFFFFFFF)
            img = augment(img, rng)
            label = rng.random() < 0.5
        else:
            rng = None
            label = (i * 2654435761 + self.seed) % 2 == 1
        img = resize_to_height(img, self.H)
        win = pad_or_window(img, self.W, rng=rng if self.train else random.Random(i))[0]
        x = to_tensor(win)
        if label:
            x = rot180(x)
        return x, torch.tensor(float(label))


def make_windows(path: str | Path, H: int, W: int, max_windows: int = 3) -> list[torch.Tensor]:
    """Инференс: все окна картинки (без переворота)."""
    img = resize_to_height(load_gray(path), H)
    return [to_tensor(w) for w in pad_or_window(img, W, rng=None, max_windows=max_windows)]
