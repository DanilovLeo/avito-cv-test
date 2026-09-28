"""
Модели для классификации ориентации текста.

TinyOrientNet — компактная свёрточная сеть "с нуля" (~0.3M параметров, ~50 MMAC
на вход 1x32x256). Задача простая (различить форму букв в двух ориентациях),
поэтому ImageNet-предобучение не нужно, а отсутствие внешних весов упрощает
воспроизводимость.

mobilenet_v3_small — вариант побольше (~1.5M) для проверки, нужна ли ёмкость.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_bn(cin: int, cout: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyOrientNet(nn.Module):
    """Вход [N,1,32,256] -> логит [N]. Пять понижений разрешения до карты 1x8,
    затем global average pooling: модель усредняет локальные "голоса" по ширине,
    что согласуется с идеей, что ориентация видна в любом фрагменте строки."""

    def __init__(self, width: int = 1.0, dropout: float = 0.2):
        super().__init__()
        c = [max(8, int(round(v * width))) for v in (16, 32, 64, 128, 128)]
        self.features = nn.Sequential(
            conv_bn(1, c[0]), conv_bn(c[0], c[0]), nn.MaxPool2d(2),        # 16 x 128
            conv_bn(c[0], c[1]), nn.MaxPool2d(2),                           # 8 x 64
            conv_bn(c[1], c[2]), conv_bn(c[2], c[2]), nn.MaxPool2d(2),      # 4 x 32
            conv_bn(c[2], c[3]), nn.MaxPool2d(2),                           # 2 x 16
            conv_bn(c[3], c[4]), nn.MaxPool2d(2),                           # 1 x 8
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c[4], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x).mean(dim=(2, 3))
        return self.head(f).squeeze(1)


class MobileNetV3Small(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        m = mobilenet_v3_small(weights=None)
        # первый слой под 1 канал
        old = m.features[0][0]
        m.features[0][0] = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        self.features = m.features
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(576, 1))

    def forward(self, x):
        f = self.features(x).mean(dim=(2, 3))
        return self.head(f).squeeze(1)


def build_model(name: str = "tiny", **kw) -> nn.Module:
    if name == "tiny":
        return TinyOrientNet(**kw)
    if name == "mnv3s":
        return MobileNetV3Small(**kw)
    raise ValueError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for n in ["tiny", "mnv3s"]:
        m = build_model(n)
        print(n, count_params(m), m(torch.zeros(2, 1, 32, 256)).shape)
