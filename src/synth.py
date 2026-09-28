"""
Генератор синтетических кропов текста (только в правильной ориентации, 0°).

Переворот на 180° и соответствующая метка делаются на этапе датасета
(см. data.py), поэтому здесь мы заботимся только о реализме и разнообразии:
  * тексты: русские / английские слова и фразы, числа, артикулы, цены;
  * шрифты: все системные TTF/OTF, у которых есть кириллица;
  * два стиля: "photo" (цветные фоны, тени, обводка, перспектива, сильные
    деградации) и "doc" (скан документа: светлый фон, мелкий шрифт, длинная строка);
  * деградации: blur, motion blur, пикселизация, JPEG, шум, яркость/контраст,
    инверсия, обрезанные соседние строки сверху/снизу.

Использование:
    gen = SynthGenerator(seed=0)
    img = gen.sample()            # PIL.Image RGB, высота ~ 12..150 px
    generate_to_dir("synth", n=80000, seed=0, workers=2)
"""
from __future__ import annotations

import io
import os
import random
import re
import string
from pathlib import Path
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from fontTools.ttLib import TTFont

# ----------------------------------------------------------------------------- fonts

_CYR_PROBE = "ЯжфыЁ"  # если шрифт умеет эти символы — считаем, что кириллица есть


def _has_cyrillic(path: str) -> bool:
    try:
        f = TTFont(path, fontNumber=0, lazy=True)
        cmap = f.getBestCmap() or {}
        return all(ord(c) in cmap for c in _CYR_PROBE)
    except Exception:
        return False


# CJK / битмапные / символьные шрифты: огромные и не похожи на наш домен
_EXCLUDE = __import__("re").compile(r"CJK|unifont|ipag|ipam|ipaexg|ipaexm|Symbol|Emoji|Math", re.I)


def find_fonts(dirs=("fonts",)):
    """Собираем все шрифты с кириллицей из папки репозитория fonts/ (там лежат
    OFL/GPL/Apache-шрифты из Debian-пакетов, см. README). Сортировка по имени файла —
    порядок влияет на rng.choice, поэтому он фиксирован, чтобы генерация была воспроизводима."""
    out = []
    for d in dirs:
        for p in Path(d).rglob("*"):
            if p.suffix.lower() in {".ttf", ".otf"} and not _EXCLUDE.search(p.name) and _has_cyrillic(str(p)):
                out.append(str(p))
    return sorted(set(out), key=lambda s: Path(s).name)


# ----------------------------------------------------------------------------- words

def load_words(path: str | None, fallback: list[str]) -> list[str]:
    """Файл частотного словаря вида 'слово частота' (hermitdave/FrequencyWords)."""
    if path and os.path.exists(path):
        words = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.split()[0]
                if 2 <= len(w) <= 16 and w.isalpha():
                    words.append(w)
        if len(words) > 100:
            return words
    return fallback


_FALLBACK_RU = ("продажа магазин цена доставка новый качество товар работа автомобиль запчасти "
                "россия москва производитель одежда обувь детские мебель ремонт телефон услуги "
                "оригинал состояние отличное торг возможен гарантия скидка вход выход внимание").split()
_FALLBACK_EN = ("sale shop price new original quality made in china parts contract honey right with "
                "auto moto power black white blue red size model number brand limited edition").split()


# ----------------------------------------------------------------------------- helpers

def _rand_color(rng: random.Random, low=0, high=255):
    return tuple(rng.randint(low, high) for _ in range(3))


def _luma(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _contrast_color(rng: random.Random, bg, min_diff=60):
    """Цвет текста, достаточно контрастный к фону (иногда — совсем слабый контраст)."""
    if rng.random() < 0.08:
        min_diff = 25
    for _ in range(20):
        c = _rand_color(rng)
        if abs(_luma(c) - _luma(bg)) >= min_diff:
            return c
    return (0, 0, 0) if _luma(bg) > 128 else (255, 255, 255)


def _perspective_coeffs(src, dst):
    """Коэффициенты для PIL Image.transform(PERSPECTIVE): dst -> src."""
    matrix = []
    for (x, y), (u, v) in zip(dst, src):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    A = np.array(matrix, dtype=np.float64)
    b = np.array(src, dtype=np.float64).reshape(8)
    return np.linalg.solve(A, b).tolist()


# ----------------------------------------------------------------------------- generator

class SynthGenerator:
    def __init__(self, seed: int = 0, fonts: list[str] | None = None,
                 ru_words: str | None = "words/ru_50k.txt", en_words: str | None = "words/en_50k.txt",
                 p_doc: float = 0.3):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.fonts = fonts or find_fonts()
        if not self.fonts:
            raise RuntimeError("Не найдено ни одного шрифта с кириллицей")
        self.ru = load_words(ru_words, _FALLBACK_RU)
        self.en = load_words(en_words, _FALLBACK_EN)
        self.p_doc = p_doc
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    # ---- text ---------------------------------------------------------------
    def _word(self, lang: str) -> str:
        return self.rng.choice(self.ru if lang == "ru" else self.en)

    def _code(self) -> str:
        """Артикулы / номера / коды: 'E4G16-3701010BA', 'TC9222FX', '06-07-12'."""
        r = self.rng
        kind = r.random()
        if kind < 0.3:
            return "-".join(str(r.randint(0, 99)).zfill(2) for _ in range(3))
        if kind < 0.6:
            return "".join(r.choice(string.ascii_uppercase + string.digits) for _ in range(r.randint(4, 10)))
        parts = ["".join(r.choice(string.ascii_uppercase + string.digits) for _ in range(r.randint(3, 6)))
                 for _ in range(r.randint(2, 3))]
        return "-".join(parts)

    def _price(self) -> str:
        r = self.rng
        n = r.choice([r.randint(10, 999), r.randint(1000, 99999), r.randint(100000, 5000000)])
        s = f"{n:,}".replace(",", " ") if r.random() < 0.5 else str(n)
        return s + r.choice([" ₽", " руб.", " руб", "р.", " $", "", " %"])

    def sample_text(self, style: str) -> str:
        r = self.rng
        lang = "ru" if r.random() < 0.65 else "en"
        u = r.random()
        if style == "doc":
            n_words = r.randint(3, 12)
            words = [self._word(lang) for _ in range(n_words)]
            if r.random() < 0.3:
                words[r.randrange(n_words)] = self._code() if r.random() < 0.5 else self._price()
            text = " ".join(words)
            if r.random() < 0.6:
                text = text.capitalize()
            if r.random() < 0.5:
                text += r.choice([".", ",", ":", ";"])
            return text
        if u < 0.45:
            text = self._word(lang)
        elif u < 0.75:
            text = " ".join(self._word(lang) for _ in range(r.randint(2, 4)))
        elif u < 0.88:
            text = self._code()
        else:
            text = self._price()
        # регистр: в фото-кропах много КАПСА (вывески, логотипы, упаковки)
        c = r.random()
        if c < 0.4:
            text = text.upper()
        elif c < 0.7:
            text = text.capitalize()
        return text

    # ---- rendering ------------------------------------------------------------
    def _font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (path, size)
        if key not in self._font_cache:
            if len(self._font_cache) > 512:
                self._font_cache.clear()
            self._font_cache[key] = ImageFont.truetype(path, size)
        return self._font_cache[key]

    def _render_text(self, text: str, font, color, stroke, stroke_color, spacing: float) -> Image.Image:
        """RGBA-слой с текстом (возможно, с обводкой) и произвольным межбуквенным интервалом."""
        # PIL не умеет letter-spacing, рисуем по символам
        widths = [font.getlength(ch) for ch in text]
        gap = spacing
        total_w = int(sum(widths) + gap * (len(text) - 1)) + 2 * stroke + 8
        asc, desc = font.getmetrics()
        h = asc + desc + 2 * stroke + 8
        layer = Image.new("RGBA", (max(total_w, 4), max(h, 4)), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        x = 4 + stroke
        y = 4 + stroke
        for ch, w in zip(text, widths):
            d.text((x, y), ch, font=font, fill=color, stroke_width=stroke, stroke_fill=stroke_color)
            x += w + gap
        return layer

    def _background(self, w: int, h: int, style: str) -> tuple[Image.Image, tuple]:
        r = self.rng
        if style == "doc":
            base = r.randint(215, 255)
            bg = (base + r.randint(-6, 0), base + r.randint(-6, 0), base + r.randint(-10, 0))
            img = Image.new("RGB", (w, h), bg)
            if r.random() < 0.5:  # лёгкая бумажная текстура
                noise = self.np_rng.normal(0, r.uniform(2, 8), (h, w, 1)).repeat(3, 2)
                img = Image.fromarray(np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8))
            return img, bg
        kind = r.random()
        bg = _rand_color(r)
        if kind < 0.45:  # однотонный
            img = Image.new("RGB", (w, h), bg)
        elif kind < 0.75:  # градиент
            c2 = _rand_color(r)
            t = np.linspace(0, 1, w if r.random() < 0.6 else h, dtype=np.float32)
            grad = np.outer(1 - t, bg) + np.outer(t, c2)
            arr = np.broadcast_to(grad[None, :, :], (h, w, 3)) if len(t) == w else np.broadcast_to(grad[:, None, :], (h, w, 3))
            img = Image.fromarray(arr.astype(np.uint8))
            bg = tuple(int(v) for v in (np.array(bg) + np.array(c2)) / 2)
        else:  # шумная текстура (имитация ткани/картона/пикселей)
            arr = self.np_rng.normal(0, r.uniform(10, 40), (h, w, 3)) + np.array(bg)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if r.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(r.uniform(1, 4)))
        # случайные крупные фигуры/полосы, как элементы упаковки
        if r.random() < 0.3:
            d = ImageDraw.Draw(img)
            for _ in range(r.randint(1, 3)):
                x0, y0 = r.randint(-w // 2, w), r.randint(-h, h)
                d.rectangle([x0, y0, x0 + r.randint(w // 4, w), y0 + r.randint(h // 4, h)],
                            fill=_rand_color(r))
        return img, bg

    def _degrade(self, img: Image.Image, style: str) -> Image.Image:
        r = self.rng
        w, h = img.size
        # пикселизация: уменьшить и вернуть (очень частый артефакт в тесте)
        if r.random() < (0.35 if style == "photo" else 0.2):
            f = r.uniform(0.25, 0.7)
            small = img.resize((max(1, int(w * f)), max(1, int(h * f))),
                               r.choice([Image.BILINEAR, Image.NEAREST, Image.BOX]))
            img = small.resize((w, h), r.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST]))
        if r.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 1.8)))
        if r.random() < 0.15:  # motion blur
            k = r.randint(3, 9)
            kernel = np.zeros((k, k), np.float32)
            if r.random() < 0.7:
                kernel[k // 2, :] = 1
            else:
                kernel[:, k // 2] = 1
            img = Image.fromarray(cv2.filter2D(np.asarray(img), -1, kernel / kernel.sum()))
        if r.random() < 0.5:
            arr = np.asarray(img, np.float32)
            arr += self.np_rng.normal(0, r.uniform(2, 14), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        if r.random() < 0.6:  # яркость / контраст / гамма
            arr = np.asarray(img, np.float32) / 255.0
            arr = arr ** r.uniform(0.6, 1.6)
            arr = (arr - 0.5) * r.uniform(0.6, 1.3) + 0.5 + r.uniform(-0.15, 0.15)
            img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
        if r.random() < 0.6:  # JPEG
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=r.randint(20, 90))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        if r.random() < 0.1:
            img = ImageOps.grayscale(img).convert("RGB")
        if r.random() < 0.03:
            img = ImageOps.invert(img)
        return img

    def sample(self) -> Image.Image:
        r = self.rng
        style = "doc" if r.random() < self.p_doc else "photo"
        text = self.sample_text(style)
        font_path = r.choice(self.fonts)
        size = r.randint(14, 34) if style == "doc" else r.randint(16, 72)
        font = self._font(font_path, size)

        # фон рисуем позже, но цвет нужен уже для текста
        bg_probe = (245, 245, 245) if style == "doc" else _rand_color(r)
        if style == "doc":
            color = (r.randint(0, 60),) * 3
            stroke, stroke_color = 0, None
        else:
            color = _contrast_color(r, bg_probe)
            stroke = r.choice([0, 0, 0, 1, 2, 3]) if size > 20 else 0
            stroke_color = _contrast_color(r, color, 80) if stroke else None
        spacing = r.uniform(-0.5, 3.0) if style == "photo" else r.uniform(0, 0.6)
        if r.random() < 0.1:
            spacing = r.uniform(3, 8)

        layer = self._render_text(text, font, color, stroke, stroke_color, spacing)
        tw, th = layer.size

        # отступы: детектор обычно даёт бокс чуть больше текста; иногда обрезает его
        pad_x = int(tw * r.uniform(-0.02, 0.10)) + r.randint(0, 6)
        pad_y_top = int(th * r.uniform(-0.12, 0.25)) + r.randint(0, 4)
        pad_y_bot = int(th * r.uniform(-0.12, 0.25)) + r.randint(0, 4)
        W = max(tw + 2 * pad_x, 16)
        H = max(th + pad_y_top + pad_y_bot, 10)
        bg, bg_color = self._background(W, H, style)
        if style == "photo":
            # перекрасим текст под реальный фон, если контраст плохой
            if abs(_luma(color) - _luma(bg_color)) < 40 and r.random() < 0.9:
                color = _contrast_color(r, bg_color)
                layer = self._render_text(text, font, color, stroke, stroke_color, spacing)
            # тень
            if r.random() < 0.25:
                sh = Image.new("RGBA", layer.size, (0, 0, 0, 0))
                sh.paste(Image.new("RGBA", layer.size, (0, 0, 0, r.randint(80, 200))), mask=layer.split()[3])
                sh = sh.filter(ImageFilter.GaussianBlur(r.uniform(0.5, 2)))
                bg.paste(sh, (pad_x + r.randint(1, 4), pad_y_top + r.randint(1, 4)), sh)
        bg.paste(layer, (pad_x, pad_y_top), layer)

        # обрезанные соседние строки сверху/снизу (в тесте это встречается часто)
        if r.random() < (0.45 if style == "doc" else 0.2):
            other = self._render_text(self.sample_text(style), font, color, stroke, stroke_color, spacing)
            oh = other.size[1]
            if r.random() < 0.5:
                y = -oh + r.randint(2, max(3, th // 4))            # верхняя строка, виден низ
            else:
                y = H - r.randint(2, max(3, th // 4))               # нижняя строка, виден верх
            bg.paste(other, (r.randint(-tw // 3, W // 3), y), other)

        img = bg
        # геометрия: небольшой наклон / перспектива (только "photo")
        if style == "photo" and r.random() < 0.5:
            w, h = img.size
            jit = lambda s: r.uniform(-s, s)
            dx, dy = w * 0.06, h * 0.15
            src = [(0, 0), (w, 0), (w, h), (0, h)]
            dst = [(jit(dx), jit(dy)), (w + jit(dx), jit(dy)), (w + jit(dx), h + jit(dy)), (jit(dx), h + jit(dy))]
            fill = tuple(int(v) for v in bg_color)
            img = img.transform((w, h), Image.PERSPECTIVE, _perspective_coeffs(src, dst),
                                Image.BILINEAR, fillcolor=fill)
        if r.random() < 0.4:
            img = img.rotate(r.uniform(-4, 4), Image.BILINEAR, expand=False, fillcolor=tuple(int(v) for v in bg_color))

        img = self._degrade(img, style)

        # итоговый масштаб: распределение высот примерно как в тесте (медиана ~40)
        target_h = int(np.clip(self.np_rng.lognormal(np.log(38), 0.55), 12, 160))
        w, h = img.size
        img = img.resize((max(12, int(w * target_h / h)), target_h), Image.BILINEAR)
        return img


# ----------------------------------------------------------------------------- batch generation

def _worker(args):
    out_dir, start, n, seed = args
    gen = SynthGenerator(seed=seed)
    for i in range(start, start + n):
        img = gen.sample()
        img.save(Path(out_dir) / f"synth_{i:06d}.png")
    return n


def generate_to_dir(out_dir: str, n: int, seed: int = 0, workers: int = 2):
    """Детерминированно (по seed) генерирует n картинок в out_dir."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    chunk = (n + workers - 1) // workers
    jobs = [(out_dir, i * chunk, min(chunk, n - i * chunk), seed * 1000 + i) for i in range(workers) if i * chunk < n]
    with Pool(workers) as p:
        done = sum(p.map(_worker, jobs))
    return done


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    generate_to_dir("synth_preview", n, seed=0, workers=1)
    print("ok")
