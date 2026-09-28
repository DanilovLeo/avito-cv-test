"""
Скачиваем подвыборку реальных кропов слов из TextOCR (Singh et al., CVPR 2021),
уже нарезанных по полигонам и выложенных на HuggingFace: MiXaiLL76/TextOCR_OCR.

Все кропы в TextOCR — в правильной ориентации (0°), поэтому метки для нашей
задачи получаются бесплатно: переворот на 180° в датасете -> метка 1.

Стримим датасет, чтобы не качать 2 ГБ целиком, и берём N первых подходящих
кропов (детерминированно). Фильтры:
  * горизонтальные (w/h >= 1.2), как в тесте;
  * не слишком мелкие (h >= 10);
  * текст читаемый (в TextOCR нечитаемые слова помечены '.').

python src/get_real_data.py --out data/textocr --n 40000
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main(out: str, n: int, split: str = "train", min_h: int = 10, min_ar: float = 1.2):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("MiXaiLL76/TextOCR_OCR", split=split, streaming=True)
    kept = 0
    with open(out_dir / "labels.tsv", "w", encoding="utf-8") as f:
        for ex in tqdm(ds, total=None):
            img, text = ex["image"], ex["text"]
            w, h = img.size
            if h < min_h or w / h < min_ar or text.strip() in {"", "."}:
                continue
            name = f"textocr_{kept:06d}.png"
            img = img.convert("RGB")
            img.info.pop("icc_profile", None)  # битые/огромные ICC-профили ломают чтение PNG
            img.save(out_dir / name)
            f.write(f"{name}\t{text}\n")
            kept += 1
            if kept >= n:
                break
    print(f"saved {kept} crops to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/textocr")
    p.add_argument("--n", type=int, default=40000)
    p.add_argument("--split", default="train")
    a = p.parse_args()
    main(a.out, a.n, a.split)
