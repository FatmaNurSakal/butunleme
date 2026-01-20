import os
import glob
import json
import argparse
from datetime import datetime

from .utils import ensure_dir


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", type=str, default="results/presentation_assets")
    parser.add_argument("--out", type=str, default=None, help="metrics.txt output path")
    args = parser.parse_args()

    assets_dir = args.assets_dir
    ensure_dir(assets_dir)

    eval_files = sorted(glob.glob(os.path.join(assets_dir, "*_eval_summary.json")))
    if not eval_files:
        raise FileNotFoundError(f"No *_eval_summary.json found in {assets_dir}")

    rows = []
    for ef in eval_files:
        prefix = os.path.basename(ef).replace("_eval_summary.json", "")
        train_f = os.path.join(assets_dir, f"{prefix}_train_summary.json")

        eval_j = _read_json(ef)
        train_j = _read_json(train_f) if os.path.exists(train_f) else {}

        rows.append({
            "name": prefix,
            "model": train_j.get("model", prefix),
            "seed": train_j.get("seed", None),
            "aug": train_j.get("use_augmentation", None),
            "img_size": train_j.get("img_size", None),
            "best_val_acc": train_j.get("best_val_acc", None),
            "test_acc": eval_j.get("test_acc", None),
        })

    rows = sorted(rows, key=lambda r: (r["test_acc"] is not None, r["test_acc"]), reverse=True)

    out_path = args.out or os.path.join(assets_dir, "metrics.txt")

    lines = []
    lines.append(f"METRICS SUMMARY (generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    lines.append(f"assets_dir: {assets_dir}")
    lines.append("-" * 72)
    lines.append(f"{'rank':<4} {'model':<15} {'aug':<5} {'seed':<6} {'val_acc':<8} {'test_acc':<8} {'img':<4}")
    lines.append("-" * 72)

    for i, r in enumerate(rows, start=1):
        model_s = str(r["model"])[:15]
        aug_s = str(r["aug"])
        seed_s = str(r["seed"])
        val_s = _fmt(r["best_val_acc"])
        test_s = _fmt(r["test_acc"])
        img_s = str(r["img_size"])

        lines.append(f"{i:<4} {model_s:<15} {aug_s:<5} {seed_s:<6} {val_s:<8} {test_s:<8} {img_s:<4}")

    lines.append("-" * 72)
    lines.append("Not: Bu dosya *_train_summary.json ve *_eval_summary.json üzerinden üretilmiştir.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"metrics.txt yazıldı: {out_path}")


if __name__ == "__main__":
    main()
