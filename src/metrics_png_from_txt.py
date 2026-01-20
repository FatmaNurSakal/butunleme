import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .utils import ensure_dir


def _to_float(s: str):
    s = s.strip()
    if s.upper() == "N/A":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_metrics_txt(path: str):
    """
    Beklenen header:
    rank model run acc prec rec f1 auc img
    Satırlar fixed-width yazılmış. Biz regex ile yakalıyoruz.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.txt not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    rows = []
    # Örnek satır:
    # 1    resnet18        resnet18_aug_on_....     0.7683  0.7777  0.7683  0.7700  0.9500  224
    # rank model run acc prec rec f1 auc img
    pat = re.compile(
        r"^\s*(\d+)\s+(\S+)\s+(.+?)\s+([0-9.]+|N/A)\s+([0-9.]+|N/A)\s+([0-9.]+|N/A)\s+([0-9.]+|N/A)\s+([0-9.]+|N/A)\s+(\d+)\s*$"
    )

    for ln in lines:
        m = pat.match(ln)
        if not m:
            continue

        rank = int(m.group(1))
        model = m.group(2)
        run = m.group(3).strip()
        acc = _to_float(m.group(4))
        prec = _to_float(m.group(5))
        rec = _to_float(m.group(6))
        f1 = _to_float(m.group(7))
        auc = _to_float(m.group(8))
        img = int(m.group(9))

        rows.append({
            "rank": rank,
            "model": model,
            "run": run,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "auc": auc,
            "img": img
        })

    if not rows:
        raise RuntimeError(
            "metrics.txt parse edilemedi. Dosya formatı beklenen tabloya uymuyor.\n"
            "İpucu: metrics.txt içindeki satırlarda rank/model/run/acc/prec/rec/f1/auc/img alanları olmalı."
        )

    # rank'e göre sırala (dosyada zaten sıralı olabilir)
    rows = sorted(rows, key=lambda r: r["rank"])
    return rows


def _fmt(x):
    if x is None:
        return "N/A"
    return f"{x:.4f}"


def _save_table_png(rows, out_path: str, title: str):
    headers = ["Rank", "Model", "Run", "Acc", "Prec", "Rec", "F1", "ROC-AUC", "Img"]
    table_data = []
    for r in rows:
        table_data.append([
            str(r["rank"]),
            str(r["model"])[:14],
            str(r["run"])[:34],
            _fmt(r["acc"]),
            _fmt(r["prec"]),
            _fmt(r["rec"]),
            _fmt(r["f1"]),
            _fmt(r["auc"]),
            str(r["img"])
        ])

    n = len(table_data)
    fig_h = max(2.8, 0.65 + 0.42 * (n + 1))
    plt.figure(figsize=(15, fig_h))
    plt.axis("off")
    plt.title(title, fontsize=14, pad=12)

    tbl = plt.table(
        cellText=table_data,
        colLabels=headers,
        loc="center",
        cellLoc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def _bar_png(values, labels, out_path: str, title: str, y_label: str):
    x = np.arange(len(values))
    plt.figure(figsize=(10, 4.5))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics_txt",
        type=str,
        default="results/presentation_assets/metrics.txt",
        help="metrics.txt yolu"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/presentation_assets/metrics",
        help="PNG çıktıları buraya"
    )
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    rows = _parse_metrics_txt(args.metrics_txt)

    # tablo PNG
    table_png = os.path.join(args.out_dir, "metrics_table_from_txt.png")
    _save_table_png(rows, table_png, title="CIFAR-10 — Metrics Table (from metrics.txt)")

    labels = [r["model"] for r in rows]

    # bar chart'lar
    _bar_png([r["acc"] for r in rows], labels,
             os.path.join(args.out_dir, "bar_accuracy_from_txt.png"),
             "Accuracy Comparison (from metrics.txt)", "accuracy")

    _bar_png([r["prec"] for r in rows], labels,
             os.path.join(args.out_dir, "bar_precision_from_txt.png"),
             "Precision (Macro) Comparison (from metrics.txt)", "precision (macro)")

    _bar_png([r["rec"] for r in rows], labels,
             os.path.join(args.out_dir, "bar_recall_from_txt.png"),
             "Recall (Macro) Comparison (from metrics.txt)", "recall (macro)")

    _bar_png([r["f1"] for r in rows], labels,
             os.path.join(args.out_dir, "bar_f1_from_txt.png"),
             "F1-Score (Macro) Comparison (from metrics.txt)", "f1 (macro)")

    # AUC: N/A olanları filtrele (None kalırsa bar çizilmez)
    auc_rows = [r for r in rows if isinstance(r["auc"], (int, float))]
    if len(auc_rows) >= 2:
        _bar_png([r["auc"] for r in auc_rows],
                 [r["model"] for r in auc_rows],
                 os.path.join(args.out_dir, "bar_roc_auc_from_txt.png"),
                 "ROC-AUC (OVR, Macro) Comparison (from metrics.txt)", "roc-auc (ovr, macro)")

    print("✅ PNG'ler metrics.txt üzerinden üretildi (NO retraining):")
    print(f"metrics.txt: {args.metrics_txt}")
    print(f"out_dir: {args.out_dir}")
    print(" - metrics_table_from_txt.png")
    print(" - bar_accuracy_from_txt.png")
    print(" - bar_precision_from_txt.png")
    print(" - bar_recall_from_txt.png")
    print(" - bar_f1_from_txt.png")
    print(" - (optional) bar_roc_auc_from_txt.png")


if __name__ == "__main__":
    main()
