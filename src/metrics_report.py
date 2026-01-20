import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)

from .model import build_model
from .utils import get_device, get_cifar10_loaders, ensure_dir


def _read_ckpt(run_dir: str) -> str:
    ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"best.pt not found: {ckpt}")
    return ckpt


def _read_eval_acc(run_dir: str):
    p = os.path.join(run_dir, "metrics", "eval_summary.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        v = j.get("test_acc", None)
        return float(v) if isinstance(v, (int, float)) else None
    except Exception:
        return None


@torch.no_grad()
def _compute_metrics_from_ckpt(ckpt_path: str, batch: int = 256):
    device = get_device("auto")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "resnet18")
    cfg = ckpt.get("cfg", {})
    num_classes = int(cfg.get("num_classes", 10))
    data_dir = cfg.get("data_dir", "data")
    img_size = int(cfg.get("img_size", 224))

    model = build_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch,
        num_workers=2,
        use_augmentation=False,
        img_size=img_size,
    )

    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)  # (B, C)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()

        pred = np.argmax(probs, axis=1)
        y_true = y.numpy()

        y_true_all.append(y_true)
        y_pred_all.append(pred)
        y_prob_all.append(probs)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_prob = np.concatenate(y_prob_all, axis=0)

    # Accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Precision / Recall / F1 (macro + weighted)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # ROC-AUC (multiclass OVR)
    # y_true one-hot yerine labels ile de çalışır (sklearn ovrs)
    try:
        auc_ovr_macro = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
        auc_ovr_weighted = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        )
    except Exception:
        # Bazı edge-case'lerde (tek sınıf vs) patlayabilir; CIFAR10'da normalde patlamaz.
        auc_ovr_macro = None
        auc_ovr_weighted = None

    return {
        "model_name": model_name,
        "img_size": img_size,
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "roc_auc_ovr_macro": auc_ovr_macro,
        "roc_auc_ovr_weighted": auc_ovr_weighted,
    }


def _fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run klasörleri (results/runs/...)"
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument(
        "--out",
        type=str,
        default="results/presentation_assets/metrics.txt",
        help="Toplu metrik raporu yolu"
    )
    parser.add_argument(
        "--per_run_dir",
        type=str,
        default=None,
        help="Verirsen her run icin ayri txt yazilir (or: results/presentation_assets/metrics_per_run)"
    )
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.out))
    if args.per_run_dir:
        ensure_dir(args.per_run_dir)

    rows = []
    for run_dir in args.runs:
        run_dir = run_dir.strip().strip('"')
        ckpt_path = _read_ckpt(run_dir)
        m = _compute_metrics_from_ckpt(ckpt_path, batch=args.batch)

        # İstersen eval_summary.json’daki acc ile kıyaslamak için ek bilgi
        eval_acc = _read_eval_acc(run_dir)

        rows.append({
            "run": os.path.basename(run_dir),
            "path": run_dir,
            "model": m["model_name"],
            "img": m["img_size"],
            "acc": m["accuracy"],
            "prec_m": m["precision_macro"],
            "rec_m": m["recall_macro"],
            "f1_m": m["f1_macro"],
            "auc_m": m["roc_auc_ovr_macro"],
            "prec_w": m["precision_weighted"],
            "rec_w": m["recall_weighted"],
            "f1_w": m["f1_weighted"],
            "auc_w": m["roc_auc_ovr_weighted"],
            "eval_acc_json": eval_acc,
        })

        # per-run txt (opsiyonel)
        if args.per_run_dir:
            out_one = os.path.join(args.per_run_dir, f"{os.path.basename(run_dir)}_metrics.txt")
            with open(out_one, "w", encoding="utf-8") as f:
                f.write(f"RUN: {run_dir}\n")
                f.write(f"MODEL: {m['model_name']} | IMG: {m['img_size']}x{m['img_size']}\n\n")
                f.write("METRICS (TEST)\n")
                f.write(f"Accuracy: { _fmt(m['accuracy']) }\n")
                f.write(f"Precision (macro): { _fmt(m['precision_macro']) }\n")
                f.write(f"Recall (macro): { _fmt(m['recall_macro']) }\n")
                f.write(f"F1-Score (macro): { _fmt(m['f1_macro']) }\n")
                f.write(f"ROC-AUC OVR (macro): { _fmt(m['roc_auc_ovr_macro']) }\n\n")
                f.write("EXTRA (weighted)\n")
                f.write(f"Precision (weighted): { _fmt(m['precision_weighted']) }\n")
                f.write(f"Recall (weighted): { _fmt(m['recall_weighted']) }\n")
                f.write(f"F1-Score (weighted): { _fmt(m['f1_weighted']) }\n")
                f.write(f"ROC-AUC OVR (weighted): { _fmt(m['roc_auc_ovr_weighted']) }\n")

    # En önemli metrik: accuracy’ye göre sırala
    rows = sorted(rows, key=lambda r: r["acc"], reverse=True)

    # Tek dosya: metrics.txt (sunumluk)
    lines = []
    lines.append("METRICS SUMMARY (CIFAR-10) — Test set")
    lines.append("Metrikler: Accuracy, Precision, Recall, F1-Score, ROC-AUC (multiclass OVR)")
    lines.append("-" * 110)
    lines.append(f"{'rank':<4} {'model':<14} {'run':<34} {'acc':<8} {'prec':<8} {'rec':<8} {'f1':<8} {'auc':<8} {'img':<4}")
    lines.append("-" * 110)

    for i, r in enumerate(rows, start=1):
        lines.append(
            f"{i:<4} "
            f"{str(r['model'])[:14]:<14} "
            f"{str(r['run'])[:34]:<34} "
            f"{_fmt(r['acc']):<8} "
            f"{_fmt(r['prec_m']):<8} "
            f"{_fmt(r['rec_m']):<8} "
            f"{_fmt(r['f1_m']):<8} "
            f"{_fmt(r['auc_m']):<8} "
            f"{str(r['img']):<4}"
        )

    lines.append("-" * 110)
    lines.append("Notlar:")
    lines.append(" - Precision/Recall/F1 burada MACRO average olarak raporlandı (sınıflar eşit ağırlıklı).")
    lines.append(" - ROC-AUC: multiclass One-vs-Rest (OVR), softmax olasılıklarından hesaplandı.")
    lines.append(" - İstersen weighted değerler de per-run dosyalarda var (per_run_dir verirsen).")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ metrics.txt yazıldı:")
    print(f"➡ {args.out}")
    if args.per_run_dir:
        print("✅ per-run metrics yazıldı:")
        print(f"➡ {args.per_run_dir}")


if __name__ == "__main__":
    main()
