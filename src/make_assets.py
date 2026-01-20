import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .train import main as train_main
from .eval import main as eval_main
from .utils import ensure_dir, load_json


def _latest_run_dir() -> str:
    runs_root = "results/runs"
    latest = sorted(
        [os.path.join(runs_root, d) for d in os.listdir(runs_root)],
        key=os.path.getmtime
    )[-1]
    return latest


def _run_train_eval(model: str, epochs: int, seed: int, use_aug: bool, run_name: str, img_size: int) -> str:
    import sys

    # TRAIN
    sys.argv = [
        "train",
        "--model", model,
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--img_size", str(img_size),
        "--run_name", run_name
    ]
    if not use_aug:
        sys.argv += ["--no_aug"]
    train_main()

    run_dir = _latest_run_dir()
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")

    # EVAL
    sys.argv = [
        "eval",
        "--ckpt", ckpt_path,
        "--out_dir", run_dir
    ]
    eval_main()

    return run_dir


def _copy_key_figures(run_dir: str, out_assets: str, prefix: str):
    curves = os.path.join(run_dir, "curves")
    figs = os.path.join(run_dir, "figures")
    metrics = os.path.join(run_dir, "metrics")

    mapping = [
        (os.path.join(curves, "loss.png"), os.path.join(out_assets, f"{prefix}_loss.png")),
        (os.path.join(curves, "accuracy.png"), os.path.join(out_assets, f"{prefix}_accuracy.png")),
        (os.path.join(figs, "confusion_matrix.png"), os.path.join(out_assets, f"{prefix}_confusion_matrix.png")),
        (os.path.join(metrics, "train_summary.json"), os.path.join(out_assets, f"{prefix}_train_summary.json")),
        (os.path.join(metrics, "eval_summary.json"), os.path.join(out_assets, f"{prefix}_eval_summary.json")),
    ]
    for src, dst in mapping:
        if os.path.exists(src):
            shutil.copyfile(src, dst)


def _bar_chart(values, labels, title, out_path, y_label="test accuracy"):
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)     # ✅ deneme için
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    out_assets = "results/presentation_assets"
    ensure_dir(out_assets)

    # ✅ karşılaştırılacak 5 model (Aug ON)
    models = [
        ("simplecnn", "SimpleCNN"),
        ("customcnn", "CustomCNN"),
        ("resnet18", "ResNet-18"),
        ("vit_tiny", "ViT-Tiny"),
        ("convnext_tiny", "ConvNeXt-Tiny"),
    ]

    model_accs = []
    model_labels = []

    for key, label in models:
        run_dir = _run_train_eval(
            model=key,
            epochs=args.epochs,
            seed=args.seed_base,
            use_aug=True,
            run_name=f"{key}_aug_on",
            img_size=args.img_size
        )
        _copy_key_figures(run_dir, out_assets, f"{key}_aug_on")

        acc = load_json(os.path.join(run_dir, "metrics", "eval_summary.json"))["test_acc"]
        model_accs.append(float(acc))
        model_labels.append(label)

    # ✅ tek grafikte hepsini karşılaştır
    _bar_chart(
        model_accs,
        model_labels,
        "CIFAR-10 Model Comparison (Test Accuracy, Aug ON)",
        os.path.join(out_assets, "compare_all_models.png"),
        y_label="test accuracy"
    )

    # (opsiyonel) Aug ablation: ResNet-18 ON/OFF
    run_r18_off = _run_train_eval(
        model="resnet18",
        epochs=args.epochs,
        seed=args.seed_base,
        use_aug=False,
        run_name="resnet18_aug_off",
        img_size=args.img_size
    )
    _copy_key_figures(run_r18_off, out_assets, "resnet18_aug_off")
    r18_off_acc = load_json(os.path.join(run_r18_off, "metrics", "eval_summary.json"))["test_acc"]
    r18_on_acc = model_accs[model_labels.index("ResNet-18")]

    _bar_chart(
        [float(r18_off_acc), float(r18_on_acc)],
        ["ResNet-18 (Aug OFF)", "ResNet-18 (Aug ON)"],
        "Ablation: Data Augmentation (ResNet-18)",
        os.path.join(out_assets, "ablation_resnet18_aug_on_off.png"),
        y_label="test accuracy"
    )

    print("\n✅ SUNUM PNG'leri üretildi:")
    print(f"➡ {out_assets}")
    for fn in sorted(os.listdir(out_assets)):
        print(" -", fn)


if __name__ == "__main__":
    main()
