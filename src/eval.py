import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from .model import build_model
from .utils import (
    get_device, get_cifar10_loaders, ensure_dir, CIFAR10_CLASSES, save_json
)


def plot_confusion_matrix(cm: np.ndarray, class_names, out_path: str):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, str(val), ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir for eval assets")
    args = parser.parse_args()

    device = get_device("auto")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model_name", "resnet18")
    cfg = ckpt.get("cfg", {})
    num_classes = cfg.get("num_classes", 10)

    model = build_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data_dir = cfg.get("data_dir", "data")
    img_size = int(cfg.get("img_size", 224))

    # ✅ test loader: augmentation kapalı, img_size aynı
    _, test_loader = get_cifar10_loaders(
        data_dir,
        batch_size=args.batch,
        num_workers=2,
        use_augmentation=False,
        img_size=img_size,
    )

    all_y, all_p = [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        p = torch.argmax(logits, dim=1).cpu().numpy()
        all_p.append(p)
        all_y.append(y.numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    acc = float((y_true == y_pred).mean())
    print(f"✅ Test Accuracy: {acc:.4f}")

    out_dir = args.out_dir or os.path.dirname(os.path.dirname(args.ckpt))
    fig_dir = os.path.join(out_dir, "figures")
    metrics_dir = os.path.join(out_dir, "metrics")
    ensure_dir(fig_dir)
    ensure_dir(metrics_dir)

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(fig_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, CIFAR10_CLASSES, cm_path)

    save_json(os.path.join(metrics_dir, "eval_summary.json"), {"test_acc": acc, "confusion_matrix_path": cm_path})

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CIFAR10_CLASSES))


if __name__ == "__main__":
    main()
