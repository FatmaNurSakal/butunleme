import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from .model import build_model
from .utils import (
    get_device, get_cifar10_loaders, ensure_dir, CIFAR10_CLASSES
)


def _denorm(img: np.ndarray, mean, std):
    # img: (H,W,C) float
    img = img * np.array(std)[None, None, :] + np.array(mean)[None, None, :]
    return np.clip(img, 0.0, 1.0)


def _make_grid(images, titles, out_path, ncols=8, title="Inference Samples (Correct + Wrong)"):
    n = len(images)
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(ncols * 2.2, nrows * 2.2))
    for i in range(n):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(images[i])
        ax.set_title(titles[i], fontsize=8)
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--num_correct", type=int, default=16)
    parser.add_argument("--num_wrong", type=int, default=16)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", type=str, default=None, help="Default: <run_dir>/figures/inference_grid.png")
    args = parser.parse_args()

    # reproducible sampling
    rng = np.random.default_rng(args.seed)

    device = get_device("auto")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model_name", "resnet18")
    cfg = ckpt.get("cfg", {})
    num_classes = int(cfg.get("num_classes", 10))
    data_dir = cfg.get("data_dir", "data")
    img_size = int(cfg.get("img_size", 224))

    # Build model + load weights
    model = build_model(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # test loader (augmentation kapalı)
    _, test_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=args.batch,
        num_workers=2,
        use_augmentation=False,
        img_size=img_size,
    )

    # ckpt path -> run_dir/figures
    run_dir = os.path.dirname(os.path.dirname(args.ckpt))  # .../checkpoints/best.pt -> .../<run_dir>
    fig_dir = os.path.join(run_dir, "figures")
    ensure_dir(fig_dir)

    out_path = args.out_path or os.path.join(fig_dir, "inference_grid.png")

    # We need normalization values to denormalize for plotting.
    # These are the ones used in utils.py (kept local to avoid import cycles).
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

    correct_pool = []
    wrong_pool = []

    # Collect candidates across batches until we have enough
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_np = y.numpy()

        # Move images to CPU numpy for visualization (B,C,H,W) -> (B,H,W,C)
        x_cpu = x.detach().cpu().numpy().transpose(0, 2, 3, 1)

        for i in range(x_cpu.shape[0]):
            img = _denorm(x_cpu[i], CIFAR10_MEAN, CIFAR10_STD)
            true_id = int(y_np[i])
            pred_id = int(pred[i])

            item = (img, true_id, pred_id)

            if pred_id == true_id:
                correct_pool.append(item)
            else:
                wrong_pool.append(item)

        if len(correct_pool) >= args.num_correct and len(wrong_pool) >= args.num_wrong:
            break

    if len(correct_pool) == 0:
        raise RuntimeError("No correct samples found (unexpected).")
    if len(wrong_pool) == 0:
        raise RuntimeError("No wrong samples found (unexpected).")

    # Random choose from pools (to avoid always same first samples)
    correct_idx = rng.choice(len(correct_pool), size=min(args.num_correct, len(correct_pool)), replace=False)
    wrong_idx = rng.choice(len(wrong_pool), size=min(args.num_wrong, len(wrong_pool)), replace=False)

    chosen = [correct_pool[i] for i in correct_idx] + [wrong_pool[i] for i in wrong_idx]

    images = []
    titles = []

    # first correct then wrong
    for k, (img, true_id, pred_id) in enumerate(chosen):
        images.append(img)
        true_name = CIFAR10_CLASSES[true_id] if 0 <= true_id < len(CIFAR10_CLASSES) else str(true_id)
        pred_name = CIFAR10_CLASSES[pred_id] if 0 <= pred_id < len(CIFAR10_CLASSES) else str(pred_id)

        if k < len(correct_idx):
            tag = "OK"
        else:
            tag = "WRONG"

        titles.append(f"{tag}\nT:{true_name}\nP:{pred_name}")

    _make_grid(
        images,
        titles,
        out_path,
        ncols=8,
        title=f"{model_name} | {img_size}x{img_size} | Correct:{len(correct_idx)} + Wrong:{len(wrong_idx)}"
    )

    print("✅ Inference grid saved:")
    print(f"➡ {out_path}")


if __name__ == "__main__":
    main()
