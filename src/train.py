import os
import argparse
from typing import Dict, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from .config import TrainConfig
from .model import build_model
from .utils import (
    set_seed, get_device, ensure_dir, make_run_dir,
    get_cifar10_loaders, accuracy_top1, save_checkpoint, save_json
)


def train_one_epoch(model, loader, optimizer, criterion, device) -> Dict[str, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in tqdm(loader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return {"loss": total_loss / n, "acc": total_acc / n}


def plot_curves(history: Dict[str, List[float]], out_dir: str):
    ensure_dir(out_dir)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy.png"), dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="simplecnn | customcnn | resnet18 | vit_tiny | convnext_tiny"
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None, help="default 224")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--run_name", type=str, default="train", help="Run folder prefix")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.model: cfg.model_name = args.model
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.batch is not None: cfg.batch_size = args.batch
    if args.lr is not None: cfg.lr = args.lr
    if args.seed is not None: cfg.seed = args.seed
    if args.img_size is not None: cfg.img_size = args.img_size
    if args.no_aug: cfg.use_augmentation = False

    set_seed(cfg.seed)
    device = get_device(cfg.device)

    ensure_dir(cfg.data_dir)
    ensure_dir("results")

    run_dir = make_run_dir("results/runs", name=args.run_name)
    curves_dir = os.path.join(run_dir, "curves")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    metrics_dir = os.path.join(run_dir, "metrics")

    train_loader, val_loader = get_cifar10_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_augmentation=cfg.use_augmentation,
        img_size=cfg.img_size,
    )

    model = build_model(cfg.model_name, cfg.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va = eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f}"
        )

        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            save_checkpoint(os.path.join(ckpt_dir, "best.pt"), {
                "model_name": cfg.model_name,
                "state_dict": model.state_dict(),
                "val_acc": float(best_val_acc),
                "cfg": cfg.__dict__,
            })

    plot_curves(history, curves_dir)

    summary = {
        "model": cfg.model_name,
        "seed": cfg.seed,
        "use_augmentation": cfg.use_augmentation,
        "img_size": cfg.img_size,
        "best_val_acc": float(best_val_acc),
        "run_dir": run_dir,
    }
    save_json(os.path.join(metrics_dir, "train_summary.json"), summary)

    print(f"\nBest Val Acc: {best_val_acc:.4f}")
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()

