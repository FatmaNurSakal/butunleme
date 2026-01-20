import os
import json
import random
from datetime import datetime
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES: List[str] = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(root: str = "results/runs", name: str = "run") -> str:
    ensure_dir(root)
    run_dir = os.path.join(root, f"{name}_{now_tag()}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "curves"))
    ensure_dir(os.path.join(run_dir, "checkpoints"))
    ensure_dir(os.path.join(run_dir, "metrics"))
    ensure_dir(os.path.join(run_dir, "figures"))
    return run_dir


def save_json(path: str, obj: Dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    use_augmentation: bool = True,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader]:
    # ✅ Aynı şart: herkes 224x224 görür (CIFAR-10 32x32 -> Resize)
    # Aug ON: RandomResizedCrop + HFlip (tüm modeller için ortak)
    if use_augmentation:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def save_checkpoint(path: str, state: Dict):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)
