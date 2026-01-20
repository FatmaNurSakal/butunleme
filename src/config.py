from dataclasses import dataclass

@dataclass
class TrainConfig:
    # data
    data_dir: str = "data"
    num_workers: int = 2
    img_size: int = 224  # ✅ tüm modeller için ortak giriş boyutu

    # model
    model_name: str = "resnet18"  # simplecnn | customcnn | resnet18 | vit_tiny | convnext_tiny
    num_classes: int = 10

    # training
    seed: int = 42
    epochs: int = 3          # ✅ deneme için
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 5e-4
    use_augmentation: bool = True

    # device
    device: str = "auto"  # "auto" | "cpu" | "cuda"
