import torch.nn as nn
import torchvision.models as tvm
import timm


class SimpleCNN(nn.Module):
    """
    Basit CNN (AdaptiveAvgPool ile giriş boyutundan bağımsız)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class CustomCNN(nn.Module):
    """
    Daha güçlü bir Custom CNN (BatchNorm + daha derin bloklar)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.backbone = nn.Sequential(
            block(3, 64),
            nn.MaxPool2d(2),

            block(64, 128),
            nn.MaxPool2d(2),

            block(128, 256),
            nn.MaxPool2d(2),

            block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    m = model_name.lower().strip()

    # alias
    if m in ["simplecnn", "simple_cnn"]:
        return SimpleCNN(num_classes=num_classes)

    if m in ["customcnn", "custom_cnn"]:
        return CustomCNN(num_classes=num_classes)

    if m == "resnet18":
        net = tvm.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net

    # ✅ timm modelleri
    if m in ["vit_tiny", "vit-tiny"]:
        # timm adı: vit_tiny_patch16_224
        return timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

    if m in ["convnext_tiny", "convnext-tiny"]:
        return timm.create_model("convnext_tiny", pretrained=False, num_classes=num_classes)

    raise ValueError(
        f"Unknown model_name={model_name}. Options: simplecnn, customcnn, resnet18, vit_tiny, convnext_tiny"
    )
