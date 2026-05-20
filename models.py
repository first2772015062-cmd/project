import torch
from torch import nn
from torchvision.models import resnet18


class MyCNN(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_channels: int = 64, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            self._block(input_channels, 16),
            self._block(16, 32),
            self._block(32, hidden_channels),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(hidden_channels, num_classes),
        )

    @staticmethod
    def _block(input_channels: int, output_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_resnet(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    name = model_name.lower()
    if name == "cnn":
        return MyCNN(input_channels=1, hidden_channels=64, num_classes=num_classes)
    if name == "resnet":
        return build_resnet(num_classes=num_classes)
    raise ValueError(f"Unsupported model '{model_name}'. Choose from: cnn, resnet.")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
