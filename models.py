import torch
from torch import nn
from torchvision.models import resnet18


class MyCNN(nn.Module):
    """CNN used by the MNIST notebook framework.

    Input shape: [batch_size, 1, 28, 28]
    Output shape: [batch_size, 10]
    """

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_resnet(num_classes: int = 10) -> nn.Module:
    try:
        model = resnet18(weights=None, num_classes=num_classes)
    except TypeError:
        model = resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    name = model_name.lower()
    if name == "cnn":
        return MyCNN(input_channels=1, num_classes=num_classes)
    if name == "resnet":
        return build_resnet(num_classes=num_classes)
    raise ValueError(f"Unsupported model '{model_name}'. Choose from: cnn, resnet.")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
