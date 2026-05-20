import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model, count_trainable_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CNN or ResNet on MNIST.")
    parser.add_argument("--model", choices=["cnn", "resnet"], default="cnn", help="Model architecture to train.")
    parser.add_argument("--data-dir", default="./data", help="Directory that stores MNIST.")
    parser.add_argument("--output-dir", default="./outputs", help="Directory for checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Optional quick-run limit.")
    parser.add_argument("--limit-test-batches", type=int, default=None, help="Optional quick-run limit.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=args.download, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=args.download, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    if total == 0:
        raise RuntimeError("No batches were processed. Check dataloader or batch limits.")
    return total_loss / total, correct / total


def save_artifacts(
    model: nn.Module,
    args: argparse.Namespace,
    metrics: dict,
    parameter_count: int,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{args.model}.pth"
    metrics_path = output_dir / f"{args.model}_metrics.json"

    torch.save(
        {
            "model": args.model,
            "state_dict": model.state_dict(),
            "num_classes": 10,
            "parameter_count": parameter_count,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = make_dataloaders(args)
    model = build_model(args.model, num_classes=10).to(device)
    parameter_count = count_trainable_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)

    history = []
    best_test_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.limit_train_batches,
        )
        test_loss, test_accuracy = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.limit_test_batches,
        )
        scheduler.step()
        best_test_accuracy = max(best_test_accuracy, test_accuracy)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_metrics)

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}"
        )

    metrics = {
        "model": args.model,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "trainable_parameters": parameter_count,
        "best_test_accuracy": best_test_accuracy,
        "history": history,
    }
    save_artifacts(model, args, metrics, parameter_count, Path(args.output_dir))
    print(f"trainable_parameters={parameter_count}")
    print(f"best_test_accuracy={best_test_accuracy:.4f}")


if __name__ == "__main__":
    main()
