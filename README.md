# MNIST CNN and ResNet

This project trains and evaluates two MNIST classifiers:

- `cnn`: a small custom `MyCNN` model.
- `resnet`: a ResNet-18 adapted for one-channel 28x28 MNIST images.

## Train and evaluate

```powershell
python train.py --model cnn --epochs 20
python train.py --model resnet --epochs 20
```

The script reports loss, accuracy, trainable parameter count, and saves:

- `outputs/cnn.pth`
- `outputs/cnn_metrics.json`
- `outputs/resnet.pth`
- `outputs/resnet_metrics.json`

## Quick smoke test

```powershell
python train.py --model cnn --epochs 1 --limit-train-batches 2 --limit-test-batches 2 --no-download
python train.py --model resnet --epochs 1 --limit-train-batches 2 --limit-test-batches 2 --no-download
```
