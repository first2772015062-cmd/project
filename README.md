# 基于深度学习的 MNIST 数字图片分类项目

本项目按照 `code_mnist_notes.ipynb` 的实验框架整理为 `.py` 文件：导入库、设置参数、加载 MNIST 数据集、搭建模型、训练模型、评估准确率、保存/读取模型、单张图片预测和可选导出图片数据。

命名方式保持不变：

- `cnn.py`：使用自定义卷积神经网络 `MyCNN`
- `resnet.py`：使用适配 MNIST 单通道输入的 `ResNet18`

## 文件说明

```text
models.py    # MyCNN、ResNet 和参数量统计
train.py     # 通用训练、评估、保存、读取、预测逻辑
cnn.py       # MyCNN 项目入口
resnet.py    # ResNet 项目入口
```

## 运行项目

训练并评估 MyCNN：

```powershell
python cnn.py
```

训练并评估 ResNet：

```powershell
python resnet.py
```

也可以手动指定参数：

```powershell
python cnn.py --epochs 10 --batch-size 128 --learning-rate 0.0001
python resnet.py --epochs 10 --batch-size 128 --learning-rate 0.0001
```

## 输出结果

训练完成后会在 `outputs/` 下保存：

```text
cnn.pth
cnn_metrics.json
resnet.pth
resnet_metrics.json
```

`*_metrics.json` 中包含训练损失、测试损失、训练准确率、测试准确率、最佳测试准确率和可训练参数量。

## 单张图片预测

```powershell
python cnn.py --predict-image ./MNIST/mnist_test/1_label_2.png
python resnet.py --predict-image ./MNIST/mnist_test/1_label_2.png
```

## 导出 MNIST 图片

如果需要像参考 notebook 一样把 MNIST 数据集保存成 PNG 图片：

```powershell
python cnn.py --export-images
```

## 快速验证

只跑少量 batch，用于确认代码能正常训练、测试和保存：

```powershell
python cnn.py --epochs 1 --limit-train-batches 2 --limit-test-batches 2 --no-download
python resnet.py --epochs 1 --limit-train-batches 2 --limit-test-batches 2 --no-download
```
