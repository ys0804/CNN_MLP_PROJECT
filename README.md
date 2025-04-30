# MNIST 手写数字识别

## 项目说明

本项目实现了 MNIST 手写数字识别任务，使用了 CNN 模型。

## 代码结构

-   `dataset_explore.ipynb`: 用于探索 MNIST 数据集的 Jupyter Notebook。
-   `mynn/`: 包含自定义神经网络模块的文件夹。
    -   `__init__.py`: 初始化文件。
    -   `lr_scheduler.py`: 学习率调度器。
    -   `models.py`: 模型定义，包括 `Model_MLP` 和 `Model_CNN`。
    -   `op.py`: 神经网络操作，如 `Linear`、`conv2D`、`ReLU` 等。
    -   `optimizer.py`: 优化器，如 `SGD` 和 `Adam`。
    -   `runner.py`: 训练和评估模型的 Runner 类。
-   `test_model.py`: 用于加载预训练模型并评估模型性能的脚本。
-   `test_train.py`: 用于训练模型的脚本。
-   `draw_tools/`: 包含绘图工具的文件夹。
    -   `plot.py`: 绘图函数。

## 数据集和模型

MNIST 数据集和训练好的模型文件需要手动下载。

-   MNIST 数据集：https://drive.google.com/drive/folders/1UXBAJP_NCJBkp43vbNfZdjswmhrdSxdT?usp=sharing
-   训练好的模型文件：https://drive.google.com/drive/folders/17YM0Z9woXUFX2gkSZzfcY7MN1px9ViWz?usp=sharing

请将下载的 MNIST 数据集解压后放置在 `dataset/MNIST/` 目录下，并将训练好的模型文件 `best_model.pickle` 放置在 `saved_models/` 目录下。

## 环境配置

1.  安装 Python 3.8 或更高版本。
2.  安装以下依赖库：

    ```bash
    pip install numpy matplotlib scikit-learn
    ```

## 使用方法

1.  **下载代码**：从 GitHub 仓库下载代码。
2.  **下载数据和模型**：从上方链接下载 MNIST 数据集和训练好的模型文件，并放置在指定位置。
3.  **运行 `test_model.py`**：运行 `test_model.py` 脚本，评估模型性能。

    ```bash
    python test_model.py
    ```

    脚本将输出模型在测试集上的准确率。
