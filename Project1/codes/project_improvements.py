import numpy as np
import matplotlib.pyplot as plt
import mynn as nn
from struct import unpack
import gzip
import pickle
from draw_tools.plot import plot

class ImprovedModel:
    def __init__(self):
        # 基础模型参数
        self.input_size = 28 * 28
        self.hidden_sizes = [600]  # 可调整的隐藏层大小
        self.output_size = 10
        self.learning_rate = 0.06
        self.weight_decay = 1e-4  # L2正则化参数
        self.dropout_rate = 0.2  # Dropout率
        
    def load_data(self):
        """加载MNIST数据集"""
        train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
        train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'
        test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
        test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

        # 加载训练数据
        with gzip.open(train_images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        
        with gzip.open(train_labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            train_labs = np.frombuffer(f.read(), dtype=np.uint8)

        # 加载测试数据
        with gzip.open(test_images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
        
        with gzip.open(test_labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            test_labs = np.frombuffer(f.read(), dtype=np.uint8)

        # 数据归一化
        train_imgs = train_imgs / 255.0
        test_imgs = test_imgs / 255.0

        # 划分验证集
        idx = np.random.permutation(np.arange(len(train_imgs)))
        valid_imgs = train_imgs[:10000]
        valid_labs = train_labs[:10000]
        train_imgs = train_imgs[10000:]
        train_labs = train_labs[10000:]

        return (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs)

    def train_with_improvements(self):
        """使用各种改进方法训练模型"""
        # 加载数据
        (train_imgs, train_labs), (valid_imgs, valid_labs), (test_imgs, test_labs) = self.load_data()

        # 创建模型
        model = nn.models.Model_MLP(
            [self.input_size] + self.hidden_sizes + [self.output_size],
            'ReLU',
            [self.weight_decay] * (len(self.hidden_sizes) + 1)
        )

        # 使用SGD优化器
        optimizer = nn.optimizer.SGD(
            init_lr=self.learning_rate,
            model=model
        )

        # 使用简单的学习率调度器
        scheduler = nn.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=1000,  # 每1000次迭代降低一次学习率
            gamma=0.5  # 每次降低为原来的0.5倍
        )

        # 使用交叉熵损失
        loss_fn = nn.op.MultiCrossEntropyLoss(
            model=model,
            max_classes=self.output_size
        )

        # 创建训练器
        runner = nn.runner.RunnerM(
            model,
            optimizer,
            nn.metric.accuracy,
            loss_fn,
            scheduler=scheduler
        )

        # 训练模型
        runner.train(
            [train_imgs, train_labs],
            [valid_imgs, valid_labs],
            num_epochs=5,
            log_iters=100,
            save_dir=r'./best_models'
        )

        # 测试模型
        logits = model(test_imgs)
        test_accuracy = nn.metric.accuracy(logits, test_labs)
        print(f"测试集准确率: {test_accuracy:.4f}")

        # 可视化训练过程
        _, axes = plt.subplots(1, 2)
        axes.reshape(-1)
        _.set_tight_layout(1)
        plot(runner, axes)
        plt.show()

        # 保存模型
        model.save_model(r'./saved_models/best_model_improved.pickle')

    def visualize_weights(self):
        """可视化模型权重"""
        model = nn.models.Model_MLP()
        model.load_model(r'./saved_models/best_model_improved.pickle')

        # 获取权重
        weights = []
        for layer in model.layers:
            if hasattr(layer, 'params') and 'W' in layer.params:
                weights.append(layer.params['W'])

        # 可视化第一层权重
        plt.figure(figsize=(20, 20))
        for i in range(min(100, weights[0].shape[1])):
            plt.subplot(10, 10, i+1)
            plt.imshow(weights[0][:, i].reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle('第一层权重可视化')
        plt.show()

if __name__ == "__main__":
    # 固定随机种子
    np.random.seed(309)
    
    # 创建改进模型实例
    improved_model = ImprovedModel()
    
    # 训练模型
    improved_model.train_with_improvements()
    
    # 可视化权重
    improved_model.visualize_weights() 