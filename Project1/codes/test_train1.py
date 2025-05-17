# 导入必要的模块
from sklearn.metrics import r2_score
from mynn.op import MultiCrossEntropyLoss
from mynn.models import Model_MLP
from mynn.models import Model_CNN
from mynn.lr_scheduler import MultiStepLR
from mynn.optimizer import SGD
from mynn.runner import RunnerM
from draw_tools.plot import plot

# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 标准化数据
train_mean = np.mean(train_imgs)
train_std = np.std(train_imgs)
train_imgs = (train_imgs - train_mean) / train_std
valid_imgs = (valid_imgs - train_mean) / train_std

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 修改数据形状
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

# 初始化 CNN 模型
linear_model = Model_CNN()
optimizer = nn.optimizer.Adam(init_lr=0.001, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, batch_size=128,scheduler=scheduler)

num_epochs = 20  # 定义 num_epochs

# runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=20, log_iters=100, save_dir=r'./saved_models/best_model.pickle')

# 添加以下代码
train_losses = []
valid_losses = []
train_scores = []
valid_scores = []

# num_epochs = 20  # 定义 num_epochs  <- remove this line
batch_size = 128  # 定义 batch_size
train_x = train_imgs
# 修改 runner.train 方法
for epoch in range(num_epochs):
    trn_loss, trn_score, val_loss, val_score = runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=1, log_iters=100, save_dir=r'./best_models')

    # 记录损失值和得分
    train_losses.append(trn_loss)
    valid_losses.append(val_loss)
    train_scores.append(trn_score)
    valid_scores.append(val_score)

# 绘制损失值和得分曲线图
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(train_scores, label='Train Score')
plt.plot(valid_scores, label='Valid Score')
plt.legend()
plt.title('Score Curve')

plt.tight_layout()
plt.show()