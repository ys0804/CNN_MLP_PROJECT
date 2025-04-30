from abc import abstractmethod
import numpy as np

class Layer():
    """
    The base class of layers.
    """
    def __init__(self) -> None:
        self.params = {}
        self.grads = {}
        self.weight_decay = False
        self.weight_decay_lambda = 0.0

    @abstractmethod
    def __call__(self, X):
        """
        The forward of the layer.
        """
        pass

    @abstractmethod
    def backward(self, loss_grad):
        """
        The backward of the layer.
        """
        pass

class Linear(Layer):
    """
    The linear layer.
    """
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.params['weight'] = np.random.normal(0.0, np.sqrt(2/in_dim), size=(in_dim, out_dim))
        self.params['bias'] = np.zeros(out_dim)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return X @ self.params['weight'] + self.params['bias']

    def backward(self, loss_grad):
        self.grads['weight'] = self.X.T @ loss_grad
        self.grads['bias'] = np.sum(loss_grad, axis=0)
        return loss_grad @ self.params['weight'].T

class conv2D(Layer):
    """
    The 2D convolutional layer.
    """
    def __init__(self, in_channel, out_channel, kernel_size, padding=0) -> None:  # 添加 padding 参数
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.padding = padding  # 保存 padding 值
        self.params['weight'] = np.random.normal(0.0, np.sqrt(2/(in_channel*kernel_size*kernel_size)), size=(out_channel, in_channel, kernel_size, kernel_size))
        self.params['bias'] = np.zeros(out_channel)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        # implement the forward of conv2d
        # naive implementation
        N, C, H, W = X.shape
        OC, IC, KH, KW = self.params['weight'].shape
        OH = H - KH + 1 + 2 * self.padding
        OW = W - KW + 1 + 2 * self.padding
        outputs = np.zeros((N, OC, OH, OW))

        # 添加 padding
        X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for n in range(N):
            for oc in range(OC):
                for oh in range(OH):
                    for ow in range(OW):
                        outputs[n, oc, oh, ow] = np.sum(X_padded[n, :, oh:oh+KH, ow:ow+KW] * self.params['weight'][oc]) + self.params['bias'][oc]
        return outputs

    def backward(self, loss_grad):
        # implement the backward of conv2d
        # naive implementation
        N, OC, OH, OW = loss_grad.shape
        OC, IC, KH, KW = self.params['weight'].shape
        N, C, H, W = self.X.shape
        self.grads['weight'] = np.zeros_like(self.params['weight'])
        self.grads['bias'] = np.zeros_like(self.params['bias'])
        dX = np.zeros_like(self.X)

        # 添加 padding
        dX_padded = np.pad(dX, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        N, C, H_padded, W_padded = dX_padded.shape

        for n in range(N):
            for oc in range(OC):
                self.grads['bias'][oc] += np.sum(loss_grad[n, oc])
                for oh in range(OH):
                    for ow in range(OW):
                        self.grads['weight'][oc] += self.X[n, :, oh:oh+KH, ow:ow+KW] * loss_grad[n, oc, oh, ow]
                        dX_padded[n, :, oh:oh+KH, ow:ow+KW] += self.params['weight'][oc] * loss_grad[n, oc, oh, ow]

        # 移除 padding
        dX = dX_padded[:, :, self.padding:H_padded-self.padding, self.padding:W_padded-self.padding]
        return dX

class ReLU(Layer):
    """
    The ReLU layer.
    """
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, loss_grad):
        dX = loss_grad * (self.X > 0)
        return dX

class MaxPool2D(Layer):
    """
    The Max Pooling layer.
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        OH = (H - self.kernel_size) // self.stride + 1
        OW = (W - self.kernel_size) // self.stride + 1
        outputs = np.zeros((N, C, OH, OW))
        for n in range(N):
            for c in range(C):
                for oh in range(OH):
                    for ow in range(OW):
                        outputs[n, c, oh, ow] = np.max(X[n, c, oh*self.stride:oh*self.stride+self.kernel_size, ow*self.stride:ow*self.stride+self.kernel_size])
        return outputs

    def backward(self, loss_grad):
        N, C, OH, OW = loss_grad.shape
        N, C, H, W = self.X.shape
        dX = np.zeros_like(self.X)
        for n in range(N):
            for c in range(C):
                for oh in range(OH):
                    for ow in range(OW):
                        # 找到最大值的位置
                        window = self.X[n, c, oh*self.stride:oh*self.stride+self.kernel_size, ow*self.stride:ow*self.stride+self.kernel_size]
                        max_idx = np.argmax(window)
                        max_h = max_idx // self.kernel_size
                        max_w = max_idx % self.kernel_size
                        # 将梯度传递给最大值的位置
                        dX[n, c, oh*self.stride+max_h, ow*self.stride+max_w] = loss_grad[n, c, oh, ow]
        return dX

class Flatten(Layer):
    """
    The Flatten layer.
    """
    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, loss_grad):
        return loss_grad.reshape(self.shape)

class MultiCrossEntropyLoss():
    """
    The cross entropy loss layer.
    """
    def __init__(self, model, max_classes):
        self.model = model
        self.max_classes = max_classes

    def __call__(self, logit, label):
        return self.forward(logit, label)

    def forward(self, logit, label):
        self.label = label
        self.batch_size = logit.shape[0]
        self.num_classes = logit.shape[1]
        self.p = np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)
        loss = -np.sum(np.log(self.p[np.arange(self.batch_size), label])) / self.batch_size
        return loss

    def backward(self, loss_grad=1.0):
        dLogit = self.p.copy()
        dLogit[np.arange(self.batch_size), self.label] -= 1
        dLogit /= self.batch_size
        return dLogit