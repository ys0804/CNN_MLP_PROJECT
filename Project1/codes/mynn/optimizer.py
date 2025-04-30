from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    if hasattr(layer, 'grads') and layer.grads is not None and key in layer.grads and layer.grads[key] is not None:
                        layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]

class Adam(Optimizer):
    def __init__(self, init_lr, model, beta1=0.9, beta2=0.999, epsilon=1e-8, clipnorm=1.0):
        super().__init__(init_lr, model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clipnorm = clipnorm  # 梯度裁剪阈值
        self.m = {}
        self.v = {}
        for layer in self.model.layers:
            if hasattr(layer, 'params'):
                self.m[layer] = {key: np.zeros_like(val) for key, val in layer.params.items()}
                self.v[layer] = {key: np.zeros_like(val) for key, val in layer.params.items()}
        self.t = 0

    def step(self):
        self.t += 1
        for layer in self.model.layers:
            if hasattr(layer, 'params'):
                for key in layer.params:
                    if hasattr(layer, 'grads') and layer.grads is not None and key in layer.grads and layer.grads[key] is not None:
                        grad = layer.grads[key]

                        # 梯度裁剪
                        norm = np.linalg.norm(grad)
                        if norm > self.clipnorm:
                            grad = grad * self.clipnorm / norm

                        self.m[layer][key] = self.beta1 * self.m[layer][key] + (1 - self.beta1) * grad
                        self.v[layer][key] = self.beta2 * self.v[layer][key] + (1 - self.beta2) * (grad ** 2)

                        m_hat = self.m[layer][key] / (1 - self.beta1 ** self.t)
                        v_hat = self.v[layer][key] / (1 - self.beta2 ** self.t)

                        layer.params[key] = layer.params[key] - self.init_lr * m_hat / (np.sqrt(v_hat) + self.epsilon)