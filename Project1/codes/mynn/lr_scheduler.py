from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones, gamma=0.1) -> None:
        """
        optimizer: 优化器对象
        milestones: 一个列表，表示在哪些 step 调整学习率
        gamma: 学习率调整的乘法因子
        """
        super().__init__(optimizer)
        self.milestones = set(milestones)  # 转为集合以加速查找
        self.gamma = gamma

    def step(self) -> None:
        """
        每次调用时增加 step 计数，并在达到 milestone 时调整学习率。
        """
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(scheduler):
    pass