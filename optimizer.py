
import torch

class ManualSGD:
    def __init__(self, parameters, lr=1e-2):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self, grads):
        for p, g in zip(self.parameters, grads):
            if p.grad is not None:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()
