from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi
import numpy as np

class CosineLR(_LRScheduler):
    """SGD with cosine annealing.
    """

    def __init__(self, optimizer, step_size_min=1e-5, t0=100, tmult=2, curr_epoch=-1, last_epoch=-1):
        self.step_size_min = step_size_min
        self.t0 = t0
        self.tmult = tmult
        self.epochs = curr_epoch
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.epochs += 1

        if self.epochs > self.t0:
            self.t0 *= self.tmult
            self.epochs = 0

        lrs = [self.step_size_min + (0.5 * (base_lr - self.step_size_min) * (1 + cos(self.epochs * pi / self.t0)))
                for base_lr in self.base_lrs]

        return lrs