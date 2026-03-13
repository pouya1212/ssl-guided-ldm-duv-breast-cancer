# coding=utf-8
"""
Learning rate scheduler classes with warmup and decay strategies.
Includes Constant, Linear, and Cosine schedules.
"""
import logging
import math

from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        """ Initialize the constant schedule. """
        super().__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        """ Initialize the warmup constant schedule. """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        """ Compute the learning rate factor for the current step. """
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        """ Initialize the warmup linear schedule. """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        """ Compute the learning rate factor for the current step. """
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        num = float(self.t_total - step)
        den = float(max(1.0, self.t_total - self.warmup_steps))
        return max(0.0, num / den)


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. following a cosine curve.
    """
    # pylint: disable=too-many-arguments
    def __init__(self, optimizer, warmup_steps, t_total, *, cycles=0.5, last_epoch=-1):
        """ Initialize the warmup cosine schedule. """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        """ Compute the learning rate factor for the current step. """
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        denominator = float(max(1, self.t_total - self.warmup_steps))
        progress = float(step - self.warmup_steps) / denominator
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
