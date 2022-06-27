from typing import List
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: int,
                 total_epochs: int,
                 steps_per_epoch: int,
                 init_lr: float,
                 max_lr: float,
                 final_lr: float):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """

        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = int(self.warmup_epochs * self.steps_per_epoch)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        if self.current_step <= self.warmup_steps:
            self.lr = self.init_lr + self.current_step * self.linear_increment
        elif self.current_step <= self.total_steps:
            self.lr = self.max_lr * (self.exponential_gamma ** (self.current_step - self.warmup_steps))
        else:  # theoretically this case should never be reached since training should stop at total_steps
            self.lr = self.final_lr
        # print('learning rate is: ', self.lr)

        self.optimizer.param_groups[0]['lr'] = self.lr

def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.

    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_optimizer(model: nn.Module, freeze=False) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param freeze: Freezing parameters with grad=False.
    :return: An initialized Optimizer.
    """
    if not freeze:
        params = [{'params': model.parameters(), 'lr': 0.0001, 'weight_decay': 0}]
    else:
        params = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': 0.0001, 'weight_decay': 0}]

    return Adam(params)


def build_lr_scheduler(optimizer:Optimizer,
                      warmup_epochs: int,
                      total_epochs: int,
                      train_data_size: int,
                      batch_size: int,
                      init_lr: float,
                      max_lr: float,
                      final_lr: float):
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=train_data_size // batch_size,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr
    )