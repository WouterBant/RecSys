from torch.optim import Optimizer
import math
import wandb


class CosineWarmupScheduler:
    def __init__(self, optimizer: Optimizer, max_lr: float, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        elif self.current_step <= self.total_steps:
            lr = self.max_lr * 0.5 * (1 + math.cos(math.pi * (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        else:
            lr = 0

        wandb.log({'lr': lr})
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr