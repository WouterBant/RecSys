import math


class CosineWarmupScheduler:
    """
    The iterative application of LayerNorms and Dropout in the Transformer model
    lead to a high variance in the gradients in the early training steps.
    To cope with this, the learning rate is increased linearly for a few steps
    before being decayed using a standard learning rate decay schedule. In this
    work, we use a cosine annealing schedule.
    """

    def __init__(self, optimizer, max_lr, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:  # Linear warmup phase
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        elif self.current_step <= self.total_steps:  # Cosine annealing phase
            done_fraction = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            cosine_value = math.cos(math.pi * done_fraction)
            lr = self.max_lr * 0.5 * (1 + cosine_value)
        else:
            lr = 0

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr  # Used for wandb logging
