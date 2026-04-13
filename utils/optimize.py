import torch.optim as optim
import math


def get_optimizer(config, parameters):
    """Create optimizer from config.

    Supports Adam, AdamW (recommended), RMSProp, and SGD.

    Args:
        config: Config namespace with optim.* fields.
        parameters: Model parameters to optimize.

    Returns:
        torch.optim.Optimizer instance.
    """
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(
            parameters, lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps
        )
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            parameters, lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(0.9, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps
        )
    elif config.optim.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(
            parameters, lr=config.optim.lr,
            weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == 'SGD':
        optimizer = optim.SGD(
            parameters, lr=config.optim.lr, momentum=0.9
        )
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer)
        )

    return optimizer


def get_scheduler(config, optimizer, total_steps):
    """Create learning rate scheduler from config.

    Supports:
        - cosine_warmup: Linear warmup → cosine annealing (recommended)
        - cosine: Cosine annealing without warmup
        - step: Step decay at 50% and 75% of training
        - none: No scheduling

    Args:
        config: Config namespace with optim.scheduler and training.warmup_steps.
        optimizer: The optimizer to schedule.
        total_steps: Total number of training steps.

    Returns:
        torch.optim.lr_scheduler instance or None.
    """
    scheduler_type = getattr(config.optim, 'scheduler', 'none')

    if scheduler_type == 'cosine_warmup':
        warmup_steps = getattr(config.training, 'warmup_steps', 500)

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler

    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-7
        )
        return scheduler

    elif scheduler_type == 'step':
        milestones = [int(total_steps * 0.5), int(total_steps * 0.75)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        )
        return scheduler

    elif scheduler_type == 'none':
        return None

    else:
        raise NotImplementedError(
            'Scheduler {} not understood.'.format(scheduler_type)
        )
