""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from solver.cosine_lr import CosineLRScheduler

def create_scheduler(args, optimizer):
    num_epochs = args.num_epoches
    lr_min = args.lr_min * args.lr
    warmup_lr_init = args.warmup_init * args.lr

    # warmup_t = cfg.SOLVER.WARMUP_EPOCHS
    warmup_t = args.warmup_epoch_num
    noise_range = None

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            lr_min=lr_min,
            t_mul=1.,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr_init,
            warmup_t=warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=noise_range,
            noise_pct=0.67,
            noise_std=1.,
            noise_seed=42,
        )

    return lr_scheduler
