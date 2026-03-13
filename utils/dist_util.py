# coding=utf-8
"""
Distributed training utility functions for retrieving process rank,
world size, and formatting training steps.
"""
import torch.distributed as dist


def get_rank():
    """
    Retrieves the rank (identifier) of the current process.
    Returns 0 if distributed training is not available or initialized.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Returns the total number of processes in the distributed setup.
    Returns 1 if distributed training is not available or initialized.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    """
    Checks if the current process is the main process (rank 0).
    """
    return get_rank() == 0


def format_step(step):
    """
    Formats a string describing training and validation steps.
    Accepts a string or a list/tuple containing (epoch, train_iter, val_iter).
    """
    if isinstance(step, str):
        return step
    s = ""
    if len(step) > 0:
        s += f"Training Epoch: {step[0]} "
    if len(step) > 1:
        s += f"Training Iteration: {step[1]} "
    if len(step) > 2:
        s += f"Validation Iteration: {step[2]} "
    return s.strip()
