import numpy as np  # For simple_accuracy to handle arrays

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset() # This is done to ensure that the instance starts with clean state.

    def reset(self):
        self.val = 0 
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val # This keeps track of the latest value added.
        self.sum += val * n   #n is count of value
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean() # giving the fraction of correct predictions out of the total predictions.
