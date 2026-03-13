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


def compute_wsi_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (y_true == y_pred).mean()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    return acc, sens, spec, prec, f1


