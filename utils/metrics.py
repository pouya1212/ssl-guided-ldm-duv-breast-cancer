# coding=utf-8
"""
Utility functions for performance metrics, including accuracy, 
WSI-level metrics, and tracking average values.
"""
import numpy as np
from sklearn.metrics import confusion_matrix


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        """Initialize the meter and reset all values."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """Resets all tracked values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the current value and recalculates the average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    """
    Computes simple accuracy by comparing predictions to labels.
    Gives the fraction of correct predictions out of the total.
    """
    return (preds == labels).mean()


def compute_wsi_metrics(y_true, y_pred):
    """
    Computes WSI-level metrics including Accuracy, Sensitivity, 
    Specificity, Precision, and F1-score.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    return acc, sens, spec, prec, f1
