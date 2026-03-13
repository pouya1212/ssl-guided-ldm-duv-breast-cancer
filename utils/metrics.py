# coding=utf-8
"""
Utility functions for performance metrics, including accuracy, 
WSI-level metrics, and tracking average values.
"""
import numpy as np
from sklearn.metrics import confusion_matrix


class AverageMeter:
    """Computes and stores the average and current value."""

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
    
    Returns:
        float: The fraction of correct predictions.
    """
    return (preds == labels).mean()


def compute_wsi_metrics(y_true, y_pred):
    """
    Computes WSI-level metrics: Accuracy, Sensitivity, Specificity, Precision, and F1.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        tuple: (accuracy, sensitivity, specificity, precision, f1_score)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Ensure confusion matrix is always 2x2 for binary classification
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1_score = (2 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0
    
    return acc, sens, spec, prec, f1_score
