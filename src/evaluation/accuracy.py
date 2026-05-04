import numpy as np


def accuracy(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    
    Returns:
    --------
    accuracy : float
        Accuracy score between 0 and 1.
        Accuracy = (# correct predictions) / (# total predictions)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total
