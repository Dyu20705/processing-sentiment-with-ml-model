import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix for classification.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    
    Returns:
    --------
    cm : np.ndarray of shape (n_classes, n_classes)
        Confusion matrix where rows are true labels and columns are predicted labels.
        Element [i, j] = number of samples with true label i and predicted label j.
    classes : np.ndarray
        Array of unique class labels in sorted order.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Create class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Initialize confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill confusion matrix
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm, classes
