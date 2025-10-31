
import numpy as np

def f1(y_true, y_pred, labels=[0, 1]):
    """
    Compute the F1 score between true and predicted labels.

    Args:
        y_true : numpy array of shape (N,)
        y_pred : numpy array of shape (N,)
        labels : list, either [0,1] or [-1,1]

    Returns:
        f1 : float, F1 score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # handle label sets
    if labels == [-1, 1]:
        # convert to 0/1 for easier counting
        y_true = (y_true + 1) // 2
        y_pred = (y_pred + 1) // 2
    elif labels != [0, 1]:
        raise ValueError("labels must be [0,1] or [-1,1]")

    # compute confusion terms
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))


    # precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # f1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def accuracy(y_true, y_pred, labels=[0, 1]):
    """
    Compute the accuracy between true and predicted labels.

    Args:
        y_true : numpy array of shape (N,)
        y_pred : numpy array of shape (N,)
        labels : list, either [0,1] or [-1,1]

    Returns:
        acc : float, accuracy score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # handle label sets
    if labels == [-1, 1]:
        # convert to 0/1 for comparison consistency
        y_true = (y_true + 1) // 2
        y_pred = (y_pred + 1) // 2
    elif labels != [0, 1]:
        raise ValueError("labels must be [0,1] or [-1,1]")

    # compute accuracy
    correct = np.sum(y_true == y_pred)
    acc = correct / len(y_true) if len(y_true) > 0 else 0.0

    return acc