from sklearn.metrics import roc_curve, auc, make_scorer


def gini(y_true, y_score, pos_label=None):
    """
    Calculate gini index.

    Args:
            y_true:
            y_score:
            pos_label:

    Returns:

    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    return 2 * auc(fpr, tpr) - 1


def scorer():
    return make_scorer(gini, needs_proba=True)