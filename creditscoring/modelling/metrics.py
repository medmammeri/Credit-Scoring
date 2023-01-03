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


def hello(name: str) -> str:
    """Just a greetings example.

    Args:
        name (str): Name to greet.

    Returns:
        str: greeting message

    Examples:
        .. code:: python

            >>> hello("Roman")
            'Hello Roman!'
    """
    return f"Hello {name}!"
