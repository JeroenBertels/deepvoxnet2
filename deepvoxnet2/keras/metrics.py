from pymirc.metrics.tf_metrics import generalized_dice_coeff


def binary_dice_score(y_true, y_pred, sample_weight=None, **kwargs):
    if "threshold" not in kwargs:
        kwargs["threshold"] = 0.5

    return generalized_dice_coeff(y_true, y_pred, **kwargs)
