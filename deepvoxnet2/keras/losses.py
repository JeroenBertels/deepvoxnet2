from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from deepvoxnet2.keras.metrics import binary_dice_score, categorical_dice_score


def binary_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return BinaryCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_loss(y_true, y_pred, sample_weight=None, **kwargs):
    kwargs["threshold"] = kwargs.get("threshold", None)
    assert kwargs["threshold"] is None, "A dice loss function needs to use soft outputs and thus no thresholding operation can be done."
    return 1 - binary_dice_score(y_true, y_pred, sample_weight=sample_weight, **kwargs)


def categorical_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return CategoricalCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def categorical_dice_loss(y_true, y_pred, sample_weight=None, **kwargs):
    kwargs["threshold"] = kwargs.get("threshold", None)
    assert kwargs["threshold"] is None, "A dice loss function needs to use soft outputs and thus no thresholding operation can be done."
    return 1 - categorical_dice_score(y_true, y_pred, sample_weight=sample_weight, **kwargs)
