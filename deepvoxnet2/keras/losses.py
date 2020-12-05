from tensorflow.keras.losses import BinaryCrossentropy
from deepvoxnet2.keras.metrics import binary_dice_score


def binary_crossentropy(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_loss(y_true, y_pred, sample_weight=None, **kwargs):
    assert "threshold" not in kwargs or kwargs["threshold"] is None
    return 1 - binary_dice_score(y_true, y_pred, sample_weight=sample_weight, **kwargs)
