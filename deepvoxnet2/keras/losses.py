from functools import partial
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from deepvoxnet2.keras.metrics import categorical_dice_score, binary_dice_score


def binary_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return BinaryCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_loss(y_true, y_pred, sample_weight=None, threshold=None, **kwargs):
    return 1 - binary_dice_score(y_true, y_pred, sample_weight=sample_weight, threshold=threshold, **kwargs)


def get_binary_dice_loss(threshold=None, **kwargs):
    loss = partial(binary_dice_loss, threshold=threshold, **kwargs)
    loss.__name__ = "binary_dice_loss"
    return loss


def categorical_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return CategoricalCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def categorical_dice_loss(y_true, y_pred, sample_weight=None, threshold=None, **kwargs):
    return 1 - categorical_dice_score(y_true, y_true * y_pred, sample_weight=sample_weight, threshold=threshold, **kwargs)
