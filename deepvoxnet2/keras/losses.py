from functools import partial
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from deepvoxnet2.keras.metrics import categorical_dice_score, binary_dice_score


def binary_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return BinaryCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_loss(y_true, y_pred, sample_weight=None, threshold=None, **kwargs):
    return 1 - binary_dice_score(y_true, y_pred, sample_weight=sample_weight, threshold=threshold, **kwargs)


def categorical_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return CategoricalCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def categorical_dice_loss(y_true, y_pred, sample_weight=None, threshold=None, **kwargs):
    return 1 - categorical_dice_score(y_true, y_true * y_pred, sample_weight=sample_weight, threshold=threshold, **kwargs)


def get_loss(loss_name, **kwargs):
    if loss_name == "binary_crossentropy":
        loss = binary_crossentropy

    elif loss_name == "binary_dice_loss":
        loss = binary_dice_loss

    elif loss_name == "categorical_crossentropy":
        loss = categorical_crossentropy

    elif loss_name == "categorical_dice_loss":
        loss = categorical_dice_loss

    else:
        raise ValueError("The requested loss is unknown.")

    loss = partial(loss, **kwargs)
    loss.__name__ = loss_name
    return loss
