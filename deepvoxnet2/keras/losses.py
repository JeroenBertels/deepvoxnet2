from functools import partial
from deepvoxnet2.keras.metrics import categorical_dice_score, binary_dice_score
from tensorflow.python.keras import backend as K


def binary_crossentropy(y_true, y_pred, from_logits=False, **kwargs):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1, keepdims=True)


def binary_dice_loss(y_true, y_pred, threshold=None, **kwargs):
    return 1 - binary_dice_score(y_true, y_pred, threshold=threshold, **kwargs)


def categorical_crossentropy(y_true, y_pred, from_logits=False, **kwargs):
    return K.expand_dims(K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits))


def categorical_dice_loss(y_true, y_pred, threshold=None, **kwargs):
    return 1 - categorical_dice_score(y_true, y_true * y_pred, threshold=threshold, **kwargs)


def get_loss(loss_name, custom_loss_name=None, **kwargs):
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
    loss.__name__ = loss_name if custom_loss_name is None else custom_loss_name
    return loss
