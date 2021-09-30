import tensorflow as tf
from functools import partial
from deepvoxnet2.keras.metrics import get_metric


def l1_loss(y_true, y_pred, **kwargs):
    return get_metric("absolute_error", **kwargs)(y_true, y_pred)


def l2_loss(y_true, y_pred, **kwargs):
    return get_metric("squared_error", **kwargs)(y_true, y_pred) / 2


def cross_entropy(y_true, y_pred, **kwargs):
    return get_metric("cross_entropy", **kwargs)(y_true, y_pred)


def dice_loss(y_true, y_pred, threshold=None, **kwargs):
    return 1 - get_metric("dice_coefficient", threshold=threshold, **kwargs)(y_true, y_pred)


def get_loss(loss_name, reduction_mode="mean", custom_loss_name=None, **kwargs):
    if loss_name == "l1_loss":
        loss = l1_loss

    elif loss_name == "l2_loss":
        loss = l2_loss

    elif loss_name == "cross_entropy":
        loss = cross_entropy

    elif loss_name == "dice_loss":
        loss = dice_loss

    else:
        raise NotImplementedError("The requested loss is not implemented.")

    loss = partial(loss, reduction_mode=reduction_mode, **kwargs)
    loss.__name__ = loss_name if custom_loss_name is None else custom_loss_name
    return loss


def get_combined_loss(losses, loss_weights=None, custom_combined_loss_name=None):
    if loss_weights is None:
        loss_weights = [1 / len(losses) for _ in losses]

    else:
        assert len(losses) == len(loss_weights)

    def loss_fn(y_true, y_pred):
        weighted_losses = [tf.cast(loss_weight, tf.float32) * loss(y_true, y_pred) for loss, loss_weight in zip(losses, loss_weights)]
        return tf.math.add_n(weighted_losses)

    if custom_combined_loss_name is not None:
        loss_fn.__name__ = custom_combined_loss_name

    return loss_fn
