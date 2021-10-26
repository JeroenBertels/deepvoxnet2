import tensorflow as tf
from deepvoxnet2.backwards_compatibility.metrics import categorical_dice_score, binary_dice_score
from tensorflow.python.keras import backend as K


def l2_loss(y_true, y_pred, **kwargs):
    return tf.math.squared_difference(y_true, y_pred) / 2


def binary_crossentropy(y_true, y_pred, from_logits=False, **kwargs):
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1, keepdims=True)


def binary_dice_loss(y_true, y_pred, threshold=None, **kwargs):
    return 1 - binary_dice_score(y_true, y_pred, threshold=threshold, **kwargs)


def categorical_crossentropy(y_true, y_pred, from_logits=False, **kwargs):
    return K.expand_dims(K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits))


def categorical_dice_loss(y_true, y_pred, threshold=None, **kwargs):
    return 1 - categorical_dice_score(y_true, y_true * y_pred, threshold=threshold, **kwargs)
