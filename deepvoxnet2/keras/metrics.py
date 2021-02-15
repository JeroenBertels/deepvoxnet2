import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC, CategoricalAccuracy
from pymirc.metrics.tf_metrics import generalized_dice_coeff


def auc(y_true, y_pred, sample_weight=None, **kwargs):
    return AUC(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_accuracy(y_true, y_pred, sample_weight=None, **kwargs):
    return BinaryAccuracy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return BinaryCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_score(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    return generalized_dice_coeff(y_true, y_pred, reduce_along_features=True, threshold=threshold, **kwargs)


# def binary_volume_difference(y_true, y_pred, sample_weight=None, threshold=None):
#     if threshold is not None:
#         y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
#         y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
#
#     return tf.math.reduce_sum(y_pred, axis=(1, 2, 3, 4)) - tf.math.reduce_sum(y_true, axis=(1, 2, 3, 4))


def categorical_accuracy(y_true, y_pred, sample_weight=None, **kwargs):
    return CategoricalAccuracy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def categorical_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
    return CategoricalCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


# def categorical_dice_score(y_true, y_pred, sample_weight=None, **kwargs):
#     kwargs["threshold"] = kwargs.get("threshold", 0.5)
#     kwargs["reduce_along_features"] = kwargs.get("reduce_along_features", False)
#     assert kwargs["reduce_along_features"] is False, "The categorical version of the dice loss/score should treat every feature/class as one sample to compute the dice loss/score on."
#     return generalized_dice_coeff(y_true, y_pred, **kwargs)
