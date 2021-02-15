import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from pymirc.metrics.tf_metrics import generalized_dice_coeff


def binary_accuracy(y_true, y_pred, sample_weight=None, **kwargs):
    return BinaryAccuracy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_score(y_true, y_pred, sample_weight=None, **kwargs):
    kwargs["threshold"] = kwargs.get("threshold", 0.5)
    return generalized_dice_coeff(y_true, y_pred, **kwargs)


def binary_auc(y_true, y_pred, sample_weight=None, **kwargs):
    kwargs["multi_label"] = kwargs.get("multi_label", False)
    assert kwargs['multi_label'] is False, "A binary AUC is just treating every point in the array as a sample. However it makes no difference here, please be consistent and put multi_label to False."
    return AUC(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_volume_difference(y_true, y_pred, sample_weight=None, threshold=None):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.math.reduce_sum(y_pred, axis=(1, 2, 3, 4)) - tf.math.reduce_sum(y_true, axis=(1, 2, 3, 4))


def categorical_dice_score(y_true, y_pred, sample_weight=None, **kwargs):
    kwargs["threshold"] = kwargs.get("threshold", 0.5)
    kwargs["reduce_along_features"] = kwargs.get("reduce_along_features", False)
    assert kwargs['reduce_along_features'] is False, "If a reduction is performed across the feature/class dimension, then you are actually calculating a binary dice score on N+1 dimensions."
    return generalized_dice_coeff(y_true, y_pred, **kwargs)


def categorical_auc(y_true, y_pred, sample_weight=None, **kwargs):
    kwargs["multi_label"] = kwargs.get("multi_label", True)
    assert kwargs['multi_label'] is True, "As opposed to the binary AUC, the categorical AUC should result in the average AUC of each feature/class and thus multi_label should be set to True."
    return AUC(**kwargs)(y_true, y_pred, sample_weight=sample_weight)
