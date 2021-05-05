from functools import partial
import tensorflow as tf
from pymirc.metrics.tf_metrics import generalized_dice_coeff


def binary_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(tf.cast(tf.math.equal(y_true, y_pred), y_true.dtype), axis=(1, 2, 3, 4), keepdims=True)


def binary_dice_score(y_true, y_pred, threshold=0.5, **kwargs):
    return generalized_dice_coeff(y_true, y_pred, keepdims=True, threshold=threshold, **kwargs)


def binary_true_positives(y_true, y_pred, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
    
    return tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4), keepdims=True)


def binary_true_negatives(y_true, y_pred, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=(1, 2, 3, 4), keepdims=True)


def binary_false_positives(y_true, y_pred, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum((1 - y_true) * y_pred, axis=(1, 2, 3, 4), keepdims=True)


def binary_false_negatives(y_true, y_pred, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(y_true * (1 - y_pred), axis=(1, 2, 3, 4), keepdims=True)


def binary_true_volume(y_true, y_pred, threshold=None, voxel_volume=1, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
    
    return tf.reduce_sum(y_true, axis=(1, 2, 3, 4), keepdims=True) * tf.cast(voxel_volume, y_true.dtype)


def binary_pred_volume(y_true, y_pred, threshold=None, voxel_volume=1, **kwargs):
    if threshold is not None:
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(y_pred, axis=(1, 2, 3, 4), keepdims=True) * tf.cast(voxel_volume, y_pred.dtype)


def binary_volume_difference(y_true, y_pred, threshold=0.5, voxel_volume=1, **kwargs):
    return binary_pred_volume(y_true, y_pred, threshold=threshold, voxel_volume=voxel_volume) - binary_true_volume(y_true, y_pred, threshold=threshold, voxel_volume=voxel_volume)


def binary_abs_volume_difference(y_true, y_pred, threshold=0.5, voxel_volume=1, **kwargs):
    return tf.abs(binary_pred_volume(y_true, y_pred, threshold=threshold, voxel_volume=voxel_volume) - binary_true_volume(y_true, y_pred, threshold=threshold, voxel_volume=voxel_volume))


def categorical_dice_score(y_true, y_pred, exclude_background=True, threshold='argmax',  **kwargs):
    if threshold == "argmax":
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), y_pred.shape[-1])
        threshold = None

    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]

    return generalized_dice_coeff(y_true, y_pred, keepdims=True, threshold=threshold,  **kwargs)


def get_metric(metric_name, **kwargs):
    if metric_name == "binary_accuracy":
        metric = binary_accuracy

    elif metric_name == "binary_dice_score":
        metric = binary_dice_score

    elif metric_name == "binary_true_positives":
        metric = binary_true_positives

    elif metric_name == "binary_true_negatives":
        metric = binary_true_negatives

    elif metric_name == "binary_false_positives":
        metric = binary_false_positives

    elif metric_name == "binary_false_negatives":
        metric = binary_false_negatives

    elif metric_name == "binary_true_volume":
        metric = binary_true_volume

    elif metric_name == "binary_pred_volume":
        metric = binary_pred_volume

    elif metric_name == "binary_volume_difference":
        metric = binary_volume_difference

    elif metric_name == "binary_abs_volume_difference":
        metric = binary_abs_volume_difference

    elif metric_name == "categorical_dice_score":
        metric = categorical_dice_score

    else:
        raise ValueError("The requested metric is unknown.")

    metric = partial(metric, **kwargs)
    metric.__name__ = metric_name
    return metric
