import tensorflow as tf
from tensorflow.keras.metrics import AUC, CategoricalAccuracy, BinaryCrossentropy, CategoricalCrossentropy
from pymirc.metrics.tf_metrics import generalized_dice_coeff


# def auc(y_true, y_pred, sample_weight=None, **kwargs):
#     return AUC(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_accuracy(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    sample_weight = tf.ones_like(y_true) * tf.cast(sample_weight, y_true.dtype) if sample_weight is not None else tf.ones_like(y_true)
    return tf.reduce_sum(sample_weight * tf.cast(tf.math.equal(y_true, y_pred), y_true.dtype), axis=(1, 2, 3, 4)) / tf.reduce_sum(sample_weight, axis=(1, 2, 3, 4))


# def binary_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
#     return BinaryCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_score(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError

    return generalized_dice_coeff(y_true, y_pred, threshold=threshold, **kwargs)


def binary_true_positives(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError
    
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
    
    return tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))


def binary_true_negatives(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError
    
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=(1, 2, 3, 4))


def binary_false_positives(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError
    
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum((1 - y_true) * y_pred, axis=(1, 2, 3, 4))


def binary_false_negatives(y_true, y_pred, sample_weight=None, threshold=0.5, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError

    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(y_true * (1 - y_pred), axis=(1, 2, 3, 4))


def binary_true_volume(y_true, y_pred, sample_weight=None, threshold=None, voxel_volume=1, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError

    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
    
    return tf.reduce_sum(y_true, axis=(1, 2, 3, 4)) * tf.cast(voxel_volume, y_true.dtype)


def binary_pred_volume(y_true, y_pred, sample_weight=None, threshold=None, voxel_volume=1, **kwargs):
    if sample_weight is not None:
        raise NotImplementedError

    if threshold is not None:
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(y_pred, axis=(1, 2, 3, 4)) * tf.cast(voxel_volume, y_pred.dtype)


def binary_volume_difference(y_true, y_pred, sample_weight=None, threshold=0.5, voxel_volume=1, **kwargs):
    return binary_pred_volume(y_true, y_pred, sample_weight=sample_weight, threshold=threshold, voxel_volume=voxel_volume) - binary_true_volume(y_true, y_pred, sample_weight=sample_weight, threshold=threshold, voxel_volume=voxel_volume)


def binary_abs_volume_difference(y_true, y_pred, sample_weight=None, threshold=0.5, voxel_volume=1, **kwargs):
    return tf.abs(binary_pred_volume(y_true, y_pred, sample_weight=sample_weight, threshold=threshold, voxel_volume=voxel_volume) - binary_true_volume(y_true, y_pred, sample_weight=sample_weight, threshold=threshold, voxel_volume=voxel_volume))


# def categorical_accuracy(y_true, y_pred, sample_weight=None, **kwargs):
#     return CategoricalAccuracy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


# def categorical_crossentropy(y_true, y_pred, sample_weight=None, **kwargs):
#     return CategoricalCrossentropy(**kwargs)(y_true, y_pred, sample_weight=sample_weight)


def categorical_dice_score(y_true, y_pred, sample_weight=None, exclude_background=True, threshold='argmax',  **kwargs):
    if threshold == "argmax":
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), y_pred.shape[-1])
        threshold = None
    if exclude_background:
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    return generalized_dice_coeff(y_true, y_pred, threshold=threshold,  **kwargs)
