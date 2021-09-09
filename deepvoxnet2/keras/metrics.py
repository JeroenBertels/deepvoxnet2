import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from pymirc.metrics.tf_metrics import generalized_dice_coeff


def mean_squared_error(y_true, y_pred, **kwargs):
    return tf.math.squared_difference(y_true, y_pred)


def mean_absolute_error(y_true, y_pred, **kwargs):
    return tf.math.abs(y_pred - y_true)


def median_absolute_error(y_true, y_pred, percentile=50, interpolation='midpoint', **kwargs):
    return tfp.stats.percentile(tf.math.abs(y_pred - y_true), percentile, interpolation=interpolation, keepdims=True)


def coefficient_of_determination(y_true, y_pred, **kwargs):
    ss_res = tf.reduce_sum(tf.math.square(y_true - y_pred), keepdims=True)
    ss_tot = tf.reduce_sum(tf.math.square(y_true - tf.reduce_mean(y_true)), keepdims=True)
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


def binary_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(tf.cast(tf.math.equal(y_true, y_pred), y_true.dtype), axis=(1, 2, 3, 4), keepdims=True)


def binary_ece(y_true, y_pred, nbins=10, threshold=0.5, quantiles_as_bins=False, **kwargs):
    y_true = tf.reshape(y_true, [y_true.shape[0], -1, 1])
    y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1, 1])
    y_true_threshold = tf.math.greater(y_true, threshold)
    y_pred_threshold = tf.math.greater(y_pred, threshold)
    if quantiles_as_bins:
        y_pred_hits = tf.math.equal(y_true_threshold, y_pred_threshold)
        y_pred_log = tf.math.log(y_pred)
        return tf.map_fn(lambda x: tfp.stats.expected_calibration_error_quantiles(x[0], x[1], nbins)[0], (y_pred_hits, y_pred_log), fn_output_signature=tf.float32)

    else:
        y_true_labels = tf.cast(y_true_threshold, tf.int32)[..., 0]
        y_pred_logits = tf.pad(tf.math.log(y_pred / (1 - y_pred)), tf.constant([[0, 0], [0, 0], [1, 0]]))
        return tf.map_fn(lambda x: tfp.stats.expected_calibration_error(nbins, x[0], x[1]), (y_pred_logits, y_true_labels), fn_output_signature=tf.float32)


# def binary_auc(y_true, y_pred, num_thresholds=26, curve='PR', thresholds=None, **kwargs):
#     y_true = tf.reshape(y_true, [y_true.shape[0], -1])
#     y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1])
#     return tf.map_fn(lambda x: tf.keras.metrics.AUC(num_thresholds=num_thresholds, curve=curve, thresholds=thresholds)(x[0], x[1]), (y_true, y_pred), fn_output_signature=tf.float32)


def binary_auc(y_true, y_pred, num_thresholds=101, curve='PR', thresholds=None, reduce_batch_dim_before_summation=True, **kwargs):
    if thresholds is None:
        thresholds = tf.linspace(-tf.keras.backend.epsilon(), 1 + tf.keras.backend.epsilon(), num_thresholds)

    else:
        thresholds = tf.convert_to_tensor(thresholds, dtype=tf.float32)

    tps = tf.map_fn(lambda x: binary_true_positives(y_true, y_pred, threshold=x), thresholds)
    fps = tf.map_fn(lambda x: binary_false_positives(y_true, y_pred, threshold=x), thresholds)
    tns = tf.map_fn(lambda x: binary_true_negatives(y_true, y_pred, threshold=x), thresholds)
    fns = tf.map_fn(lambda x: binary_false_negatives(y_true, y_pred, threshold=x), thresholds)
    specificities = (tns + tf.keras.backend.epsilon()) / (tns + fps + tf.keras.backend.epsilon())
    precisions = (tps + tf.keras.backend.epsilon()) / (tps + fps + tf.keras.backend.epsilon())
    recalls = (tps + tf.keras.backend.epsilon()) / (tps + fns + tf.keras.backend.epsilon())
    if curve =="PR":
        x, y = recalls, precisions

    elif curve == "ROC":
        x, y = specificities, recalls

    else:
        raise ValueError("The requested curve is not yet implemented.")

    if reduce_batch_dim_before_summation:
        x, y = tf.reduce_mean(x, axis=1, keepdims=True), tf.reduce_mean(y, axis=1, keepdims=True)

    x, y = tf.transpose(x), tf.transpose(y)
    sort_order = tf.argsort(x, axis=1)
    x = tf.map_fn(lambda x: tf.gather(x[1], x[0]), (sort_order, x), fn_output_signature=tf.float32)
    y = tf.map_fn(lambda x: tf.gather(x[1], x[0]), (sort_order, y), fn_output_signature=tf.float32)
    x, y = tf.transpose(x), tf.transpose(y)
    x = tf.pad(x, tf.constant([[1, 0], [0, 0]]))
    x = tf.pad(x, tf.constant([[0, 1], [0, 0]]), constant_values=1)
    y = tf.pad(y, tf.constant([[1, 1], [0, 0]]), "SYMMETRIC")
    return tf.reduce_sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2, axis=0)


def binary_dice_score(y_true, y_pred, threshold=0.5, **kwargs):
    return _generalized_dice_coeff(y_true, y_pred, keepdims=True, threshold=threshold, **kwargs)


def _hausdorff_distance(y_true, y_pred, threshold=0.5, voxel_sizes=1, percentile=95, interpolation='midpoint'):
    y_true = tf.expand_dims(tf.cast(tf.math.greater(y_true, threshold), tf.float32), 0)
    y_pred = tf.expand_dims(tf.cast(tf.math.greater(y_pred, threshold), tf.float32), 0)
    y_true = tf.pad(y_true, tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]))
    y_pred = tf.pad(y_pred, tf.constant([[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]))
    y_true_edge = tf.abs(tf.nn.conv3d(y_true, tf.constant([[[[[1]]]], [[[[-1]]]]], tf.float32), [1, 1, 1, 1, 1], "SAME")) + \
                  tf.abs(tf.nn.conv3d(y_true, tf.constant([[[[[1]]], [[[-1]]]]], tf.float32), [1, 1, 1, 1, 1], "SAME")) + \
                  tf.abs(tf.nn.conv3d(y_true, tf.constant([[[[[1]], [[-1]]]]], tf.float32), [1, 1, 1, 1, 1], "SAME"))
    y_pred_edge = tf.abs(tf.nn.conv3d(y_pred, tf.constant([[[[[1]]]], [[[[-1]]]]], tf.float32), [1, 1, 1, 1, 1], "SAME")) + \
                  tf.abs(tf.nn.conv3d(y_pred, tf.constant([[[[[1]]], [[[-1]]]]], tf.float32), [1, 1, 1, 1, 1], "SAME")) + \
                  tf.abs(tf.nn.conv3d(y_pred, tf.constant([[[[[1]], [[-1]]]]], tf.float32), [1, 1, 1, 1, 1], "SAME"))
    y_true_contour = tf.cast(tf.where(tf.not_equal(y_true_edge[0, ..., 0], tf.constant(0, tf.float32))), tf.float32) - tf.constant(0.5, tf.float32)
    y_pred_contour = tf.cast(tf.where(tf.not_equal(y_pred_edge[0, ..., 0], tf.constant(0, tf.float32))), tf.float32) - tf.constant(0.5, tf.float32)
    hdd = tf.cond(
        tf.equal(tf.size(y_true_contour), 0),
        lambda: tf.cond(
            tf.equal(tf.size(y_pred_contour), 0),
            lambda: tf.constant(0, tf.float32),
            lambda: tf.constant(np.inf, tf.float32)
        ),
        lambda: tf.cond(
            tf.equal(tf.size(y_pred_contour), 0),
            lambda: tf.constant(np.inf, tf.float32),
            lambda: (tfp.stats.percentile(
                tf.reduce_min(tf.sqrt(tf.reduce_sum(((tf.expand_dims(y_true_contour, axis=-2) - tf.expand_dims(y_pred_contour, axis=-3)) * tf.constant(voxel_sizes, tf.float32)) ** 2, axis=-1)), axis=-1),
                percentile,
                interpolation=interpolation) +
                    tfp.stats.percentile(
                tf.reduce_min(tf.sqrt(tf.reduce_sum(((tf.expand_dims(y_true_contour, axis=-2) - tf.expand_dims(y_pred_contour, axis=-3)) * tf.constant(voxel_sizes, tf.float32)) ** 2, axis=-1)), axis=-2),
                percentile,
                interpolation=interpolation)) / 2
        )
    )
    return hdd


def binary_hausdorff_distance(y_true, y_pred, threshold=0.5, voxel_sizes=1, percentile=95, interpolation='midpoint', **kwargs):
    return tf.map_fn(lambda x: _hausdorff_distance(x[0], x[1], threshold=threshold, voxel_sizes=voxel_sizes, percentile=percentile, interpolation=interpolation), (y_true, y_pred), fn_output_signature=tf.float32)


def binary_true_positives(y_true, y_pred, threshold=0.5, **kwargs):
    y_true = tf.reshape(y_true, [y_true.shape[0], -1])
    y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1])
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)
    
    return tf.reduce_sum(y_true * y_pred, axis=1, keepdims=False)


def binary_true_negatives(y_true, y_pred, threshold=0.5, **kwargs):
    y_true = tf.reshape(y_true, [y_true.shape[0], -1])
    y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1])
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=1, keepdims=False)


def binary_false_positives(y_true, y_pred, threshold=0.5, **kwargs):
    y_true = tf.reshape(y_true, [y_true.shape[0], -1])
    y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1])
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum((1 - y_true) * y_pred, axis=1, keepdims=False)


def binary_false_negatives(y_true, y_pred, threshold=0.5, **kwargs):
    y_true = tf.reshape(y_true, [y_true.shape[0], -1])
    y_pred = tf.reshape(y_pred, [y_pred.shape[0], -1])
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

    return tf.reduce_sum(y_true * (1 - y_pred), axis=1, keepdims=False)


def binary_true_volume(y_true, y_pred, threshold=0.5, voxel_volume=1, **kwargs):
    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
    
    return tf.reduce_sum(y_true, axis=(1, 2, 3, 4), keepdims=True) * tf.cast(voxel_volume, y_true.dtype)


def binary_pred_volume(y_true, y_pred, threshold=0.5, voxel_volume=1, **kwargs):
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

    return _generalized_dice_coeff(y_true, y_pred, keepdims=True, threshold=threshold,  **kwargs)


def _generalized_dice_coeff(y_true, y_pred, **kwargs):
    return generalized_dice_coeff(
        y_true,
        y_pred,
        eps=kwargs.get("eps", tf.keras.backend.epsilon()),
        reduce_along_batch=kwargs.get("reduce_along_batch", False),
        reduce_along_features=kwargs.get("reduce_along_features", True),
        feature_weights=kwargs.get("feature_weights", None),
        threshold=kwargs.get("threshold", None),
        keepdims=kwargs.get("keepdims", False))


def get_metric(metric_name, custom_metric_name=None, **kwargs):
    if metric_name == "mean_squared_error":
        metric = mean_squared_error

    elif metric_name == "mean_absolute_error":
        metric = mean_absolute_error

    elif metric_name == "median_absolute_error":
        metric = median_absolute_error

    elif metric_name == "coefficient_of_determination":
        metric = coefficient_of_determination

    elif metric_name == "binary_accuracy":
        metric = binary_accuracy

    elif metric_name == "binary_ece":
        metric = binary_ece

    elif metric_name == "binary_auc":
        metric = binary_auc

    elif metric_name == "binary_dice_score":
        metric = binary_dice_score

    elif metric_name == "binary_soft_dice_score":
        metric = binary_dice_score
        kwargs["threshold"] = None

    elif metric_name == "binary_hausdorff_distance":
        metric = binary_hausdorff_distance

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
    metric.__name__ = metric_name if custom_metric_name is None else custom_metric_name
    return metric
