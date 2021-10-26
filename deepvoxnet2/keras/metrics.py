import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from functools import partial
from collections import Iterable
from pymirc.metrics.tf_metrics import generalized_dice_coeff
from deepvoxnet2.backwards_compatibility.metrics import *


def _expand_binary(y_true, y_pred):
    y_true_shape, y_pred_shape = tf.keras.backend.int_shape(y_true), tf.keras.backend.int_shape(y_pred)
    if y_true_shape[-1] == 1 or y_pred_shape[-1] == 1:
        assert y_true_shape[-1] == 1 and y_pred_shape[-1] == 1
        y_true = tf.concat([1 - y_true, y_true], axis=-1)
        y_pred = tf.concat([1 - y_pred, y_pred], axis=-1)

    return y_true, y_pred


def error(y_true, y_pred, **kwargs):
    return y_pred - y_true


def absolute_error(y_true, y_pred, **kwargs):
    return tf.math.abs(error(y_true, y_pred))


def squared_error(y_true, y_pred, **kwargs):
    err = error(y_true, y_pred)
    return err ** 2


def cross_entropy(y_true, y_pred, from_logits=False, **kwargs):
    y_true, y_pred = _expand_binary(y_true, y_pred)
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)[..., None]


def accuracy(y_true, y_pred, **kwargs):
    tp = tf.reduce_sum(true_positive(y_true, y_pred), axis=-1, keepdims=True)
    tn = tf.reduce_sum(true_negative(y_true, y_pred), axis=-1, keepdims=True)
    return (tp + tn) / y_true.shape[-1]


def true_positive(y_true, y_pred, **kwargs):
    return y_true * y_pred


def true_negative(y_true, y_pred, **kwargs):
    return (1 - y_true) * (1 - y_pred)


def false_positive(y_true, y_pred, **kwargs):
    return (1 - y_true) * y_pred


def false_negative(y_true, y_pred, **kwargs):
    return y_true * (1 - y_pred)


def true_volume(y_true, y_pred, voxel_volume=1, **kwargs):
    return tf.reduce_sum(y_true, axis=(1, 2, 3), keepdims=True) * voxel_volume


def pred_volume(y_true, y_pred, voxel_volume=1, **kwargs):
    return tf.reduce_sum(y_pred, axis=(1, 2, 3), keepdims=True) * voxel_volume


def volume_error(y_true, y_pred, voxel_volume=1, **kwargs):
    return tf.reduce_sum(error(y_true, y_pred), axis=(1, 2, 3), keepdims=True) * voxel_volume


def absolute_volume_error(y_true, y_pred, voxel_volume=1, **kwargs):
    return tf.reduce_sum(absolute_error(y_true, y_pred), axis=(1, 2, 3), keepdims=True) * voxel_volume


def positive_predictive_value(y_true, y_pred, eps=tf.keras.backend.epsilon(), **kwargs):
    tp = tf.reduce_sum(true_positive(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    fp = tf.reduce_sum(false_positive(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    return (tp + eps) / (tp + fp + eps)


def negative_predictive_value(y_true, y_pred, eps=tf.keras.backend.epsilon(), **kwargs):
    tn = tf.reduce_sum(true_negative(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    fn = tf.reduce_sum(false_negative(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    return (tn + eps) / (tn + fn + eps)


def true_positive_rate(y_true, y_pred, eps=tf.keras.backend.epsilon(), **kwargs):
    tp = tf.reduce_sum(true_positive(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    fn = tf.reduce_sum(false_negative(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    return (tp + eps) / (tp + fn + eps)


def true_negative_rate(y_true, y_pred, eps=tf.keras.backend.epsilon(), **kwargs):
    tn = tf.reduce_sum(true_negative(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    fp = tf.reduce_sum(false_positive(y_true, y_pred, **kwargs), axis=(1, 2, 3), keepdims=True)
    return (tn + eps) / (tn + fp + eps)


def dice_coefficient(y_true, y_pred, eps=tf.keras.backend.epsilon(), reduce_along_batch=False, reduce_along_features=False, feature_weights=None, **kwargs):
    return generalized_dice_coeff(y_true, y_pred, keepdims=True, eps=eps, reduce_along_batch=reduce_along_batch, reduce_along_features=reduce_along_features, feature_weights=feature_weights)


def coefficient_of_determination(y_true, y_pred, **kwargs):
    ss_res = tf.reduce_sum(tf.math.square(y_true - y_pred), axis=(1, 2, 3), keepdims=True)
    ss_tot = tf.reduce_sum(tf.math.square(y_true - tf.reduce_mean(y_true, axis=(1, 2, 3), keepdims=True)), axis=(1, 2, 3), keepdims=True)
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


def ece_from_bin_stats(y_true, y_pred, **kwargs):
    y_true = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    bin_confidence = y_true[:, :, :1, ...]
    bin_accuracy = y_true[:, :, 1:2, ...]
    bin_count = y_true[:, :, 2:, ...]
    return tf.reduce_sum(tf.abs(bin_confidence - bin_accuracy) * bin_count, axis=1, keepdims=True) / tf.reduce_sum(bin_count, axis=1, keepdims=True)


def ece(y_true, y_pred, nbins=10, quantiles_as_bins=False, return_bin_stats=False, from_bin_stats=False, **kwargs):
    y_true, y_pred = _expand_binary(y_true, y_pred)
    y_true = tf.reshape(y_true, [-1, y_true.shape[4]])
    y_pred = tf.reshape(y_pred, [-1, y_pred.shape[4]])
    labels_true = tf.math.argmax(y_true, axis=-1)
    labels_pred = tf.math.argmax(y_pred, axis=-1)
    confidence = tf.math.reduce_max(y_pred, axis=-1)
    hits = tf.equal(labels_true, labels_pred)
    if quantiles_as_bins:
        edges = tfp.stats.percentile(confidence, tf.linspace(0, 100, nbins), interpolation="midpoint")
        if not return_bin_stats and not from_bin_stats:
            return tfp.stats.expected_calibration_error_quantiles(hits, tf.math.log(confidence), nbins)[0][None, None, None, None, None]

    else:
        edges = tf.cast(tf.linspace(0, 1, nbins), tf.float32)
        if not return_bin_stats and not from_bin_stats:
            return tfp.stats.expected_calibration_error(nbins, tf.math.log(y_pred), labels_true)[None, None, None, None, None]

    bins = tfp.stats.find_bins(confidence, edges=edges)
    stats = []
    for i in range(nbins):
        mask = tf.equal(bins, i)
        bin_confidence = tf.reduce_mean(tf.boolean_mask(confidence, mask))
        bin_accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(hits, tf.float32), mask))
        bin_count = tf.reduce_sum(tf.cast(mask, tf.float32))
        stats.append([bin_confidence, bin_accuracy, bin_count])

    bin_stats = tf.convert_to_tensor(stats, tf.float32)[None, ..., None, None]
    if return_bin_stats:
        return bin_stats

    else:
        return ece_from_bin_stats(bin_stats, bin_stats)


def riemann_sum(y_true, y_pred, reduce_mean_axes=(1, 2, 3, 4), **kwargs):
    x, y = y_true, y_pred
    assert len(reduce_mean_axes) == 4
    x, y = tf.reduce_mean(x, axis=reduce_mean_axes), tf.reduce_mean(y, axis=reduce_mean_axes)
    indices = tf.argsort(x)
    x, y = tf.gather(x, indices), tf.gather(y, indices)
    return tf.reduce_sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2, axis=0, keepdims=True)[None, None, None, None]


def auc(y_true, y_pred, thresholds=np.linspace(tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon(), 51), curve='PR', **kwargs):
    thresholds = tf.convert_to_tensor(thresholds, dtype=tf.float32)
    if curve == "PR":
        x = recall = get_metric_at_multiple_thresholds("true_positive_rate", thresholds, threshold_axis=1, **kwargs)(y_true, y_pred)
        y = precision = get_metric_at_multiple_thresholds("positive_predictive_value", thresholds, threshold_axis=1, **kwargs)(y_true, y_pred)

    elif curve == "ROC":
        specificity = get_metric_at_multiple_thresholds("true_negative_rate", thresholds, threshold_axis=1, **kwargs)(y_true, y_pred)
        x = 1 - specificity
        y = recall = get_metric_at_multiple_thresholds("true_positive_rate", thresholds, threshold_axis=1, **kwargs)(y_true, y_pred)

    else:
        raise ValueError("The requested curve is not yet implemented.")

    return riemann_sum(x, y, reduce_mean_axes=(0, 2, 3, 4))


def hausdorff_distance(y_true, y_pred, min_edge_diff=1, voxel_size=1, hd_percentile=95, **kwargs):
    y_true_shape = tf.keras.backend.int_shape(y_true)
    edge_filter = np.zeros((2, 1, 1, y_true_shape[4], y_true_shape[4]))
    for i in range(y_true_shape[4]):
        edge_filter[0, 0, 0, i, i], edge_filter[1, 0, 0, i, i] = 1, -1

    y_true_contour = tf.cast(tf.where(tf.math.reduce_any([
        tf.greater_equal(tf.abs(tf.nn.conv3d(y_true, tf.constant(edge_filter, tf.float32), [1, 1, 1, 1, 1], "SAME")), min_edge_diff),
        tf.greater_equal(tf.abs(tf.nn.conv3d(y_true, tf.constant(np.transpose(edge_filter, (1, 0, 2, 3, 4)), tf.float32), [1, 1, 1, 1, 1], "SAME")), min_edge_diff),
        tf.greater_equal(tf.abs(tf.nn.conv3d(y_true, tf.constant(np.transpose(edge_filter, (2, 1, 0, 3, 4)), tf.float32), [1, 1, 1, 1, 1], "SAME")), min_edge_diff)
    ], axis=0)[:, :-1, :-1, :-1, :]), tf.float32) + 0.5
    y_pred_contour = tf.cast(tf.where(tf.math.reduce_any([
        tf.greater_equal(tf.abs(tf.nn.conv3d(y_pred, tf.constant(edge_filter, tf.float32), [1, 1, 1, 1, 1], "SAME")), min_edge_diff),
        tf.greater_equal(tf.abs(tf.nn.conv3d(y_pred, tf.constant(np.transpose(edge_filter, (1, 0, 2, 3, 4)), tf.float32), [1, 1, 1, 1, 1], "SAME")), min_edge_diff),
        tf.greater_equal(tf.abs(tf.nn.conv3d(y_pred, tf.constant(np.transpose(edge_filter, (2, 1, 0, 3, 4)), tf.float32), [1, 1, 1, 1, 1], "SAME")), min_edge_diff)
    ], axis=0)[:, :-1, :-1, :-1, :]), tf.float32) + 0.5
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
            lambda: (tfp.stats.percentile(tf.reduce_min(tf.sqrt(tf.reduce_sum(((tf.expand_dims(y_true_contour, axis=-2) - tf.expand_dims(y_pred_contour, axis=-3)) * tf.constant(voxel_size, tf.float32)) ** 2, axis=-1)), axis=-1), hd_percentile, interpolation="midpoint")
                     + tfp.stats.percentile(tf.reduce_min(tf.sqrt(tf.reduce_sum(((tf.expand_dims(y_true_contour, axis=-2) - tf.expand_dims(y_pred_contour, axis=-3)) * tf.constant(voxel_size, tf.float32)) ** 2, axis=-1)), axis=-2), hd_percentile, interpolation="midpoint")) / 2
        )
    )
    return hdd[None, None, None, None, None]


def _metric(y_true, y_pred, metric_name, metric, batch_dim_as_spatial_dim=False, feature_dim_as_spatial_dim=False, threshold=None, argmax=False, map_batch=False, map_features=False, reduction_mode=None, percentile=None, reduction_axes=(0, 1, 2, 3, 4), **kwargs):
    assert len(tf.keras.backend.int_shape(y_true)) == 5 and len(tf.keras.backend.int_shape(y_pred)) == 5, "The input tensors/arrays y_true and y_pred to a metric function must be 5D!"
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if batch_dim_as_spatial_dim:
        y_true = tf.reshape(y_true, [1, y_true.shape[1], y_true.shape[2], -1, y_true.shape[4]])
        y_pred = tf.reshape(y_pred, [1, y_pred.shape[1], y_pred.shape[2], -1, y_pred.shape[4]])

    if feature_dim_as_spatial_dim:
        y_true = tf.reshape(y_true, [1, y_true.shape[1], y_true.shape[2], -1, 1])
        y_pred = tf.reshape(y_pred, [1, y_pred.shape[1], y_pred.shape[2], -1, 1])

    if argmax:
        y_true = tf.one_hot(tf.math.argmax(y_true, axis=-1), y_true.shape[-1], dtype=tf.float32)
        y_pred = tf.one_hot(tf.math.argmax(y_pred, axis=-1), y_pred.shape[-1], dtype=tf.float32)

    if threshold is not None:
        y_true = tf.cast(tf.math.greater(y_true, threshold), tf.float32)
        y_pred = tf.cast(tf.math.greater(y_pred, threshold), tf.float32)

    if map_batch:
        result = tf.map_fn(lambda x: get_metric(metric_name, map_features=map_features, **kwargs)(x[0][None], x[1][None]), (y_true, y_pred), fn_output_signature=tf.float32)
        result = result[:, 0, ...]

    elif map_features:
        y_true = tf.keras.backend.permute_dimensions(y_true, (4, 0, 1, 2, 3))
        y_pred = tf.keras.backend.permute_dimensions(y_pred, (4, 0, 1, 2, 3))
        result = tf.map_fn(lambda x: get_metric(metric_name, **kwargs)(x[0][..., None], x[1][..., None]), (y_true, y_pred), fn_output_signature=tf.float32)
        result = result[..., 0]
        result = tf.keras.backend.permute_dimensions(result, (1, 2, 3, 4, 0))

    else:
        result = metric(y_true, y_pred, **kwargs)

    result = tf.cast(result, tf.float32)
    assert len(tf.keras.backend.int_shape(result)) == 5, "The output tensor/array of a metric function must be 5D!"
    if reduction_mode is None:
        return result

    elif reduction_mode == "mean":
        return tf.reduce_mean(result, axis=reduction_axes, keepdims=True)

    elif reduction_mode == "median" or (reduction_mode == "percentile" and (percentile is None or percentile == 50)):
        return tfp.stats.percentile(result, 50, axis=reduction_axes, interpolation="midpoint", keepdims=True)

    else:
        assert reduction_mode == "percentile" and percentile is not None, "Unknown reduction_mode/percentile combination requested."
        return tfp.stats.percentile(result, percentile, axis=reduction_axes, interpolation="midpoint", keepdims=True)


def get_metric(
        metric_name,
        batch_dim_as_spatial_dim=False,
        feature_dim_as_spatial_dim=False,
        threshold=None,
        argmax=False,
        map_batch=False,
        map_features=False,
        reduction_mode=None,
        percentile=None,
        reduction_axes=(0, 1, 2, 3, 4),
        custom_metric_name=None,
        **kwargs):

    if metric_name.startswith("mean_"):
        metric_name = metric_name[5:]
        reduction_mode = "mean"

    elif metric_name.startswith("median_"):
        metric_name = metric_name[7:]
        reduction_mode = "median"

    if metric_name == "error":
        metric = error

    elif metric_name == "absolute_error":
        metric = absolute_error

    elif metric_name == "squared_error":
        metric = squared_error

    elif metric_name == "cross_entropy":
        metric = cross_entropy

    elif metric_name == "accuracy":
        metric = accuracy

    elif metric_name == "true_positive":
        metric = true_positive

    elif metric_name == "true_negative":
        metric = true_negative

    elif metric_name == "false_positive":
        metric = false_positive

    elif metric_name == "false_negative":
        metric = false_negative

    elif metric_name == "true_volume":
        metric = true_volume

    elif metric_name == "pred_volume":
        metric = pred_volume

    elif metric_name == "volume_error":
        metric = volume_error

    elif metric_name == "absolute_volume_error":
        metric = absolute_volume_error

    elif metric_name == "positive_predictive_value":
        metric = positive_predictive_value

    elif metric_name == "negative_predictive_value":
        metric = negative_predictive_value

    elif metric_name == "true_positive_rate":
        metric = true_positive_rate

    elif metric_name == "true_negative_rate":
        metric = true_negative_rate

    elif metric_name == "dice_coefficient":
        metric = dice_coefficient

    elif metric_name == "coefficient_of_determination":
        metric = coefficient_of_determination

    elif metric_name == "ece_from_bin_stats":
        metric = ece_from_bin_stats

    elif metric_name == "ece":
        metric = ece

    elif metric_name == "riemann_sum":
        metric = riemann_sum

    elif metric_name == "auc":
        metric = auc

    elif metric_name == "hausdorff_distance":
        metric = hausdorff_distance

    else:
        raise NotImplementedError("The requested metric is not implemented.")

    metric = partial(
        _metric,
        metric_name=metric_name,
        metric=metric,
        batch_dim_as_spatial_dim=batch_dim_as_spatial_dim,
        feature_dim_as_spatial_dim=feature_dim_as_spatial_dim,
        threshold=threshold,
        argmax=argmax,
        map_batch=map_batch,
        map_features=map_features,
        reduction_mode=reduction_mode,
        percentile=percentile,
        reduction_axes=reduction_axes,
        **kwargs
    )
    if custom_metric_name is not None:
        metric.__name__ = custom_metric_name

    elif reduction_mode is None:
        metric.__name__ = metric_name

    elif reduction_mode == "mean":
        metric.__name__ = "mean_" + metric_name

    elif reduction_mode == "median" or (reduction_mode == "percentile" and (percentile is None or percentile == 50)):
        metric.__name__ = "median_" + metric_name

    else:
        assert reduction_mode == "percentile" and percentile is not None, "Unknown reduction_mode/percentile combination requested."
        metric.__name__ = f"p{percentile}_" + metric_name

    return metric


def _metric_at_multiple_thresholds(y_true, y_pred, metric_name, thresholds, threshold_axis=None, **kwargs):
    if not isinstance(thresholds, Iterable):
        thresholds = (thresholds,)

    metric_values = [get_metric(metric_name, threshold=threshold, **kwargs)(y_true, y_pred) for threshold in thresholds]
    if threshold_axis is None:
        return tf.stack(metric_values, axis=0)

    else:
        return tf.concat(metric_values, axis=threshold_axis)


def get_metric_at_multiple_thresholds(metric_name, thresholds, threshold_axis=None, **kwargs):
    metric_at_multiple_thresholds = partial(_metric_at_multiple_thresholds, metric_name=metric_name, thresholds=thresholds, threshold_axis=threshold_axis, **kwargs)
    metric_at_multiple_thresholds.__name__ = get_metric(metric_name, **kwargs).__name__
    return metric_at_multiple_thresholds
