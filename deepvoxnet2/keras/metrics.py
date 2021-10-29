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


def _combine_ece_bin_stats(y_true, y_pred, combine_ece_bin_stats_axis=0, **kwargs):
    ece_bin_stats = tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true)
    if combine_ece_bin_stats_axis is None:
        return ece_bin_stats

    else:
        bin_confidence = ece_bin_stats[:, :, :1, ...]
        bin_accuracy = ece_bin_stats[:, :, 1:2, ...]
        bin_count = ece_bin_stats[:, :, 2:, ...]
        bin_count_combined = tf.reduce_sum(bin_count, axis=combine_ece_bin_stats_axis, keepdims=True)
        bin_confidence_combined = (tf.reduce_sum(bin_confidence * bin_count, axis=combine_ece_bin_stats_axis, keepdims=True) + tf.keras.backend.epsilon()) / (bin_count_combined + tf.keras.backend.epsilon())
        bin_accuracy_combined = (tf.reduce_sum(bin_accuracy * bin_count, axis=combine_ece_bin_stats_axis, keepdims=True) + tf.keras.backend.epsilon()) / (bin_count_combined + tf.keras.backend.epsilon())
        return tf.concat([bin_confidence_combined, bin_accuracy_combined, bin_count_combined], axis=2)


def ece_from_bin_stats(y_true, y_pred, **kwargs):
    ece_bin_stats = _combine_ece_bin_stats(y_true, y_pred, **kwargs)
    bin_confidence = ece_bin_stats[:, :, :1, ...]
    bin_accuracy = ece_bin_stats[:, :, 1:2, ...]
    bin_count = ece_bin_stats[:, :, 2:, ...]
    return tf.reduce_sum(tf.abs(bin_confidence - bin_accuracy) * bin_count, axis=1, keepdims=True) / tf.reduce_sum(bin_count, axis=1, keepdims=True)


def ece(y_true, y_pred, nbins=21, quantiles_as_bins=False, return_bin_stats=False, from_bin_stats=True, **kwargs):
    y_true, y_pred = _expand_binary(y_true, y_pred)
    y_true, y_pred = tf.reshape(y_true, [-1, y_true.shape[4]]), tf.reshape(y_pred, [-1, y_pred.shape[4]])
    labels_true, labels_pred = tf.math.argmax(y_true, axis=-1), tf.math.argmax(y_pred, axis=-1)
    confidence = tf.math.reduce_max(y_pred, axis=-1)
    hits = tf.equal(labels_true, labels_pred)
    if quantiles_as_bins:
        edges = tfp.stats.percentile(confidence, tf.linspace(0, 100, nbins), interpolation="midpoint")
        if not return_bin_stats and not from_bin_stats:
            return tfp.stats.expected_calibration_error_quantiles(hits, tf.math.log(confidence), nbins)[0][None, None, None, None, None]

    else:
        edges = tf.cast(tf.linspace(0, 1, nbins), tf.float32)
        if not return_bin_stats and not from_bin_stats:
            return tfp.stats.expected_calibration_error(nbins, tf.math.log(y_pred), labels_true, labels_pred)[None, None, None, None, None]

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


def auc(y_true, y_pred, thresholds=np.linspace(0 - tf.keras.backend.epsilon(), 1, 51), curve='PR', auc_threshold_axis=1, return_auc_stats=False, y_true_thresholds=None, **kwargs):
    if y_true_thresholds is not None:
        if not isinstance(y_true_thresholds, Iterable):
            y_true_thresholds = [y_true_thresholds] * len(thresholds)

        assert len(y_true_thresholds) == len(thresholds), "If you specify a list of y_true_thresholds they must be of equal length to the specified list of (y_pred) thresholds."
        thresholds = list(zip(y_true_thresholds, thresholds))

    if curve == "PR":
        x = recall = get_metric_at_multiple_thresholds("true_positive_rate", thresholds, threshold_axis=auc_threshold_axis, **kwargs)(y_true, y_pred)
        y = precision = get_metric_at_multiple_thresholds("positive_predictive_value", thresholds, threshold_axis=auc_threshold_axis, **kwargs)(y_true, y_pred)

    elif curve == "ROC":
        specificity = get_metric_at_multiple_thresholds("true_negative_rate", thresholds, threshold_axis=auc_threshold_axis, **kwargs)(y_true, y_pred)
        x = 1 - specificity
        y = recall = get_metric_at_multiple_thresholds("true_positive_rate", thresholds, threshold_axis=auc_threshold_axis, **kwargs)(y_true, y_pred)

    else:
        raise ValueError("The requested curve is not yet implemented.")

    if return_auc_stats:
        return tf.concat([x, y], axis=auc_threshold_axis)

    else:
        return riemann_sum(x, y, reduce_mean_axes=tuple([i for i in range(5) if i != auc_threshold_axis]))


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
        if not isinstance(threshold, tuple):
            threshold = (threshold, threshold)

        assert len(threshold) == 2, "A threshold must either be specified as a scalar, a broadcastable array or a tuple of two of these (one for y_true and one for y_pred)."
        y_true = tf.cast(tf.math.greater(y_true, tf.convert_to_tensor(threshold[0], dtype=tf.float32)), tf.float32)
        y_pred = tf.cast(tf.math.greater(y_pred, tf.convert_to_tensor(threshold[1], dtype=tf.float32)), tf.float32)

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

    elif metric_name == "_combine_ece_bin_stats":
        metric = _combine_ece_bin_stats

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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    y_true = np.random.rand(1, 100, 100, 100, 1)
    # y_pred = np.random.rand(1, 100, 100, 100, 1)
    y_pred = np.clip(y_true + np.random.rand(1, 100, 100, 100, 1) / 2, 0, 1)
    y_true = y_true > 0.5

    # ECE checking
    ece_0 = get_metric("ece", quantiles_as_bins=False, ece_from_bin_stats=False)(y_true, y_pred)
    ece_1 = get_metric("ece", quantiles_as_bins=True, ece_from_bin_stats=False)(y_true, y_pred)
    ece_2 = get_metric("ece", quantiles_as_bins=False, ece_from_bin_stats=True)(y_true, y_pred)
    ece_3 = get_metric("ece", quantiles_as_bins=True, ece_from_bin_stats=True)(y_true, y_pred)
    print(ece_0, ece_1, ece_2, ece_3)
    bin_stats_0 = get_metric("ece", quantiles_as_bins=False, ece_from_bin_stats=False, return_bin_stats=True)(y_true, y_pred)
    bin_stats_1 = get_metric("ece", quantiles_as_bins=True, ece_from_bin_stats=False, return_bin_stats=True)(y_true, y_pred)
    confidence_0, accuracy_0 = bin_stats_0[0, :, 0, 0, 0], bin_stats_0[0, :, 1, 0, 0]
    confidence_1, accuracy_1 = bin_stats_1[0, :, 0, 0, 0], bin_stats_1[0, :, 1, 0, 0]
    plt.figure()
    plt.plot(confidence_0, accuracy_0, "b.")
    plt.plot(confidence_1, accuracy_1, "r.")
    plt.title("ECE")
    plt.show()

    # AUC checking
    auc_0 = get_metric("auc", y_true_thresholds=None)(y_true, y_pred)
    auc_1 = get_metric("auc", y_true_thresholds=0.5)(y_true, y_pred)
    print(auc_0, auc_1)
    auc_stats_0 = get_metric("auc", return_auc_stats=True, y_true_thresholds=None)(y_true, y_pred)
    auc_stats_1 = get_metric("auc", return_auc_stats=True, y_true_thresholds=0.5)(y_true, y_pred)
    x_0, y_0 = auc_stats_0[0, :auc_stats_0.shape[1] // 2, 0, 0, 0], auc_stats_0[0, auc_stats_0.shape[1] // 2:, 0, 0, 0]
    x_1, y_1 = auc_stats_1[0, :auc_stats_1.shape[1] // 2, 0, 0, 0], auc_stats_1[0, auc_stats_1.shape[1] // 2:, 0, 0, 0]
    plt.figure()
    plt.plot(x_0, y_0, "b.")
    plt.plot(x_1, y_1, "r.")
    plt.title("AUC")
    plt.show()
