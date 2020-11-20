import tensorflow as tf


def binary_dice_score(y_true, y_pred, sample_weight=None, per_image=False, mode="hard", smoothing=1e-7):
    if mode == "hard":
        y_true = tf.round(y_true)
        y_pred = tf.round(y_pred)

    if sample_weight is None:
        sample_weight = 1

    intersection = tf.math.reduce_sum(y_true * y_pred * sample_weight, axis=(1, 2, 3, 4) if per_image else None, keepdims=True)
    union = tf.math.reduce_sum(y_true * sample_weight, axis=(1, 2, 3, 4) if per_image else None, keepdims=True) + tf.math.reduce_sum(y_pred * sample_weight, axis=(1, 2, 3, 4) if per_image else None, keepdims=True)
    return (2 * intersection + smoothing) / (union + smoothing)
