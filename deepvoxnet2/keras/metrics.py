import tensorflow as tf


def binary_dice_score(y_true, y_pred, sample_weight=None, **kwargs):
  if "threshold" not in kwargs:
    kwargs["threshold"] = 0.5

  return generalized_dice_coeff(y_true, y_pred, **kwargs)


def generalized_dice_coeff(y_true,
                           y_pred,
                           eps                   = tf.keras.backend.epsilon(),
                           reduce_along_batch    = False,
                           reduce_along_features = True,
                           feature_weights       = None,
                           threshold             = None):
  """ Generalized Dice coefficient for a tensor containing a batch of ndim images with multiple features
      Sudre et al. "Generalised Dice overlap as a deep learning loss function for highly
                    unbalanced segmentations" https://arxiv.org/abs/1707.03237

  Parameters
  ----------

  y_true : tf tensor
    containing the label data. dimensions (n_batch, n0, n1, ...., n_feat)

  y_pred : tf tensor
    containing the predicted data. dimensions (n_batch, n0, n1, ...., n_feat)

  eps : float, default tf.keras.backend.epsilon()
    a small constant that prevents division by 0

  reduce_along_batch : bool, default False
    reduce (sum) the loss values along the batch dimension

  reduce_along_features : bool, default True
    reduce (sum) the loss values along the feature dimension

  feature_weights : None or 1D tf tensor of length n_feat
    feature weights as defined in Generalized Dice LOSS
    If None, every feature gets the same weight (1/n_feat) -> standard Dice coefficient
    If 1D tf array, the "Generalized Dice Score" including feature weighting is calculated

    This only has an effect if reduce_along_features is True.

  threshold : None or float, default None
    if None, no thresholding is done and thus the dice coefficient is 'soft' (depending on the input). If a threshold is provided the volumes are thresholded first.

  Returns
  -------
    a tensor of shape (n_batch) containing the generalized (feature weighted) Dice coefficient
    per batch sample.
  """

  # ndim = len(y_true.shape)
  ndim = tf.rank(y_true)
  ax = tf.range(1, ndim - 1)
  if threshold is not None:
    y_true = tf.cast(tf.math.greater(y_true, threshold), y_true.dtype)
    y_pred = tf.cast(tf.math.greater(y_pred, threshold), y_pred.dtype)

  intersection = tf.math.reduce_sum(y_true * y_pred, axis = ax)

  denom = tf.math.reduce_sum(y_true, axis = ax) + tf.math.reduce_sum(y_pred, axis = ax)

  # now reduce the the dice coeff across the feature dimension
  if reduce_along_features:

    if feature_weights is None:
      # if no weights are given, we use the same constant weight for all features
      feature_weights = 1

    intersection = tf.math.reduce_sum(intersection*feature_weights, axis = -1)
    denom        = tf.math.reduce_sum(denom*feature_weights, axis = -1)

  if reduce_along_batch:
    intersection = tf.math.reduce_sum(intersection, axis = 0)
    denom = tf.math.reduce_sum(denom, axis = 0)

  return (2 * intersection + eps) / (denom + eps)
