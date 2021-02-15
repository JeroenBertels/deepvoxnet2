from deepvoxnet2.keras.metrics import binary_crossentropy, categorical_crossentropy, binary_dice_score


def binary_dice_loss(y_true, y_pred, sample_weight=None, **kwargs):
    return 1 - binary_dice_score(y_true, y_pred, sample_weight=sample_weight, threshold=None, **kwargs)


# def categorical_dice_loss(y_true, y_pred, sample_weight=None, **kwargs):
#     kwargs["threshold"] = kwargs.get("threshold", None)
#     assert kwargs["threshold"] is None, "A dice loss function needs to use soft outputs and thus no thresholding operation can be done."
#     kwargs["reduce_along_features"] = kwargs.get("reduce_along_features", False)
#     assert kwargs["reduce_along_features"] is False, "The categorical version of the dice loss/score should treat every feature/class as one sample to compute the dice loss/score on."
#     return 1 - categorical_dice_score(y_true, y_pred, sample_weight=sample_weight, **kwargs)
