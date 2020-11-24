from tensorflow.keras.losses import BinaryCrossentropy
from deepvoxnet2.keras.metrics import binary_dice_score


def binary_crossentropy(y_true, y_pred, sample_weight=None):
    return BinaryCrossentropy()(y_true, y_pred, sample_weight=sample_weight)


def binary_dice_loss(y_true, y_pred, sample_weight=None, per_image=False, smoothing=1e-7):
    return 1 - binary_dice_score(y_true, y_pred, sample_weight=sample_weight, per_image=per_image, mode="soft", smoothing=smoothing)
