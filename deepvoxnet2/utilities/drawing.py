"""This module provides several functions for data visualization using OpenCV.

These functions can be used to overlay masks, draw contours and figures, and can be customized with various line types, colors, and thicknesses.
This module is designed to be used with 2D and 3D numpy arrays, and can be especially useful for visualizing image segmentation and object detection results.
"""

import cv2
import numpy as np


def draw_my_own_contours(img_array, contours, line_color=(0, 255, 0), line_thickness=1, line_type="-", spacing=3):
    """Draws contours on the input image array.

    Parameters
    ----------
    img_array: np.ndarray
        The input image array.
    contours: list of np.ndarray
        The list of contours. Each contour is an array of shape (N, 1, 2), where N is the number of points.
    line_color: tuple of int
        The color of the contour lines. Default is green.
    line_thickness: int
        The thickness of the contour lines. Default is 1.
    line_type: str
        The type of the contour lines. Should be one of "-", "--", ".", "o". Default is "-".
    spacing: int
        The spacing between the points when drawing dotted lines or circles. Default is 3.

    Returns
    -------
    None
    """

    for contour in contours:
        state = False
        for point_i in range(len(contour) - 1):
            if line_type == "-":
                cv2.line(img_array, contour[point_i, 0, :], contour[point_i + 1, 0, :], color=line_color, thickness=line_thickness)

            elif line_type == "--":
                if point_i % spacing == 0:
                    state = not state

                if state:
                    cv2.line(img_array, contour[point_i, 0, :], contour[point_i + 1, 0, :], color=line_color, thickness=line_thickness)

            elif line_type == ".":
                if point_i % spacing == 0:
                    cv2.circle(img_array, contour[point_i, 0, :], line_thickness, color=line_color, thickness=-1)

            elif line_type == "o":
                if point_i % spacing == 0:
                    cv2.circle(img_array, contour[point_i, 0, :], line_thickness, color=line_color, thickness=line_thickness)

            else:
                raise ValueError("Unknown line_type.")


def overlay_mask(img_array, mask_array, line_color=(0, 255, 0), line_thickness=1, line_type="-", fill_color=None, alpha=1, combination_mode=0, **kwargs):
    """Overlays the mask on top of the input image.

    Parameters
    ----------
    img_array: np.ndarray
        The input image array of shape (H, W, 3).
    mask_array: np.ndarray
        The mask array of shape (H, W, 3) or (H, W). If it is a binary mask, it will be converted to RGB.
    line_color: tuple of int or None
        The color of the mask boundaries. If None, the boundaries will not be drawn. Default is green.
    line_thickness: int
        The thickness of the mask boundaries. Default is 1.
    line_type: str
        The type of the mask boundaries. Should be one of "-", "--", ".", "o". Default is "-".
    fill_color: tuple of int
        The color of the mask. Default is black.
    alpha: float
        The opacity of the mask. Should be between 0 and 1. Default is 1.
    combination_mode: int
        The mode of combining the image and the mask. Should be either 0 or 1. Default is 0.
    kwargs:
        Additional arguments to pass to draw_my_own_contours.

    Returns
    -------
    overlayed_array: np.ndarray
        The overlayed image array.
    """

    assert img_array.ndim == 3 and img_array.shape[-1] == 3 and np.array_equal(img_array.shape, mask_array.shape)
    assert np.min(img_array) >= 0 and np.max(img_array) <= 255 and np.min(mask_array) >= 0 and np.max(mask_array) <= 255
    if fill_color is None:
        fill_color = (0, 0, 0)
        alpha = 0

    img_array, mask_array = img_array.copy(), mask_array.copy()
    color_mask_array = (mask_array > 0) * fill_color
    if combination_mode == 0:
        img_array = np.clip(img_array - alpha * mask_array, 0, 255)
        img_array = (img_array + alpha * color_mask_array).astype("uint8")

    elif combination_mode == 1:
        img_array[mask_array > 0] = (img_array[mask_array > 0] * (1 - alpha) + color_mask_array[mask_array > 0] * alpha).astype("uint8")

    else:
        raise ValueError("Unknown combination mode.")

    if line_color is not None:
        binary_mask = (np.any(mask_array > 127, axis=-1) * 255).astype("uint8")
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img_array, contours, contourIdx=-1, color=line_color, thickness=line_thickness)
        draw_my_own_contours(img_array, contours, line_color=line_color, line_thickness=line_thickness, line_type=line_type, **kwargs)

    return img_array


def draw_figure(img_array, mask_array=None, shift=0., scale=1., clip=(0, 255), **kwargs):
    """Draws an image with an optional mask.

    Parameters
    ----------
    img_array: np.ndarray
        The input image array of shape (H, W, 3) or (H, W).
    mask_array: np.ndarray or None
        The mask array of shape (H, W, 3) or (H, W). If it is a binary mask, it will be converted to RGB. Default is None.
    shift: float
        The amount to shift the pixel values of the image. Default is 0.
    scale: float
        The amount to scale the pixel values of the image. Default is 1.
    clip: tuple of int
        The range to clip the pixel values of the image. Default is (0, 255).
    kwargs:
        Additional arguments to pass to overlay_mask.

    Returns
    -------
    output_array: np.ndarray
        The output image array.
    """

    if img_array.ndim == 3:
        assert img_array.shape[-1] == 3

    else:
        assert img_array.ndim == 2
        img_array = np.stack([img_array] * 3, axis=-1)

    img_array = np.clip((img_array + shift) * scale, *clip).astype("uint8")
    assert np.min(img_array) >= 0 and np.max(img_array) <= 255
    if mask_array is not None:
        if mask_array.ndim == 3:
            assert mask_array.shape[-1] == 3

        else:
            assert mask_array.ndim == 2
            mask_array = np.stack([mask_array] * 3, axis=-1)

        assert np.array_equal(img_array.shape, mask_array.shape)
        assert np.min(mask_array) >= 0 and np.max(mask_array) <= 1
        mask_array = np.clip(mask_array * 255, 0, 255).astype("uint8")
        img_array = overlay_mask(img_array, mask_array, **kwargs)

    return img_array


if __name__ == "__main__":
    import os
    import nibabel as nib
    from deepvoxnet2 import DEMO_DIR
    from matplotlib import pyplot as plt

    img = nib.load(os.path.join(DEMO_DIR, "brats_2018", "case_0", "FLAIR.nii.gz")).get_fdata()[:, :, 80].T
    mask = nib.load(os.path.join(DEMO_DIR, "brats_2018", "case_0", "GT_W.nii.gz")).get_fdata()[:, :, 80].T
    fig = draw_figure(img, mask, line_color=(255, 0, 255), line_thickness=1, line_type="o", spacing=2)
    plt.figure()
    plt.imshow(fig)
    plt.show()
