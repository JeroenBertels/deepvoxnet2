import cv2
import numpy as np


def draw_my_own_contours(img_array, contours, line_color=(0, 255, 0), line_thickness=1, line_type="-", spacing=3):
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
