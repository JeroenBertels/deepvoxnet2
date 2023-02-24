"""The transformations module provides functions for various data transformations and augmentations, such as affine transformations, elastic deformations, orthogonalization of image directions, and registration.

These functions are implemented using the SimpleITK and NumPy libraries, and can be used to preprocess or augment medical image data for various applications, such as computer-aided diagnosis, image segmentation, or deep learning.
"""

import os
import numpy as np
import nibabel as nib
try:
    import SimpleITK as sitk

except ModuleNotFoundError:
    pass

from tempfile import mkdtemp
from shutil import rmtree
from scipy.ndimage import zoom, gaussian_filter, uniform_filter, distance_transform_edt
from scipy.spatial.transform import Rotation
from pymirc.image_operations import aff_transform, backward_3d_warp, random_deformation_field
from .conversions import nii_to_sitk, sitk_to_nii


def downsample_array(input_array, window_sizes):
    """Downsamples an input array using a sliding window and averaging over the window.

    This function performs a sliding window with a stride equal to the window size over each dimension of the input array with the specified window size, and
    then averages over the window to produce a single output value. If the size of the input array is not divisible by the
    window size, the output array will be truncated to the largest integer size that is divisible by the window size.

    Parameters
    ----------
    input_array : numpy.ndarray
        Input array to be downsampled.
    window_sizes : tuple or list of int
        Window size for each dimension of the input array. If only one value is specified, the same window size is used
        for all dimensions.

    Returns
    -------
    output_array : numpy.ndarray
        Downsampled array.

    Raises
    ------
    AssertionError
        If `window_sizes` is not a tuple or list, or if not all dimensions in the input array have a corresponding window
        size being specified, or if any of the window sizes are not >= 1 integers.
    """

    if not isinstance(window_sizes, (tuple, list)):
        window_sizes = [window_sizes] * len(input_array.shape)

    assert len(input_array.shape) == len(window_sizes), "Not all dimensions in the input array have a corresponding window size being specified."
    assert all([isinstance(window_size, int) and window_size >= 1 for window_size in window_sizes]), "Window sizes must be >= 1 integers."
    output_shapes = []
    tmp_shapes = []
    for window_size, input_shape in zip(window_sizes, input_array.shape):
        output_shape = input_shape if window_size == 1 else int(input_shape - input_shape % window_size)
        output_shapes.append(output_shape)
        tmp_shapes += [output_shape // window_size, window_size]

    tmp_array = np.reshape(input_array[tuple([slice(output_shape) for output_shape in output_shapes])], tmp_shapes)
    output_array = np.mean(tmp_array, tuple(range(1, len(tmp_shapes), 2)))
    return output_array


def resample(input_nii, output_zooms, order=3, prefilter=True, reference_nii=None):
    """Resamples a Nifti image to have new voxel sizes defined by `output_zooms`.

    Parameters
    ----------
    input_nii : nibabel.nifti1.Nifti1Image
        The input Nifti image to resample.
    output_zooms : tuple
        The desired voxel size for the resampled Nifti image. It must be a tuple or list of same length as the number of dimensions in the original Nifti image. None can be used to specify that a certain dimension does not need to be resampled.
    order : int or str, optional
        The order of interpolation (0 to 5) or the string 'mean' to perform a downsampling based on mean (with a moving average window that gets the effective output zooms as close as possible to the requested output zooms). Default is 3.
    prefilter : bool, optional
        Whether to apply a Gaussian filter to the input data before resampling (https://stackoverflow.com/questions/35340197/box-filter-size-in-relation-to-gaussian-filter-sigma). Default is True.
    reference_nii : nibabel.nifti1.Nifti1Image or None, optional
        If not None, the resampled Nifti image will have the same dimensions as `reference_nii`. Default is None.
        E.g., when doing upsampling back to an original image space we want to make sure the image sizes are consistent. By giving a reference Nifti image, we crop or pad with zeros where necessary.

    Returns
    -------
    output_nii : nibabel.nifti1.Nifti1Image
        The resampled Nifti image.
    """

    input_zooms = input_nii.header.get_zooms()
    assert len(input_zooms) == len(output_zooms), "Number of dimensions mismatch."
    assert np.allclose(input_zooms[:3], np.linalg.norm(input_nii.affine[:3, :3], 2, axis=0)), "Inconsistency (we only support voxel size = voxel distance) in affine and zooms (spatial) of input Nifti image."
    input_array = input_nii.get_fdata()
    output_zooms = [input_zoom if output_zoom is None else output_zoom for input_zoom, output_zoom in zip(input_zooms, output_zooms)]
    if order == "mean":
        assert all([4 / 3 * output_zoom >= input_zoom for output_zoom, input_zoom in zip(output_zooms, input_zooms)]), "This function with order='mean' only supports downsampling by an integer factor (input zooms: {}).".format(input_zooms)
        zoom_factors = [1 if input_zoom > 2 / 3 * output_zoom else 1 / int(round(output_zoom / input_zoom)) for input_zoom, output_zoom in zip(input_zooms, output_zooms)]
        output_zooms = [input_zoom / zoom_factor for input_zoom, zoom_factor in zip(input_zooms, zoom_factors)]
        output_array = downsample_array(input_array, window_sizes=[int(1 / zoom_factor) for zoom_factor in zoom_factors])

    else:
        assert isinstance(order, int), "When order != 'mean', it must be an integer (see scipy.ndimage.zoom)."
        zoom_factors = [input_zoom / output_zoom for input_zoom, output_zoom in zip(input_zooms, output_zooms)]
        if prefilter:
            input_array = gaussian_filter(input_array.astype(np.float32), [np.sqrt(((1 / zoom_factor)**2 - 1) / 12) if zoom_factor < 1 else 0 for zoom_factor in zoom_factors], mode="nearest")

        output_array = zoom(input_array, zoom_factors, order=order, mode="nearest")

    if reference_nii is not None:
        assert output_zooms == list(reference_nii.header.get_zooms()), "The output zooms are not equal to the reference zooms."
        output_array_ = np.zeros_like(reference_nii.get_fdata())
        output_array_[tuple([slice(min(s, s_)) for s, s_ in zip(output_array.shape, output_array_.shape)])] = output_array[tuple([slice(min(s, s_)) for s, s_ in zip(output_array.shape, output_array_.shape)])]
        output_array = output_array_
        output_affine = reference_nii.affine

    else:
        output_affine = input_nii.affine.copy()
        output_affine[:3, :3] = output_affine[:3, :3] / zoom_factors[:3]

    output_nii = nib.Nifti1Image(output_array, affine=output_affine)
    output_nii.header.set_zooms(output_zooms)  # important to set non-spatial zooms correctly
    return output_nii


def crop(I, S_size, coordinates=None, subsample_factors=None, default_value=0, prefilter=None):
    """Crop an input array I to size S_size centered on coordinates in I with optional subsampling.

    Parameters
    ----------
    I : np.ndarray
        Input array. Must be at least as large as S_size in each dimension.
    S_size : tuple, list
        Output array size of the cropped volume.
    coordinates : tuple, list or None, optional
        Indices in I to center the crop. If None, the center of I is used.
    subsample_factors : tuple, list of int or None, optional
        Subsampling factors for each requested segment dimension.
    default_value : int or np.nan, optional
        Value to fill outside of I. If np.nan, a nearest-neighbor interpolation is used to fill the edges.
    prefilter : {'gaussian', 'uniform'} or None, optional
        Filter to use during interpolation. Uniform filter uses the `downsample_array` function.

    Returns
    -------
    np.ndarray
        Cropped array.
    """

    assert I.ndim >= len(S_size), "Number of image dimensions must be equal to or larger than the number of requested segment sizes. Extra (trailing) dimensions will not be cropped."
    if coordinates is None:
        coordinates = [s // 2 for s in I.shape]  # cropping happens around the center

    else:
        assert len(coordinates) == len(S_size), "Coordinates must specify an index for each requested segment dimension."
        coordinates = [s // 2 if c is None else c for c, s in zip(coordinates, I.shape[:len(coordinates)])]  # cropping happens around the specified coordinates
        coordinates += [s // 2 for s in I.shape[len(coordinates):]]  # cropping happens around the center of the remaining trailing dimensions

    if subsample_factors is None:
        subsample_factors = [1] * len(I.shape)  # no resampling happens

    else:
        assert len(subsample_factors) == len(S_size), "A subsample factor must be specified for each requested segment dimension."
        subsample_factors = list(subsample_factors) + [1] * (I.ndim - len(subsample_factors))

    if prefilter is not None:
        assert prefilter in ["uniform", "gaussian"]
        if prefilter == "gaussian":
            I = gaussian_filter(I.astype(np.float32), [s_f if s_f > 1 else 0 for s_f in subsample_factors], mode="nearest")

        elif prefilter == "uniform":
            I = uniform_filter(I.astype(np.float32), subsample_factors, mode="nearest")

    S_size = tuple(S_size) + tuple(I.shape[len(S_size):])
    S = np.full(S_size, fill_value=default_value, dtype=np.float32)
    idx_I = [slice(None)] * I.ndim
    idx_S = [slice(None)] * S.ndim
    for i, (d_I, d_S, c, s_f) in enumerate(zip(I.shape, S_size, coordinates, subsample_factors)):
        n_left_I = c
        n_right_I = d_I - c - 1
        n_left_S = d_S // 2
        n_right_S = d_S // 2
        if d_S % 2 == 0:
            n_right_S -= 1

        if n_left_I < n_left_S * s_f:
            n = n_left_I // s_f
            start_S = d_S // 2 - n
            start_I = c - n * s_f

        else:
            start_S = 0
            start_I = c - n_left_S * s_f

        if n_right_I < n_right_S * s_f:
            n = n_right_I // s_f
            end_S = d_S // 2 + n
            end_I = c + n * s_f

        else:
            end_S = d_S - 1
            end_I = c + n_right_S * s_f

        idx_I[i] = slice(start_I, end_I + 1, s_f)
        idx_S[i] = slice(start_S, end_S + 1)

    S[tuple(idx_S)] = I[tuple(idx_I)]
    if np.any(np.isnan(default_value)):
        S = S[tuple(distance_transform_edt(np.isnan(S), return_distances=False, return_indices=True))]

    return S


def put(I, S, coordinates=None, subsample_factors=None):
    """This function can be used to put back a segment centered on a specific coordinate in the input array (reverse of crop function).

    Parameters
    ----------
    I : np.ndarray
        The input array in which to put back the segment S. Can be of any number of dimensions.
    S : np.ndarray
        The segment array to put back in I. Same number of dimensions as I.
    coordinates : None or tuple or list
        Around what coordinate in the input array the segment array will be put back. When None, the center coordinate in I will be used.
        None can also be used for a specific axis to specify that the center coordinate along that axis must be used.
    subsample_factors : None or tuple or list
        If the segment array was subsampled, one can specify an integer subsample factor for each axis.
        None is used to denote a subsample factor of 1.

    Returns
    -------
    I : np.ndarray
        The output array I in which the segment S is now put.
    """

    assert I.ndim == S.ndim, "Number of image dimensions must be equal to number of segment dimensions."
    if coordinates is None:
        coordinates = [s // 2 for s in I.shape]  # putting happens around the center

    else:
        assert len(coordinates) <= I.ndim
        coordinates = [s // 2 if c is None else c for c, s in zip(coordinates, I.shape[:len(coordinates)])]  # putting happens around the specified coordinates
        coordinates += [s // 2 for s in I.shape[len(coordinates):]]  # putting happens around the center of the remaining trailing dimensions

    if subsample_factors is None:
        subsample_factors = [1] * len(I.shape)

    else:
        assert len(subsample_factors) == I.ndim, "A subsample factor must be specified for each image/segment dimension."

    idx_I = [slice(None)] * I.ndim
    idx_S = [slice(None)] * S.ndim
    for i, (d_I, d_S, c, s_f) in enumerate(zip(I.shape, S.shape, coordinates, subsample_factors)):
        n_left_I = c
        n_right_I = d_I - c - 1
        n_left_S = d_S // 2
        n_right_S = d_S // 2
        if d_S % 2 == 0:
            n_right_S -= 1

        if n_left_I < n_left_S * s_f:
            n = n_left_I // s_f
            start_S = d_S // 2 - n
            start_I = c - n * s_f

        else:
            start_S = 0
            start_I = c - n_left_S * s_f

        if n_right_I < n_right_S * s_f:
            n = n_right_I // s_f
            end_S = d_S // 2 + n
            end_I = c + n * s_f

        else:
            end_S = d_S - 1
            end_I = c + n_right_S * s_f

        idx_I[i] = slice(start_I, end_I + 1, s_f)
        idx_S[i] = slice(start_S, end_S + 1)

    I[tuple(idx_I)] = S[tuple(idx_S)]
    return I


def get_affine_matrix(I_shape, voxel_size=(1, 1, 1), reflection=(1, 1, 1), shear=(0, 0, 0), rotation=(0, 0, 0), translation=(0, 0, 0), scaling=(1, 1, 1), return_forward_affine=False):
    """
    Apply a certain deformation to an image, using the center of the image as origin for the scaling and rotation, and afterwards an optional translation as well.

    Parameters
    ----------
    I_shape : tuple
        The shape of the image in the form of (x, y, z). The image is centered using this shape.
    voxel_size : tuple, optional
        The voxel size in the form of (x, y, z). Default is (1, 1, 1).
    reflection : tuple, optional
        The reflection along each axis in the form of (x, y, z). Default is (1, 1, 1).
    shear : tuple, optional
        The shear angle in radians for each axis in the form of (x, y, z). Default is (0, 0, 0).
    rotation : tuple, optional
        The rotation angles in degrees for each axis in the form of (x, y, z). Default is (0, 0, 0).
    translation : tuple, optional
        The translation along each axis in the form of (x, y, z). Default is (0, 0, 0).
    scaling : tuple, optional
        The scaling along each axis in the form of (x, y, z). Default is (1, 1, 1).
    return_forward_affine : bool, optional
        Whether to return the forward or inverse transformation matrix. Default is False (i.e., returns the inverse).

    Returns
    -------
    affine : np.ndarray
        The affine transformation matrix that describes the specified transformation.
    """

    Tw, Sw, Rw, S2w, T2c, SHw, RFw = [np.eye(4) for _ in range(7)]
    T2c[:-1, -1] = - np.array(I_shape) / 2
    S2w[:-1, :-1] = np.diag(voxel_size)
    RFw[:-1, :-1] = np.diag(reflection)
    SHw[0, 1], SHw[0, 2], SHw[1, 0], SHw[1, 2], SHw[2, 0], SHw[2, 1] = shear[0], shear[0], shear[1], shear[1], shear[2], shear[2]
    Rw[:-1, :-1] = Rotation.from_euler('zyx', rotation).as_matrix()
    Sw[:-1, :-1] = np.diag(scaling)
    Tw[:-1, -1] = np.array(translation)
    affine = np.linalg.inv(T2c) @ np.linalg.inv(S2w) @ Tw @ Sw @ Rw @ SHw @ RFw @ S2w @ T2c
    if not return_forward_affine:
        affine = np.linalg.inv(affine)

    return np.round(affine, 5)


def affine_deformation(I, affine, output_shape=None, order=1, cval=0, oversampling_factors=(1, 1, 1)):
    """Apply an affine deformation to the input array I.

    Parameters
    ----------
    I: np.ndarray
        The input array to be deformed.
    affine: np.ndarray
        The 4x4 affine transformation matrix specifying the transformation to apply.
    output_shape: tuple, None
        The shape of the output array. If None, the output shape will be the same as the input array I.
    order: int
        The order of the interpolation used in the transformation. Can be 0, 1, 2, 3, 4, or 5.
    cval: int
        The value to fill in for points outside the boundaries of the input array. Default is 0.
    oversampling_factors: tuple
        The oversampling factors in each dimension. Default is (1, 1, 1).

    Returns
    -------
    deformed_array: np.ndarray
        The deformed array I after applying the specified affine transformation.
    """

    return aff_transform(I, np.round(affine, 5), output_shape=output_shape or I.shape, trilin=order, cval=cval, os0=oversampling_factors[0], os1=oversampling_factors[1], os2=oversampling_factors[2])


def get_deformation_field(I_shape, shift=(2, 2, 2), nsize=(30, 30, 30), npad=(5, 5, 5), std=(6, 6, 6)):
    """Generate an elastic deformation field.

    Parameters
    ----------
    I_shape: tuple
        The shape of the image for which to generate the deformation field.
    shift: tuple
        The maximum shift in each dimension.
    nsize: tuple
        The number of points in each dimension for which to calculate the deformation field.
    npad: tuple
        The padding in each dimension for which to calculate the deformation field.
    std: tuple
        The standard deviation of the Gaussian smoothing kernel in each dimension.

    Returns
    -------
    deformation_field: np.ndarray
        The elastic deformation field as a 4D numpy array.
    """

    deformation_field = np.zeros(tuple(I_shape) + (3,))
    deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2] = random_deformation_field(I_shape, shift=shift, n=nsize, npad=npad, gaussian_std=std)
    return deformation_field


def elastic_deformation(I, deformation_field, cval=0, order=1):
    """Apply an elastic deformation to the input array I.

    Parameters
    ----------
    I: np.ndarray
        The input array to be deformed.
    deformation_field: np.ndarray
        The 4D elastic deformation field specifying the deformation to apply.
    cval: int
        The value to fill in for points outside the boundaries of the input array. Default is 0.
    order: int
        The order of the interpolation used in the transformation. Can be 0, 1, 2, 3, 4, or 5.

    Returns
    -------
    deformed_array: np.ndarray
        The deformed array I after applying the specified elastic deformation.
    """

    return backward_3d_warp(I, deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2], trilin=order, cval=cval)


def orthogonalize(input_nii, invalid_value):
    """Orthogonalize by having a new z-axis that is orthogonal to the x and y axis.

    This function takes a nibabel.Nifti1Image as input, which is a 3D image with x, y, and z axes.
    The function then finds a new z-axis that is orthogonal to the x and y axes of the input image, keeping the dimensions of the image the same.
    It returns the orthogonalized image as another nibabel.Nifti1Image.
    The invalid_value parameter is used to specify the value to fill in outside of the image.
    Keeps the dimensions of the image.

    Parameters
    ----------
    input_nii : nib.Nifti1Image
        Input image.
    invalid_value : float
        Outside image values.

    Returns
    -------
    output_nii : nib.Nifti1Image
        Orthogonalized input image.
    """

    image = nii_to_sitk(input_nii)
    direction_matrix = np.array(image.GetDirection()).reshape((3, 3))
    z_direction = direction_matrix[:, 2]
    new_z_direction = np.cross(direction_matrix[:, 0], direction_matrix[:, 1])
    new_z_spacing = np.dot(z_direction, new_z_direction) * image.GetSpacing()[2]
    new_direction_matrix = np.array(direction_matrix)
    new_direction_matrix[:, 2] = new_z_direction
    resampleImageFilter = sitk.ResampleImageFilter()
    resampleImageFilter.SetOutputDirection(tuple(new_direction_matrix.flatten()))
    resampleImageFilter.SetOutputOrigin(image.GetOrigin())
    resampleImageFilter.SetOutputSpacing(image.GetSpacing()[:2] + (new_z_spacing,))
    resampleImageFilter.SetSize(image.GetSize())
    resampleImageFilter.SetDefaultPixelValue(invalid_value)
    image = resampleImageFilter.Execute(image)
    output_nii = sitk_to_nii(image)
    return output_nii


def registration_quality(fixed_array, moved_array, mask_array=None):
    """Computes the quality of registration between two arrays using the correlation coefficient.

    Parameters
    ----------
    fixed_array: np.ndarray
        The reference array to which the `moved_array` is registered.
    moved_array: np.ndarray
        The array that has been registered to the `fixed_array`.
    mask_array: None or np.ndarray, optional
        A mask array with 0 values where the corresponding voxels should be ignored in the computation.

    Returns
    -------
    correlation_coefficient: float
        The correlation coefficient between the flattened `fixed_array` and `moved_array`.
    """

    if mask_array is not None:
        fixed_array = fixed_array[mask_array]
        moved_array = moved_array[mask_array]

    correlation_coefficient = np.corrcoef(fixed_array.flatten(), moved_array.flatten())[0, 1]
    return correlation_coefficient


def move(moving_nii, fixed_nii=None, transform_parameter_map=None, fixed_nii_mask=None, moving_nii_mask=None, parameter_map=None, initial_transform_parameter_map=None):
    """Apply a registration or a transform to a moving image using SimpleElastix.

    Parameters
    ----------
    moving_nii: nib.Nifti1Image
        The moving image to apply the transformation to.
    fixed_nii: None or nib.Nifti1Image
        If specified, the function will do registration of the moving image with the fixed image using SimpleElastix.
    transform_parameter_map: None or sitk.Transform, dict, or sitk.ParameterMap
        If no fixed image is specified, the parameter map (file, dict or SimpleITK object) specifying the transformation is applied to the moving image.
    fixed_nii_mask: None or nib.Nifti1Image
        If fixed image is specified, use this image as a mask for the fixed image.
    moving_nii_mask: None or nib.Nifti1Image
        If fixed image is specified, use this image as a mask for the moving image.
    parameter_map: None, "default", "elastic", "translation", "rigid", "affine", sitk.ParameterMap or dict
        If fixed image is specified, use this SimpleElastix parameter map for registration. Default value is "default".
    initial_transform_parameter_map: None or sitk.ParameterMap
        If specified, this initial transformation parameter map is used for the registration.

    Returns
    -------
    moved_nii: None or nib.Nifti1Image
        The output Nifti image with the transformation applied. Is None if the transformation is unsuccessful.
    transform_parameter_map: None or sitk.ParameterMap
        The transform parameter map. Is None if the transformation is unsuccessful.

    See Also
    --------
    sitk.ElastixImageFilter: Class that is used under the hood to perform a registration.
    sitk.TransformixImageFilter: Class that is used under the hood to perform a move.
    """

    assert hasattr(sitk, "ElastixImageFilter"), "Please also install Simple Elastix (see documentation of this move function for installation instructions)."
    if fixed_nii is not None:
        assert transform_parameter_map is None, "When a fixed image is specified a registration is done and no transformation map will be used."
        elastix_image_filter = sitk.ElastixImageFilter()
        if parameter_map is None or parameter_map in ["default", "elastic"]:
            parameter_map = "default"

        elif parameter_map in ["translation", "rigid", "affine", "bspline"]:
            elastix_image_filter.SetParameterMap(sitk.GetDefaultParameterMap(parameter_map))

        elif isinstance(parameter_map, (dict, sitk.ParameterMap)):
            elastix_image_filter.SetParameterMap(parameter_map)

        else:
            raise NotImplementedError

        elastix_image_filter.SetFixedImage(sitk.Cast(nii_to_sitk(fixed_nii), sitk.sitkFloat32))
        if fixed_nii_mask is not None:
            elastix_image_filter.SetFixedMask(sitk.Cast(nii_to_sitk(fixed_nii_mask), sitk.sitkUInt8))

        if moving_nii_mask is not None:
            elastix_image_filter.SetMovingMask(sitk.Cast(nii_to_sitk(moving_nii_mask), sitk.sitkUInt8))

        if initial_transform_parameter_map is not None:
            tmpdir = mkdtemp()
            sitk.WriteParameterFile(initial_transform_parameter_map, os.path.join(tmpdir, "initial_transform_parameter_map.txt"))
            elastix_image_filter.SetInitialTransformParameterFileName(os.path.join(tmpdir, "initial_transform_parameter_map.txt"))

        image_filter = elastix_image_filter

    else:
        assert transform_parameter_map is not None and parameter_map is None and fixed_nii_mask is None and moving_nii_mask is None and initial_transform_parameter_map is None, "When no fixed image is specified only a transformation map will be used."
        transformix_image_filter = sitk.TransformixImageFilter()
        transformix_image_filter.SetTransformParameterMap(transform_parameter_map)
        image_filter = transformix_image_filter

    image_filter.SetMovingImage(sitk.Cast(nii_to_sitk(moving_nii), sitk.sitkFloat32))
    try:
        image_filter.Execute()
        transform_parameter_map = image_filter.GetTransformParameterMap()
        moved_nii = sitk_to_nii(image_filter.GetResultImage())

    except RuntimeError:
        moved_nii = None
        transform_parameter_map = None
        print("WARNING: transformation unsuccessful")

    if initial_transform_parameter_map is not None:
        rmtree(tmpdir)

    return moved_nii, transform_parameter_map
