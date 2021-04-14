import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tempfile import mkdtemp
from shutil import rmtree
from scipy.ndimage import zoom, gaussian_filter, uniform_filter, distance_transform_edt
from scipy.spatial.transform import Rotation
from pymirc.image_operations import aff_transform, backward_3d_warp, random_deformation_field
from .conversions import nii_to_sitk, sitk_to_nii


def downsample_array(input_array, window_sizes):
    """
    Performs a moving average filter across an array with stride equal to window size.

    Parameters
    ----------
    input_array: np.ndarray
        The array to be downsampled.
    window_sizes: int, tuple, list
        The window size to be used for the moving average filter. Must be specified as integers > 0 or a iterable as such.

    Returns
    -------
    output_array: np.ndarray
        Downsampled version of the input array.
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
    """
    This function resamples a Nifti image to the desired zooms.

    Parameters
    ----------
    input_nii: nib.Nifti1Image
        The input Nifti image to be resampled.
    output_zooms: tuple, list
        The desired output zooms. It must be a tuple or list of same length as the number of dimensions in the original Nifti image. None can be used to specify that a certain dimension does not need to be resampled.
    order: int, str
        Order of the interpolation that will be used. See supported orders in NumPy's zoom function (order=0, 1, 2, 3). If you choose order='mean' instead, than it will be downsampling with a moving average window that gets the effective output zooms as close as possible to the requested output zooms.
    prefilter: bool
        Use gaussian prefilter when downsampling (https://stackoverflow.com/questions/35340197/box-filter-size-in-relation-to-gaussian-filter-sigma).
    reference_nii: nib.Nifti1Image
        E.g. when doing upsampling back to an original image space we want to make sure the image sizes are consistent. By giving a reference Nifti image, we crop or pad with zeros where necessary.

    Returns
    -------
    output_nii: nib.Nifti1Image
        Resampled version of the original Nifti image.

    See Also
    --------
    scipy.ndimage.zoom: The core function that is used under the hood.
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

    output_affine = input_nii.affine.copy()
    output_affine[:3, :3] = output_affine[:3, :3] / zoom_factors[:3]
    output_nii = nib.Nifti1Image(output_array, affine=output_affine)
    output_nii.header.set_zooms(output_zooms)  # important to set non-spatial zooms correctly
    return output_nii


def crop(I, S_size, coordinates=None, subsample_factors=None, default_value=0, prefilter=None):
    """
    This function can be used to extract an array of a specific size centered on a specific coordinate from the input array.

    Parameters
    ----------
    I: np.ndarray
        The input array. Can be of any number of dimensions, but the number of axes specified in the other arguments must (i) all be the same and (ii) less than or equal to the number of image dimensions. Remaining trailing dimensions will not be cropped.
    S_size: tuple, list
        The requested output array size of the cropped volume.
    coordinates: None, tuple, list
        Around what coordinate in the input array the array will be cropped. When None, the center coordinate will be used. None can also be used for a specific axis to specify that the center coordinate along that axis must be used.
    subsample_factors: None, tuple, list
        If the cropped array must be subsampled, one can specify an integer subsample factor for each axis. None is used to denote a subsample factor of 1.
    default_value: int
        What value to pad with outside the input array, if np.nan a nearest interpolation is done at the borders.
    prefilter: None, str
        One of "gaussian" and "uniform", the interpolation used in case prefilter is not None (the "uniform" prefilter is implemented with downsample_array function).

    Returns
    -------
    S: np.ndarray
        The output array crop.
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
    if np.isnan(default_value):
        S = S[tuple(distance_transform_edt(np.isnan(S), return_distances=False, return_indices=True))]

    return S


def put(I, S, coordinates=None, subsample_factors=None):
    """
    This function can be used to put back a segment centered on a specific coordinate in the input array (reverse of crop function).

    Parameters
    ----------
    I: np.ndarray
        The input array in which to put back the segment S. Can be of any number of dimensions.
    S: np.ndarray
        The segment array to put back in I. Same number of dimensions as I.
    coordinates: None, tuple, list
        Around what coordinate in the input array the segment array will be put back. When None, the center coordinate in I will be used. None can also be used for a specific axis to specify that the center coordinate along that axis must be used.
    subsample_factors: None, tuple, list
        If the segment array was subsampled, one can specify an integer subsample factor for each axis. None is used to denote a subsample factor of 1.

    Returns
    -------
    I: np.ndarray
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
    ---->   Therefor voxel space has its origin at the corner of the image (x = y = z = 0) and the units of the axes are the pixels/voxels.
            In world space the origin is at the center of the image and the axes are scaled according to the voxel size. It is there that the transformation is defined.

          --------- p_v voxel space p_v' <--------
         |                                        |
         | W                                      | W-1
         v                                        |
    world space p_w       == T_w ==>         world space' p_w'

    Step 1:
        The image is centered (T2c = Translation to center) and then the axes are scaled according to the voxel size (S2w = Scaled to world (real size)) => W = S2w T2c
    Step 2:
        The image as such is "augmented": rotated (Rw = rotated in world), scaled (Sw = scaled in world) and translated (Tw = translated in ) in that order => T_w = Tw Sw Rw
    Step 3:
        The image is put back on its original place (T2c-1) with the original voxel size (S2w-1) => W-1 = T2c-1 S2w-1
    This means the transformation matrix = matrix-1 for the augmentation is:
        matrix-1 = W-1 T_w W = T2c-1 . S2w-1 . Tw . Sw . Rw . S2w . T2c
    Because the function takes the transformation matrix from deformed space to original space, we take the inverse tranformation matrix = matrix as input to the function.

    p_w         = W p_v
    p_w'        = T_w W p_v
    p_v'        = W-1 T_w W p_v
                = T2c-1 S2w-1 Tw Sw Rw S2w T2c p_v
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
    return aff_transform(I, np.round(affine, 5), output_shape=output_shape or I.shape, trilin=order, cval=cval, os0=oversampling_factors[0], os1=oversampling_factors[1], os2=oversampling_factors[2])


def get_deformation_field(I_shape, shift=(2, 2, 2), nsize=(30, 30, 30), npad=(5, 5, 5), std=(6, 6, 6)):
    deformation_field = np.zeros(tuple(I_shape) + (3,))
    deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2] = random_deformation_field(I_shape, shift=shift, n=nsize, npad=npad, gaussian_std=std)
    return deformation_field


def elastic_deformation(I, deformation_field, cval=0, order=1):
    return backward_3d_warp(I, deformation_field[..., 0], deformation_field[..., 1], deformation_field[..., 2], trilin=order, cval=cval)


def orthogonalize(input_nii, invalid_value):
    """
    Orthogonalize by having a new z-axis that is orthogonal to the x and y axis. Keeps the dimensions of the image.

    Parameters
    ----------
    input_nii: nib.Nifti1Image
        Input image.
    invalid_value: float
        Outside image values.

    Returns
    -------
    output_nii: nib.Nifti1Image
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
    """
    Checking registration quality of two arrays, optionally using a mask.

    Parameters
    ----------
    fixed_array: np.ndarray
        First array.
    moved_array: np.ndarray
        Second array.
    mask_array: np.ndarray
        Mask array (optional).

    Returns
    -------
    correlation_coefficient: float
        Correlation coefficient of the two, optionally masked, arrays.

    See Also
    --------
    np.corrcoef: The NumPy function that is used under the hood.
    """
    if mask_array is not None:
        fixed_array = fixed_array[mask_array]
        moved_array = moved_array[mask_array]

    correlation_coefficient = np.corrcoef(fixed_array.flatten(), moved_array.flatten())[0, 1]
    return correlation_coefficient


def move(moving_nii, fixed_nii=None, transform_parameter_map=None, fixed_nii_mask=None, moving_nii_mask=None, parameter_map=None, initial_transform_parameter_map=None):
    """
    Function to move a 'moving image'. When a 'fixed image' is specified, this function is an actual registration. Otherwise, it will be simply applying a transformation to the image.
    To install Simple ITK with Simple Elastix there are two options:
    1) Following the installation instructions from https://simpleelastix.readthedocs.io/index.html with some small tweaks:
        $ git clone https://github.com/SuperElastix/SimpleElastix
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_CXX_COMPILER:STRING=/usr/bin/clang++ -DCMAKE_C_COMPILER:STRING=/usr/bin/clang -DWRAP_JAVA:BOOL=OFF -DWRAP_LUA:BOOL=OFF -DWRAP_R:BOOL=OFF -DWRAP_RUBY:BOOL=OFF -DWRAP_TCL:BOOL=OFF -DSimpleITK_PYTHON_USE_VIRTUALENV:BOOL=OFF ../SimpleElastix/SuperBuild
        $ make -j4
        Then activate your virtual environment and do the following (make sure there is not yet any Simple ITK version installed in this environment!)
        $ {BUILD_DIRECTORY}/SimpleITK-build/Wrapping/Python
        $ python Packaging/setup.py install
    2) Simply installing the simpleitk-elastix package via PIP in your virtual environment:
        $ pip install simpleitk-elastix

    Parameters
    ----------
    moving_nii: nib.Nifti1Image
        'moving image' Nifti.
    fixed_nii: None, nib.Nifti1Image
       'fixed image' Nifti. When specified, a registration is done.
    transform_parameter_map: None, dict, list of dicts
        A dictionary, or list of dictionaries, with the transformation parameter map(s) to be used to move the 'moving image'. When specified, a simple move of the 'moving image' will be done.
    fixed_nii_mask: None, nib.Nifti1Image
        'fixed image mask' Nifti. Only useful when doing a registration.
    moving_nii_mask: None, nib.Nifti1Image
        'moving image mask' Nifti. Only useful when doing a registration.
    parameter_map: None, str, dict, sitk.ParameterMap
        The type of registration that needs to be performed.
    initial_transform_parameter_map: None, sitk.ParameterMap
        Optional initial parameter map to be used.

    Returns
    -------
    moved_nii: nib.Nifti1Image
        'moved image' Nifti.
    transform_parameter_map: dict
        The transformation parameter map that was used. In case of a requested registration this is the resulting transformation for the registration. Otherwise, this is simply the inputted transformation parameter map.

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
