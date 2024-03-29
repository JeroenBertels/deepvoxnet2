"""The conversions module provides a collection of functions for common image and file format conversions, loading, and manipulation.

These functions are designed to make it easier to work with medical imaging data and to provide a standardized way of converting between different file formats and representations.
The module includes functions for converting DICOM files to NIfTI format, converting between NumPy arrays and SimpleITK images, and converting between different coordinate systems.
Additionally, the module includes several utility functions for working with files and directories.
"""

import os
import glob
import pymirc
import transforms3d
import numpy as np
import nibabel as nib
try:
    import SimpleITK as sitk

except ModuleNotFoundError:
    pass

from pydicom import dcmread
from tempfile import mkdtemp
from shutil import rmtree


def check_dcm_list(dcm_list):
    """Check if the DICOM files in the given list satisfy certain criteria.

    The following criteria are checked:
    1. All dcm slice positions are unique
    2. All dcm slice positions are equidistant
    3. There is no time difference greater than 60 seconds between the earliest and latest dcm slice.

    Parameters
    ----------
    dcm_list : List[str]
        List of paths to DICOM files to be checked.

    Raises
    ------
    AssertionError
        If any of the above criteria is not met.

    Returns
    -------
    None
    """

    slice_locations = []
    slice_times = []
    for dcm_file in dcm_list:
        dcm = dcmread(dcm_file)
        slice_locations.append(float(dcm.SliceLocation))
        slice_times.append(float(dcm[0x0019, 0x1024].value) if (0x0019, 0x1024) in dcm else float(dcm.AcquisitionTime if "AcquisitionTime" in dcm else 0))

    slice_locations = sorted(slice_locations)
    assert len(slice_locations) == len(set(slice_locations)), "Not all dcm slice positions are unique."
    assert all([slice_locations[1] - slice_locations[0] == slice_locations[i] - slice_locations[i - 1] for i in range(1, len(slice_locations))]), "Not all dcm slice positions are equidistant."
    assert max(slice_times) - min(slice_times) <= 60, "There is a time difference {} s > 1 min between the earliest and latest dcm slice.".format(max(slice_times) - min(slice_times))


def file_dir_to_list(file_dir_or_list, extension):
    """Takes a directory path or a list of file paths and returns a list of file paths with the given extension.

    Parameters
    ----------
    file_dir_or_list : str or list of str
        The path of the directory or a list of file paths.
    extension : str
        The file extension to filter the files.

    Returns
    -------
    list of str
        A list of file paths that have the given file extension.

    Raises
    ------
    AssertionError
        If `file_dir_or_list` is a list of file paths and any of the files does not have the given extension, or
        if `file_dir_or_list` is a path to a directory that does not exist.

    """

    if isinstance(file_dir_or_list, (list, tuple)):
        file_list = file_dir_or_list
        assert all([file.endswith(extension) for file in file_list]), "Not all files in the list have the extension '{}'".format(extension)

    else:
        file_dir = file_dir_or_list
        assert os.path.isdir(file_dir), "The input file directory '{}' is not a valid directory".format(file_dir)
        file_list = glob.glob(os.path.join(file_dir, "*" + extension))

    return file_list


def lps_to_ras(img_affine):
    """Converts an affine matrix from LPS (Left-Posterior-Superior) coordinate system to RAS (Right-Anterior-Superior) coordinate system.

    Parameters
    ----------
    img_affine : numpy.ndarray
        Affine matrix to convert from LPS to RAS coordinate system.

    Returns
    -------
    numpy.ndarray
        Affine matrix converted from LPS to RAS coordinate system.
    """
    xyflip = np.eye(4)
    xyflip[:2, :2] *= -1
    img_affine_ras = xyflip @ img_affine
    return img_affine_ras


def ras_to_lps(img_affine):
    """Converts an affine matrix from RAS (Right-Anterior-Superior) coordinate system to LPS (Left-Posterior-Superior) coordinate system.

    Parameters
    ----------
    img_affine : numpy.ndarray
        Affine matrix to convert from RAS to LPS coordinate system.

    Returns
    -------
    numpy.ndarray
        Affine matrix converted from RAS to LPS coordinate system.
    """

    xyflip = np.eye(4)
    xyflip[:2, :2] *= -1
    img_affine_lps = xyflip @ img_affine
    return img_affine_lps


def dcm_to_array(dcm_list, dcm_check=False, **kwargs):
    """Convert a list of DICOM files to a 3D NumPy array and affine matrix.

    Parameters
    ----------
    dcm_list : list
        A list of DICOM file paths.
    dcm_check : bool, optional
        Whether to check the DICOM files for equidistant slices and unique slice locations, by default False
    **kwargs : dict, optional
        Optional keyword arguments to be passed to pymirc.fileio.DicomVolume constructor.

    Returns
    -------
    tuple
        A tuple consisting of:
        - 3D NumPy array containing the DICOM image data.
        - Affine matrix of the DICOM image.

    Raises
    ------
    AssertionError
        If dcm_check is True and the DICOM files do not have equidistant slices or unique slice locations.
    """

    if dcm_check:
        check_dcm_list(dcm_list)

    dcm_img = pymirc.fileio.DicomVolume(dcm_list, **kwargs)
    dcm_array = dcm_img.get_data()
    T, R, Z, S = [np.round(transformation, 1) for transformation in transforms3d.affines.decompose44(dcm_img.affine)]
    if not np.allclose(S, [0, 0, 0]):
        print("WARNING: the image data is not orthogonal!")

    return dcm_array, dcm_img.affine


def sitk_to_nii(sitk_img):
    """Convert a SimpleITK image to a Nifti1Image object.

    Parameters
    ----------
    sitk_img : SimpleITK.Image
        The SimpleITK image to be converted.

    Returns
    -------
    nibabel.nifti1.Nifti1Image
        A Nifti1Image object.

    """

    tmpdir = mkdtemp()
    sitk.WriteImage(sitk_img, os.path.join(tmpdir, "nii_img.nii"))
    nii_img = nib.load(os.path.join(tmpdir, "nii_img.nii"))
    nii_img = nib.Nifti1Image(nii_img.get_fdata(), nii_img.affine)
    rmtree(tmpdir)
    return nii_img


def nii_to_sitk(nii_img):
    """Convert a nifti image to a SimpleITK image.

    Parameters
    ----------
    nii_img : nib.Nifti1Image
        The nifti image to be converted.

    Returns
    -------
    sitk.Image
        The SimpleITK image.

    """

    tmpdir = mkdtemp()
    nib.save(nii_img, os.path.join(tmpdir, "nii_img.nii"))
    sitk_img = sitk.ReadImage(os.path.join(tmpdir, "nii_img.nii"))
    rmtree(tmpdir)
    return sitk_img


def dcm_to_nii(dcm_dir_or_list, nii_path=None, dcm_check=True, **kwargs):
    """Convert a DICOM image or a directory of DICOM images to a Nifti1Image.

    Parameters
    ----------
    dcm_dir_or_list : str, list
        The directory containing DICOM images, or a list of DICOM file paths.
    nii_path : str, optional
        The path to save the Nifti1Image. Default is None.
    dcm_check : bool, optional
        Whether to check the consistency of the DICOM images. Default is True.
    **kwargs
        Additional keyword arguments for dcm_to_array.

    Returns
    -------
    nib.Nifti1Image
        The Nifti1Image.
    """

    dcm_list = file_dir_to_list(dcm_dir_or_list, extension=".dcm")
    img_array, img_affine = dcm_to_array(dcm_list, dcm_check=dcm_check, **kwargs)
    nii_img = nib.Nifti1Image(img_array, lps_to_ras(img_affine))  # nifti uses RAS instead of LPS so we have to convert the array and the affine
    if nii_path is not None:
        nib.save(nii_img, nii_path)

    return nii_img


def rt_to_nii(rt_path, reference_dir_or_list=None, reference_affine=None, reference_shape=None, nii_path=None, contour_indices=None, roi_number_as_label=False, dcm_check=True, **kwargs):
    """Converts a DICOM RTSTRUCT file to a NIfTI label file.

    Parameters
    ----------
    rt_path : str
        Path to the DICOM RTSTRUCT file.
    reference_dir_or_list : str or list of str, optional
        Path to the reference DICOM directory or a list of paths to the DICOM files.
    reference_affine : ndarray, optional
        The affine transformation of the reference image.
    reference_shape : tuple of int, optional
        The shape of the reference image.
    nii_path : str, optional
        Path to save the resulting NIfTI label file.
    contour_indices : list of int, optional
        List of contour indices to include in the output. If None, all contours are included.
    roi_number_as_label : bool, optional
        Whether to use the ROI number as the label in the output label file. If False, sequential integer labels are used instead.
    dcm_check : bool, optional
        Whether to check that the DICOM files in the reference directory or list are valid. Default is True.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `dcm_to_array` function.

    Returns
    -------
    rt_nii : Nifti1Image
        The NIfTI label image.
    """

    if reference_dir_or_list is None:
        assert reference_affine is not None and reference_shape is not None, "Please specify a reference affine and shape when not using a reference dcm dir or list."

    else:
        assert reference_affine is None and reference_shape is None, "Please do not specify a reference affine and shape when using a reference dcm dir or list."
        dcm_list = file_dir_to_list(reference_dir_or_list, extension=".dcm")
        reference_array, reference_affine = dcm_to_array(dcm_list, dcm_check=dcm_check, **kwargs)
        reference_shape = reference_array.shape

    # load rtstruct file
    contour_data = pymirc.fileio.read_rtstruct_contour_data(rt_path)  # read the ROI contours (in world coordinates)
    if len(contour_data) > 0 and contour_indices is not None:
        print([[cd['ROIName'], i] for i, cd in enumerate(contour_data)])
        contour_data = [contour_data[i] for _, i in sorted([[cd['ROIName'], i] for i, cd in enumerate(contour_data)])]
        contour_data = [contour_data[i] for i in contour_indices]

    roi_indices = pymirc.fileio.convert_contour_data_to_roi_indices(contour_data, reference_affine, reference_shape)  # convert contour data to index arrays (voxel space)
    # create a label array
    roi_array = np.zeros(reference_shape)
    for i in range(len(roi_indices)):
        roi_array[roi_indices[i]] = int(contour_data[i]['ROINumber']) if roi_number_as_label else i + 1

    # convert the label array to nifti and optionally save
    rt_nii = nib.Nifti1Image(roi_array.astype(np.uint8), lps_to_ras(reference_affine))  # nifti uses RAS instead of LPS so we have to convert the array and the affine
    if nii_path is not None:
        nib.save(rt_nii, nii_path)

    return rt_nii


def dcmperf_to_nii(dcm_dir_or_list, nii_path=None, verbose=True, **kwargs):
    """Convert a folder with DICOM perfusion images to a 4D NIfTI file with the time series.

    Parameters
    ----------
    dcm_dir_or_list : str or list of str
        Path to a directory containing DICOM files or a list of paths to individual DICOM files.
    nii_path : str or None, optional
        Path to save the NIfTI file or None to not save the NIfTI file (default is None).
    verbose : bool, optional
        Whether to print diagnostic messages (default is True).
    **kwargs : dict, optional
        Optional keyword arguments to pass to dcm_to_array.

    Returns
    -------
    perfusion_niis : list of NIfTI1Image
        A list of NIfTI1Image objects containing the perfusion data.
    perfusion_times : list of ndarray
        A list of ndarrays containing the perfusion times in seconds for each voxel.
    mean_perfusion_times : list of float
        A list of mean perfusion times for each time point.
    time_resolution : float
        The time resolution of the perfusion time series in seconds.

    Raises
    ------
    AssertionError
        If the time periods differ more than 5% compared to the average time period.
        If not all slice locations have an equal number of time points.
        If no integer division of number of unique slice locations with the number of slice locations per time point.
        If the perfusion scan type is not recognized.

    """

    dcm_list = file_dir_to_list(dcm_dir_or_list, extension=".dcm")
    # we load all dcm files in memory and store their original file paths and their acquisition times (i.e. SliceTime) and z location (i.e. SliceLocation)
    dcms = []
    for dcm_file in dcm_list:
        dcm = dcmread(dcm_file)
        dcms.append({
            "FilePath": dcm_file,
            "DcmObject": dcm,
            "SliceTime": float(dcm[0x0019, 0x1024].value),
            "SliceLocation": float(dcm.SliceLocation)
        })

    # we sort the list of dcm dictionaries primary according to acquisition time and secondary according to slice location and get some interesting properties
    dcms = sorted(dcms, key=lambda dcm: (dcm["SliceTime"], dcm["SliceLocation"]))
    unique_slice_locations, unique_slice_locations_counts = np.unique([dcm["SliceLocation"] for dcm in dcms], return_counts=True)
    unique_slice_times, unique_slice_times_counts = np.unique([dcm["SliceTime"] for dcm in dcms], return_counts=True)
    assert np.all(unique_slice_locations_counts == unique_slice_locations_counts[0]), "Not all slice locations have an equal number of time points."

    # let's check what type of perfusion scan it is in order to know whether this dcmperf_to_nii implementation can actually be used
    if len(unique_slice_times) == len(dcms):
        nb_time_points = unique_slice_locations_counts[0]
        if verbose:
            print("Helical perfusion scan.")

    elif unique_slice_times_counts[0] > 1:
        assert np.all(unique_slice_times_counts == unique_slice_times_counts[0]), "Not all time points have an equal number of slice locations."
        assert len(unique_slice_locations) % unique_slice_times_counts[0] == 0, "No integer division of number of unique slice locations with the number of slice locations per time point."
        jog_mode = len(unique_slice_locations) // unique_slice_times_counts[0]
        nb_time_points = unique_slice_times_counts[0] * jog_mode
        if verbose:
            print("Fixed area perfusion scan (i.e. jog mode == 1)." if jog_mode == 1 else "Jogged perfusion scan (jog mode == {}).".format(jog_mode))

    else:
        raise NotImplementedError

    # now we create a list of dictionaries, one for each time point of the perfusion series, containing the mean acquisition time and a sorted list (bottom to top) of the initial dcm dicts
    perfusion_series = []
    for time_point_i in range(nb_time_points):
        perfusion_series.append({
            "VolumeTime": (dcms[len(unique_slice_locations) * time_point_i]["SliceTime"] + dcms[len(unique_slice_locations) * (time_point_i + 1) - 1]["SliceTime"]) / 2,
            "Dcms": [[dcm for dcm in dcms[len(unique_slice_locations) * time_point_i: len(unique_slice_locations) * (time_point_i + 1)] if dcm["SliceLocation"] == slice_location][0] for slice_location in unique_slice_locations]
        })

    # perform a double check that every time period is almost equal to the mean time period (i.e. the time resolution)
    time_periods = [perfusion_series[i + 1]["VolumeTime"] - perfusion_series[i]["VolumeTime"] for i in range(nb_time_points - 1)]
    time_resolution = np.mean(time_periods)
    assert np.allclose(time_periods, time_resolution, rtol=0.05, atol=0), "The time periods differ more than 5 % compared to the average time period (i.e. the time resolution)."
    if verbose:
        print("\t- # slices: {}".format(len(unique_slice_locations)))
        print("\t- span: {} mm".format((max([dcm["SliceLocation"] for dcm in dcms]) - min([dcm["SliceLocation"] for dcm in dcms])) / (len(unique_slice_locations) - 1) * len(unique_slice_locations)))
        print("\t- # time points: {}".format(nb_time_points))
        print("\t- time resolution: {} s".format(np.round(time_resolution, 2)))

    perfusion_niis = []
    perfusion_times = []
    mean_perfusion_times = []
    for time_point_i in range(nb_time_points):
        perfusion_niis.append(dcm_to_nii([dcm["FilePath"] for dcm in perfusion_series[time_point_i]["Dcms"]], verbose=False, **kwargs))
        tmpdir = mkdtemp()
        for slice_location_i, slice_location in enumerate(unique_slice_locations):
            perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].decompress()
            slope = perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].RescaleSlope if "RescaleSlope" in perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"] else 1
            intercept = perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].RescaleIntercept if "RescaleIntercept" in perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"] else 0
            perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].pixel_array[...] = (perfusion_series[time_point_i]["Dcms"][slice_location_i]["SliceTime"] - intercept) / slope
            perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].PixelData = perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].pixel_array.tobytes()
            perfusion_series[time_point_i]["Dcms"][slice_location_i]["DcmObject"].save_as(os.path.join(tmpdir, "{}_{}.dcm".format(slice_location_i, time_point_i)))
            perfusion_series[time_point_i]["Dcms"][slice_location_i]["FilePath"] = os.path.join(tmpdir, "{}_{}.dcm".format(slice_location_i, time_point_i))

        perfusion_times.append(dcm_to_nii([dcm["FilePath"] for dcm in perfusion_series[time_point_i]["Dcms"]], verbose=False, **kwargs).get_fdata())
        mean_perfusion_times.append(perfusion_series[time_point_i]["VolumeTime"])
        rmtree(tmpdir)

    assert all([np.allclose(perfusion_nii.affine, perfusion_niis[0].affine) for perfusion_nii in perfusion_niis]), "Not all affines are equal."
    if nii_path is not None:
        perfusion_nii = nib.Nifti1Image(np.stack([perfusion_nii.get_fdata() for perfusion_nii in perfusion_niis], axis=-1), perfusion_niis[0].affine)
        perfusion_nii.header.set_zooms(perfusion_niis[0].header.get_zooms()[:3] + (time_resolution,))
        nib.save(perfusion_nii, nii_path)

    return perfusion_niis, perfusion_times, mean_perfusion_times, time_resolution
