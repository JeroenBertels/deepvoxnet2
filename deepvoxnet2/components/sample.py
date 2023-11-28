"""This module contains the `Sample` class, which represents a sample of image data with an affine matrix to transform the sample between image and world space.

It is a fundamental class in the DVN2 package, which provides a framework for handling medical image data.
The `Sample` class is used in various other classes of the package, including `Modality` subclasses that represent specific modalities, such as `NiftyModality` and `ArrayModality`, and the `Record` class that organizes multiple modalities from a single case.
Additionally, the `Dataset` and `Case` classes use the `Sample` class to represent individual samples of a dataset and a case, respectively.
Finally, the `Mirc` class is the entry point to the DVN2 package, which provides functionalities for reading, writing, and processing medical image data stored in the MIRC (Medical Imaging Research Center) format.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class Sample(np.ndarray):
    """Represents an image sample as a numpy array with affine matrix information.

    Attributes:
    -----------
    affine : numpy.ndarray, shape (n, 4, 4)
        Affine matrices representing the spatial transformation of the image.

    Methods:
    --------
    __new__(cls, array, affine=None, dtype=np.float32):
        Creates a new Sample object from the input array and affine matrix.

    __array_finalize__(self, obj):
        Finalizes the creation of a Sample object and sets the affine matrix attribute.

    aff_to_144(affine):
        Converts an affine matrix to a shape of (n, 4, 4).

    nd_to_5d(array):
        Converts an input numpy array to a shape of (n, x, y, z, f) where n is the batch size,
        (x, y, z) are the spatial dimensions, and f is the feature dimension.

    update_affine(affine=None, transformation_matrix=None, reflection=None, shear=None,
                  rotation=None, translation=None, scaling=None):
        Updates the affine matrix based on a given transformation matrix or various
        affine transformation parameters.
    """

    def __new__(cls, array, affine=None, dtype=np.float32):
        """Creates a new Sample object from a 4D numpy array and an affine transformation.

        Parameters
        ----------
        array : ndarray
            A 4D numpy array representing the sample.
        affine : ndarray or None, optional
            A 4x4 affine transformation matrix that maps voxel coordinates to world coordinates.
            Default is None.
        dtype : dtype, optional
            The data type of the created ndarray. Default is np.float32.

        Returns
        -------
        Sample
            A new Sample object.
        """

        obj = np.asarray(cls.nd_to_5d(array), dtype=dtype).view(cls)
        affine = obj.aff_to_144(affine)
        obj.affine = np.broadcast_to(affine, (len(obj), 4, 4)).copy()
        return obj

    def __array_finalize__(self, obj):
        """Finalizes the creation of the Sample object.

        Parameters
        ----------
        self : Sample
            The Sample object being created.
        obj : ndarray or None
            The input array from which the Sample object is being created.
        """

        if obj is not None:
            self.affine = getattr(obj, 'affine', None)

    @staticmethod
    def aff_to_144(affine):
        """Converts the input affine transformation to a 4x4 matrix.

        Parameters
        ----------
        affine : ndarray or None
            The input affine transformation. Can be None, a 2D array with shape (4, 4), or a 3D array with shape (n, 4, 4).

        Returns
        -------
        ndarray
            A 3D numpy array with shape (n, 4, 4) representing the input affine transformations.
        """

        if affine is None:
            return np.stack([np.eye(4, dtype=np.float32)])

        elif affine.ndim == 2:
            return np.stack([np.array(affine, dtype=np.float32)])

        else:
            assert affine.ndim == 3, "Affines cannot have more than 3 dimensions."
            return np.array(affine, dtype=np.float32)

    @staticmethod
    def nd_to_5d(array):
        """Converts the input array to a 5D numpy array with shape (n, c, x, y, z).

        Parameters
        ----------
        array : ndarray
            The input numpy array. Can have between 0 and 5 dimensions.

        Returns
        -------
        ndarray
            A 5D numpy array with shape (n, c, x, y, z) representing the input array.
        """

        array = np.asarray(array)
        if array.ndim == 0:
            return array[None, None, None, None, None]

        elif array.ndim == 1:
            return array[None, ..., None, None, None]

        elif array.ndim == 2:
            return array[None, ..., None, None]

        elif array.ndim == 3:
            return array[None, ..., None]

        elif array.ndim == 4:
            return array[None, ...]

        else:
            assert array.ndim == 5, "DVN2 only supports up to 5 dimensions (i.e. 3 spatial and 1 feature dimension)."
            return array

    @staticmethod
    def ln_to_l5(shape):
        """Converts the shape to a shape of length 5.

        Parameters
        ----------
        shape : tuple
            The input shape. Can have a length between 0 and 5.

        Returns
        -------
        tuple
            A length 5 shape representing the input shape.
        """

        shape = tuple(shape)
        if len(shape) == 0:
            return (1, 1, 1, 1, 1)

        elif len(shape) == 1:
            return (1,) + shape + (1, 1, 1)

        elif len(shape) == 2:
            return (1,) + shape + (1, 1)

        elif len(shape) == 3:
            return (1,) + shape + (1,)

        elif len(shape) == 4:
            return (1,) + shape

        else:
            assert len(shape) == 5, "DVN2 only supports up to 5 dimensions (i.e. 3 spatial and 1 feature dimension)."
            return shape
        
    @staticmethod
    def update_affine(affine=None, transformation_matrix=None, reflection=None, shear=None, rotation=None, translation=None, scaling=None):
        """Update the affine of a Sample.

        Parameters:
        -----------
        affine : np.ndarray or None
            The current affine matrix.
        transformation_matrix : np.ndarray or None
            The transformation matrix to apply to the current affine.
        reflection : list or None
            A list of three values indicating whether to reflect (invert) each axis. If True, the axis is reflected.
        shear : list or None
            A list of three values indicating the shear to apply to each axis.
        rotation : list or None
            A list of three values indicating the rotation to apply to each axis.
        translation : list or None
            A list of three values indicating the translation to apply to each axis.
        scaling : list or None
            A list of three values indicating the scaling to apply to each axis.

        Returns:
        --------
        affine : np.ndarray
            The new affine matrix.
        """

        assert len([arg for arg in [transformation_matrix, reflection, shear, rotation, translation, scaling] if arg is not None]) <= 1
        affine = Sample.aff_to_144(affine)
        if transformation_matrix is None:
            transformation_matrix = np.eye(4)

        if reflection is not None:
            transformation_matrix[:-1, :-1] = np.diag(reflection)

        elif shear is not None:
            transformation_matrix[0, 1] = shear[0]
            transformation_matrix[0, 2] = shear[0]
            transformation_matrix[1, 0] = shear[1]
            transformation_matrix[1, 2] = shear[1]
            transformation_matrix[2, 0] = shear[2]
            transformation_matrix[2, 1] = shear[2]

        elif rotation is not None:
            transformation_matrix[:-1, :-1] = Rotation.from_euler('zyx', rotation).as_matrix()

        elif translation is not None:
            transformation_matrix[:-1, -1] = np.array(translation)

        elif scaling is not None:
            transformation_matrix[:-1, :-1] = np.diag(scaling)

        for i, affine_ in enumerate(affine):
            affine[i, ...] = affine_ @ transformation_matrix

        return affine
