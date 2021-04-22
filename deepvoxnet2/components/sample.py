import numpy as np
from scipy.spatial.transform import Rotation


class Sample(np.ndarray):
    def __new__(cls, array, affine=None, dtype=np.float32):
        obj = np.asarray(cls.nd_to_5d(array), dtype=dtype).view(cls)
        affine = obj.aff_to_144(affine)
        obj.affine = np.broadcast_to(affine, (len(obj), 4, 4)).copy()
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.affine = getattr(obj, 'affine', None)

    @staticmethod
    def aff_to_144(affine):
        if affine is None:
            return np.stack([np.eye(4, dtype=np.float32)])

        elif affine.ndim == 2:
            return np.stack([np.array(affine, dtype=np.float32)])

        else:
            assert affine.ndim == 3, "Affines cannot have more than 3 dimensions."
            return np.array(affine, dtype=np.float32)

    @staticmethod
    def nd_to_5d(array):
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
    def update_affine(affine=None, transformation_matrix=None, reflection=None, shear=None, rotation=None, translation=None, scaling=None):
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
