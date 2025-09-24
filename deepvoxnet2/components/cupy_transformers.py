import random
import numpy as np
import cupy as cp
from scipy.spatial.transform import Rotation
from cupyx.scipy.ndimage import affine_transform, gaussian_filter, uniform_filter, distance_transform_edt
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.components.transformers import Transformer


class CupyThreshold(Transformer):
    def __init__(self, lower_threshold=0, upper_threshold=np.inf, **kwargs):
        super(CupyThreshold, self).__init__(**kwargs)
        self.lower_threshold = np.array(lower_threshold)
        self.upper_threshold = np.array(upper_threshold)

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            cp_sample = cp.asarray(sample)
            cp_sample[...] = (cp_sample > self.lower_threshold) * (cp_sample < self.upper_threshold)
            self.outputs[idx][idx_] = Sample(cp_sample.get(), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class CupyClip(Transformer):
    def __init__(self, lower_clip=-np.inf, higher_clip=np.inf, **kwargs):
        super(CupyClip, self).__init__(**kwargs)
        self.lower_clip = lower_clip
        self.higher_clip = higher_clip

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            cp_sample = cp.asarray(sample)
            cp.clip(cp_sample, self.lower_clip, self.higher_clip, out=cp_sample)
            self.outputs[idx][idx_] = Sample(cp_sample.get(), sample.affine)
            # del cp_sample

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class CupyIntensityTransform(Transformer):
    def __init__(self, mean_shift=0, std_shift=0, mean_scale=1, std_scale=0, **kwargs):
        super(CupyIntensityTransform, self).__init__(**kwargs)
        self.mean_shift = mean_shift
        self.std_shift = std_shift
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        self.shift = None
        self.scale = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            cp_sample = cp.asarray(sample)
            cp_sample[...] += self.shift
            cp_sample[...] *= self.scale
            self.outputs[idx][idx_] = Sample(cp_sample.get(), sample.affine)
            # del cp_sample

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        self.shift = np.random.normal(self.mean_shift, self.std_shift)
        self.scale = np.random.normal(self.mean_scale, self.std_scale)


class CupyResampledAffineCropper(Transformer):
    def __init__(
            self, 
            reference_connection, 
            segment_size,
            voxel_size,
            shear_window_width=(0, 0, 0), 
            rotation_window_width=(0, 0, 0), 
            translation_window_width=(0, 0, 0), 
            scaling_window_width=(0, 0, 0), 
            flip_probabilities=(0, 0, 0),
            nonzero=1,
            cval=0, 
            order=1, 
            **kwargs):
        super(CupyResampledAffineCropper, self).__init__(extra_connections=reference_connection, **kwargs)
        assert self.n is not None
        self.reference_connection = reference_connection
        self.segment_size = segment_size
        self.voxel_size = voxel_size
        self.shear_window_width = shear_window_width
        self.rotation_window_width = rotation_window_width
        self.translation_window_width = translation_window_width
        self.scaling_window_width = scaling_window_width
        self.flip_probabilities = flip_probabilities
        self.nonzero = nonzero
        self.cval = cval
        self.order = order
        self.backward_affine = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            assert self.backward_affine is not None
            transformed_sample = cp.zeros((sample.shape[0], *self.segment_size, sample.shape[-1]))
            for batch_i in range(len(sample)):
                for feature_i in range(sample.shape[-1]):
                    cval = self.cval[idx] if isinstance(self.cval, (tuple, list)) else self.cval
                    cval = cval[feature_i] if isinstance(cval, (tuple, list)) else cval
                    affine_transform(
                        cp.asarray(sample[batch_i, ..., feature_i]),
                        cp.asarray(self.backward_affine),
                        output_shape=self.segment_size,
                        cval=cval,
                        order=self.order[idx] if isinstance(self.order, (tuple, list)) else self.order,
                        output=transformed_sample[batch_i, ..., feature_i]
                    )

            transformed_affine = Sample.update_affine(sample.affine, transformation_matrix=self.backward_affine)
            self.outputs[idx][idx_] = Sample(transformed_sample.get(), transformed_affine)
            # del transformed_sample

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [tuple([output_shape_ if axis_i not in [1, 2, 3] else (self.segment_size[idx][axis_i - 1] if isinstance(self.segment_size, list) else self.segment_size[axis_i - 1]) for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
        if self.n_ == 0:
            self.coordinates = list(zip(*np.nonzero(np.any(self.reference_connection[0] != 0, axis=(0, -1)))))
            assert len(self.coordinates) > 0

        if random.random() < self.nonzero:
            coordinates = random.choice(self.coordinates)

        else:
            coordinates = tuple([random.choice(range(self.reference_connection[0].shape[i])) for i in range(1, 4)])
        
        scaling = np.linalg.norm(self.reference_connection[0].affine[0, :3, :3], 2, axis=1) / self.voxel_size
        translation = np.array(self.segment_size) // 2
        shear = [random.uniform(-w, w) for w in self.shear_window_width]

        rotation = [random.uniform(-w, w) for w in self.rotation_window_width]
        translation = [t + random.uniform(-w, w) for t, w in zip(translation, self.translation_window_width)]
        scaling = [s * (1 + random.uniform(-w, w)) for s, w in zip(scaling, self.scaling_window_width)] 
        flip = [-1 if random.random() < p else 1 for p in self.flip_probabilities]

        # # via forward affine
        # T2c, Sc, R, Sh, F, T2s = np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        # T2c[:-1, -1] = -np.array(coordinates)
        # Sc[:-1, :-1] = np.diag(scaling)
        # R[:-1, :-1] = Rotation.from_euler('zyx', rotation).as_matrix()
        # Sh[0, 1], Sh[0, 2], Sh[1, 0], Sh[1, 2], Sh[2, 0], Sh[2, 1] = shear[0], shear[0], shear[1], shear[1], shear[2], shear[2]
        # F[:-1, :-1] = np.diag(flip)
        # T2s[:-1, -1] = np.array(translation)
        # affine = T2s @ F @ Sh @ R @ Sc @ T2c
        # self.backward_affine = np.round(np.linalg.inv(affine), 5)

        # straight as backward affine
        T2c, Sc, R, Sh, F, T2s = np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        T2c[:-1, -1] = np.array(coordinates)
        Sc[:-1, :-1] = np.diag(1 / np.array(scaling))
        R[:-1, :-1] = Rotation.from_euler('zyx', -np.array(rotation)).as_matrix()
        Sh[0, 1], Sh[0, 2], Sh[1, 0], Sh[1, 2], Sh[2, 0], Sh[2, 1] = -shear[0], -shear[0], -shear[1], -shear[1], -shear[2], -shear[2]
        F[:-1, :-1] = np.diag(flip)
        T2s[:-1, -1] = -np.array(translation)
        self.backward_affine = T2c @ Sc @ R @ Sh @ F @ T2s


class CupyPut(Transformer):
    def __init__(self, reference_connection, caching=True, cval=0, order=0, keep_counts=False, gaussian_counts=False, **kwargs):
        super(CupyPut, self).__init__(extra_connections=reference_connection, **kwargs)
        self.reference_connection = reference_connection
        self.caching = caching
        self.cval = cval
        self.order = order
        self.prev_references = [None] * len(reference_connection)
        self.output_array_counts = None
        self.keep_counts = keep_counts
        self.gaussian_counts = gaussian_counts

    def _update_idx(self, idx):
        for idx_, (reference, sample) in enumerate(zip(self.reference_connection.get(), self.connections[idx][0].get())):
            cp_sample = cp.asarray(sample)
            cp_output = cp.asarray(self.outputs[idx][idx_])
            for i in range(sample.shape[0]):
                backward_affine = cp.asarray(np.linalg.inv(sample.affine[i]) @ reference.affine[0])
                transformed_array = cp.stack([affine_transform(cp_sample[i, ..., j], backward_affine, output_shape=reference.shape[1:4], cval=self.cval, order=self.order) for j in range(sample.shape[4])], axis=-1)[None, ...]
                if self.keep_counts:
                    kernel = cp.ones_like(cp_sample[i, ..., 0])
                    if self.gaussian_counts:
                        kernel = gaussian_filter(kernel, sigma=[max(1, s // 2) for s in kernel.shape])

                    transformed_array_counts = affine_transform(kernel, backward_affine, output_shape=reference.shape[1:4], cval=0, order=self.order)[None, ..., None]
                    transformed_array = self.output_array_counts[idx][idx_] / (self.output_array_counts[idx][idx_] + transformed_array_counts) * cp_output + transformed_array_counts / (self.output_array_counts[idx][idx_] + transformed_array_counts) * transformed_array
                    self.output_array_counts[idx][idx_] += transformed_array_counts
                
                mask = (transformed_array != self.cval).get()
                self.outputs[idx][idx_][mask] = transformed_array[mask].get()
        
    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        assert len(self.connections[idx][0]) == len(self.reference_connection), "The length of the connection to be put must be equal to the length of the reference connection."
        return [reference_shape[:4] + sample_shape[4:] for reference_shape, sample_shape in zip(self.reference_connection.shapes, self.connections[idx][0].shapes)]

    def _randomize(self):
        if self.output_array_counts is None and self.keep_counts:
            self.output_array_counts = [[None] * len(self.reference_connection) for _ in self.connections]

        for idx, connection in enumerate(self.connections):
            assert len(connection[0]) == len(self.reference_connection), "The length of the connection to be put must be equal to the length of the reference connection."
            for idx_, sample in enumerate(connection[0]):
                if not self.caching or self.prev_references[idx_] is not self.reference_connection[idx_]:
                    assert self.reference_connection[idx_].shape[0] == 1, "The batch dimension of a reference sample must be 1."
                    self.prev_references[idx_] = self.reference_connection[idx_]
                    self.outputs[idx][idx_] = Sample(np.full(self.reference_connection[idx_].shape[:4] + sample.shape[4:], self.cval), self.reference_connection[idx_].affine)
                    if self.keep_counts:
                        self.output_array_counts[idx][idx_] = cp.full(self.reference_connection[idx_].shape[:4] + (1,), 1e-7)


class CupyCrop(Transformer):
    def __init__(self, reference_connection, segment_size, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, **kwargs):
        super(CupyCrop, self).__init__(extra_connections=reference_connection, **kwargs)
        self.ncrops = self.n
        self.segment_size = segment_size
        self.reference_connection = reference_connection
        self.subsample_factors = subsample_factors
        self.default_value = default_value
        self.prefilter = prefilter
        self.coordinates = None

    def _update_idx(self, idx):
        segment_size = self.segment_size[idx] if isinstance(self.segment_size, list) else self.segment_size
        subsample_factors = self.subsample_factors[idx] if isinstance(self.subsample_factors, list) else self.subsample_factors
        backward_affine = Sample.update_affine(translation=self.coordinates[self.n_]) @ Sample.update_affine(scaling=subsample_factors[:3]) @ Sample.update_affine(translation=[-(segment_size_ // 2) for segment_size_ in segment_size[:3]])
        for idx_, sample in enumerate(self.connections[idx][0]):
            transformed_sample = CupyCrop.cupy_crop(
                sample,
                (len(sample),) + segment_size,
                (None,) + self.coordinates[self.n_] + (None,) * (len(segment_size) - len(self.coordinates[self.n_])),
                (1,) + subsample_factors,
                self.default_value[idx] if isinstance(self.default_value, (tuple, list)) else self.default_value,
                self.prefilter[idx] if isinstance(self.prefilter, (tuple, list)) else self.prefilter
            )
            transformed_affine = Sample.update_affine(sample.affine, transformation_matrix=backward_affine)
            self.outputs[idx][idx_] = Sample(transformed_sample, transformed_affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [tuple([output_shape_ if axis_i not in [1, 2, 3] else (self.segment_size[idx][axis_i - 1] if isinstance(self.segment_size, list) else self.segment_size[axis_i - 1]) for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            self.coordinates = [tuple([shape // 2 for shape in self.connections[0][0][0].shape[1:4]])] * self.n

    @staticmethod
    def cupy_crop(I, S_size, coordinates=None, subsample_factors=None, default_value=0, prefilter=None):
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
        
        subsample_factors = [int(s_f) for s_f in subsample_factors]
        I = cp.asarray(I)
        if prefilter is not None:
            assert prefilter in ["uniform", "gaussian"]
            if prefilter == "gaussian":
                I = gaussian_filter(I.astype(cp.float64), [s_f if s_f > 1 else 0 for s_f in subsample_factors], mode="nearest").astype(cp.float32)

            elif prefilter == "uniform":
                I = uniform_filter(I.astype(cp.float64), subsample_factors, mode="nearest").astype(cp.float32)

        S_size = tuple(S_size) + tuple(I.shape[len(S_size):])
        S = cp.full(S_size, fill_value=default_value, dtype=cp.float32)
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
        if cp.any(cp.isnan(default_value)):
            S = S[tuple(distance_transform_edt(cp.isnan(S), return_distances=False, return_indices=True))]

        return S.get()


class CupyResampledAffineCropperV2(Transformer):
    def __init__(
            self, 
            reference_connection, 
            segment_size,
            voxel_size,
            shear_window_width=(0, 0, 0), 
            rotation_window_width=(0, 0, 0), 
            translation_window_width=(0, 0, 0), 
            scaling_window_width=(0, 0, 0), 
            flip_probabilities=(0, 0, 0),
            nonzero=1,
            cval=0, 
            order=1, 
            **kwargs):
        super(CupyResampledAffineCropperV2, self).__init__(extra_connections=reference_connection, **kwargs)
        assert self.n is not None
        self.reference_connection = reference_connection
        self.segment_size = segment_size
        self.voxel_size = voxel_size
        self.shear_window_width = shear_window_width
        self.rotation_window_width = rotation_window_width
        self.translation_window_width = translation_window_width
        self.scaling_window_width = scaling_window_width
        self.flip_probabilities = flip_probabilities
        self.nonzero = nonzero
        self.cval = cval
        self.order = order
        self.backward_affine = None
        self.nonzero_coordinates = None
        self.ranges = None
        self.hss = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            assert self.backward_affine is not None
            transformed_sample = cp.zeros((sample.shape[0], *self.segment_size, sample.shape[-1]))
            for batch_i in range(len(sample)):
                for feature_i in range(sample.shape[-1]):
                    cval = self.cval[idx] if isinstance(self.cval, (tuple, list)) else self.cval
                    cval = cval[feature_i] if isinstance(cval, (tuple, list)) else cval
                    affine_transform(
                        cp.asarray(sample[batch_i, ..., feature_i]),
                        cp.asarray(self.backward_affine),
                        output_shape=self.segment_size,
                        cval=cval,
                        order=self.order[idx] if isinstance(self.order, (tuple, list)) else self.order,
                        output=transformed_sample[batch_i, ..., feature_i]
                    )

            transformed_affine = Sample.update_affine(sample.affine, transformation_matrix=self.backward_affine)
            self.outputs[idx][idx_] = Sample(transformed_sample.get(), transformed_affine)
            # del transformed_sample

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [tuple([output_shape_ if axis_i not in [1, 2, 3] else (self.segment_size[idx][axis_i - 1] if isinstance(self.segment_size, list) else self.segment_size[axis_i - 1]) for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
        if self.n_ == 0:
            self.nonzero_coordinates = cp.argwhere(cp.any(cp.asarray(self.reference_connection[0]) != 0, axis=(0, -1))).get()
            self.hss = (np.array(self.segment_size) * self.voxel_size / np.linalg.norm(self.reference_connection[0].affine[0, :3, :3], 2, axis=1)) // 2
            self.ranges = []
            for i in range(3):
                self.ranges.append(list(range(int(self.hss[i]), self.reference_connection[0].shape[i + 1] - int(self.hss[i]))) or [self.reference_connection[0].shape[i + 1] // 2])

        if len(self.nonzero_coordinates) == 0:
            coordinates = [random.choice(r) for r in self.ranges]

        elif random.random() < self.nonzero:
            tries = 0
            while True:
                coordinates = np.array([random.choice(r) for r in self.ranges])
                c0 = coordinates - self.hss
                c1 = coordinates + self.hss
                if np.any(np.all(np.greater(self.nonzero_coordinates, c0) * np.less(self.nonzero_coordinates, c1), axis=1)):
                    break
                
                tries += 1
                if tries == 1000:
                    coordinates = [random.choice(r) for r in self.ranges]
                    break

        else:
            coordinates = [random.choice(r) for r in self.ranges]
        
        scaling = np.linalg.norm(self.reference_connection[0].affine[0, :3, :3], 2, axis=1) / self.voxel_size
        translation = np.array(self.segment_size) // 2
        shear = [random.uniform(-w, w) for w in self.shear_window_width]

        rotation = [random.uniform(-w, w) for w in self.rotation_window_width]
        translation = [t + random.uniform(-w, w) for t, w in zip(translation, self.translation_window_width)]
        scaling = [s * (1 + random.uniform(-w, w)) for s, w in zip(scaling, self.scaling_window_width)] 
        flip = [-1 if random.random() < p else 1 for p in self.flip_probabilities]

        # straight as backward affine
        T2c, Sc, R, Sh, F, T2s = np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4)
        T2c[:-1, -1] = np.array(coordinates)
        Sc[:-1, :-1] = np.diag(1 / np.array(scaling))
        R[:-1, :-1] = Rotation.from_euler('zyx', -np.array(rotation)).as_matrix()
        Sh[0, 1], Sh[0, 2], Sh[1, 0], Sh[1, 2], Sh[2, 0], Sh[2, 1] = -shear[0], -shear[0], -shear[1], -shear[1], -shear[2], -shear[2]
        F[:-1, :-1] = np.diag(flip)
        T2s[:-1, -1] = -np.array(translation)
        self.backward_affine = T2c @ Sc @ R @ Sh @ F @ T2s
