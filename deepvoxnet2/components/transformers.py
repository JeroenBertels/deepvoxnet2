import random
import uuid
import numpy as np
from scipy.ndimage import distance_transform_edt
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.utilities import transformations


class Connection(object):
    def __init__(self, transformer, idx):
        self.transformer = transformer
        self.idx = idx

    def __len__(self):
        return len(self.get())

    def __getitem__(self, item):
        return self.get()[item]

    def __iter__(self):
        return iter(self.get())

    def get(self):
        return self.transformer.outputs[self.idx]

    def eval(self, sample_id=None):
        return self.transformer.eval(sample_id)[self.idx]


class Transformer(object):
    def __init__(self, n=1, output_len=None, extra_connections=None):
        self.n = n
        self.n_ = 0
        self.connections = []
        self.outputs = []
        self.extra_connections = [] if extra_connections is None else (extra_connections if isinstance(extra_connections, list) else [extra_connections])
        self.active_indices = None
        self.generator = None
        self.sample_id = None
        self.output_len = output_len

    def __call__(self, *connections):
        if len(connections) == 0:
            input_connections = [Connection(self, idx) for idx in range(len(self.outputs))]

        else:
            input_connections = []
            for connection in connections:
                if self.__class__.__name__ in ["Group", "Multiply"]:
                    assert isinstance(connection, list)
                    self.connections.append(connection)

                else:
                    assert not isinstance(connection, list)
                    self.connections.append([connection])

                self.outputs.append([None] * self.get_output_len(len(self.connections) - 1))
                input_connections.append(Connection(self, len(self.outputs) - 1))

        return input_connections if len(input_connections) > 1 else (input_connections[0] if len(input_connections) == 1 else None)

    def get_output_len(self, idx):
        if self.output_len is None and len(self.connections[idx]) == 1:
            return len(self.connections[idx][0])

        elif self.output_len is None and len(self.connections[idx]) > 1:
            return np.sum([len(connection) for connection in self.connections[idx]])

        else:
            assert self.output_len is not None
            return self.output_len

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        return self.outputs[item]

    def __iter__(self):
        return iter(self.outputs)

    def eval(self, sample_id=None):
        if sample_id is None:
            sample_id = uuid.uuid4()

        if self.sample_id is None or sample_id != self.sample_id:
            self.sample_id = sample_id
            if self.generator is not None and (self.n is None or self.n_ < self.n):
                next(self.generator)

            else:
                for connection in self.connections + [self.extra_connections]:
                    for connection_ in connection:
                        if connection_.transformer.active_indices is None or len(connection_.transformer.active_indices) > 0:
                            connection_.transformer.eval(self.sample_id)

                self.generator = self.update()
                next(self.generator)

        return self.outputs

    def update(self):
        self.n_ = 0
        while self.n is None or self.n_ < self.n:
            self._randomize()
            for idx, connection in enumerate(self.connections):
                if self.active_indices is None or idx in self.active_indices:
                    self._update(idx, connection)

            self.n_ += 1
            yield self.outputs

    def _update(self, idx, connection):
        raise NotImplementedError

    def _randomize(self):
        raise NotImplementedError


class _Input(Transformer):
    def __init__(self, n=1, output_len=None):
        super(_Input, self).__init__(n, output_len)
        self.outputs.append([None] * output_len)
        self.n_resets = 0

    def __call__(self):
        return super(_Input, self).__call__()

    def load(self, identifier):
        raise NotImplementedError

    def _update(self, idx, connection):
        pass

    def _randomize(self):
        if self.n_ == 0:
            self.n_resets += 1


class _MircInput(_Input):
    def __init__(self, modality_ids, n=1):
        self.modality_ids = modality_ids if isinstance(modality_ids, list) else [modality_ids]
        super(_MircInput, self).__init__(n, len(self.modality_ids))

    def load(self, identifier):
        for idx_, modality_id in enumerate(self.modality_ids):
            self.outputs[0][idx_] = identifier.mirc[identifier.dataset_id][identifier.case_id][identifier.record_id][modality_id].load()


class MircInput(_MircInput):
    def __new__(cls, modality_ids, n=1):
        return _MircInput(modality_ids, n)()


class _SampleInput(_Input):
    def __init__(self, samples, n=1):
        self.samples = samples if isinstance(samples, list) else [samples]
        super(_SampleInput, self).__init__(n, len(self.samples))

    def load(self, identifier):
        for idx_, sample in enumerate(self.samples):
            self.outputs[0][idx_] = sample


class SampleInput(_SampleInput):
    def __new__(cls, samples, n=1):
        return _SampleInput(samples, n)()


class Group(Transformer):
    def __init__(self):
        super(Group, self).__init__(1)

    def _update(self, idx, connection):
        idx_ = 0
        for connection_ in connection:
            for sample in connection_:
                self.outputs[idx][idx_] = sample
                idx_ += 1

    def _randomize(self):
        pass


class Split(Transformer):
    def __init__(self, indices=(0,)):
        self.indices = indices if isinstance(indices, tuple) else (indices,)
        super(Split, self).__init__(1, len(self.indices))

    def _update(self, idx, connection):
        for idx_, j in enumerate(self.indices):
            self.outputs[idx][idx_] = connection[0][j]

    def _randomize(self):
        pass


class Concat(Transformer):
    def __init__(self, axis=-1):
        super(Concat, self).__init__(1, 1)
        assert axis in [0, 4, -1]
        self.axis = axis

    def _update(self, idx, connection):
        self.outputs[idx][0] = Sample(np.concatenate([sample for sample in connection[0]], axis=self.axis), connection[0][0].affine if self.axis != 0 else np.concatenate([sample.affine for sample in connection[0]]))

    def _randomize(self):
        pass


class Mean(Transformer):
    def __init__(self, axis=-1):
        super(Mean, self).__init__(1)
        assert axis in [0, 4, -1]
        self.axis = axis

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            self.outputs[idx][idx_] = sample.mean(axis=self.axis, keepdims=True)

    def _randomize(self):
        pass


class Threshold(Transformer):
    def __init__(self, lower_threshold=0, upper_threshold=np.inf):
        super(Threshold, self).__init__(1)
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            affine = sample.affine
            array = (sample > self.lower_threshold) * (sample < self.upper_threshold)
            self.outputs[idx][idx_] = Sample(array, affine)

    def _randomize(self):
        pass


class Multiply(Transformer):
    def __init__(self):
        super(Multiply, self).__init__(1)

    def get_output_len(self, idx):
        return len(self.connections[idx][0])

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            array = sample
            for i in range(1, len(connection)):
                array = array * connection[i][idx_]

            self.outputs[idx][idx_] = Sample(array, sample.affine)

    def _randomize(self):
        pass


class FillNan(Transformer):
    def __init__(self, fill_value):
        super(FillNan, self).__init__(1)
        self.fill_value = fill_value

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            self.outputs[idx][idx_] = Sample(np.nan_to_num(sample, nan=self.fill_value), sample.affine)

    def _randomize(self):
        pass


class Clip(Transformer):
    def __init__(self, lower_clip=-np.inf, higher_clip=np.inf):
        super(Clip, self).__init__(1)
        self.lower_clip = lower_clip
        self.higher_clip = higher_clip

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            self.outputs[idx][idx_] = Sample(np.clip(sample, self.lower_clip, self.higher_clip), sample.affine)

    def _randomize(self):
        pass


class Round(Transformer):
    def __init__(self, decimals=0):
        super(Round, self).__init__(1)
        self.decimals = decimals

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            self.outputs[idx][idx_] = Sample(np.round(sample, self.decimals), sample.affine)

    def _randomize(self):
        pass


class Normalize(Transformer):
    def __init__(self, shift, scale):
        super(Normalize, self).__init__(1)
        self.shift = shift
        self.scale = scale

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            self.outputs[idx][idx_] = Sample((sample + self.shift) * self.scale, sample.affine)

    def _randomize(self):
        pass


class WindowNormalize(Transformer):
    def __init__(self, lower_window=0, higher_window=1):
        super(WindowNormalize, self).__init__(1)
        self.lower_window = lower_window
        self.higher_window = higher_window

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            lower_window = np.min(sample) if self.lower_window is None else self.lower_window
            higher_window = np.max(sample) if self.higher_window is None else self.higher_window
            self.outputs[idx][idx_] = Sample(sample * (higher_window - lower_window) / (np.max(sample) - np.min(sample)) + (lower_window - np.min(sample)), sample.affine)

    def _randomize(self):
        pass


class Remove(Transformer):
    def __init__(self, remove_probability, fill_value, axis=-1, n=1):
        super(Remove, self).__init__(n)
        self.remove_probability = remove_probability
        self.fill_value = fill_value
        self.axis = axis
        self.remove_state = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            removed_array = sample.copy()
            removed_array = np.moveaxis(removed_array, self.axis, 0)
            for i, remove_state in enumerate(self.remove_sate):
                if remove_state:
                    removed_array[i, ...] = self.fill_value

            self.outputs[idx][idx_] = Sample(np.moveaxis(removed_array, 0, self.axis), sample.affine)

    def _randomize(self):
        shapes = [sample.shape[self.axis] for connection in self.connections for sample in connection[0] if sample is not None]
        assert all([shapes[0] == shape for shape in shapes])
        self.remove_sate = [random.random() < self.remove_probability for _ in range(shapes[0])]


class ClassWeights(Transformer):
    def __init__(self, class_weights_dict, one_hot=False):
        super(ClassWeights, self).__init__(1)
        self.class_weights_dict = class_weights_dict
        self.one_hot = one_hot
        k = np.array(list(class_weights_dict.keys()))
        v = np.array(list(class_weights_dict.values()))
        self.mapping_array = np.zeros(np.max(k) + 1, dtype=np.float32)
        self.mapping_array[k] = v

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            if self.one_hot:
                assert sample.shape[-1] > 1
                sample = Sample(np.argmax(sample, axis=-1)[..., None], sample.affine)

            else:
                sample = np.round(sample).astype(int)

            self.outputs[idx][idx_] = Sample(self.mapping_array[sample], sample.affine)

    def _randomize(self):
        pass


class Recalibrate(Transformer):
    def __init__(self, source_priors, target_priors):
        super(Recalibrate, self).__init__(1)
        self.source_priors = source_priors if isinstance(source_priors, list) else [source_priors]
        self.target_priors = target_priors if isinstance(target_priors, list) else [target_priors]
        assert len(self.source_priors) == len(self.target_priors)

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            assert sample.shape[-1] == len(self.source_priors)
            if sample.shape[-1] == 1:
                tp, sp = self.target_priors[0], self.source_priors[0]
                recalibrated_sample = tp / sp * sample / (tp / sp * sample + (1 - tp) / (1 - sp) * (1 - sample))

            else:
                assert np.max(sample) <= 1
                tp, sp = np.array(self.target_priors), np.array(self.source_priors)
                recalibrated_sample = tp / sp * sample / np.sum(tp / sp * sample, axis=-1, keepdims=True)

            self.outputs[idx][idx_] = Sample(recalibrated_sample, sample.affine)

    def _randomize(self):
        pass


class Shift(Transformer):
    """
    TODO: If axis in [1, 2, 3] modify affine
    """
    def __init__(self, max_shift_forward, max_shift_backward, axis=-1, n=1):
        super(Shift, self).__init__(n)
        self.max_shift_forward = max_shift_forward
        self.max_shift_backward = max_shift_backward
        self.axis = axis
        self.shift_state = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            shifted_array = sample.copy()
            shifted_array = np.moveaxis(shifted_array, self.axis, 0)
            if self.shift_state < 0:
                shifted_array[:self.shift_state, ...] = shifted_array[-self.shift_state:, ...]
                shifted_array[self.shift_state:, ...] = shifted_array[-1:, ...]

            elif self.shift_state > 0:
                shifted_array[self.shift_state:, ...] = shifted_array[:-self.shift_state, ...]
                shifted_array[:self.shift_state, ...] = shifted_array[:1, ...]

            self.outputs[idx][idx_] = Sample(np.moveaxis(shifted_array, 0, self.axis), sample.affine)

    def _randomize(self):
        shapes = [sample.shape[self.axis] for connection in self.connections for sample in connection[0] if sample is not None]
        assert all([shapes[0] == shape and shape > self.max_shift_forward and shape > self.max_shift_backward for shape in shapes])
        self.shift_state = int(np.round(random.uniform(-self.max_shift_forward, self.max_shift_backward)))


class Contrast(Transformer):
    def __init__(self, mean_log_scale=0, std_log_scale=0, axis=-1, n=1):
        super(Contrast, self).__init__(n)
        self.mean_log_scale = mean_log_scale
        self.std_log_scale = std_log_scale
        self.axis = axis
        self.contrast_state = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            contrasted_array = sample.copy()
            contrasted_array = np.moveaxis(contrasted_array, self.axis, 0)
            contrast = contrasted_array[1:, ...] - contrasted_array[:1, ...]
            contrasted_array[1:, ...] += (self.contrast_state - 1) * contrast
            self.outputs[idx][idx_] = Sample(np.moveaxis(contrasted_array, 0, self.axis), sample.affine)

    def _randomize(self):
        shapes = [sample.shape[self.axis] for connection in self.connections for sample in connection[0] if sample is not None]
        assert all([shapes[0] == shape and shape >= 2 for shape in shapes])
        self.contrast_state = np.random.lognormal(self.mean_log_scale, self.std_log_scale)


class Extrapolate(Transformer):
    def __init__(self, fixed_length, mode="back", axis=-1):
        super(Extrapolate, self).__init__(1)
        self.fixed_length = fixed_length
        self.mode = mode
        self.axis = axis

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            extrapolated_array = sample.copy()
            extrapolated_array = np.moveaxis(extrapolated_array, self.axis, 0)
            extrapolated_array = extrapolated_array[:self.fixed_length, ...]
            extrapolated_array = np.concatenate([extrapolated_array, *[extrapolated_array[-1:, ...]] * (self.fixed_length - len(extrapolated_array))])
            self.outputs[idx][idx_] = Sample(np.moveaxis(extrapolated_array, 0, self.axis), sample.affine)

    def _randomize(self):
        pass


class Subsample(Transformer):
    """
    TODO: If axis in [1, 2, 3] modify affine
    """
    def __init__(self, factor, mode="mean", axis=-1):
        super(Subsample, self).__init__(1)
        self.factor = factor
        assert mode in ["mean", "nearest"]
        self.mode = mode
        self.axis = axis

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            subsampled_array = np.moveaxis(sample, self.axis, 0)
            if self.mode == "nearest":
                subsampled_array = np.concatenate([subsampled_array] + [subsampled_array[-1:, ...]] * (self.factor - len(subsampled_array) % self.factor))
                subsampled_array = subsampled_array[slice(0, None, self.factor), ...]

            else:
                if len(subsampled_array) < self.factor:
                    subsampled_array = np.mean(subsampled_array, axis=0, keepdims=True)

                else:
                    subsampled_array = np.concatenate([subsampled_array] + [subsampled_array[-1:, ...]] * (self.factor - len(subsampled_array) % self.factor))
                    subsampled_array = transformations.downsample_array(subsampled_array, (self.factor, 1, 1, 1, 1))

            self.outputs[idx][idx_] = Sample(np.moveaxis(subsampled_array, 0, self.axis), sample.affine)

    def _randomize(self):
        pass


class Flip(Transformer):
    def __init__(self, flip_probabilities=(0.5, 0.5, 0.5), n=1):
        super(Flip, self).__init__(n)
        self.flip_probabilities = flip_probabilities
        self.flip_state = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            flipped_array = sample[:, ::self.flip_state[0], ::self.flip_state[1], ::self.flip_state[2], :]
            backward_affine = Sample.update_affine(reflection=self.flip_state) @ Sample.update_affine(translation=[1 - shape if state == -1 else 0 for state, shape in zip(self.flip_state, sample.shape[1:4])])
            flipped_affine = Sample.update_affine(sample.affine, transformation_matrix=backward_affine)
            self.outputs[idx][idx_] = Sample(flipped_array, flipped_affine)

    def _randomize(self):
        self.flip_state = [-1 if random.random() < flip_probability else 1 for flip_probability in self.flip_probabilities]


class GaussianNoise(Transformer):
    def __init__(self, mean=0, std=1, n=1):
        super(GaussianNoise, self).__init__(n)
        self.mean = mean
        self.std = std
        self.gaussian_noise = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            self.outputs[idx][idx_] = sample + np.random.normal(self.mean, self.std, sample.shape)

    def _randomize(self):
        pass


class AffineDeformation(Transformer):
    def __init__(self, reference_connection, voxel_size=(1, 1, 1), shear_window_width=(0, 0, 0), rotation_window_width=(0, 0, 0), translation_window_width=(0, 0, 0), scaling_window_width=(0, 0, 0), cval=0, order=1, n=1):
        super(AffineDeformation, self).__init__(n, extra_connections=reference_connection)
        self.reference_connection = reference_connection
        self.voxel_size = voxel_size
        self.shear_window_width = shear_window_width
        self.rotation_window_width = rotation_window_width
        self.translation_window_width = translation_window_width
        self.scaling_window_width = scaling_window_width
        self.cval = cval
        self.order = order
        self.backward_affine = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            transformed_sample = np.zeros_like(sample)
            for batch_i in range(len(sample)):
                for feature_i in range(sample.shape[-1]):
                    transformed_sample[batch_i, ..., feature_i] = transformations.affine_deformation(sample[batch_i, ..., feature_i], self.backward_affine, cval=self.cval, order=self.order)

            transformed_affine = Sample.update_affine(sample.affine, transformation_matrix=self.backward_affine)
            self.outputs[idx][idx_] = Sample(transformed_sample, transformed_affine)

    def _randomize(self):
        assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
        self.backward_affine = transformations.get_affine_matrix(
            I_shape=self.reference_connection[0].shape[1:4],
            voxel_size=self.voxel_size,
            shear=[random.uniform(-w, w) for w in self.shear_window_width],
            rotation=[random.uniform(-w, w) for w in self.rotation_window_width],
            translation=[random.uniform(-w, w) for w in self.translation_window_width],
            scaling=[1 + random.uniform(-w, w) for w in self.scaling_window_width],
        )


class ElasticDeformation(Transformer):
    def __init__(self, reference_connection, shift=(2, 2, 2), nsize=(30, 30, 30), npad=(5, 5, 5), std=(6, 6, 6), cval=0, order=1, n=1):
        super(ElasticDeformation, self).__init__(n, extra_connections=reference_connection)
        self.reference_connection = reference_connection
        self.shift = shift
        self.nsize = nsize
        self.npad = npad
        self.std = std
        self.cval = cval
        self.order = order
        self.deformation_field = None

    def _update(self, idx, connection):
        for idx_, sample in enumerate(connection[0]):
            transformed_sample = np.zeros_like(sample)
            for batch_i in range(len(sample)):
                for feature_i in range(sample.shape[-1]):
                    transformed_sample[batch_i, ..., feature_i] = transformations.elastic_deformation(sample[batch_i, ..., feature_i], self.deformation_field, cval=self.cval, order=self.order)

            self.outputs[idx][idx_] = Sample(transformed_sample, sample.affine)

    def _randomize(self):
        assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
        self.deformation_field = transformations.get_deformation_field(
            I_shape=self.reference_connection[0].shape[1:4],
            shift=self.shift,
            nsize=self.nsize,
            npad=self.npad,
            std=self.std
        )


class Crop(Transformer):
    def __init__(self, reference_connection, segment_size, subsample_factors=(1, 1, 1), default_value=0, n=1):
        super(Crop, self).__init__(n, extra_connections=reference_connection)
        self.segment_size = segment_size
        self.reference_connection = reference_connection
        self.subsample_factors = subsample_factors
        self.default_value = default_value
        self.coordinates = None

    def _update(self, idx, connection):
        backward_affine = Sample.update_affine(translation=self.coordinates[self.n_]) @ Sample.update_affine(scaling=self.subsample_factors[:3]) @ Sample.update_affine(translation=[-(segment_size // 2) for segment_size in self.segment_size[:3]])
        for idx_, sample in enumerate(connection[0]):
            transformed_sample = transformations.crop(sample, (len(sample),) + self.segment_size, (None,) + self.coordinates[self.n_] + (None,) * (len(self.segment_size) - len(self.coordinates[self.n_])), (1,) + self.subsample_factors, self.default_value)
            transformed_affine = Sample.update_affine(sample.affine, transformation_matrix=backward_affine)
            self.outputs[idx][idx_] = Sample(transformed_sample, transformed_affine)

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            self.coordinates = [tuple([shape // 2 for shape in self.connections[0][0][0].shape[1:4]])] * self.n


class RandomCrop(Crop):
    def __init__(self, reference_connection, segment_size, n, nonzero=False, subsample_factors=(1, 1, 1), default_value=0):
        super(RandomCrop, self).__init__(reference_connection, segment_size, subsample_factors, default_value, n)
        self.nonzero = nonzero

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            if self.nonzero:
                coordinates = list(zip(*np.nonzero(np.any(self.reference_connection[0] != 0, axis=(0, -1)))))
                self.coordinates = [tuple(random.choice(coordinates)) for _ in range(self.n)]

            else:
                self.coordinates = [tuple([random.choice(range(self.reference_connection[0].shape[i])) for i in range(1, 4)]) for _ in range(self.n)]


class GridCrop(Crop):
    def __init__(self, reference_connection, segment_size, n=None, grid_size=None, strides=None, nonzero=False, subsample_factors=(1, 1, 1), default_value=0):
        super(GridCrop, self).__init__(reference_connection, segment_size, subsample_factors, default_value, n)
        self.ncrops = n
        self.grid_size = segment_size if grid_size is None else grid_size
        self.strides = self.grid_size if strides is None else strides
        self.nonzero = nonzero

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            self.coordinates = []
            starts = [-(self.grid_size[i] // 2) for i in range(3)]
            stops = [self.reference_connection[0].shape[i + 1] - self.grid_size[i] // 2 - 1 for i in range(3)]
            for x in list(range(starts[0], stops[0], self.strides[0])) + [stops[0]]:
                for y in list(range(starts[1], stops[1], self.strides[1])) + [stops[1]]:
                    for z in list(range(starts[2], stops[2], self.strides[2])) + [stops[2]]:
                        if self.nonzero:
                            if np.any(self.reference_connection[0][:, max(0, x):x + self.grid_size[0], max(0, y):y + self.grid_size[1], max(0, z):z + self.grid_size[2], :]):
                                self.coordinates.append((x + self.grid_size[0] // 2, y + self.grid_size[1] // 2, z + self.grid_size[2] // 2))

                        else:
                            self.coordinates.append((x + self.grid_size[0] // 2, y + self.grid_size[1] // 2, z + self.grid_size[2] // 2))

            if self.ncrops is None:
                self.n = len(self.coordinates)

            else:
                self.coordinates = [random.choice(self.coordinates) for _ in range(self.n)]


class KerasModel(Transformer):
    def __init__(self, keras_model, output_affines=None):
        super(KerasModel, self).__init__(1, len(keras_model.output_names))
        self.keras_model = keras_model
        self.output_affines = output_affines if isinstance(output_affines, list) else [output_affines]
        assert len(self.output_affines) == self.output_len

    def _update(self, idx, connection):
        y = self.keras_model.predict(connection[0].get())
        y = y if isinstance(y, list) else [y]
        for idx_, (y_, output_affine) in enumerate(zip(y, self.output_affines)):
            if output_affine is None:
                output_affine = Sample.update_affine(translation=[-(out_shape // 2) + (in_shape // 2) for in_shape, out_shape in zip(connection[0][0].shape[1:4], y_.shape[1:4])])

            self.outputs[idx][idx_] = Sample(y_, Sample.update_affine(connection[0][0].affine, transformation_matrix=output_affine))

    def _randomize(self):
        pass


class Put(Transformer):
    def __init__(self, reference_connection, cashing=True, cval=0, order=1):
        super(Put, self).__init__(1, extra_connections=reference_connection)
        self.reference_connection = reference_connection
        self.cashing = cashing
        self.cval = cval
        self.order = order
        self.prev_references = [None] * len(reference_connection)
        self.output_array_counts = None

    def _update(self, idx, connection):
        for idx_, (reference, sample) in enumerate(zip(self.reference_connection.get(), connection[0].get())):
            for i in range(sample.shape[0]):
                transformed_array_counts = transformations.affine_deformation(np.ones_like(sample[i, ..., 0]), np.linalg.inv(sample.affine[i]) @ reference.affine[0], output_shape=reference.shape[1:4], cval=0, order=self.order)[None, ..., None]
                transformed_array = np.stack([transformations.affine_deformation(sample[i, ..., j], np.linalg.inv(sample.affine[i]) @ reference.affine[0], output_shape=reference.shape[1:4], cval=self.cval, order=self.order) for j in range(sample.shape[4])], axis=-1)[None, ...]
                if np.isnan(self.cval):
                    transformed_array = transformed_array[tuple(distance_transform_edt(np.isnan(transformed_array), return_distances=False, return_indices=True))]

                self.outputs[idx][idx_][...] = self.output_array_counts[idx][idx_] / (self.output_array_counts[idx][idx_] + transformed_array_counts) * self.outputs[idx][idx_] + transformed_array_counts / (self.output_array_counts[idx][idx_] + transformed_array_counts) * transformed_array
                self.output_array_counts[idx][idx_] += transformed_array_counts

    def _randomize(self):
        if self.output_array_counts is None:
            self.output_array_counts = [[None] * len(self.reference_connection) for _ in self.connections]

        for idx, connection in enumerate(self.connections):
            assert len(connection[0]) == len(self.reference_connection)
            for idx_, sample in enumerate(connection[0]):
                if not self.cashing or self.prev_references[idx_] is not self.reference_connection[idx_]:
                    assert self.reference_connection[idx_].shape[0] == 1
                    self.prev_references[idx_] = self.reference_connection[idx_]
                    self.outputs[idx][idx_] = Sample(np.zeros(self.reference_connection[idx_].shape[:4] + sample.shape[4:]), self.reference_connection[idx_].affine)
                    self.output_array_counts[idx][idx_] = np.full_like(self.outputs[idx][idx_], 1e-7)
