import os
import json
import random
import uuid
import transforms3d
import numpy as np
import scipy.ndimage
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.utilities import transformations


class Connection(object):
    def __init__(self, transformer, idx):
        self.transformer = transformer
        self.idx = idx
        self.shapes = self.get_shapes()

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

    def get_shapes(self):
        return [sample_shape for sample_shape in self.transformer.output_shapes[self.idx]]

    @staticmethod
    def trace(connections, only_active=False, clear_active_indices=False, set_active_indices=False, set_names=False, reset_transformers=False):
        traced_transformers = []
        traced_connections = []
        connections = [connection for connection in connections]
        while len(connections) > 0:
            connection = connections.pop(0)
            if connection not in traced_connections:
                traced_connections.append(connection)
                if connection.transformer not in traced_transformers:
                    if clear_active_indices:
                        connection.transformer.active_indices = []

                    if reset_transformers:
                        connection.transformer.reset()

                    if set_names:
                        if connection.transformer.name is None:
                            connection.transformer.name = "{}_{}".format(connection.transformer.__class__.__name__, len([traced_transformer for traced_transformer in traced_transformers if traced_transformer.__class__.__name__ == connection.transformer.__class__.__name__]))

                        assert connection.transformer.name not in [traced_transformer.name for traced_transformer in traced_transformers], "In a transformer network you cannot use the same name for more than one Transformer."

                    traced_transformers.append(connection.transformer)

                if set_active_indices and connection.idx not in connection.transformer.active_indices:
                    connection.transformer.active_indices.append(connection.idx)

                for connection__ in connection.transformer.extra_connections:
                    if connection__ not in traced_connections and connection__ not in connections:
                        connections.append(connection__)

                for idx, connection_ in enumerate(connection.transformer.connections):
                    for connection__ in connection_:
                        if connection__ not in traced_connections and connection__ not in connections and (not only_active or idx == connection.idx):
                            connections.append(connection__)

        return traced_transformers, traced_connections


class Transformer(object):
    def __init__(self, n=1, extra_connections=None, name=None):
        self.n = n
        self.extra_connections = [] if extra_connections is None else (extra_connections if isinstance(extra_connections, list) else [extra_connections])
        self.name = name
        self.n_ = 0
        self.connections = []
        self.outputs = []
        self.output_shapes = []
        self.active_indices = []
        self.generator = None
        self.sample_id = None

    def __call__(self, *connections):
        output_connections = []
        if len(connections) == 0:
            for idx in range(len(self.outputs)):
                output_connections.append(Connection(self, idx))

        else:
            for connections_ in connections:
                connections_ = connections_ if isinstance(connections_, list) else [connections_]
                self.connections.append(connections_)
                self.output_shapes.append(self._calculate_output_shape_at_idx(len(self.connections) - 1))
                self.outputs.append([None] * len(self.output_shapes[-1]))
                self.active_indices.append(len(self.connections) - 1)
                output_connections.append(Connection(self, len(self.connections) - 1))

        return output_connections if len(output_connections) > 1 else (output_connections[0] if len(output_connections) == 1 else None)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, item):
        return self.outputs[item]

    def __iter__(self):
        return iter(self.outputs)

    def get_output_shapes(self):
        return self.output_shapes

    def eval(self, sample_id=None):
        if sample_id is None:
            sample_id = uuid.uuid4()

        if sample_id != self.sample_id:
            self.sample_id = sample_id
            if self.generator is None or self.n_ == self.n:
                if not isinstance(self, _Input):
                    for connections in self.connections + [self.extra_connections]:
                        for connection in connections:
                            if len(connection.transformer.active_indices) > 0:
                                connection.transformer.eval(self.sample_id)

                elif self.n_ == self.n:
                    raise StopIteration

                self.generator = self._update()

            next(self.generator)

        return self.outputs

    def reset(self):
        self.n_ = 0
        self.sample_id = None
        self.generator = None

    def _update(self):
        self.n_ = 0
        while self.n is None or self.n_ < self.n:
            self._randomize()
            for idx in self.active_indices:
                self._update_idx(idx)

            self.n_ += 1
            yield self.outputs

    def _update_idx(self, idx):
        raise NotImplementedError

    def _calculate_output_shape_at_idx(self, idx):
        raise NotImplementedError

    def _randomize(self):
        raise NotImplementedError


class _Input(Transformer):
    def __init__(self, output_shapes, **kwargs):
        super(_Input, self).__init__(**kwargs)
        for i, output_shapes_ in enumerate(output_shapes):
            assert isinstance(output_shapes_, list) and all([isinstance(output_shape, tuple) and len(output_shape) == 5 for output_shape in output_shapes_]), "The given output_shapes fot the _Input transformer are not in the correct format."
            self.connections.append([])
            self.outputs.append([None] * len(output_shapes_))
            self.output_shapes.append(output_shapes_)
            self.active_indices.append(i)

    def load(self, identifier=None):
        raise NotImplementedError

    def _update_idx(self, idx):
        pass

    def _calculate_output_shape_at_idx(self, idx):
        return self.output_shapes[idx]

    def _randomize(self):
        pass


class _MircInput(_Input):
    def __init__(self, modality_ids, output_shapes=None, **kwargs):
        self.modality_ids = modality_ids if isinstance(modality_ids, list) else [modality_ids]
        if output_shapes is None:
            output_shapes = [(None, ) * 5, ] * len(self.modality_ids)

        assert len(output_shapes) == len(self.modality_ids)
        super(_MircInput, self).__init__([output_shapes], **kwargs)

    def load(self, identifier=None):
        assert identifier is not None
        for idx_, modality_id in enumerate(self.modality_ids):
            self.outputs[0][idx_] = identifier.mirc[identifier.dataset_id][identifier.case_id][identifier.record_id][modality_id].load()


class MircInput(_MircInput):
    def __new__(cls, modality_ids, output_shapes=None, **kwargs):
        return _MircInput(modality_ids, output_shapes, **kwargs)()


class _SampleInput(_Input):
    def __init__(self, samples=None, output_shapes=None, **kwargs):
        if samples is not None:
            samples = samples if isinstance(samples, list) else [samples]
            output_shapes = [sample.shape for sample in samples] if output_shapes is None else output_shapes
            assert all([np.array_equal(sample.shape, output_shape) for sample, output_shape in zip(samples, output_shapes)])
            super(_SampleInput, self).__init__([output_shapes], **kwargs)
            for idx_, sample in enumerate(samples):
                self.outputs[0][idx_] = sample

        else:
            assert output_shapes is not None, "When the samples are not given as constructor arguments, the output_shapes must be given (can be None, but this is necessary for the length of samples)."
            super(_SampleInput, self).__init__([output_shapes], **kwargs)

    def load(self, identifier=None):
        if identifier is not None:
            for idx_, sample in enumerate(identifier.sample if isinstance(identifier.sample, list) else [identifier.sample]):
                self.outputs[0][idx_] = sample


class SampleInput(_SampleInput):
    def __new__(cls, samples=None, output_shapes=None, **kwargs):
        return _SampleInput(samples, output_shapes, **kwargs)()


class Buffer(Transformer):
    def __init__(self, buffer_size=None, axis=0, drop_remainder=False, **kwargs):
        assert "n" not in kwargs, "A Buffer does not accept n. It just buffers so it cannot create n samples from 1 input."
        super(Buffer, self).__init__(n=1, **kwargs)
        self.buffer_size = buffer_size
        assert axis in [0, 4, -1]
        self.axis = axis if axis != -1 else 4
        self.drop_remainder = drop_remainder
        self.buffered_outputs = None

    def _update_idx(self, idx):
        for idx_ in range(len(self.outputs[idx])):
            self.outputs[idx][idx_] = Sample(np.concatenate([output[idx_] for output in self.buffered_outputs[idx]], axis=self.axis), self.buffered_outputs[idx][0][idx_].affine if self.axis != 0 else np.concatenate([output[idx_].affine for output in self.buffered_outputs[idx]]))

        self.buffered_outputs[idx] = None

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        assert all([output_shape_ is not None for output_shape in self.connections[idx][0].shapes for axis_i, output_shape_ in enumerate(output_shape) if axis_i != self.axis]), "A buffer can only used on a connection with fully specified shapes (except for the concatenation axis)."
        return [tuple([output_shape_ if axis_i != self.axis else (self.buffer_size * output_shape_ if self.buffer_size is not None and self.drop_remainder and output_shape_ is not None else None) for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        self.buffered_outputs = [[[sample for sample in self.connections[idx][0]]] if idx in self.active_indices else None for idx in range(len(self.outputs))]
        while self.buffer_size is None or len(self.buffered_outputs[0]) < self.buffer_size:
            try:
                sample_id = uuid.uuid4()
                for idx in self.active_indices:
                    self.buffered_outputs[idx].append([sample for sample in self.connections[idx][0].eval(sample_id)])

            except (StopIteration, RuntimeError):
                break

        if self.buffer_size is not None and self.drop_remainder and len(self.buffered_outputs[0]) < self.buffer_size:
            raise StopIteration

        else:
            for transformer in Connection.trace([connection for connections in self.connections for connection in connections], only_active=True)[0]:
                transformer.sample_id = self.sample_id


class Group(Transformer):
    def __init__(self, **kwargs):
        super(Group, self).__init__(**kwargs)

    def _update_idx(self, idx):
        idx_ = 0
        for connection in self.connections[idx]:
            for sample in connection:
                self.outputs[idx][idx_] = sample
                idx_ += 1

    def _calculate_output_shape_at_idx(self, idx):
        return [shape for connection in self.connections[idx] for shape in connection.shapes]

    def _randomize(self):
        pass


class Split(Transformer):
    def __init__(self, indices=(0,), **kwargs):
        self.indices = indices if isinstance(indices, tuple) else (indices,)
        super(Split, self).__init__(**kwargs)

    def _update_idx(self, idx):
        for idx_, j in enumerate(self.indices):
            self.outputs[idx][idx_] = self.connections[idx][0][j]

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [self.connections[idx][0].shapes[i] for i in self.indices]

    def _randomize(self):
        pass


class Concat(Transformer):
    def __init__(self, axis=-1, **kwargs):
        super(Concat, self).__init__(**kwargs)
        assert axis in [0, 4, -1]
        self.axis = axis if axis != -1 else 4

    def _update_idx(self, idx):
        for idx_ in range(len(self.connections[idx][0])):
            self.outputs[idx][idx_] = Sample(np.concatenate([connection[idx_] for connection in self.connections[idx]], axis=self.axis), self.connections[idx][0][idx_].affine if self.axis != 0 else np.concatenate([connection[idx_].affine for connection in self.connections[idx]]))

    def _calculate_output_shape_at_idx(self, idx):
        assert all(len(self.connections[idx][0]) == len(connection) for connection in self.connections[idx]), "All input connections must have the same output length."
        output_shapes = [list(output_shape) for output_shape in self.connections[idx][0].shapes]
        for output_shape_i, output_shape in enumerate(output_shapes):
            for connection in self.connections[idx][1:]:
                for axis_i, (output_shape_, output_shape__) in enumerate(zip(output_shape, connection.shapes[output_shape_i])):
                    if axis_i == self.axis:
                        if output_shape_ is None or output_shape__ is None:
                            output_shape[axis_i] = None

                        else:
                            output_shape[axis_i] = output_shape_ + output_shape__

                    else:
                        assert output_shape_ is not None and output_shape__ is not None and output_shape_ == output_shape__, "The shapes of the shared axes should be identical and different from None."

        return [tuple(output_shape) for output_shape in output_shapes]

    def _randomize(self):
        pass


class Mean(Transformer):
    def __init__(self, axis=-1, **kwargs):
        super(Mean, self).__init__(**kwargs)
        assert axis in [4, -1]
        self.axis = axis if axis != -1 else 4

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = sample.mean(axis=self.axis, keepdims=True)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [tuple([output_shape_ if axis_i != self.axis else 1 for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass


class Resample(Transformer):
    def __init__(self, voxel_sizes, order=1, prefilter=True, **kwargs):
        super(Resample, self).__init__(**kwargs)
        assert len(voxel_sizes) == 3, "The Resample Transformer is a spatial resampler."
        self.voxel_sizes = voxel_sizes
        assert all([order_ in [0, 1, 2, 3] for order_ in (order if isinstance(order, (tuple, list)) else [order])]), "Scipy's zoom is used internally. Please refer to that documentation."
        self.order = order
        self.prefilter = prefilter

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            affine = sample.affine.copy()
            input_zooms = np.linalg.norm(affine[:, :3, :3], 2, axis=1)
            assert np.all(input_zooms == input_zooms[:1, :])
            zoom_factors = [1 if output_zoom is None else input_zoom / output_zoom for input_zoom, output_zoom in zip(input_zooms[0], self.voxel_sizes)]
            if self.prefilter[idx] if isinstance(self.prefilter, (tuple, list)) else self.prefilter:
                sample = scipy.ndimage.gaussian_filter(sample, [0] + [np.sqrt(((1 / zoom_factor) ** 2 - 1) / 12) if zoom_factor < 1 else 0 for zoom_factor in zoom_factors] + [0], mode="nearest")

            sample = scipy.ndimage.zoom(
                sample,
                [1] + zoom_factors + [1],
                order=self.order[idx] if isinstance(self.order, (tuple, list)) else self.order,
                mode="nearest"
            )
            affine[:, :3, :3] = affine[:, :3, :3] / zoom_factors
            self.outputs[idx][idx_] = Sample(sample, affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [(input_shape[0], *[input_shape[i + 1] if self.voxel_sizes[i] is None else None for i in range(3)], input_shape[4]) for input_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass


class Threshold(Transformer):
    def __init__(self, lower_threshold=0, upper_threshold=np.inf, **kwargs):
        super(Threshold, self).__init__(**kwargs)
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            affine = sample.affine
            array = (sample > self.lower_threshold) * (sample < self.upper_threshold)
            self.outputs[idx][idx_] = Sample(array, affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Multiply(Transformer):
    def __init__(self, **kwargs):
        super(Multiply, self).__init__(**kwargs)

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            array = sample
            for connection in self.connections[idx][1:]:
                array = array * connection[idx_]

            self.outputs[idx][idx_] = Sample(array, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert all(len(self.connections[idx][0]) == len(connection) for connection in self.connections[idx]), "All input connections must have the same output length."
        output_shapes = self.connections[idx][0].shapes
        for output_shape_i, output_shape in enumerate(output_shapes):
            for connection in self.connections[idx][1:]:
                for axis_i, (output_shape_, output_shape__) in enumerate(zip(output_shape, connection.shapes[output_shape_i])):
                    assert output_shape_ is not None or output_shape__ is not None or output_shape_ == output_shape__, "The shapes of the shared axes should be identical and different from None."

        return output_shapes

    def _randomize(self):
        pass


class Sum(Transformer):
    def __init__(self, **kwargs):
        super(Sum, self).__init__(**kwargs)

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            array = sample
            for connection in self.connections[idx][1:]:
                array = array + connection[idx_]

            self.outputs[idx][idx_] = Sample(array, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert all(len(self.connections[idx][0]) == len(connection) for connection in self.connections[idx]), "All input connections must have the same output length."
        output_shapes = self.connections[idx][0].shapes
        for output_shape_i, output_shape in enumerate(output_shapes):
            for connection in self.connections[idx][1:]:
                for axis_i, (output_shape_, output_shape__) in enumerate(zip(output_shape, connection.shapes[output_shape_i])):
                    assert output_shape_ is not None or output_shape__ is not None or output_shape_ == output_shape__, "The shapes of the shared axes should be identical and different from None."

        return output_shapes

    def _randomize(self):
        pass


class FillNan(Transformer):
    def __init__(self, fill_value, **kwargs):
        super(FillNan, self).__init__(**kwargs)
        self.fill_value = fill_value

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample(np.nan_to_num(sample, nan=self.fill_value), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Clip(Transformer):
    def __init__(self, lower_clip=-np.inf, higher_clip=np.inf, **kwargs):
        super(Clip, self).__init__(**kwargs)
        self.lower_clip = lower_clip
        self.higher_clip = higher_clip

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample(np.clip(sample, self.lower_clip, self.higher_clip), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class LogClassCounts(Transformer):
    def __init__(self, logs_dir, class_values_dict, one_hot=False, **kwargs):
        super(LogClassCounts, self).__init__(**kwargs)
        self.logs_dir = logs_dir
        self.class_values_dict = class_values_dict
        self.one_hot = one_hot

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = sample
            if os.path.isdir(self.logs_dir):
                file_path = os.path.join(self.logs_dir, f"class_counts_i{idx}_s{idx_}.txt")
                if os.path.isfile(file_path):
                    while True:
                        try:
                            with open(file_path, "r") as f:
                                class_counts_dict = json.load(f)

                            class_counts_dict["voxel_count"] += float(np.prod(sample.shape[:4]))
                            for class_ in self.class_values_dict:
                                class_counts_dict[str(class_)] += float(np.sum((sample[..., class_] if self.one_hot else sample) == self.class_values_dict[class_]).item())

                            break

                        except:
                            pass

                else:
                    class_counts_dict["voxel_count"] = float(np.prod(sample.shape[:4]))
                    class_counts_dict = {str(class_): float(np.sum((sample[..., class_] if self.one_hot else sample) == self.class_values_dict[class_]).item()) for class_ in self.class_values_dict}

                while True:
                    try:
                        with open(file_path, "w") as f:
                            json.dump(class_counts_dict, f, indent=2)

                        break

                    except:
                        pass

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Round(Transformer):
    def __init__(self, decimals=0, **kwargs):
        super(Round, self).__init__(**kwargs)
        self.decimals = decimals

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample(np.round(sample, self.decimals), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Normalize(Transformer):
    def __init__(self, shift, scale, **kwargs):
        super(Normalize, self).__init__(**kwargs)
        self.shift = shift
        self.scale = scale

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample((sample + self.shift) * self.scale, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class WindowNormalize(Transformer):
    def __init__(self, lower_window=0, higher_window=1, **kwargs):
        super(WindowNormalize, self).__init__(**kwargs)
        self.lower_window = lower_window
        self.higher_window = higher_window

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            lower_window = np.min(sample) if self.lower_window is None else self.lower_window
            higher_window = np.max(sample) if self.higher_window is None else self.higher_window
            self.outputs[idx][idx_] = Sample(sample * (higher_window - lower_window) / (np.max(sample) - np.min(sample)) + (lower_window - np.min(sample)), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class IntensityTransform(Transformer):
    def __init__(self, mean_shift=0, std_shift=0, mean_scale=1, std_scale=0, **kwargs):
        super(IntensityTransform, self).__init__(**kwargs)
        self.mean_shift = mean_shift
        self.std_shift = std_shift
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        self.shift = None
        self.scale = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample((sample + self.shift) * self.scale, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        self.shift = np.random.normal(self.mean_shift, self.std_shift)
        self.scale = np.random.normal(self.mean_scale, self.std_scale)


class Remove(Transformer):
    def __init__(self, remove_probability, fill_value, axis=-1, **kwargs):
        super(Remove, self).__init__(**kwargs)
        self.remove_probability = remove_probability
        self.fill_value = fill_value
        self.axis = axis if axis != -1 else 4
        self.remove_state = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            removed_array = sample.copy()
            removed_array = np.moveaxis(removed_array, self.axis, 0)
            for i, remove_state in enumerate(self.remove_sate):
                if remove_state:
                    removed_array[i, ...] = self.fill_value

            self.outputs[idx][idx_] = Sample(np.moveaxis(removed_array, 0, self.axis), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        shapes = [sample.shape[self.axis] for connection in self.connections for sample in connection[0] if sample is not None]
        assert all([shapes[0] == shape for shape in shapes])
        self.remove_sate = [random.random() < self.remove_probability for _ in range(shapes[0])]


class RemoveBorder(Transformer):
    def __init__(self, remove_widths, fill_value=0, limit_to_fraction_of_input_shape=0.5, **kwargs):
        super(RemoveBorder, self).__init__(**kwargs)
        if not isinstance(remove_widths, (tuple, list)):
            remove_widths = [(remove_widths,) * 2] * 3

        assert len(remove_widths) == 3, "A remove width must be specified for each spatial axis."
        self.remove_widths = [remove_widths_ if isinstance(remove_widths_, (tuple, list)) else (remove_widths_,) * 2 for remove_widths_ in remove_widths]
        self.fill_value = fill_value
        self.limit_to_fraction_of_input_shape = limit_to_fraction_of_input_shape
        self.remove_state = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            removed_array = sample.copy()
            fill_value = self.fill_value[idx] if isinstance(self.fill_value, (tuple, list)) else self.fill_value
            removed_array[:, :self.remove_state[0][0], ...] = fill_value
            removed_array[:, -self.remove_state[0][1]:, ...] = fill_value
            removed_array[:, :, :self.remove_state[1][0], ...] = fill_value
            removed_array[:, :, -self.remove_state[1][1]:, ...] = fill_value
            removed_array[..., :self.remove_state[2][0], :] = fill_value
            removed_array[..., -self.remove_state[2][1]:, :] = fill_value
            self.outputs[idx][idx_] = removed_array

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        shapes = [sample.shape for connection in self.connections for sample in connection[0]]
        assert all([shapes[0][i] == shape[i] for shape in shapes for i in range(1, 4)])
        self.remove_state = [(
            random.randint(0, min(remove_widths_[0], int(self.limit_to_fraction_of_input_shape * shapes[0][i + 1]))),
            random.randint(0, min(remove_widths_[1], int(self.limit_to_fraction_of_input_shape * shapes[0][i + 1])))
        ) for i, remove_widths_ in enumerate(self.remove_widths)]


class ClassWeights(Transformer):
    def __init__(self, class_weights_dict, one_hot=False, **kwargs):
        super(ClassWeights, self).__init__(**kwargs)
        self.class_weights_dict = class_weights_dict
        self.one_hot = one_hot
        k = np.array(list(class_weights_dict.keys()))
        v = np.array(list(class_weights_dict.values()))
        self.mapping_array = np.zeros(np.max(k) + 1, dtype=np.float32)
        self.mapping_array[k] = v

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            if self.one_hot:
                assert sample.shape[-1] > 1
                sample = Sample(np.argmax(sample, axis=-1)[..., None], sample.affine)

            else:
                sample = np.round(sample).astype(int)

            self.outputs[idx][idx_] = Sample(self.mapping_array[sample], sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Recalibrate(Transformer):
    def __init__(self, source_priors, target_priors, **kwargs):
        super(Recalibrate, self).__init__(**kwargs)
        self.source_priors = source_priors if isinstance(source_priors, list) else [source_priors]
        self.target_priors = target_priors if isinstance(target_priors, list) else [target_priors]
        assert len(self.source_priors) == len(self.target_priors)

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            assert sample.shape[-1] == len(self.source_priors)
            if sample.shape[-1] == 1:
                tp, sp = self.target_priors[0], self.source_priors[0]
                recalibrated_sample = tp / sp * sample / (tp / sp * sample + (1 - tp) / (1 - sp) * (1 - sample))

            else:
                assert np.max(sample) <= 1
                tp, sp = np.array(self.target_priors), np.array(self.source_priors)
                recalibrated_sample = tp / sp * sample / np.sum(tp / sp * sample, axis=-1, keepdims=True)

            self.outputs[idx][idx_] = Sample(recalibrated_sample, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Shift(Transformer):
    def __init__(self, max_shift_forward, max_shift_backward, axis=-1, order=0, mode="nearest", **kwargs):
        super(Shift, self).__init__(**kwargs)
        self.max_shift_forward = max_shift_forward
        self.max_shift_backward = max_shift_backward
        assert axis in [-1, 4], "Currently Shift only supports shifting in the final/feature dimension."
        self.axis = axis if axis != -1 else 4
        self.order = order
        self.mode = mode
        self.shift_state = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            shifted_array = scipy.ndimage.shift(sample, shift=[self.shift_state if i == self.axis else 0 for i in range(5)], order=self.order, mode=self.mode)
            self.outputs[idx][idx_] = Sample(shifted_array, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        shapes = [sample.shape[self.axis] for connection in self.connections for sample in connection[0] if sample is not None]
        assert all([shapes[0] == shape and shape > self.max_shift_forward and shape > self.max_shift_backward for shape in shapes])
        self.shift_state = random.randint(-self.max_shift_forward, self.max_shift_backward)


class Contrast(Transformer):
    def __init__(self, mean_log_scale=0, std_log_scale=0, axis=-1, **kwargs):
        super(Contrast, self).__init__(**kwargs)
        self.mean_log_scale = mean_log_scale
        self.std_log_scale = std_log_scale
        self.axis = axis if axis != -1 else 4
        self.contrast_state = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            contrasted_array = sample.copy()
            contrasted_array = np.moveaxis(contrasted_array, self.axis, 0)
            contrast = contrasted_array[1:, ...] - contrasted_array[:1, ...]
            contrasted_array[1:, ...] += (self.contrast_state - 1) * contrast
            self.outputs[idx][idx_] = Sample(np.moveaxis(contrasted_array, 0, self.axis), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        shapes = [sample.shape[self.axis] for connection in self.connections for sample in connection[0] if sample is not None]
        assert all([shapes[0] == shape and shape >= 2 for shape in shapes])
        self.contrast_state = np.random.lognormal(self.mean_log_scale, self.std_log_scale)


class Extrapolate(Transformer):
    def __init__(self, fixed_length, mode="back", axis=-1, **kwargs):
        super(Extrapolate, self).__init__(**kwargs)
        self.fixed_length = fixed_length
        self.mode = mode
        self.axis = axis if axis != -1 else 4

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            extrapolated_array = sample.copy()
            extrapolated_array = np.moveaxis(extrapolated_array, self.axis, 0)
            extrapolated_array = extrapolated_array[:self.fixed_length, ...]
            extrapolated_array = np.concatenate([extrapolated_array, *[extrapolated_array[-1:, ...]] * (self.fixed_length - len(extrapolated_array))])
            self.outputs[idx][idx_] = Sample(np.moveaxis(extrapolated_array, 0, self.axis), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [tuple([output_shape_ if axis_i != self.axis else self.fixed_length for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass


class Subsample(Transformer):
    """
    TODO: If axis in [1, 2, 3] modify affine
    """
    def __init__(self, factor, mode="mean", axis=-1, **kwargs):
        super(Subsample, self).__init__(**kwargs)
        self.factor = factor
        assert mode in ["mean", "nearest"]
        self.mode = mode
        self.axis = axis if axis != -1 else 4

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            subsampled_array = np.moveaxis(sample, self.axis, 0)
            if self.mode == "nearest":
                subsampled_array = subsampled_array[slice(0, None, self.factor), ...]

            elif len(subsampled_array) < self.factor:
                subsampled_array = np.mean(subsampled_array, axis=0, keepdims=True)

            else:
                subsampled_array = np.pad(subsampled_array, ((0, int(self.factor - len(subsampled_array) % self.factor if len(subsampled_array) % self.factor else 0)), (0, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
                subsampled_array = transformations.downsample_array(subsampled_array, (self.factor, 1, 1, 1, 1))

            self.outputs[idx][idx_] = Sample(np.moveaxis(subsampled_array, 0, self.axis), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [tuple([output_shape_ if axis_i != self.axis else int(np.ceil(output_shape_ / self.factor)) for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass


class Flip(Transformer):
    def __init__(self, flip_probabilities=(0.5, 0.5, 0.5), **kwargs):
        super(Flip, self).__init__(**kwargs)
        self.flip_probabilities = flip_probabilities
        self.flip_state = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            flipped_array = sample[:, ::self.flip_state[0], ::self.flip_state[1], ::self.flip_state[2], :]
            backward_affine = Sample.update_affine(reflection=self.flip_state) @ Sample.update_affine(translation=[1 - shape if state == -1 else 0 for state, shape in zip(self.flip_state, sample.shape[1:4])])
            flipped_affine = Sample.update_affine(sample.affine, transformation_matrix=backward_affine)
            self.outputs[idx][idx_] = Sample(flipped_array, flipped_affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        self.flip_state = [-1 if random.random() < flip_probability else 1 for flip_probability in self.flip_probabilities]


class GaussianNoise(Transformer):
    def __init__(self, mean=0, std=1, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = sample + np.random.normal(self.mean, self.std, sample.shape)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Filter(Transformer):
    def __init__(self, filter_size, method="uniform", mode="nearest", cval=0, **kwargs):
        super(Filter, self).__init__(**kwargs)
        self.filter_size = filter_size if isinstance(filter_size, (tuple, list)) else [filter_size]
        assert method in ["uniform", "gaussian"]
        self.method = method
        self.mode = mode
        self.cval = cval

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            if self.method == "uniform":
                filtered_array = scipy.ndimage.uniform_filter(sample, self.filter_size, mode=self.mode, cval=self.cval)

            else:
                filtered_array = scipy.ndimage.gaussian_filter(sample, [s_f if s_f > 1 else 0 for s_f in self.filter_size], mode=self.mode, cval=self.cval)

            self.outputs[idx][idx_] = Sample(filtered_array, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class Resize(Transformer):
    def __init__(self, output_shape, order=0, **kwargs):
        super(Resize, self).__init__(**kwargs)
        assert isinstance(output_shape, tuple) and len(output_shape) == 3, "Resize needs a shape for every spatial dimension."
        self.output_shape = output_shape
        self.order = order

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            zoom_factors = np.array([1, *[self.output_shape[i] / sample.shape[i + 1] for i in range(3)], 1])
            self.outputs[idx][idx_] = Sample(scipy.ndimage.zoom(sample, zoom_factors, order=self.order), sample.affine / zoom_factors[None, 1:])

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [(shape[0], *self.output_shape, shape[4]) for shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass


class AffineDeformation(Transformer):
    def __init__(self, reference_connection, voxel_size=(1, 1, 1), shear_window_width=(0, 0, 0), rotation_window_width=(0, 0, 0), translation_window_width=(0, 0, 0), scaling_window_width=(0, 0, 0), cval=0, order=1, width_as_std=False, transform_probability=1, **kwargs):
        super(AffineDeformation, self).__init__(extra_connections=reference_connection, **kwargs)
        self.reference_connection = reference_connection
        self.voxel_size = voxel_size
        self.shear_window_width = shear_window_width
        self.rotation_window_width = rotation_window_width
        self.translation_window_width = translation_window_width
        self.scaling_window_width = scaling_window_width
        self.cval = cval
        self.order = order
        self.width_as_std = width_as_std if isinstance(width_as_std, (tuple, list)) else (width_as_std,) * 4
        assert len(self.width_as_std) == 4, "When specifying width_as_std as a tuple/list it must be of length 4 (shear, rotation, translation, scaling)"
        self.transform_probability = transform_probability
        self.backward_affine = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            if self.backward_affine is not None:
                transformed_sample = np.zeros_like(sample)
                for batch_i in range(len(sample)):
                    for feature_i in range(sample.shape[-1]):
                        transformed_sample[batch_i, ..., feature_i] = transformations.affine_deformation(
                            sample[batch_i, ..., feature_i],
                            self.backward_affine,
                            cval=self.cval[idx] if isinstance(self.cval, (tuple, list)) else self.cval,
                            order=self.order[idx] if isinstance(self.order, (tuple, list)) else self.order
                        )

                transformed_affine = Sample.update_affine(sample.affine, transformation_matrix=self.backward_affine)
                self.outputs[idx][idx_] = Sample(transformed_sample, transformed_affine)

            else:
                self.outputs[idx][idx_] = sample.copy()

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
        if random.random() < self.transform_probability:
            self.backward_affine = transformations.get_affine_matrix(
                I_shape=self.reference_connection[0].shape[1:4],
                voxel_size=self.voxel_size,
                shear=[np.random.normal(0, w) if self.width_as_std[0] else random.uniform(-w, w) for w in self.shear_window_width],
                rotation=[np.random.normal(0, w) if self.width_as_std[1] else random.uniform(-w, w) for w in self.rotation_window_width],
                translation=[np.random.normal(0, w) if self.width_as_std[2] else random.uniform(-w, w) for w in self.translation_window_width],
                scaling=[1 + (np.random.normal(0, w) if self.width_as_std[3] else random.uniform(-w, w)) for w in self.scaling_window_width],
            )

        else:
            self.backward_affine = None


class ElasticDeformation(Transformer):
    def __init__(self, reference_connection, shift=(2, 2, 2), nsize=(30, 30, 30), npad=(5, 5, 5), std=(6, 6, 6), cval=0, order=1, **kwargs):
        super(ElasticDeformation, self).__init__(extra_connections=reference_connection, **kwargs)
        self.reference_connection = reference_connection
        self.shift = shift
        self.nsize = nsize
        self.npad = npad
        self.std = std
        self.cval = cval
        self.order = order
        self.deformation_field = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            transformed_sample = np.zeros_like(sample)
            for batch_i in range(len(sample)):
                for feature_i in range(sample.shape[-1]):
                    transformed_sample[batch_i, ..., feature_i] = transformations.elastic_deformation(
                        sample[batch_i, ..., feature_i],
                        self.deformation_field,
                        cval=self.cval[idx] if isinstance(self.cval, (tuple, list)) else self.cval,
                        order=self.order[idx] if isinstance(self.order, (tuple, list)) else self.order
                    )

            self.outputs[idx][idx_] = Sample(transformed_sample, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

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
    def __init__(self, reference_connection, segment_size, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, **kwargs):
        super(Crop, self).__init__(extra_connections=reference_connection, **kwargs)
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
            transformed_sample = transformations.crop(
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


class RandomCrop(Crop):
    def __init__(self, reference_connection, segment_size, nonzero=False, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, **kwargs):
        super(RandomCrop, self).__init__(reference_connection, segment_size, subsample_factors, default_value, prefilter, **kwargs)
        self.nonzero = nonzero

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            if self.nonzero:
                self.coordinates = list(zip(*np.nonzero(np.any(self.reference_connection[0] != 0, axis=(0, -1)))))
                if len(self.coordinates) > 0 and self.ncrops is not None:
                    self.coordinates = [random.choice(self.coordinates) for _ in range(self.ncrops)]

            else:
                self.coordinates = [tuple([random.choice(range(self.reference_connection[0].shape[i])) for i in range(1, 4)]) for _ in range(np.prod(self.reference_connection[0].shape[1:4]) if self.ncrops is None else self.ncrops)]

            if len(self.coordinates) > 0:
                self.n = len(self.coordinates)

            else:
                raise StopIteration


class GridCrop(Crop):
    def __init__(self, reference_connection, segment_size, n=None, grid_size=None, strides=None, nonzero=False, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, **kwargs):
        super(GridCrop, self).__init__(reference_connection, segment_size, subsample_factors, default_value, prefilter, n=n, **kwargs)
        self.grid_size = segment_size if grid_size is None else grid_size
        self.strides = self.grid_size if strides is None else strides
        self.nonzero = nonzero

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            self.coordinates = []
            for x in range(0, self.reference_connection[0].shape[1] + self.strides[0] + 1, self.strides[0]):
                for y in range(0, self.reference_connection[0].shape[2] + self.strides[1] + 1, self.strides[1]):
                    for z in range(0, self.reference_connection[0].shape[3] + self.strides[2] + 1, self.strides[2]):
                        if self.nonzero:
                            if np.any(self.reference_connection[0][:, max(0, x):x + self.grid_size[0], max(0, y):y + self.grid_size[1], max(0, z):z + self.grid_size[2], :]):
                                self.coordinates.append((x + self.grid_size[0] // 2, y + self.grid_size[1] // 2, z + self.grid_size[2] // 2))

                        else:
                            self.coordinates.append((x + self.grid_size[0] // 2, y + self.grid_size[1] // 2, z + self.grid_size[2] // 2))

                        if z + self.grid_size[2] >= self.reference_connection[0].shape[3]:
                            break

                    if y + self.grid_size[1] >= self.reference_connection[0].shape[2]:
                        break

                if x + self.grid_size[0] >= self.reference_connection[0].shape[1]:
                    break

            if len(self.coordinates) > 0:
                if self.ncrops is not None:
                    self.coordinates = [random.choice(self.coordinates) for _ in range(self.ncrops)]

                self.n = len(self.coordinates)

            else:
                raise StopIteration


class KerasModel(Transformer):
    def __init__(self, keras_model, output_affines=None, **kwargs):
        super(KerasModel, self).__init__(**kwargs)
        self.keras_model = keras_model
        self.output_affines = output_affines if isinstance(output_affines, list) else [output_affines] * len(self.keras_model.outputs)
        assert len(self.output_affines) == len(self.keras_model.outputs)

    def _update_idx(self, idx):
        y = self.keras_model.predict(self.connections[idx][0].get())
        y = y if isinstance(y, list) else [y]
        for idx_, (y_, output_affine) in enumerate(zip(y, self.output_affines)):
            if output_affine is None:
                output_affine = Sample.update_affine(translation=[-(out_shape // 2) + (in_shape // 2) for in_shape, out_shape in zip(self.connections[idx][0][0].shape[1:4], y_.shape[1:4])])

            self.outputs[idx][idx_] = Sample(y_, Sample.update_affine(self.connections[idx][0][0].affine, transformation_matrix=output_affine))

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        output_shapes = self.keras_model.output_shape if isinstance(self.keras_model.output_shape, list) else [self.keras_model.output_shape]
        for idx_, output_shape in enumerate(output_shapes):
            output_shapes[idx_] = (output_shape[0] or self.connections[idx][0].shapes[0][0], output_shape[1], output_shape[2], output_shape[3], output_shape[4])

        return output_shapes

    def _randomize(self):
        pass


class Put(Transformer):
    def __init__(self, reference_connection, caching=True, cval=0, order=0, keep_counts=False, **kwargs):
        super(Put, self).__init__(extra_connections=reference_connection, **kwargs)
        self.reference_connection = reference_connection
        self.caching = caching
        self.cval = cval
        self.order = order
        self.prev_references = [None] * len(reference_connection)
        self.output_array_counts = None
        self.keep_counts = keep_counts

    def _update_idx(self, idx):
        for idx_, (reference, sample) in enumerate(zip(self.reference_connection.get(), self.connections[idx][0].get())):
            for i in range(sample.shape[0]):
                backward_affine = np.linalg.inv(sample.affine[i]) @ reference.affine[0]
                T, R, Z, S = [np.round(transformation, 2) for transformation in transforms3d.affines.decompose44(backward_affine)]
                if np.allclose(R, np.eye(3)) and np.allclose(Z, [1, 1, 1]) and np.allclose(S, [0, 0, 0]):
                    coordinates = [int(round(s // 2 - t)) for s, t in zip(sample.shape[1:4], T)]
                    if self.keep_counts:
                        transformed_array = transformations.put(np.zeros(self.outputs[idx][idx_].shape[1:]), sample[i, ...], coordinates=coordinates)[None, ...]
                        transformed_array_counts = transformations.put(np.zeros(self.output_array_counts[idx][idx_].shape[1:4]), np.ones_like(sample[i, ..., 0]), coordinates=coordinates)[None, ..., None]
                        # transformed_array = np.stack([transformations.put(np.zeros(reference.shape[1:4]), sample[i, ..., j], coordinates=coordinates) for j in range(sample.shape[4])], axis=-1)[None, ...]

                    else:
                        transformed_array = transformations.put(self.outputs[idx][idx_][0, ...], sample[i, ...], coordinates=coordinates)[None, ...]

                else:
                    transformed_array = np.stack([transformations.affine_deformation(sample[i, ..., j], backward_affine, output_shape=reference.shape[1:4], cval=self.cval, order=self.order) for j in range(sample.shape[4])], axis=-1)[None, ...]
                    if self.keep_counts:
                        transformed_array_counts = transformations.affine_deformation(np.ones_like(sample[i, ..., 0]), backward_affine, output_shape=reference.shape[1:4], cval=0, order=self.order)[None, ..., None]

                if self.keep_counts:
                    transformed_array = self.output_array_counts[idx][idx_] / (self.output_array_counts[idx][idx_] + transformed_array_counts) * self.outputs[idx][idx_] + transformed_array_counts / (self.output_array_counts[idx][idx_] + transformed_array_counts) * transformed_array
                    self.output_array_counts[idx][idx_] += transformed_array_counts

                self.outputs[idx][idx_][transformed_array != self.cval] = transformed_array[transformed_array != self.cval]

            if np.isnan(self.outputs[idx][idx_]).any():
                self.outputs[idx][idx_][...] = self.outputs[idx][idx_][tuple(scipy.ndimage.distance_transform_edt(np.isnan(self.outputs[idx][idx_]), return_distances=False, return_indices=True))]

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.reference_connection.shapes

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
                        self.output_array_counts[idx][idx_] = np.full(self.reference_connection[idx_].shape[:4] + (1,), 1e-7)
