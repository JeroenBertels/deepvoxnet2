import os
import uuid
import copy
import pickle
import numpy as np
import nibabel as nib
from deepvoxnet2.components.transformers import Connection, _Input, KerasModel
from PIL import Image


class Creator(object):
    def __init__(self, outputs):
        self.outputs = Creator.deepcopy(outputs if isinstance(outputs, list) else [outputs])
        self.transformers, self.connections = self.trace(self.outputs, clear_active_indices=True, set_names=True)
        self.trace(self.outputs, set_active_indices=True)
        self.active_transformers, self.active_connections = self.trace(self.outputs, only_active=True)
        self.active_input_transformers = [transformer for transformer in self.active_transformers if isinstance(transformer, _Input)]
        self.inputs = [active_connection for active_connection in self.active_connections if isinstance(active_connection.transformer, _Input)]
        self.output_shapes = [output_connection.shapes for output_connection in self.outputs]
        self.sample_id = None

    def eval(self, identifier=None):
        for transformer in self.active_transformers:
            transformer.reset()
            if transformer in self.active_input_transformers:
                transformer.load(identifier)

        while True:
            try:
                self.sample_id = uuid.uuid4()
                for output_connection in self.outputs:
                    output_connection.eval(self.sample_id)

            except (StopIteration, RuntimeError):
                break

            yield [[sample.copy() for sample in output] for output in self.outputs]

    def reset(self):
        for transformer in self.transformers:
            transformer.reset()

    def get_output_shapes(self):
        return self.output_shapes

    @staticmethod
    def trace(output_connections, only_active=False, clear_active_indices=False, set_active_indices=False, set_names=False, reset_transformers=False):
        return Connection.trace(output_connections, only_active=only_active, clear_active_indices=clear_active_indices, set_active_indices=set_active_indices, set_names=set_names, reset_transformers=reset_transformers)

    @staticmethod
    def deepcopy(output_connections):
        keras_models = {}
        transformers, _ = Creator.trace(output_connections, reset_transformers=True)  # All transformers have to be reset (i.e. generator attributes of the transformers set back to None because generators can not be pickled)
        for transformer in transformers:
            if isinstance(transformer, KerasModel):
                keras_model = transformer.keras_model
                transformer.keras_model = uuid.uuid4()
                keras_models[transformer.keras_model] = keras_model

        output_connections = copy.deepcopy(output_connections)
        transformers_, _ = Creator.trace(output_connections)
        for transformer in transformers + transformers_:
            if isinstance(transformer, KerasModel):
                transformer.keras_model = keras_models[transformer.keras_model]

        return output_connections

    def summary(self, only_active=True):
        print('_' * 260)
        print('{:<50} {:<10} {:<75} {:<75} {:<50}'.format('Transformer (type) (n)', 'idx', 'Output shape', 'Connected to', 'Extra connected to'))
        print('=' * 260)
        for transformer in reversed(self.active_transformers if only_active else self.transformers):
            for i, idx in enumerate(transformer.active_indices if only_active else range(len(transformer))):
                print('{:<50} {:<10} {:<75} {:<75} {:<50}'.format(
                    '"' if i > 0 else f"{transformer.name} ({transformer.__class__.__name__}) ({transformer.n})",
                    idx,
                    str(transformer.output_shapes[idx]),
                    str([f"{connection.transformer.name}[{connection.idx}]" for connection in transformer.connections[idx]]),
                    '"' if i > 0 else str([f"{connection.transformer.name}[{connection.idx}]" for connection in transformer.extra_connections])))

            print('_' * 260)

        print('')

    def write_transformer_outputs(self, output_dir, file_format=".nii.gz"):
        assert self.sample_id is not None, "The creator is not yet updated and thus there are no transformer outputs yet."
        sample_dir = os.path.join(output_dir, str(self.sample_id))
        assert not os.path.isdir(sample_dir), "The transformer outputs have already been written to the specified directory."
        os.makedirs(sample_dir)
        for transformer in self.active_transformers:
            for idx in transformer.active_indices:
                for idx_, sample in enumerate(transformer[idx]):
                    for b, sample_ in enumerate(sample):
                        if file_format in [".nii.gz", ".nii"]:
                            img = nib.Nifti1Image(sample_, affine=sample.affine[b])
                            nib.save(img, os.path.join(sample_dir, f"{transformer.name}_i{idx}_s{idx_}_b{b}" + file_format))

                        elif file_format in [".png", ".jpg"]:
                            assert sample_.shape[2] == 1, "Can only save 2D images as png."
                            for f in range(sample_.shape[-1]):
                                img = Image.fromarray(((sample_[:, :, 0, f] - np.min(sample_[..., f])) / np.max(sample_[..., f]) * 255).astype(np.uint8))
                                img.save(os.path.join(sample_dir, f"{transformer.name}_i{idx}_s{idx_}_b{b}_f{f}" + file_format))

                        else:
                            raise NotImplementedError

    def clear_keras_models(self):
        keras_models = {}
        for transformer in self.transformers:
            if isinstance(transformer, KerasModel):
                keras_models[transformer.name] = transformer.keras_model
                transformer.keras_model = None

        return keras_models

    def set_keras_models(self, keras_models):
        for transformer in self.transformers:
            if isinstance(transformer, KerasModel):
                transformer.keras_model = keras_models if not isinstance(keras_models, dict) else keras_models.get(transformer.name, None)

    def save(self, file_path):
        self.save_creator(self, file_path)

    @staticmethod
    def save_creator(creator, file_path):
        creator = Creator(creator.outputs)
        creator.clear_keras_models()
        with open(file_path, "wb") as f:
            pickle.dump(creator, f)

    @staticmethod
    def load_creator(file_path):
        with open(file_path, "rb") as f:
            creator = pickle.load(f)

        return creator
