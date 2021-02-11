import os
import uuid
import copy
import nibabel as nib
from deepvoxnet2.components.transformers import _Input, KerasModel


class Creator(object):
    def __init__(self, outputs):
        outputs = outputs if isinstance(outputs, list) else [outputs]
        self.outputs = Creator.deepcopy(outputs)
        self.active_transformers, self.active_connections = self.trace(self.outputs, set_active_indices=True, reset_transformers=True, set_names=True)
        self.active_input_transformers = [transformer for transformer in self.active_transformers if isinstance(transformer, _Input)]
        self.sample_id = None

    def eval(self, identifier=None):
        for transformer in self.active_transformers:
            Creator.reset_transformer(transformer)
            if transformer in self.active_input_transformers:
                transformer.input_transformers = self.active_input_transformers
                transformer.load(identifier)

        while True:
            try:
                self.sample_id = uuid.uuid4()
                for output_connection in self.outputs:
                    output_connection.eval(self.sample_id)

            except StopIteration:
                break

            yield copy.deepcopy([output.get() for output in self.outputs])

    @staticmethod
    def reset_transformer(transformer, clear_active_indices=False):
        transformer.n_ = 0
        transformer.sample_id = None
        transformer.generator = None
        if isinstance(transformer, _Input):
            transformer.n_resets = 0
            transformer.input_transformers = None

        if clear_active_indices:
            transformer.active_indices = []

        return transformer

    @staticmethod
    def trace(output_connections, set_active_indices=False, reset_transformers=False, set_names=False):
        traced_transformers = []
        traced_connections = []
        connections = [output_connection for output_connection in output_connections]
        while len(connections) > 0:
            connection = connections.pop(0)
            if connection not in traced_connections:
                traced_connections.append(connection)
                if connection.transformer not in traced_transformers:
                    if reset_transformers:
                        Creator.reset_transformer(connection.transformer, clear_active_indices=set_active_indices)
                    
                    if set_names:
                        if connection.transformer.name is None:
                            connection.transformer.name = "{}_{}".format(connection.transformer.__class__.__name__, len([traced_transformer for traced_transformer in traced_transformers if traced_transformer.__class__.__name__ == connection.transformer.__class__.__name__]))

                        assert connection.transformer.name not in [traced_transformer.name for traced_transformer in traced_transformers], "In a Creator you cannot use the same name for more than one Transformer."

                    traced_transformers.append(connection.transformer)

                if set_active_indices and connection.idx not in connection.transformer.active_indices:
                    connection.transformer.active_indices.append(connection.idx)

                for connection__ in connection.transformer.extra_connections:
                    if connection__ not in traced_connections and connection__ not in connections:
                        connections.append(connection__)

                for idx, connection_ in enumerate(connection.transformer.connections):
                    for connection__ in connection_:
                        if connection__ not in traced_connections and connection__ not in connections and (not set_active_indices or idx == connection.idx):
                            connections.append(connection__)

        return traced_transformers, traced_connections

    @staticmethod
    def deepcopy(output_connections):
        keras_models = {}
        transformers, _ = Creator.trace(output_connections)
        for transformer in transformers:
            Creator.reset_transformer(transformer)  # has to be reset (i.e. generator attributes of the transformers set back to None because generators can not be pickled)
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

    def summary(self):
        print('_' * 260)
        print('{:<50} {:<10} {:<75} {:<75} {:<50}'.format('Transformer (type) (n)', 'idx', 'Output shape', 'Connected to', 'Extra connected to'))
        print('=' * 260)
        for transformer in reversed(self.active_transformers):
            for i, idx in enumerate(transformer.active_indices):
                print('{:<50} {:<10} {:<75} {:<75} {:<50}'.format(
                    '"' if i > 0 else f"{transformer.name} ({transformer.__class__.__name__}) ({transformer.n})",
                    idx,
                    str(transformer.output_shapes[idx]),
                    str([f"{connection.transformer.name}[{connection.idx}]" for connection in transformer.connections[idx]]),
                    '"' if i > 0 else str([f"{connection.transformer.name}[{connection.idx}]" for connection in transformer.extra_connections])))

            print('_' * 260)

        print('')

    def write_transformer_outputs(self, output_dir, file_format=".nii.gz"):
        if file_format not in [".nii.gz"]:
            raise ValueError

        sample_dir = os.path.join(output_dir, str(self.sample_id))
        assert not os.path.isdir(sample_dir), "The transformer outputs have already been written to the specified directory."
        os.makedirs(sample_dir)
        for transformer in self.active_transformers:
            for idx in transformer.active_indices:
                for idx_, sample in enumerate(transformer[idx]):
                    if len(sample) > 1:
                        print(f"WARNING: encountered a batch size = {len(sample)} > 1 for {transformer.name} at idx {idx} and idx_ {idx_} --> only batch element 0 is saved.")

                    img = nib.Nifti1Image(sample[0], affine=sample.affine[0])
                    nib.save(img, os.path.join(sample_dir, f"{transformer.name}_idx-{idx}_output-{idx_}" + file_format))
