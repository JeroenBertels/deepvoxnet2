"""The Creator class in DeepVoxNet2 is used to build models by creating a computational graph of Transformers, i.e. a Transformer pipeline, much like Keras' models are constructed with layers.

To build such a model, you create a list of output Connection objects, which represent the different outputs of the Transformer pipeline. The Creator object is then constructed with this list of output connections, and it automatically builds the corresponding computational graph by tracing the inputs of the output connections, in a manner similar to the way a Keras model is built.

Once the Creator object is built, it can be used to generate output for the specified output connections. It has a eval method which can be called repeatedly to generate the output for each input Identifier.

The Creator class also provides methods to get information about the output shapes, reset the transformers in the pipeline, get a summary of the pipeline, save the model, load the saved model, etc. Additionally, it can write the output of each transformer in the pipeline to disk.
"""

import os
import uuid
import copy
import pickle
import numpy as np
import nibabel as nib
from deepvoxnet2.components.transformers import Connection, _Input, KerasModel
from PIL import Image


class Creator(object):
    """A `Creator` object takes a list of transformer outputs (in the form of `Connection` objects)
    and creates a Creator object.

    This object has some useful properties, such as the ability to save it, and to calculate the state/output of all the requested outputs.

    Attributes:
    -----------
    outputs: List
        List of output `Connection` objects.

    transformers: List
        List of all transformers in the creator.

    connections: List
        List of all `Connection` objects in the creator.

    active_transformers: List
        List of all active transformers in the creator.

    active_connections: List
        List of all active `Connection` objects in the creator.

    active_input_transformers: List
        List of all active input transformers in the creator.

    inputs: List
        List of all input connections in the creator.

    output_shapes: List
        List of the output shapes for each `Connection` in the creator.

    sample_id: UUID
        Unique identifier for each sample in the creator.

    Methods:
    --------
    eval(identifier=None)
        Evaluates the current state/output of all the requested outputs.

    reset()
        Resets all transformers in the creator.

    get_output_shapes()
        Returns a list of the output shapes for each `Connection` in the creator.

    trace(output_connections, only_active=False, clear_active_indices=False, set_active_indices=False,
        set_names=False, reset_transformers=False)
        Traces the transformers and connections for the given output connections.

    deepcopy(output_connections)
        Returns a deep copy of the given output connections.

    summary(only_active=True)
        Prints a summary of all transformers in the creator.

    write_transformer_outputs(output_dir, file_format=".nii.gz")
        Writes the transformer outputs to the specified directory.

    clear_keras_models()
        Clears all Keras models in the creator.

    set_keras_models(keras_models)
        Sets the Keras models in the creator.

    save(file_path)
        Saves the creator to the specified file path.

    save_creator(creator, file_path)
        Saves the given creator to the specified file path.

    load_creator(file_path)
        Loads a creator from the specified file path.
    """

    def __init__(self, outputs):
        """Initializes the Creator object.

        Traces the specified outputs to determine all transformers and connections involved in the process.
        Determines the active transformers and connections.
        Initializes some instance variables such as inputs, output_shapes, etc.
        """

        self.outputs = Creator.deepcopy(outputs if isinstance(outputs, list) else [outputs])
        self.transformers, self.connections = self.trace(self.outputs, clear_active_indices=True, set_names=True)
        self.trace(self.outputs, set_active_indices=True)
        self.active_transformers, self.active_connections = self.trace(self.outputs, only_active=True)
        self.active_input_transformers = [transformer for transformer in self.active_transformers if isinstance(transformer, _Input)]
        self.inputs = [active_connection for active_connection in self.active_connections if isinstance(active_connection.transformer, _Input)]
        self.output_shapes = [output_connection.shapes for output_connection in self.outputs]
        self.sample_id = None

    def eval(self, identifier=None):
        """Evaluates the Creator object.

        Calls the eval() method of each active transformer and returns the calculated state/output of all the requested transformer outputs.
        Accepts an optional identifier parameter which is passed to the _Input transformers to load the input data.

        Parameters:
        -----------
        identifier: UUID, Identifier
            Unique identifier for each sample in the creator.

        Yields:
        ------
        List of output samples for each requested output.
        """

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
        """Resets all transformers involved in the Creator object.
        """

        for transformer in self.transformers:
            transformer.reset()

    def get_output_shapes(self):
        """Returns the output shapes of all requested transformer outputs.
        """

        return self.output_shapes

    @staticmethod
    def trace(output_connections, only_active=False, clear_active_indices=False, set_active_indices=False, set_names=False, reset_transformers=False):
        """Static method that traces the specified output_connections to determine all transformers and connections involved in the process.

        The method returns a tuple of two lists: transformers and connections.
        transformers is a list of all transformers involved in the process, while connections is a list of all connections involved in the process.
        """

        return Connection.trace(output_connections, only_active=only_active, clear_active_indices=clear_active_indices, set_active_indices=set_active_indices, set_names=set_names, reset_transformers=reset_transformers)

    @staticmethod
    def deepcopy(output_connections):
        """Static method that creates a deep copy of the specified output_connections.

        All transformers are reset (i.e. generator attributes of the transformers set back to None because generators can not be pickled). This is necessary for pickling the Creator object.
        """

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
        """Prints a summary of the Creator object.

        Shows a table of all transformers and their output shapes.
        """

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
        """Writes the output of all transformers involved in the Creator object to the specified directory.

        The output format can be specified using the file_format parameter.
        Currently, ".nii.gz", ".nii", ".png", and ".jpg" formats are supported.
        """

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
        """Clears all Keras models used in the Creator object.
        """

        keras_models = {}
        for transformer in self.transformers:
            if isinstance(transformer, KerasModel):
                keras_models[transformer.name] = transformer.keras_model
                transformer.keras_model = None

        return keras_models

    def set_keras_models(self, keras_models):
        """Sets the Keras models used in the Creator object.
        """

        for transformer in self.transformers:
            if isinstance(transformer, KerasModel):
                transformer.keras_model = keras_models if not isinstance(keras_models, dict) else keras_models.get(transformer.name, None)

    def save(self, file_path):
        """Saves the Creator object to the specified file path.
        """

        self.save_creator(self, file_path)

    @staticmethod
    def save_creator(creator, file_path):
        """Static method that saves the specified creator object to the specified file path.
        """

        creator = Creator(creator.outputs)
        creator.clear_keras_models()
        with open(file_path, "wb") as f:
            pickle.dump(creator, f)

    @staticmethod
    def load_creator(file_path):
        """Static method that loads a Creator object from the specified file path.
        """

        with open(file_path, "rb") as f:
            creator = pickle.load(f)

        return creator
