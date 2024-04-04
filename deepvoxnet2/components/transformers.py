"""In summary, transformers are objects that can be connected to form a network to preprocess and postprocess data.

They operate on lists of Sample objects, and can include a transformer that wraps a Keras Model to form an end-to-end pipeline. Connection objects are used to specify how the transformers are connected.
"""

import os
import json
import random
import uuid
import transforms3d
import numpy as np
import scipy.ndimage
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.utilities import transformations
from tensorflow.keras.utils import to_categorical


class Connection(object):
    """Connection between transformers.

    Attributes:
    -----------
    transformer: Transformer
        Transformer instance that the connection belongs to.
    idx: int
        The index of the output in the list of the transformer's outputs.
    shapes: List[tuple]
        List of output shapes of the connected transformer.

    Methods:
    --------
    __len__():
        Returns the length of the output of the connected transformer.
    __getitem__(item):
        Returns the item of the output of the connected transformer.
    __iter__():
        Returns an iterator of the output of the connected transformer.
    get():
        Returns the output of the connected transformer.
    eval(sample_id=None):
        Returns the evaluated output of the connected transformer for the given `sample_id`.
    get_shapes():
        Returns the shapes of the outputs of the connected transformer.
    trace(connections, only_active=False, clear_active_indices=False, set_active_indices=False, set_names=False, reset_transformers=False):
        Static method that traces the connections between transformers.
    """

    def __init__(self, transformer, idx):
        """Initializes a new Connection instance.

        Parameters:
        -----------
        transformer: Transformer
            Transformer instance that the connection belongs to.
        idx: int
            The index of the output in the list of the transformer's outputs.
        """

        self.transformer = transformer
        self.idx = idx
        self.shapes = self.get_shapes()

    def __len__(self):
        """Returns the length of the output of the connected transformer.

        Returns:
        --------
        int: Length of the output.
        """

        return len(self.get())

    def __getitem__(self, item):
        """Returns the item of the output of the connected transformer.

        Parameters:
        -----------
        item: Any
            Index or slice object to select a specific item(s) of the output.

        Returns:
        --------
        Any: Selected item(s) of the output.
        """

        return self.get()[item]

    def __iter__(self):
        """Returns an iterator of the output of the connected transformer.

        Returns:
        --------
        Iterator: Iterator of the output.
        """

        return iter(self.get())

    def get(self):
        """Returns the output of the connected transformer.

        Returns:
        --------
        Any: Output of the connected transformer.
        """

        return self.transformer.outputs[self.idx]

    def eval(self, sample_id=None):
        """Returns the evaluated output of the connected transformer for the given `sample_id`.

        Parameters:
        -----------
        sample_id: Any, optional
            Identifier of the sample to be evaluated.

        Returns:
        --------
        Any: Evaluated output of the connected transformer for the given `sample_id`.
        """

        return self.transformer.eval(sample_id)[self.idx]

    def get_shapes(self):
        """Returns the shapes of the outputs of the connected transformer.

        Returns:
        --------
        List[tuple]: Shapes of the outputs of the connected transformer.
        """

        return [sample_shape for sample_shape in self.transformer.output_shapes[self.idx]]

    @staticmethod
    def trace(connections, only_active=False, clear_active_indices=False, set_active_indices=False, set_names=False, reset_transformers=False):
        """Trace the connections between transformers and connection indices in a transformer network.

        Parameters:
        -----------
        connections : list of Connection objects
            The connections to trace.

        only_active : bool, optional
            If True, only trace connections to active indices. Default is False.

        clear_active_indices : bool, optional
            If True, clear the active indices of each transformer before tracing. Default is False.

        set_active_indices : bool, optional
            If True, set the active indices of the transformers. Default is False.

        set_names : bool, optional
            If True, set the name of each transformer. Default is False.

        reset_transformers : bool, optional
            If True, reset the transformers. Default is False.

        Returns:
        --------
        tuple of (list of Transformer objects, list of Connection objects)
            A tuple containing the list of transformers and their connections.
        """

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
                        if connection__ not in traced_connections and connection__ not in connections and (not set_active_indices or idx == connection.idx):
                            connections.append(connection__)

        if only_active:
            traced_transformers = [traced_transformer for traced_transformer in traced_transformers if len(traced_transformer.active_indices) > 0]
            traced_connections = [traced_connection for traced_connection in traced_connections if traced_connection.idx in traced_connection.transformer.active_indices]

        return traced_transformers, traced_connections


class Transformer(object):
    """Abstract base class for creating transformers in a transformer network.

    Attributes:
    -----------
    n: int
        The number of times to apply the transformer to the input data.
    extra_connections: list of Connection
        The additional input connections to be used for the transformer.
    name: str
        The name of the transformer.
    n_: int
        The current number of iterations.
    connections: list of list of Connection
        The input connections to the transformer.
    outputs: list of list of Sample
        The output samples of the transformer.
    output_shapes: list of tuple of int
        The shapes of the output samples.
    active_indices: list of int
        The indices of the active input connections.
    generator: iterator
        The iterator object used for transforming the input data.
    sample_id: uuid.UUID
        The ID of the current sample being processed.

    Methods:
    --------
    __call__(*connections) -> Connection or List of Connection:
        Add input connections to the transformer and return output connection(s).
    __len__() -> int:
        Return the number of output connections of the transformer.
    __getitem__(item) -> List:
        Return the output(s) at the index item.
    __iter__() -> Iterator:
        Return an iterator over the output connections.
    get_output_shapes() -> List:
        Return a list of tuples that describe the output shapes for each output.
    eval(sample_id=None) -> List:
        Generate the output of the transformer and return it. This method is the main entry point to execute the transformer.
    reset() -> None:
        Reset the transformer for a new evaluation run.
    _update() -> Iterator:
        Generate the output of the transformer in the form of a generator. This method is called by eval() to execute the generator.
    _update_idx(idx) -> None:
        Update the output at the index idx of the transformer.
    _calculate_output_shape_at_idx(idx) -> tuple:
        Calculate the output shape at the index idx of the transformer.
    _randomize() -> None:
        Generate new samples to be used as inputs for the transformer.
    """

    def __init__(self, n=1, extra_connections=None, name=None):
        """Initializes a new instance of the Transformer class.

        Parameters:
        -----------
        n: int, optional
            The number of times to apply the transformer to the input data.
            Default is 1.
        extra_connections: Connection or list of Connection, optional
            Additional input connections to be used for the transformer.
            Default is None.
        name: str, optional
            The name of the transformer.
            Default is None.
        """

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
        """
        Add input connection(s) to the transformer and return output connection(s).

        Parameters:
        -----------
        *connections: tuple of Connection
            The input connection(s) to add to the transformer.

        Returns:
        --------
        output_connections: Connection or List of Connection
            The output connection(s) of the transformer.
        """

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
        """Return the number of output connections of the transformer.

        Returns:
        --------
        length: int
            The number of output connections of the transformer.
        """

        return len(self.outputs)

    def __getitem__(self, item):
        """Returns the output of this transformer at the given index.

        Parameters:
        -----------
        item: int
            The index of the output to be returned.

        Returns:
        --------
        The output at the given index.
        """

        return self.outputs[item]

    def __iter__(self):
        """Returns an iterator over the outputs of this transformer.

        Returns:
        --------
        An iterator over the outputs of this transformer.
        """

        return iter(self.outputs)

    def get_output_shapes(self):
        """Returns the shapes of the outputs of this transformer.

        Returns:
        --------
        A list of tuples, where each tuple corresponds to the shape of an output.
        """

        return self.output_shapes

    def eval(self, sample_id=None):
        """Evaluates this transformer and returns its outputs.

        Parameters:
        -----------
        sample_id: uuid.UUID, optional
            An optional identifier for the sample being processed.

        Returns:
        --------
        The outputs of this transformer.
        """

        # print(f"--> {self.name}")
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

        # print(f"<-- {self.name}")
        return self.outputs

    def reset(self):
        """Resets the state of this transformer, allowing it to be re-evaluated.
        """

        self.n_ = 0
        self.sample_id = None
        self.generator = None

    def _update(self):
        """Helper method that updates the state of this transformer and yields its outputs.

        Yields:
        -------
        The outputs of this transformer.
        """

        self.n_ = 0
        while self.n is None or self.n_ < self.n:
            self._randomize()
            for idx in self.active_indices:
                self._update_idx(idx)

            self.n_ += 1
            yield self.outputs

    def _update_idx(self, idx):
        """Updates the output at the given index.

        Parameters:
        -----------
        idx: int
            The index of the output to be updated.
        """

        raise NotImplementedError

    def _calculate_output_shape_at_idx(self, idx):
        """Calculates the shape of the output at the given index.

        Parameters:
        -----------
        idx: int
            The index of the output whose shape is to be calculated.

        Returns:
        --------
        The shape of the output at the given index.
        """

        raise NotImplementedError

    def _randomize(self):
        """Helper method that randomizes the state of this transformer in preparation for a new evaluation.
        """

        raise NotImplementedError


class _Input(Transformer):
    """A special type of transformer which serves as the input to the transformer network.

    It has no input connections and forms the start of the transformer network.

    Parameters:
    -----------
    output_shapes : list of lists of tuples
        The list of output shapes for each output. Each output shape must be in the format of a tuple of 5 integers
        (batch_size, depth, height, width, channels).

    Other Parameters:
    ------------------
    name : str, optional
        A name for the transformer.
    n : int, optional
        The number of times this transformer is expected to generate the output before stopping. If None, it will run
        indefinitely.
    extra_connections : list or transformer or None, optional
        A list of transformers or a single transformer that this transformer is not directly connected to but is
        required for its computations.
    """

    def __init__(self, output_shapes, **kwargs):
        """Initializes the _Input transformer.

        Parameters:
        -----------
        output_shapes : list of lists of tuples
            The list of output shapes for each output. Each output shape must be in the format of a tuple of 5 integers
            (batch_size, depth, height, width, channels).
        """

        super(_Input, self).__init__(**kwargs)
        for i, output_shapes_ in enumerate(output_shapes):
            assert isinstance(output_shapes_, list) and all([isinstance(output_shape, tuple) and len(output_shape) == 5 for output_shape in output_shapes_]), "The given output_shapes fot the _Input transformer are not in the correct format."
            self.connections.append([])
            self.outputs.append([None] * len(output_shapes_))
            self.output_shapes.append(output_shapes_)
            self.active_indices.append(i)

    def load(self, identifier=None):
        """A method to load data from a source into the _Input transformer. It must be implemented in subclasses.

        Parameters:
        -----------
        identifier : object, optional
            An identifier that can be used to identify the data source.
        """

        raise NotImplementedError

    def _update_idx(self, idx):
        """A method that updates a particular output connection of the _Input transformer.

        Parameters:
        -----------
        idx : int
            The index of the output connection to update.
        """

        pass

    def _calculate_output_shape_at_idx(self, idx):
        """A method that returns the output shape at a given output connection index.

        Parameters:
        -----------
        idx : int
            The index of the output connection for which the output shape is requested.

        Returns:
        --------
        output_shape : list of tuples
            The output shape for the given output connection.
        """

        return self.output_shapes[idx]

    def _randomize(self):
        """A method that does nothing as the _Input transformer does not require any randomization.
        """
        pass


class _MircInput(_Input):
    """A Transformer that loads MIRC data into the transformer network.

    This Transformer is an input transformer, meaning it has no input connections.

    Parameters:
    -----------
    modality_ids : str or List[str]
        The modality ID(s) of the MIRC dataset to load.
    output_shapes : List[Tuple[int]]
        A list of output shapes of the loaded data. Default is None and is set to [(None, ) * 5] * len(self.modality_ids).
    **kwargs : dict
        Optional keyword arguments for the superclass.

    Attributes:
    -----------
    modality_ids : List[str]
        The modality ID(s) of the MIRC dataset to load.

    Methods:
    --------
    load(identifier=None)
        Loads the MIRC dataset from the provided `identifier` and stores the loaded data in `self.outputs`.
        Raises an error if the `identifier` is not provided.

    Inherited Methods:
    ------------------
    __init__(output_shapes, **kwargs)
        Initializes the object by constructing the necessary connections and outputs for each modality.
    _calculate_output_shape_at_idx(idx)
        Returns the output shape for the given index `idx`.
    _randomize()
        Method that will be called during initialization to randomly set the output tensor.
    _update_idx(idx)
        Raises a NotImplementedError as it should be implemented by the subclass.
    """

    def __init__(self, modality_ids, output_shapes=None, **kwargs):
        """Initialize the transformer with the specified modality IDs and expected output shapes.

        Raises an assertion error if the length of the output_shapes list is not equal to the length of the modality_ids list.

        Parameters
        ----------
        modality_ids: str or list of str
            The modality ID(s) of the data to be loaded from the MIRC source(s).
        output_shapes: list of tuple of int, optional
            The expected output shape(s) of the input data. If not provided, it defaults to [(None, ) * 5, ] * len(modality_ids).
        **kwargs
            Additional keyword arguments to be passed to the parent Transformer class.
        """

        self.modality_ids = modality_ids if isinstance(modality_ids, list) else [modality_ids]
        if output_shapes is None:
            output_shapes = [(None, ) * 5, ] * len(self.modality_ids)

        assert len(output_shapes) == len(self.modality_ids)
        super(_MircInput, self).__init__([output_shapes], **kwargs)

    def load(self, identifier=None):
        """Load the data from the MIRC source based on the identifier.

        Raises an assertion error if the identifier is not provided.

        Parameters
        ----------
        identifier: Identifier, optional
            An identifier object that specifies the dataset, case, and record to load the data from.
        """

        assert identifier is not None
        for idx_, modality_id in enumerate(self.modality_ids):
            self.outputs[0][idx_] = identifier.mirc[identifier.dataset_id][identifier.case_id][identifier.record_id][modality_id].load()


class MircInput(_MircInput):
    """Instantiates the _MircInput transformer with given `modality_ids` and `output_shapes`.

    Parameters:
    -----------
    modality_ids : str or List[str]
        The modality ID(s) of the MIRC dataset to load.
    output_shapes : List[Tuple[int]]
        A list of output shapes of the loaded data. Default is None and is set to [(None, ) * 5] * len(self.modality_ids).
    **kwargs : dict
        Optional keyword arguments for the superclass.
    """

    def __new__(cls, modality_ids, output_shapes=None, **kwargs):
        return _MircInput(modality_ids, output_shapes, **kwargs)()


class _SampleInput(_Input):
    """An input transformer for a list of samples.

    Parameters
    ----------
    samples: list of ndarrays or None
        The input samples.
    output_shapes: list of tuples or None
        The output shapes of the samples. Must be specified if `samples` is not given.
    **kwargs: dict
        Additional keyword arguments to be passed to the superclass constructor.

    Attributes
    ----------
    samples : list of ndarrays
        The input samples.
    output_shapes : list of tuples
        The output shapes of the samples.

    Methods
    -------
    load(identifier=None)
        Load the input samples.

    Notes
    -----
    The output shape of each sample is determined either by `output_shapes` or by the shape of the input samples.
    """

    def __init__(self, samples=None, output_shapes=None, **kwargs):
        """Initialize the _SampleInput instance.

        Parameters
        ----------
        samples: list of ndarrays or None
            The input samples.
        output_shapes: list of tuples or None
            The output shapes of the samples. Must be specified if `samples` is not given.
        **kwargs: dict
            Additional keyword arguments to be passed to the superclass constructor.
        """

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
        """Load the input samples.

        Parameters
        ----------
        identifier: Identifier or None
            An identifier that contains information on where to load the samples from.
        """

        if identifier is not None:
            for idx_, sample in enumerate(identifier.sample if isinstance(identifier.sample, list) else [identifier.sample]):
                self.outputs[0][idx_] = sample


class SampleInput(_SampleInput):
    """Instantiates the _SampleInput transformer with given `samples` and `output_shapes`.

    Parameters:
    -----------
    samples: list of ndarrays or None
            The input samples.
    output_shapes: list of tuples or None
            The output shapes of the samples. Must be specified if `samples` is not given.
    **kwargs: dict
        Additional keyword arguments to be passed to the superclass constructor.
    """
    def __new__(cls, samples=None, output_shapes=None, **kwargs):
        return _SampleInput(samples, output_shapes, **kwargs)()


class Buffer(Transformer):
    """Buffer class to concatenate incoming samples along the specified axis and buffer them until the buffer is full.

    Parameters
    ----------
    buffer_size : int, optional
        The size of the buffer. If set to None, the buffer will not be constrained by its size. (default: None)
    axis : int, optional
        The axis along which to concatenate incoming samples. Must be either 0 or 4 for an image. (default: 0)
    drop_remainder : bool, optional
        Whether to drop the last batch if the incoming samples do not fill up the buffer. (default: False)
    """

    def __init__(self, buffer_size=None, axis=0, drop_remainder=False, **kwargs):
        """Initializes the Buffer class with specified parameters.

        Parameters
        ----------
        buffer_size : int, optional
            The size of the buffer. If set to None, the buffer will not be constrained by its size. (default: None)
        axis : int, optional
            The axis along which to concatenate incoming samples. Must be either 0 or 4 for an image. (default: 0)
        drop_remainder : bool, optional
            Whether to drop the last batch if the incoming samples do not fill up the buffer. (default: False)
        **kwargs
            Additional arguments for the parent class `Transformer`.

        Raises
        ------
        AssertionError
            If `n` is provided as a keyword argument.
            If `axis` is not 0 or 4.
        """

        assert "n" not in kwargs, "A Buffer does not accept n. It just buffers so it cannot create n samples from 1 input."
        super(Buffer, self).__init__(n=1, **kwargs)
        self.buffer_size = buffer_size
        assert axis in [0, 4, -1]
        self.axis = axis if axis != -1 else 4
        self.drop_remainder = drop_remainder
        self.buffered_outputs = None

    def _update_idx(self, idx):
        """Updates the buffer and its output values for a given index.

        Parameters
        ----------
        idx : int
            The index of the transformer.
        """

        for idx_ in range(len(self.outputs[idx])):
            self.outputs[idx][idx_] = Sample(np.concatenate([output[idx_] for output in self.buffered_outputs[idx]], axis=self.axis), self.buffered_outputs[idx][0][idx_].affine if self.axis != 0 else np.concatenate([output[idx_].affine for output in self.buffered_outputs[idx]]))

        self.buffered_outputs[idx] = None

    def _calculate_output_shape_at_idx(self, idx):
        """Calculates the output shape at a given index.

        Parameters
        ----------
        idx : int
            The index of the transformer.

        Returns
        -------
        list of tuples
            A list of tuples representing the output shape for each output connection.
        """

        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        # assert all([output_shape_ is not None for output_shape in self.connections[idx][0].shapes for axis_i, output_shape_ in enumerate(output_shape) if axis_i != self.axis]), "A buffer can only used on a connection with fully specified shapes (except for the concatenation axis)."
        return [tuple([output_shape_ if axis_i != self.axis else (self.buffer_size * output_shape_ if self.buffer_size is not None and self.drop_remainder and output_shape_ is not None else None) for axis_i, output_shape_ in enumerate(output_shape)]) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        """Randomizes the buffer by filling it with new samples.

        If buffer_size is not None, it continues filling the buffer until it reaches that size.

        Raises
        ------
        StopIteration
            If `drop_remainder` is True and there are not enough samples to fill the buffer completely.
        """
        initial_sample_ids = [transformer.sample_id for transformer in Connection.trace([connection for idx in self.active_indices for connection in self.connections[idx]], only_active=True)[0]]
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
            for i, transformer in enumerate(Connection.trace([connection for idx in self.active_indices for connection in self.connections[idx]], only_active=True)[0]):
                if initial_sample_ids[i] != transformer.sample_id:
                    transformer.sample_id = self.sample_id


class Group(Transformer):
    """The `Group` transformer groups together all samples in all input connections at each index into one output connection at that index.
    """

    def __init__(self, **kwargs):
        """Initializes a new instance of the Group class.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments to pass to the parent class.
        """

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


class Swap(Transformer):
    """A transformer that swaps the order of samples in a connection randomly with a certain probability.
    """

    def __init__(self, swap_probability=1, **kwargs):
        """Initializes a new instance of the Swap class.

        Parameters
        ----------
        swap_probability : float, optional
            The probability that the samples in a connection are swapped. The default is 1 (i.e., always swap).
        **kwargs
            Additional keyword arguments to pass to the parent class.
        """
        super(Swap, self).__init__(**kwargs)
        self.swap_probability = swap_probability
        self.swap_state = None

    def _update_idx(self, idx):
        assert len(self.connections[idx][0]) == len(self.swap_state), "Not all connections idx have the same length!"
        for idx_, j in enumerate(self.swap_state):
            self.outputs[idx][idx_] = self.connections[idx][0][j]

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        assert all([all([self.connections[idx][0].shapes[0][axis_i] == shape[axis_i] for shape in self.connections[idx][0].shapes]) for axis_i in range(5)]), "In each connection, the shape of each Sample must be equal!"
        return self.connections[idx][0].shapes

    def _randomize(self):
        self.swap_state = list(range(len(self.connections[0][0])))
        if random.random() < self.swap_probability:
            random.shuffle(self.swap_state)


class Split(Transformer):
    """A transformer that splits the input at each index into multiple outputs.
    """

    def __init__(self, indices=(0,), **kwargs):
        """Initializes the Split transformer with the provided parameters.

        Parameters
        ----------
        indices : tuple or int, optional
            The indices at which to split the input. If an int is provided, it is automatically
            converted into a single-item tuple. Defaults to (0,).
        **kwargs
            Additional arguments to pass to the `Transformer` constructor.

        Raises
        ------
        AssertionError
            If multiple input connections are present at a single index.
        """

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
    """A transformer that concatenates the inputs along a specified axis.
    """

    def __init__(self, axis=-1, **kwargs):
        """Initializes the Concat transformer with the provided parameters.

        Parameters
        ----------
        axis : int, optional
            The axis along which to concatenate the inputs. Must be one of {0, 4, -1}.
            Default is -1.
        **kwargs
            Additional arguments to pass to the `Transformer` constructor.

        Raises
        ------
        AssertionError
            If the specified axis is not one of {0, 4, -1}.
        """

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
                        assert (output_shape_ is None or output_shape__ is None) or output_shape_ == output_shape__, "The shapes of the shared axes should be identical and different from None."

        return [tuple(output_shape) for output_shape in output_shapes]

    def _randomize(self):
        pass


class Mean(Transformer):
    """Calculates the mean of samples along the specified axis.
    """

    def __init__(self, axis=-1, **kwargs):
        """Constructs a new Mean Transformer.

        Parameters
        ----------
        axis : int, optional
            The axis along which to compute the mean. Default is -1.
        **kwargs
            Additional arguments passed to the parent class constructor.
        """

        super(Mean, self).__init__(**kwargs)
        # assert axis in [4, -1]
        self.axis = axis if axis != -1 else 4

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = sample.mean(axis=self.axis, keepdims=True) if self.axis != 0 else Sample(np.mean(sample, axis=0, keepdims=True), affine=sample.affine[0])

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
        self.lower_threshold = np.array(lower_threshold)
        self.upper_threshold = np.array(upper_threshold)

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


class NonZeroMask(Transformer):
    def __init__(self, zero_value=0, compensate_zeros=False, axis=(1, 2, 3, 4), **kwargs):
        super(NonZeroMask, self).__init__(**kwargs)
        self.zero_value = zero_value
        self.compensate_zeros = compensate_zeros
        self.axis = axis

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample(sample != self.zero_value, sample.affine)
            if self.compensate_zeros:
                self.outputs[idx][idx_] *= np.prod([s for i, s in enumerate(self.outputs[idx][idx_].shape) if i in self.axis]) / np.sum(self.outputs[idx][idx_], axis=self.axis, keepdims=True)

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
                    class_counts_dict = {str(class_): float(np.sum((sample[..., class_] if self.one_hot else sample) == self.class_values_dict[class_]).item()) for class_ in self.class_values_dict}
                    class_counts_dict["voxel_count"] = float(np.prod(sample.shape[:4]))

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


class NormalizeIndividual(Transformer):
    def __init__(self, ignore_value=None, replace_value=None, axis=(1, 2, 3), **kwargs):
        super(NormalizeIndividual, self).__init__(**kwargs)
        self.ignore_value = ignore_value
        self.replace_value = ignore_value if replace_value is None else replace_value
        self.axis = axis

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            mask = np.logical_not(np.logical_or(np.isnan(sample), np.isinf(sample))) if self.ignore_value is None or np.isnan(self.ignore_value) or np.isinf(self.ignore_value) else (sample != self.ignore_value)
            normalized_array = (sample - np.mean(sample, axis=self.axis, keepdims=True, where=mask)) / np.std(sample, axis=self.axis, keepdims=True, where=mask)
            self.outputs[idx][idx_] = Sample(np.where(mask, normalized_array, self.replace_value), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        pass


class NormalizeMask(Transformer):
    def __init__(self, reference_connection, axis=(1, 2, 3), mean_shift=0, std_shift=0, mean_scale=0, std_scale=0, **kwargs):
        super(NormalizeMask, self).__init__(extra_connections=reference_connection, **kwargs)
        self.reference_connection = reference_connection
        self.axis = axis
        self.mean_shift = mean_shift
        self.std_shift = std_shift
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        self.shift = None
        self.scale = None

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            mask = self.reference_connection[0] > 0
            shift = np.mean(sample, axis=self.axis, keepdims=True, where=mask)
            scale = np.std(sample, axis=self.axis, keepdims=True, where=mask)
            self.outputs[idx][idx_] = Sample((sample - (shift + self.shift)) / (scale + self.scale), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return self.connections[idx][0].shapes

    def _randomize(self):
        self.shift = np.random.normal(self.mean_shift, self.std_shift)
        self.scale = np.random.normal(self.mean_scale, self.std_scale)


class WindowNormalize(Transformer):
    def __init__(self, lower_window=None, higher_window=None, axis=(1, 2, 3), **kwargs):
        super(WindowNormalize, self).__init__(**kwargs)
        self.lower_window = lower_window
        self.higher_window = higher_window
        self.axis = axis

    def _update_idx(self, idx):
        if not hasattr(self, "axis"):
            self.axis = (1, 2, 3)

        for idx_, sample in enumerate(self.connections[idx][0]):
            lower_window = np.min(sample, axis=self.axis, keepdims=True) if self.lower_window is None else self.lower_window
            higher_window = np.max(sample, axis=self.axis, keepdims=True) if self.higher_window is None else self.higher_window
            self.outputs[idx][idx_] = Sample((sample - lower_window) / (higher_window - lower_window), sample.affine)

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
        shapes = [sample.shape for connection in self.connections for sample in connection[0] if sample is not None]
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
            subsampled_array = np.pad(subsampled_array, ((0, int(self.factor - len(subsampled_array) % self.factor if len(subsampled_array) % self.factor else 0)), (0, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
            if self.mode == "nearest":
                subsampled_array = subsampled_array[slice(0, None, self.factor), ...]

            else:
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
        if isinstance(flip_probabilities, str):
            assert flip_probabilities == "all" and self.n == 8

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
        if isinstance(self.flip_probabilities, str):
            if self.n_ == 0:
                self.flip_state = [1, 1, 1]
            
            elif self.n_ == 1:
                self.flip_state = [-1, 1, 1]
            
            elif self.n_ == 2:
                self.flip_state = [1, -1, 1]
            
            elif self.n_ == 3:
                self.flip_state = [-1, -1, 1]
            
            elif self.n_ == 4:
                self.flip_state = [1, 1, -1]
            
            elif self.n_ == 5:
                self.flip_state = [-1, 1, -1]
            
            elif self.n_ == 6:
                self.flip_state = [1, -1, -1]
            
            elif self.n_ == 7:
                self.flip_state = [-1, -1, -1]

            else:
                raise ValueError
                        
        else:
            self.flip_state = [-1 if random.random() < flip_probability else 1 for flip_probability in self.flip_probabilities]


class GaussianNoise(Transformer):
    def __init__(self, mean=0, std=1, **kwargs):
        super(GaussianNoise, self).__init__(**kwargs)
        mean, std = np.array(mean), np.array(std)
        assert mean.ndim <= 5 and std.ndim <= 5, "mean and std must be broadcastable with the shape of the samples that pass through and may have a maximum of 5 dimensions."
        self.mean = np.expand_dims(mean, list(range(5 - mean.ndim))) if mean.ndim < 5 else mean
        self.std = np.expand_dims(std, list(range(5 - std.ndim))) if std.ndim < 5 else std
        self.mean_std_shape = np.broadcast(self.mean, self.std).shape

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            b = np.broadcast(self.mean, self.std, sample)
            self.outputs[idx][idx_] = sample + np.random.normal(np.broadcast_to(self.mean, b.shape), np.broadcast_to(self.std, b.shape))

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        output_shapes = []
        for shape in self.connections[idx][0].shapes:
            new_shape = []
            for shape_, mean_std_shape_ in zip(shape, self.mean_std_shape):
                if shape_ is None:
                    assert mean_std_shape_ == 1, "Since the shape is unknown for this axis, the shape of the broadcasted mean and std must be one, such that it can broadcast to any shape."
                    new_shape.append(None)

                elif shape_ == 1:
                    new_shape.append(mean_std_shape_)

                else:
                    assert shape_ == mean_std_shape_ or mean_std_shape_ == 1, "Shape of sample is not broadcastable with shape of mean and/or std."
                    new_shape.append(shape_)

            output_shapes.append(tuple(new_shape))

        return output_shapes

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
                        cval = self.cval[idx] if isinstance(self.cval, (tuple, list)) else self.cval
                        cval = cval[feature_i] if isinstance(cval, (tuple, list)) else cval
                        transformed_sample[batch_i, ..., feature_i] = transformations.affine_deformation(
                            sample[batch_i, ..., feature_i],
                            self.backward_affine,
                            cval=cval,
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
                shear=[np.random.normal(0, w) if self.width_as_std[0] else random.uniform(-w, w) for w in self.shear_window_width] * (3 if len(self.shear_window_width) == 1 else 1),
                rotation=[np.random.normal(0, w) if self.width_as_std[1] else random.uniform(-w, w) for w in self.rotation_window_width] * (3 if len(self.rotation_window_width) == 1 else 1),
                translation=[np.random.normal(0, w) if self.width_as_std[2] else random.uniform(-w, w) for w in self.translation_window_width] * (3 if len(self.translation_window_width) == 1 else 1),
                scaling=[1 + (np.random.normal(0, w) if self.width_as_std[3] else random.uniform(-w, w)) for w in self.scaling_window_width] * (3 if len(self.scaling_window_width) == 1 else 1),
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
            if self.nonzero == "p":
                values, counts = np.unique(self.reference_connection[0], return_counts=True)
                p = 1 / counts / np.sum(1 / counts)
                self.coordinates, coordinates_dict = [], {}
                for v in np.random.choice(values, size=self.ncrops, p=p):
                    if v not in coordinates_dict:
                        coordinates_dict[v] = np.where(self.reference_connection[0] == v)

                    c = random.randrange(len(coordinates_dict[v][0]))
                    self.coordinates.append((coordinates_dict[v][1][c], coordinates_dict[v][2][c], coordinates_dict[v][3][c]))
                    
            elif random.random() < self.nonzero:
                self.coordinates = list(zip(*np.nonzero(np.any(self.reference_connection[0] != 0, axis=(0, -1)))))
                if len(self.coordinates) > 0 and self.ncrops is not None:
                    self.coordinates = [random.choice(self.coordinates) for _ in range(self.ncrops)]

            else:
                self.coordinates = [tuple([random.choice(range(self.reference_connection[0].shape[i])) for i in range(1, 4)]) for _ in range(np.prod(self.reference_connection[0].shape[1:4]) if self.ncrops is None else self.ncrops)]

            if len(self.coordinates) > 0:
                self.n = len(self.coordinates)

            else:
                raise StopIteration


class RandomCropMultiClassEqualSampling(Crop):
    def __init__(self, reference_connection, segment_size, nonzero=False, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, shuffle=False, **kwargs):
        super(RandomCropMultiClassEqualSampling, self).__init__(reference_connection, segment_size,subsample_factors, default_value, prefilter, **kwargs)
        self.nonzero = nonzero
        self.shuffle = shuffle

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            nb_classes = self.reference_connection[0].shape[4]
            assert self.ncrops % nb_classes == 0 and self.ncrops >= 1, "Please make sure n is a whole multiple of the number of classes!"
            n_crops_per_class = int(np.round(self.ncrops / nb_classes))
            for class_ in range(nb_classes):
                reference = self.reference_connection[0][0, :, :, :, class_]
                if self.nonzero:
                    coordinates = list(zip(*np.nonzero(reference)))
                    if len(coordinates) > 0 and self.ncrops is not None:
                         coordinates = [random.choice(coordinates) for _ in range(n_crops_per_class)]

                else:
                    coordinates = [tuple([random.choice(range(reference.shape[i])) for i in range(1, 4)]) for _ in range(np.prod(reference.shape[1:4]) if self.ncrops is None else n_crops_per_class)]

                if class_ == 0:
                    self.coordinates = coordinates

                else:
                    self.coordinates += coordinates

                if self.shuffle:
                    idx = random.sample(range(len(self.coordinates)), len(self.coordinates))
                    self.coordinates = [self.coordinates[i] for i in idx]

            if len(self.coordinates) > 0:
                self.n = len(self.coordinates)

            else:
                raise StopIteration


class GridCrop(Crop):
    def __init__(self, reference_connection, segment_size, n=None, grid_size=None, strides=None, nonzero=False, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, axes=(1, 2, 3), **kwargs):
        super(GridCrop, self).__init__(reference_connection, segment_size, subsample_factors, default_value, prefilter, n=n, **kwargs)
        self.grid_size = segment_size if grid_size is None else grid_size
        self.strides = self.grid_size if strides is None else strides
        self.nonzero = nonzero
        assert 0 not in axes and 4 not in axes, "Only the spatial dimensions (axes=1, 2, 3) can be used as a grid."
        self.axes = axes

    def _randomize(self):
        if not hasattr(self, "axes"):
            self.axes = (1, 2, 3)

        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            self.coordinates = []
            for x in range(0, self.reference_connection[0].shape[1] + self.strides[0] + 1, self.strides[0]) if 1 in self.axes else [self.reference_connection[0].shape[1] // 2 - self.grid_size[0] // 2]:
                for y in range(0, self.reference_connection[0].shape[2] + self.strides[1] + 1, self.strides[1]) if 2 in self.axes else [self.reference_connection[0].shape[2] // 2 - self.grid_size[1] // 2]:
                    for z in range(0, self.reference_connection[0].shape[3] + self.strides[2] + 1, self.strides[2]) if 3 in self.axes else [self.reference_connection[0].shape[3] // 2 - self.grid_size[2] // 2]:
                        if random.random() < self.nonzero:
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


class GeometricCrop(Crop):
    def __init__(self, reference_connection, segment_size, subsample_factors=(1, 1, 1), default_value=0, prefilter=None, **kwargs):
        super(GeometricCrop, self).__init__(reference_connection, segment_size, subsample_factors, default_value, prefilter, **kwargs)

    def _randomize(self):
        if self.n_ == 0:
            assert all([np.array_equal(self.reference_connection[0].shape[1:4], sample.shape[1:4]) for connection in self.connections for sample in connection[0] if sample is not None])
            nonzero_indices = np.where(np.any(self.reference_connection[0] != 0, axis=(0, 4)))
            if len(nonzero_indices[0]) == 0:
                raise StopIteration

            else:
                self.coordinates = [tuple([int(np.mean(idx)) for idx in nonzero_indices])] * self.n


class KerasModel(Transformer):
    def __init__(self, keras_model, output_affines=None, output_to_input=0, **kwargs):
        super(KerasModel, self).__init__(**kwargs)
        self.keras_model = keras_model
        self.output_affines = output_affines if isinstance(output_affines, list) else [output_affines] * len(self.keras_model.outputs)
        self.output_to_input = output_to_input if isinstance(output_to_input, list) else [output_to_input] * len(self.keras_model.outputs)
        assert len(self.output_affines) == len(self.output_to_input) == len(self.keras_model.outputs)

    def _update_idx(self, idx):
        if not hasattr(self, "output_to_input"):
            self.output_to_input = [0] * len(self.keras_model.outputs)

        y = self.keras_model.predict(self.connections[idx][0].get())
        y = y if isinstance(y, list) else [y]
        for idx_, (y_, output_affine, output_to_input) in enumerate(zip(y, self.output_affines, self.output_to_input)):
            if output_affine is None:
                output_affine = Sample.update_affine(translation=[-(out_shape // 2) + (in_shape // 2) for in_shape, out_shape in zip(self.connections[idx][0][output_to_input].shape[1:4], y_.shape[1:4])])

            self.outputs[idx][idx_] = Sample(y_, Sample.update_affine(self.connections[idx][0][output_to_input].affine, transformation_matrix=output_affine))

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

                    else:
                        transformations.put(self.outputs[idx][idx_][0, ...], sample[i, ...], coordinates=coordinates)  # put function modifies in-place
                        continue

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
                        self.output_array_counts[idx][idx_] = np.full(self.reference_connection[idx_].shape[:4] + (1,), 1e-7)


class ToCategorical(Transformer):
    """Transforms integer class labels to categorical labels.
    """
    def __init__(self, nb_classes, **kwargs):
        """Construct a new ToCategorical transformer.

        Parameters
        ----------
        nb_classes: int
            Number of classes.
        **kwargs:
            Additional keyword arguments passed to Transformer.
        """

        super(ToCategorical, self).__init__(**kwargs)
        self.nb_classes = nb_classes

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            categorical_sample = to_categorical(sample, num_classes=self.nb_classes)
            self.outputs[idx][idx_] = Sample(categorical_sample, sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [output_shape[:-1] + (self.nb_classes,) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass


class ArgMax(Transformer):
    """Transformer that returns the indices of the maximum values along the last axis of the input.
    """

    def __init__(self, **kwargs):
        """Initializes an instance of ArgMax transformer.

        Parameters:
        -----------
        **kwargs
            Additional arguments to pass to the `Transformer` constructor.
        """

        super(ArgMax, self).__init__(**kwargs)

    def _update_idx(self, idx):
        for idx_, sample in enumerate(self.connections[idx][0]):
            self.outputs[idx][idx_] = Sample(np.argmax(sample, axis=-1, keepdims=True), sample.affine)

    def _calculate_output_shape_at_idx(self, idx):
        assert len(self.connections[idx]) == 1, "This transformer accepts only a single connection at every idx."
        return [output_shape[:-1] + (1,) for output_shape in self.connections[idx][0].shapes]

    def _randomize(self):
        pass
