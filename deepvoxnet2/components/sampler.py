"""This module provides classes for sampling data.

The core concept is that an Identifier object is a general representation of an item that needs to be sampled, which can be a specific record or a Sample object.
A Sample object contains the actual data to be processed, and can be sampled via a SampleSampler object.
The Sampler class is an abstract object that samples Identifier objects. A specific type of Sampler, such as the MircSampler, can sample a specific type of Identifier, in this case a MircIdentifier (which is essentially just a Record from inside the Mirc class).
Overall, this module provides a flexible way to sample data from different sources for use in machine learning or other applications.
"""

import random
import numpy as np


class Identifier(dict):
    """A dictionary-based class to represent an identifier, which is a generic representation of an item that can be sampled.

    An identifier is represented as a dictionary of key-value pairs, where the keys are strings and the values can be of any
    type. This class inherits from Python's built-in `dict` class.

    Parameters:
    -----------
    **kwargs: arbitrary keyword arguments.
        The keyword arguments are added as key-value pairs to the `Identifier` object.

    Methods:
    --------
    __call__():
        Raises a `NotImplementedError` when called. This is meant to be overridden by child classes.
    """

    def __init__(self, **kwargs):
        """Initializes an identifier with the specified key-value pairs.

        Parameters:
        -----------
        **kwargs: A dictionary of key-value pairs representing the identifier.

        Returns:
        --------
        None.
        """

        super(Identifier, self).__init__()
        for key in kwargs:
            self[key] = kwargs[key]

    def __call__(self):
        """A callable method that samples the identifier. Not implemented in the base class.

        Returns:
        --------
        None.
        """

        raise NotImplementedError


class MircIdentifier(dict):
    """Represents an identifier for a single image in a MIRC dataset.

    Parameters:
    -----------
    mirc : Mirc, optional
        The MIRC instance where the dataset is hosted.
    dataset_id : str, optional
        The identifier of the dataset that the image belongs to.
    case_id : str, optional
        The identifier of the case that the image belongs to.
    record_id : str, optional
        The identifier of the image record.
    modality_id : str, optional
        The identifier of the modality that the image belongs to.

    Methods:
    --------
    __call__() -> Tuple[str, str, str, str]:
        Returns a tuple containing the dataset, case, record and modality ids of the image.
    """

    def __init__(self, mirc=None, dataset_id=None, case_id=None, record_id=None, modality_id=None):
        super(MircIdentifier, self).__init__(mirc=mirc, dataset_id=dataset_id, case_id=case_id, record_id=record_id, modality_id=modality_id)
        self.mirc = mirc
        self.dataset_id = dataset_id
        self.case_id = case_id
        self.record_id = record_id
        self.modality_id = modality_id
        """Initialize a new instance of MircIdentifier.

        Parameters:
        -----------
        mirc : Mirc, optional
            The MIRC instance where the dataset is hosted.
        dataset_id : str, optional
            The identifier of the dataset that the image belongs to.
        case_id : str, optional
            The identifier of the case that the image belongs to.
        record_id : str, optional
            The identifier of the image record.
        modality_id : str, optional
            The identifier of the modality that the image belongs to.
        """

    def __call__(self):
        """Returns a tuple containing the dataset, case, record and modality ids of the image.

        Returns:
        --------
        Tuple[str, str, str, str]:
            A tuple containing the dataset, case, record and modality ids of the image.
        """

        return self.dataset_id, self.case_id, self.record_id, self.modality_id


class SampleIdentifier(dict):
    """Identifier object for a Sample instance.

    Parameters
    ----------
    sample : Sample, optional
        The Sample object that is represented by this identifier.

    Attributes
    ----------
    sample : Sample, optional
        The Sample object that is represented by this identifier.

    Methods
    -------
    __call__()
        Returns a string indicating that the Identifier object represents a Sample instance.
    """

    def __init__(self, sample=None):
        """Constructor of the SampleIdentifier class.

        Parameters
        ----------
        sample : Sample, optional
            The Sample object that is represented by this identifier.
        """

        super(SampleIdentifier, self).__init__(sample=sample)
        self.sample = sample

    def __call__(self):
        """Returns a string indicating that the Identifier object represents a Sample instance.

        Returns
        -------
        str
            A string indicating that the Identifier object represents a Sample instance.
        """

        return "sample"


class Sampler(object):
    """Abstract class that samples identifiers.

    A Sampler is an abstract object that samples so called Identifier objects. This is a general representation of an item
    that needs to be sampled. Under this idea, you can have a specific type of Sampler, for example the MircSampler, which
    samples a specific type of Identifier, in this case a MircIdentifier (in essence just a Record from inside the Mirc class).

    Attributes:
    -----------
    identifiers : list, optional
        List of Identifier objects. Defaults to empty list.
    base_identifiers : list
        A copy of the identifiers that were present when the sampler was instantiated. This is used for shuffling purposes.
    shuffle : bool, optional
        Whether to shuffle the identifiers before returning them. Defaults to False.
    weights : list, optional
        List of weights to be used when sampling the identifiers. Defaults to None.

    Methods:
    --------
    __len__():
        Returns the number of identifiers in the sampler.
    __getitem__(idx):
        Returns the Identifier object at the given index in the identifiers list.
    __iter__():
        Returns an iterator over the identifiers list.
    randomize():
        Re-orders the identifiers list based on the specified sampling method, shuffle and weights.
    """

    def __init__(self, identifiers=None, shuffle=False, weights=None, n=None):
        """Initializes the Sampler object.

        Parameters:
        -----------
        identifiers : list, optional
            List of Identifier objects. Defaults to empty list.
        shuffle : bool, optional
            Whether to shuffle the identifiers before returning them. Defaults to False.
        weights : list, optional
            List of weights to be used when sampling the identifiers. Defaults to None.
        """

        self.identifiers = [] if identifiers is None else identifiers
        self.base_identifiers = self.identifiers.copy()
        self.shuffle = shuffle
        self.weights = weights
        self.n = n
        self._nb_randomized = 0
        self._randomizing = False
        self.randomize()
        
    def __len__(self):
        """Returns the number of identifiers in the sampler.

        Returns:
        --------
        int:
            Number of identifiers in the sampler.
        """
        while True:
            if not self._randomizing:
                return len(self.identifiers)

    def __getitem__(self, idx):
        """Returns the Identifier object at the given index in the identifiers list.

        Parameters:
        -----------
        idx : int
            Index of the Identifier object to be returned.

        Returns:
        --------
        Identifier:
            Identifier object at the given index in the identifiers list.
        """
        while True:
            if not self._randomizing:
                return self.identifiers[idx]

    def __iter__(self):
        """Returns an iterator over the identifiers list.

        Returns:
        --------
        Iterator:
            Iterator over the identifiers list.
        """
        while True:
            if not self._randomizing:
                return iter(self.identifiers)

    def randomize(self):
        """Re-orders the identifiers list based on the specified sampling method, shuffle and weights.

        If the `shuffle` attribute is set to True, the `identifiers` list is randomly shuffled. If `weights` are specified,
        `base_identifiers` and their corresponding `weights` are shuffled together, and the shuffled lists are used to
        populate `identifiers`.

        If `weights` are not specified, the `base_identifiers` list is simply shuffled and the resulting order is stored
        in `identifiers`.
        """

        self._randomizing = True
        self.identifiers = self.base_identifiers.copy()
        self._randomize()
        if self.shuffle:
            if self.weights is None and (self.n is None or self.n <= len(self.identifiers)):
                random.shuffle(self.identifiers)

            elif self.weights is None:
                self.identifiers = random.choices(self.identifiers, k=self.n)

            else:
                assert len(self.identifiers) == len(self.weights), "When sample weights are specified, you must specify a weight for each identifier."
                weights = [w[self._nb_randomized] if isinstance(w, (tuple, list)) else w for w in self.weights]
                sum_weights = np.sum(weights)
                weights = [w / sum_weights for w in weights]
                self.identifiers = random.choices(self.identifiers, weights=weights, k=self.n or len(self.identifiers))

        self.identifiers = self.identifiers[:self.n]
        self._nb_randomized += 1
        self._randomizing = False
    
    def _randomize(self):
        """Internal method used to randomly shuffle the `base_identifiers` into the `identifiers` list.

        This method can be overwritten by child classes that need to implement custom randomization.
        """

        pass


class MircSampler(Sampler):
    """A sampler for MIRC data.

    Parameters
    ----------
    mirc : Mirc
        A Mirc object.
    mode : str, optional
        Sampling mode, either 'per_record' or 'per_case'.
    **kwargs : dict, optional
        Keyword arguments for the `Sampler` class.

    Attributes
    ----------
    mirc : Mirc
        A Mirc object.
    mode : str
        Sampling mode, either 'per_record' or 'per_case'.
    identifiers : list of MircIdentifier
        List of `MircIdentifier` objects representing the samples to be drawn.
    base_identifiers : list of MircIdentifier
        The original list of `MircIdentifier` objects representing the samples.
    shuffle : bool
        Whether to shuffle the `identifiers` list before drawing samples.
    weights : list or array of float, optional
        Sample weights used for weighted sampling.

    Methods
    -------
    __len__()
        Return the number of identifiers in the sampler.
    __getitem__(idx)
        Get an identifier by index.
    __iter__()
        Get an iterator over the identifiers.
    randomize()
        Randomize the list of identifiers.
    _randomize()
        Private method that sets the `base_identifiers` attribute based on the `mode`.

    Raises
    ------
    NotImplementedError
        If an unsupported sampling mode is selected.
    """

    def __init__(self, mirc, mode="per_record", **kwargs):
        """Initialize a `MircSampler` instance.

        Parameters
        ----------
        mirc : Mirc
            A Mirc object.
        mode : str, optional
            Sampling mode, either 'per_record' or 'per_case'.
        **kwargs : dict, optional
            Keyword arguments for the `Sampler` class.
        """

        self.mirc = mirc
        self.mode = mode
        identifiers = []
        if self.mode == "per_record":
            for dataset_id in self.mirc:
                for case_id in self.mirc[dataset_id]:
                    for record_id in self.mirc[dataset_id][case_id]:
                        identifiers.append(MircIdentifier(self.mirc, dataset_id, case_id, record_id))

        elif self.mode == "per_case":
            for dataset_id in self.mirc:
                for case_id in self.mirc[dataset_id]:
                    identifiers.append([MircIdentifier(self.mirc, dataset_id, case_id, record_id) for record_id in self.mirc[dataset_id][case_id]])

        else:
            raise NotImplementedError
        
        super(MircSampler, self).__init__(identifiers=identifiers, **kwargs)
        if self.mode == "per_case" and not self.shuffle:
            print("Watch out in MircSampler: shuffle=False with mode='per_case' will result in always taking the first record!")

    def _randomize(self):
        """Set the `base_identifiers` attribute based on the selected sampling mode.

        Raises
        ------
        NotImplementedError
            If an unsupported sampling mode is selected.
        """

        if self.mode == "per_record":
            pass

        elif self.mode == "per_case":
            self.identifiers = [random.choice(_) for _ in self.identifiers]

        else:
            raise NotImplementedError


class SampleSampler(Sampler):
    """A `Sampler` for `Sample` objects.

    This class inherits from `Sampler` and accepts a list of `Sample` objects as input. It creates a list of `SampleIdentifier`
    objects and passes them to the base `Sampler` class.

    Parameters
    ----------
    samples : list of `Sample` objects
        The list of `Sample` objects to be sampled.
    kwargs : dict, optional
        Optional keyword arguments that can be passed to the base `Sampler` class.
    """

    def __init__(self, samples, **kwargs):
        """Initialize the SampleSampler object.

        Parameters
        ----------
        samples : list
            A list of Sample objects.
        shuffle : bool, optional
            Whether to shuffle the samples. Default is False.
        weights : list, optional
            A list of weights for each sample. Default is None.
        """

        identifiers = [SampleIdentifier(sample) for sample in samples]
        super(SampleSampler, self).__init__(identifiers, **kwargs)
