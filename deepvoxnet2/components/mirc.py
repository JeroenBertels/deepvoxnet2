"""The Mirc, Dataset, Case, Record, and Modality classes form a hierarchical structure for working with medical image data.

The Mirc class represents a collection of datasets, each containing a set of cases, and each case comprising one or more image records, each with one or more image modalities.
The Dataset class encapsulates a collection of Cases, the Case class stores a collection of Records, and the Record class provides metadata for an image, such as its ID and label.
The Modality class stores the raw image data and its affine transformation, with different subclasses depending on the type of data source.
"""

import os
import json
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from deepvoxnet2.components.sample import Sample
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


class SortedDict(dict):
    """A dictionary that maintains items in sorted order.

    This dictionary subclass maintains items in sorted order based on the keys. The items
    are sorted in ascending order by default, but a custom sorting function can be provided.

    Methods
    -------
    __iter__() -> iterator
        Returns an iterator over the keys of the dictionary in sorted order.
    items() -> iterator
        Returns an iterator over the (key, value) pairs of the dictionary in sorted order.
    keys() -> list
        Returns a list of the keys of the dictionary in sorted order.
    values() -> list
        Returns a list of the values of the dictionary in sorted order.
    """

    def __init__(self):
        """Constructor for the SortedDict.

        Parameters
        ----------
        *args : positional arguments, optional
            Positional arguments to be passed to the dict constructor.
        **kwargs : keyword arguments, optional
            Keyword arguments to be passed to the dict constructor.
        """

        super(SortedDict, self).__init__()

    def __iter__(self):
        """Returns an iterator over the keys of the dictionary in sorted order.

        Returns
        -------
        iterator
            An iterator over the keys of the dictionary in sorted order.
        """

        return iter(sorted(super(SortedDict, self).__iter__()))

    def items(self):
        """Returns an iterator over the (key, value) pairs of the dictionary in sorted order.

        Returns
        -------
        iterator
            An iterator over the (key, value) pairs of the dictionary in sorted order.
        """

        return iter((k, self[k]) for k in self)

    def keys(self):
        """Returns a list of the keys of the dictionary in sorted order.

        Returns
        -------
        list
            A list of the keys of the dictionary in sorted order.
        """

        return list(self)

    def values(self):
        """Returns a list of the values of the dictionary in sorted order.

        Returns
        -------
        list
            A list of the values of the dictionary in sorted order.
        """

        return [self[k] for k in self]


class Mirc(SortedDict):
    """Class representing a MIRC (Multi-institutional Radiomics Comparison) dataset.

    Parameters
    ----------
    datasets : tuple of Dataset
        The datasets to include in the Mirc object.
    """

    def __init__(self, *datasets):
        """Initializes the Mirc object.

        Parameters
        ----------
        datasets : tuple of Dataset
            The datasets to include in the Mirc object.
        """

        super(Mirc, self).__init__()
        for dataset in datasets:
            self.add(dataset)

    def __add__(self, other):
        """Combines two Mirc objects into one.

        Parameters
        ----------
        other : Mirc
            The Mirc object to be added.

        Returns
        -------
        mirc : Mirc
            The combined Mirc object.
        """

        mirc = Mirc()
        for dataset_id in self:
            mirc.add(self[dataset_id])

        for dataset_id in other:
            mirc.add(other[dataset_id])

        return mirc

    def add(self, dataset):
        """Adds a dataset to the Mirc object.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be added.
        """

        assert dataset.dataset_id not in self
        self[dataset.dataset_id] = dataset

    def get_dataset_ids(self):
        """Returns the IDs of all datasets in the Mirc object.

        Returns
        -------
        list
            A list of all dataset IDs.
        """

        return [dataset_id for dataset_id in self]

    def get_case_ids(self):
        """Returns the IDs of all cases in the Mirc object.

        Returns
        -------
        list
            A list of all case IDs.
        """

        return sorted(set([case_id for dataset_id in self for case_id in self[dataset_id]]))

    def get_record_ids(self):
        """Returns the IDs of all records in the Mirc object.

        Returns
        -------
        list
            A list of all record IDs.
        """

        return sorted(set([record_id for dataset_id in self for case_id in self[dataset_id] for record_id in self[dataset_id][case_id]]))

    def get_modality_ids(self):
        """Returns the IDs of all modalities in the Mirc object.

        Returns
        -------
        list
            A list of all modality IDs.
        """

        return sorted(set([modality_id for dataset_id in self for case_id in self[dataset_id] for record_id in self[dataset_id][case_id] for modality_id in self[dataset_id][case_id][record_id]]))

    def get_df(self, modality_id, custom_modality_id=None):
        """Returns a Pandas DataFrame containing a specific modality for all records.

        Parameters
        ----------
        modality_id : str
            The ID of the modality to include in the DataFrame.
        custom_modality_id : str, optional
            A custom ID for the modality in the DataFrame. Defaults to None.

        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame containing the specified modality for all records.
        """

        custom_modality_id = modality_id if custom_modality_id is None else custom_modality_id
        indices = pd.MultiIndex.from_tuples([(dataset_id, case_id, record_id) for dataset_id in self for case_id in self[dataset_id] for record_id in self[dataset_id][case_id]], names=["dataset_id", "case_id", "record_id"])
        columns = pd.MultiIndex.from_tuples([(custom_modality_id,)], names=["modality_id"])
        df = pd.DataFrame(index=indices, columns=columns)
        for dataset_id in self:
            for case_id in self[dataset_id]:
                for record_id in self[dataset_id][case_id]:
                    df.at[(dataset_id, case_id, record_id), (custom_modality_id,)] = self[dataset_id][case_id][record_id][modality_id].load()

        return df

    def mean_and_std(self, modality_id, n=None, clipping=(-np.inf, np.inf), return_histogram=False, fillna=None, exclude_clipping=True):
        """Calculate the mean and standard deviation of a modality across all records in this Mirc object.

        Parameters
        ----------
        modality_id : str
            The modality for which to calculate the mean and standard deviation.
        n : int or None, optional
            The number of records to sample, or None to use all records. Default is None.
        clipping : tuple of floats, optional
            The minimum and maximum values to include in the analysis. Default is (-np.inf, np.inf).
        return_histogram : bool, optional
            Whether to also return a histogram of the values. Default is False.
        fillna : float or None, optional
            If specified, missing values will be filled with this value. If None, missing values will be excluded. Default is None.
        exclude_clipping : bool, optional
            Whether to exclude clipped values from the analysis. Default is True.

        Returns
        -------
        tuple of floats or tuple of floats and numpy array
            If return_histogram is False, returns a tuple of the mean and standard deviation of the modality. If return_histogram is True, returns a tuple of the mean, standard deviation, and histogram of the modality values.
        """

        assert modality_id in self.get_modality_ids(), "The requested modality_id is not present in this Mirc object."
        checkna = False
        count, values = 0, []
        for dataset_id in self:
            for case_id in self[dataset_id]:
                for record_id in self[dataset_id][case_id]:
                    if n is None or count < n:
                        modality = np.clip(self[dataset_id][case_id][record_id][modality_id].load(), *clipping)
                        modality = modality[modality != clipping[0]]
                        modality = modality[modality != clipping[1]]
                        if np.isnan(modality).any():
                            checkna = True

                        if fillna is not None:
                            modality[np.isnan(modality)] = fillna

                        else:
                            modality = modality[np.logical_not(np.isnan(modality))]

                        if exclude_clipping:
                            modality = modality[(modality != clipping[0]) * (modality != clipping[1])]

                        values += list(modality)
                        count += 1

        if checkna:
            print(f"WARNING: NaNs encountered in {modality_id}!")

        if return_histogram:
            return np.mean(values, dtype=np.float64), np.std(values, dtype=np.float64), np.histogram(values)

        else:
            return np.mean(values, dtype=np.float64), np.std(values, dtype=np.float64)

    def inspect(self, modality_ids=None, check_affines_and_shapes=True, ns=None, clippings=(-np.inf, np.inf), fillnas=None, exclude_clippings=True, rtol=1e-5, atol=1e-8, only_check_affines_and_shapes=False):
        """Inspect the MIRC object to examine the shape, voxel size, and content of the modalities.

        Parameters
        ----------
        modality_ids : list or str or None, optional
            A list of modality IDs to inspect. If not provided, all available modalities will be inspected. Default is None.
        check_affines_and_shapes : bool, optional
            Whether to check the affines and shapes of the selected modalities. Default is True.
        ns : int or list of int, optional
            The number of samples to include in the inspection. If a single integer is provided, it will be used for all selected modalities. Default is None, which means all samples will be used.
        clippings : tuple or list of tuple, optional
            A tuple of minimum and maximum values to clip the modality values to. If a list of tuples is provided, it should have the same length as the number of selected modalities. Default is (-np.inf, np.inf).
        fillnas : float or list of float, optional
            The value to fill NaNs with in the modality. If a list of floats is provided, it should have the same length as the number of selected modalities. Default is None.
        exclude_clippings : bool or list of bool, optional
            Whether to exclude the clipped values from the inspection. If a list of booleans is provided, it should have the same length as the number of selected modalities. Default is True.
        rtol : float, optional
            The relative tolerance parameter used in checking the affines of the selected modalities. Default is 1e-5.
        atol : float, optional
            The absolute tolerance parameter used in checking the affines of the selected modalities. Default is 1e-8.
        only_check_affines_and_shapes : bool, optional
            Whether to only check the affines and shapes of the selected modalities. Default is False.

        Returns
        -------
        None
            This method only displays the inspection results and does not return anything.

        Raises
        ------
        AssertionError
            If any of the input arguments are inconsistent or invalid.
            If any of the requested modality IDs are not present in the MIRC object.
            If the affine or shape of any selected modality is not equal to that of the first selected modality.
        """

        if modality_ids is None:
            modality_ids = self.get_modality_ids()

        modality_ids = modality_ids if isinstance(modality_ids, list) else [modality_ids]
        ns = ns if isinstance(ns, list) else [ns] * len(modality_ids)
        clippings = clippings if isinstance(clippings, list) else [clippings] * len(modality_ids)
        fillnas = fillnas if isinstance(fillnas, list) else [fillnas] * len(modality_ids)
        exclude_clippings = exclude_clippings if isinstance(exclude_clippings, list) else [exclude_clippings] * len(modality_ids)
        assert len(modality_ids) == len(ns) == len(clippings) == len(fillnas) == len(exclude_clippings), "Inconsistent input arguments."
        dataset_ids = self.get_dataset_ids()
        print(f"Total number of different dataset ids: {len(dataset_ids)}")
        case_ids = self.get_case_ids()
        print(f"Total number of different case ids: {len(case_ids)}")
        record_ids = self.get_record_ids()
        print(f"Total number of different record ids: {len(record_ids)}")
        all_modality_ids = self.get_modality_ids()
        print(f"Total number of different modality ids: {len(modality_ids)}")
        assert all([modality_id in all_modality_ids for modality_id in modality_ids]), "Some of the requested modality_ids are not present in this Mirc object."
        if check_affines_and_shapes:
            img_sizes = []
            voxel_sizes = []
            for dataset_id in dataset_ids:
                for case_id in self[dataset_id]:
                    for record_id in self[dataset_id][case_id]:
                        reference_shape, reference_affine = self[dataset_id][case_id][record_id][modality_ids[0]].shape, self[dataset_id][case_id][record_id][modality_ids[0]].affine
                        for i, modality_id in enumerate(modality_ids):
                            modality_shape, modality_affine = self[dataset_id][case_id][record_id][modality_id].shape, self[dataset_id][case_id][record_id][modality_id].affine
                            assert np.allclose(modality_affine, reference_affine, rtol=rtol, atol=atol), f"Affine of {dataset_id}-{case_id}-{record_id}-{modality_id} is not equal to the reference from {modality_ids[0]}."
                            assert np.array_equal(reference_shape[1:4], modality_shape[1:4])
                            img_sizes.append(modality_shape[1:4])
                            voxel_sizes.append([s.round(2) for s in np.linalg.norm(modality_affine[0][:3, :3], 2, axis=0)])

            fig, axs = plt.subplots(3, 3, figsize=(9, 10))
            for i in range(3):
                img_sizes_dim_i = list(zip(*img_sizes))[i]
                axs[0, i].hist(img_sizes_dim_i)
                axs[0, i].set_title(f"mod. shape d{i} (# vxls)")
                voxel_sizes_dim_i = list(zip(*voxel_sizes))[i]
                axs[1, i].hist(voxel_sizes_dim_i)
                axs[1, i].set_title(f"vxl size d{i} (mm)")
                counts, bins = np.histogram([i * j for i, j in zip(img_sizes_dim_i, voxel_sizes_dim_i)])
                axs[2, i].bar(bins[:-1], counts, width=np.array(bins[1:] - bins[:-1]), align="edge")
                axs[2, i].set_title(f"mod. shape d{i} (mm)")

            plt.suptitle(" + ".join(dataset_ids))
            plt.show()

        if not only_check_affines_and_shapes:
            n_cols = min(len(modality_ids), 3)
            n_rows = int(np.ceil(len(modality_ids) / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3 + 1))
            axs = [axs] if n_rows * n_cols == 1 else axs
            for i, modality_id in enumerate(modality_ids):
                mean, std, (counts, bins) = self.mean_and_std(modality_id, return_histogram=True, n=ns[i], clipping=clippings[i], fillna=fillnas[i], exclude_clipping=exclude_clippings[i])
                print(f"Mean and standard deviation of '{modality_id}': ({mean}, {std})")
                if n_rows == 1:
                    axs[i].bar(bins[:-1], counts, width=np.array(bins[1:] - bins[:-1]), align="edge")
                    axs[i].set_title(f"{modality_id} (n={np.sum(counts)})")

                else:
                    axs[i // n_cols, i % n_cols].bar(bins[:-1], counts, width=np.array(bins[1:] - bins[:-1]), align="edge")
                    axs[i // n_cols, i % n_cols].set_title(f"{modality_id} (n={np.sum(counts)})")

            plt.suptitle(" + ".join(dataset_ids))
            plt.show()

    @staticmethod
    def from_nnUNetv2_dataset(*nnUNetv2_dataset_args, **nnUNetv2_dataset_kwargs):
        train_dataset, val_dataset, test_dataset = Dataset.from_nnUNetv2_dataset(*nnUNetv2_dataset_args, **nnUNetv2_dataset_kwargs)
        return Mirc(train_dataset), Mirc(val_dataset), Mirc(test_dataset)


class Dataset(SortedDict):
    """A class to represent a dataset, which is a collection of cases.

    Parameters
    ----------
    dataset_id : str or None, optional
        The ID of the dataset.
    dataset_dir : str or None, optional
        The directory where the dataset is stored.

    Attributes
    ----------
    dataset_id : str or None
        The ID of the dataset.
    dataset_dir : str or None
        The directory where the dataset is stored.

    Methods
    -------
    add(case)
        Adds a case to the dataset.
    """

    def __init__(self, dataset_id=None, dataset_dir=None):
        super(Dataset, self).__init__()
        self.dataset_id = dataset_id
        self.dataset_dir = dataset_dir

    def add(self, case):
        """Adds a case to the dataset.

        Parameters
        ----------
        case : Case
            The case to add to the dataset.

        Raises
        ------
        AssertionError
            If the case ID already exists in the dataset.
        """

        assert case.case_id not in self
        self[case.case_id] = case

    @staticmethod
    def from_nnUNetv2_dataset(dataset_dir, fold="all", nb_folds=5, file_extension=".nii.gz"):
        assert file_extension in [".nii", ".nii.gz"], "Currently we only support nifti files."
        images_dir_trainval = os.path.join(dataset_dir, "imagesTr")
        labels_dir_trainval = os.path.join(dataset_dir, "labelsTr")
        images_dir_test = os.path.join(dataset_dir, "imagesTs")
        labels_dir_test = os.path.join(dataset_dir, "labelsTs")
        case_ids_trainval = ["_".join(filename.split("_")[:-1]) for filename in sorted(os.listdir(images_dir_trainval)) if not filename.startswith(".") and filename.endswith(file_extension)]
        if os.path.isdir(images_dir_test):
            case_ids_test = ["_".join(filename.split("_")[:-1]) for filename in sorted(os.listdir(images_dir_test)) if not filename.startswith(".") and filename.endswith(file_extension)]
        
        else:
            case_ids_test = []

        dataset_id = os.path.split(dataset_dir)[1]

        if os.path.isfile(os.path.join(dataset_dir, "splits_final.json")):
            splits_final_path = os.path.join(dataset_dir, "splits_final.json")
        
        elif os.path.isfile(os.path.join(dataset_dir, "splits_final.json").replace("nnUNet_raw", "nnUNet_preprocessed")):
            splits_final_path = os.path.join(dataset_dir, "splits_final.json").replace("nnUNet_raw", "nnUNet_preprocessed")
        
        else:
            splits_final_path = None
        
        if splits_final_path is not None:
            with open(splits_final_path, "r") as f:
                splits_final = json.load(f)

            assert nb_folds is None or nb_folds == len(splits_final)
            nb_folds = len(splits_final)
        
        if fold == "all":
            train_dataset_id = f"{dataset_id}_train"
            val_dataset_id = f"{dataset_id}_val"
            case_ids_train = case_ids_trainval
            case_ids_val = []
        
        else:
            assert nb_folds is not None and nb_folds > 1 and fold in range(nb_folds), "Please make sure to specify a fold that's smaller than the total number of folds specified in nb_folds > 1."
            train_dataset_id = f"{dataset_id}_train_{fold}"
            val_dataset_id = f"{dataset_id}_val_{fold}"
            if splits_final_path is None:
                kfold = KFold(n_splits=nb_folds, shuffle=True, random_state=12345)
                for fold_i, (train_idx, val_idx) in enumerate(kfold.split(case_ids_trainval)):
                    if fold_i == fold:
                        case_ids_train = [case_ids_trainval[i] for i in train_idx]
                        case_ids_val = [case_ids_trainval[i] for i in val_idx]
                        break
            
            else:
                case_ids_train = splits_final[fold]["train"]
                case_ids_val = splits_final[fold]["val"]
                assert set(case_ids_trainval) == set(case_ids_train + case_ids_val)
        
        train_dataset, val_dataset = Dataset(train_dataset_id), Dataset(val_dataset_id)
        for case_id in case_ids_trainval:
            case = Case(case_id)
            record = Record("record_0")
            assert os.path.isfile(os.path.join(labels_dir_trainval, f"{case_id}{file_extension}"))
            record.add(NiftiFileModality("label", os.path.join(labels_dir_trainval, f"{case_id}{file_extension}")))
            for i, channel_path in enumerate(sorted(glob.glob(os.path.join(images_dir_trainval, f"{case_id}_*.nii.gz")))):
                record.add(NiftiFileModality("channel_{:04}".format(i), channel_path))

            case.add(record)
            if case_id in case_ids_train:
                train_dataset.add(case)
            
            else:
                assert case_id in case_ids_val
                val_dataset.add(case)

        test_dataset = Dataset(f"{dataset_id}_test")
        for case_id in case_ids_test:
            case = Case(case_id)
            record = Record("record_0")
            if os.path.isdir(labels_dir_test):
                assert os.path.isfile(os.path.join(labels_dir_test, f"{case_id}{file_extension}"))
                record.add(NiftiFileModality("label", os.path.join(labels_dir_test, f"{case_id}{file_extension}")))

            for i, channel_path in enumerate(sorted(glob.glob(os.path.join(images_dir_test, f"{case_id}_*.nii.gz")))):
                record.add(NiftiFileModality("channel_{:04}".format(i), channel_path))
            
            case.add(record)
            test_dataset.add(case)

        return train_dataset, val_dataset, test_dataset


class Case(SortedDict):
    """A dictionary subclass that stores medical imaging data for a single case.

    Parameters
    ----------
    case_id : str, optional
        A unique identifier for this case.
    case_dir : str, optional
        The path to the directory containing the case data.

    Attributes
    ----------
    case_id : str or None
        A unique identifier for this case.
    case_dir : str or None
        The path to the directory containing the case data.
    """

    def __init__(self, case_id=None, case_dir=None):
        """Initialize a new Case object.

        Parameters
        ----------
        case_id : str, optional
            A unique identifier for this case.
        case_dir : str, optional
            The path to the directory containing the case data.
        """

        super(Case, self).__init__()
        self.case_id = case_id
        self.case_dir = case_dir

    def add(self, record):
        """Add a new Record object to this Case.

        Parameters
        ----------
        record : Record
            The Record object to add to this Case.

        Raises
        ------
        AssertionError
            If the record_id of the Record object is already in this Case.
        """

        assert record.record_id not in self
        self[record.record_id] = record


class Record(SortedDict):
    """A class to represent a single record within a case in a dataset. A record contains one or more modalities.

    Args:
        record_id (str): A unique identifier for this record.
        record_dir (str): The path to the directory containing the files for this record.

    Attributes:
        record_id (str): A unique identifier for this record.
        record_dir (str): The path to the directory containing the files for this record.

    Methods:
        add(modality): Adds a modality to this record.
    """

    def __init__(self, record_id=None, record_dir=None):
        """Initializes a new instance of the Record class.

        Args:
            record_id (str): A unique identifier for this record.
            record_dir (str): The path to the directory containing the files for this record.
        """

        super(Record, self).__init__()
        self.record_id = record_id
        self.record_dir = record_dir

    def add(self, modality):
        """Adds a modality to this record.

        Args:
            modality (Modality): The modality to add to this record.

        Raises:
            AssertionError: If the modality_id of the given modality is already present in this record.
        """

        assert modality.modality_id not in self
        self[modality.modality_id] = modality


class Modality(object):
    """A class to represent a modality.

    Attributes
    ----------
    modality_id : str
        The identifier for the modality.
    modality_dir : str or None, optional
        The directory path for the modality data, by default None.

    Methods
    -------
    load()
        An abstract method to load the modality data.
    """

    def __init__(self, modality_id, modality_dir=None, shape=None, affine=None):
        """Constructor of the Modality class.

        Parameters
        ----------
        modality_id : str
            The identifier for the modality.
        modality_dir : str or None, optional
            The directory path for the modality data, by default None.
        """

        self.modality_id = modality_id
        self.modality_dir = modality_dir
        self.shape = Sample.ln_to_l5(shape)
        self.affine = Sample.aff_to_144(affine)

    def load(self):
        """An abstract method to load the modality data.

        Raises
        ------
        NotImplementedError
            This method must be implemented in the child classes.
        """

        raise NotImplementedError


class ArrayModality(Modality):
    """A class that represents a modality loaded from an array.

    Parameters
    ----------
    modality_id : str
        The identifier of the modality.
    array : numpy.ndarray
        The array containing the modality data.
    affine : numpy.ndarray, optional
        The affine transformation matrix for the modality data. Default is None.

    Attributes
    ----------
    modality_id : str
        The identifier of the modality.
    array : numpy.ndarray
        The array containing the modality data.
    affine : numpy.ndarray, optional
        The affine transformation matrix for the modality data.

    Methods
    -------
    load():
        Returns a Sample object with the modality data and affine matrix.
    """

    def __init__(self, modality_id, array, affine=None):
        """Initializes the ArrayModality object.

        Parameters
        ----------
        modality_id : str
            The identifier of the modality.
        array : numpy.ndarray
            The array containing the modality data.
        affine : numpy.ndarray, optional
            The affine transformation matrix for the modality data. Default is None.
        """

        super(ArrayModality, self).__init__(modality_id, shape=array.shape, affine=affine)
        self.array = array

    def load(self):
        """Returns a Sample object with the modality data and affine matrix.

        Returns
        -------
        Sample
            A Sample object with the modality data and affine matrix.
        """

        return Sample(self.array, self.affine)


class NiftyModality(Modality):
    """Represents a modality from a NIfTI image.

    Parameters:
    -----------
    modality_id: str
        Identifier for the modality.
    nifty: nib.Nifti1Image
        NIfTI image object containing the modality data.

    Attributes:
    -----------
    nifty: nib.Nifti1Image
        NIfTI image object containing the modality data.

    Methods:
    --------
    load():
        Returns the loaded modality as a Sample object, which is an ndarray subclass with an associated affine.
    """

    def __init__(self, modality_id, nifty, caching="unchanged"):
        """Initializes a new instance of the NiftyModality class.

        Parameters:
        -----------
        modality_id: str
            Identifier for the modality.
        nifty: nib.Nifti1Image
            NIfTI image object containing the modality data.
        """

        super(NiftyModality, self).__init__(modality_id, shape=nifty.shape, affine=nifty.affine)
        self.nifty = nifty
        self.caching = caching

    def load(self):
        """Returns the loaded modality as a Sample object, which is an ndarray subclass with an associated affine.

        Returns:
        --------
        Sample:
            ndarray subclass with an associated affine representing the loaded modality data.
        """

        return Sample(self.nifty.get_fdata(caching=self.caching), self.affine)


NiftiModality = NiftyModality


class NiftyFileModality(Modality):
    """A class representing a single modality that is stored in a single NIfTI file.

    Parameters:
    -----------
    modality_id : str
        Unique identifier of the modality.
    file_path : str
        Path to the NIfTI file containing the modality.

    Returns:
    --------
    NiftyFileModality : NiftyFileModality object
        A NiftyFileModality object representing the modality stored in the NIfTI file.

    Note:
    -----
    The NIfTI file should be 3D or 4D. In the case of a 4D NIfTI file, the fourth dimension is interpreted as the feature dimension.
    """

    def __init__(self, modality_id, file_path, caching="unchanged"):
        """Initialize a new `NiftyFileModality` object.

        Parameters
        ----------
        modality_id : str
            The modality ID.
        file_path : str
            The path to the NIfTI file containing the modality data.

        Returns
        -------
        NiftyFileModality
            The new `NiftyFileModality` object.
        """

        nifty = nib.load(file_path)
        super(NiftyFileModality, self).__init__(modality_id, os.path.dirname(file_path), shape=nifty.shape, affine=nifty.affine)
        self.file_path = file_path
        self.nifty = nifty
        self.caching = caching

    def load(self):
        """Load the modality data from the NIfTI file.

        Returns:
        --------
        Sample : Sample object
            A Sample object representing the modality data.
        """

        # return Sample(self.nifty.get_fdata(caching=self.caching), self.affine)
        return Sample(np.asarray(self.nifty.dataobj), self.affine)


NiftiFileModality = NiftyFileModality


class NiftyFileMultiModality(Modality):
    """A modality consisting of multiple NIfTI files that are loaded from disk.

    Parameters
    ----------
    modality_id : str
        Unique identifier for the modality.
    file_paths : list of str
        List of paths to NIfTI files to be loaded.
    axis : int, optional
        Axis along which to stack the volumes if `mode` is 'stack'. Default is -1.
    mode : str, optional
        Mode for combining the volumes. Possible values are 'stack' and 'concat'. If 'stack', the volumes are stacked
        along `axis`. If 'concat', the volumes are concatenated along the feature dimension. Default is 'stack'.

    Methods
    -------
    load()
        Load the NIfTI files and return them as a `Sample` object.

    Attributes
    ----------
    file_paths : list of str
        List of paths to NIfTI files to be loaded.
    axis : int
        Axis along which to stack the volumes if `mode` is 'stack'.
    mode : str
        Mode for combining the volumes.
    """

    def __init__(self, modality_id, file_paths, axis=-1, mode="stack", caching="unchanged"):
        """Initialize an instance of the `NiftyFileMultiModality` class.

        Args:
            modality_id (str): The identifier for the modality.
            file_paths (list): A list of file paths to NIfTI files for this modality.
            axis (int, optional): The axis along which to stack/concatenate the data in the NIfTI files. Defaults to -1.
            mode (str, optional): The mode to use when loading the NIfTI files. Options are 'stack' and 'concat'.
                If 'stack', the data in the files will be stacked along the specified axis. If 'concat', the data
                in the files will be concatenated along the specified axis. Defaults to 'stack'.
        """

        assert axis in [0, -1], "For now only first and last axes are supported!"
        niftys = [nib.load(file_path) for file_path in file_paths]
        assert all([np.allclose(niftys[0].affine, nii.affine) for nii in niftys[1:]]), "Not all affines are equal!"
        shape = np.concatenate([nii.shape for nii in niftys], axis=axis) if mode == "concat" else np.stack([nii.shape for nii in niftys], axis=axis)
        affine = [niftys[0].affine] if mode == "concat" else [niftys[0].affine] * len(niftys)
        super(NiftyFileMultiModality, self).__init__(modality_id, shape=shape, affine=affine)
        self.niftys = niftys
        self.file_paths = file_paths
        self.axis = axis
        self.mode = mode
        self.caching = caching

    def load(self):
        """Load the NIfTI files and return them as a `Sample` object.

        Returns
        -------
        Sample
            A `Sample` object containing the loaded volumes.

        Raises
        ------
        nibabel.filebasedimages.ImageFileError
            If any of the NIfTI files cannot be loaded.
        """

        if self.mode == "concat":
            array = np.concatenate([nii.get_fdata(caching=self.caching) for nii in self.niftys], axis=self.axis)

        else:
            array = np.stack([nii.get_fdata(caching=self.caching) for nii in self.niftys], axis=self.axis)

        return Sample(array, self.affine)


NiftiFileMultiModality = NiftyFileMultiModality


class NiftyMultiModality(Modality):
    """A modality containing multiple NIfTI files.

    Parameters
    ----------
    modality_id : str
        The unique identifier of the modality.
    niftys : list of nib.Nifti1Image
        List of NIfTI images to be combined into a single modality.
    axis : int, optional
        The axis along which to concatenate or stack the NIfTI images.
        Defaults to -1.
    mode : str, optional
        The mode for combining the NIfTI images.
        Must be one of "stack" or "concat".
        Defaults to "stack".

    Raises
    ------
    ValueError
        If the mode parameter is not "stack" or "concat".
    """

    def __init__(self, modality_id, niftys, axis=-1, mode="stack", caching="unchanged"):
        """Initializes a new instance of the NiftyMultiModality class.
        """

        assert axis in [0, -1], "For now only first and last axes are supported!"
        assert all([np.allclose(niftys[0].affine, nii.affine) for nii in niftys[1:]]), "Not all affines are equal!"
        shape = np.concatenate([nii.shape for nii in niftys], axis=axis) if mode == "concat" else np.stack([nii.shape for nii in niftys], axis=axis)
        affine = [niftys[0].affine] if mode == "concat" else [niftys[0].affine] * len(niftys)
        super(NiftyMultiModality, self).__init__(modality_id, shape=shape, affine=affine)
        self.niftys = niftys
        self.axis = axis
        self.mode = mode
        self.caching = caching

    def load(self):
        """Load the NIfTI images in this modality and return them as a Sample.

        Returns
        -------
        Sample
            The NIfTI images in this modality as a Sample.
        """

        if self.mode == "concat":
            array = np.concatenate([nii.get_fdata(caching=self.caching) for nii in self.niftys], axis=self.axis)

        else:
            array = np.stack([nii.get_fdata(caching=self.caching) for nii in self.niftys], axis=self.axis)

        return Sample(array, self.affine)


NiftiMultiModality = NiftyMultiModality


class ImageFileModality(Modality):
    """Loads an image file and converts it to a numpy array.

    Parameters
    ----------
    modality_id : str
        A unique identifier for the modality.
    file_path : str
        The path to the image file.
    **kwargs :
        Additional keyword arguments passed to PIL's `Image.convert` function.

    Returns
    -------
    Sample
        A sample object containing the image data.

    Raises
    ------
    AssertionError
        If the image data has an invalid number of dimensions.

    Notes
    -----
    The image data is converted to a numpy array with dimensions (X, Y, C) or (X, Y) depending on whether the image is
    color or grayscale, respectively.

    See Also
    --------
    PIL.Image.convert : Method used to convert the image to a numpy array.
    """

    def __init__(self, modality_id, file_path, shape=None, affine=None, **kwargs):  # check documentation of Image.convert for **kwargs: e.g. mode, which can be "1" (binary), "RGB" (color), "L" grayscale
        """Initialize the ImageFileModality.

        Parameters
        ----------
        modality_id : str
            A unique identifier for the modality.
        file_path : str
            The path to the image file.
        **kwargs :
            Additional keyword arguments passed to PIL's `Image.convert` function.
        """

        super(ImageFileModality, self).__init__(modality_id, os.path.dirname(file_path), shape=shape, affine=affine)
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        """Load the image and convert it to a numpy array.

        Returns
        -------
        Sample
            A sample object containing the image data.

        Raises
        ------
        AssertionError
            If the image data has an invalid number of dimensions.
        """

        img = Image.open(self.file_path)
        if self.kwargs:
            img = img.convert(**self.kwargs)

        img = np.asarray(img)
        assert img.ndim in [2, 3], "A 2D image file can only have a maximum of three dimensions: x, y, color."
        return Sample(img if img.ndim == 2 else img[:, :, None, :])
