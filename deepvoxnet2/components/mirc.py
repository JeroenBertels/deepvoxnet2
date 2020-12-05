import os
import random
import numpy as np
import nibabel as nib
from deepvoxnet2.components.sample import Sample


class Identifier(dict):
    def __init__(self, mirc=None, dataset_id=None, case_id=None, record_id=None, modality_id=None):
        super(Identifier, self).__init__()
        self.mirc = mirc
        self.dataset_id = dataset_id
        self["dataset_id"] = self.dataset_id
        self.case_id = case_id
        self["case_id"] = self.case_id
        self.record_id = record_id
        self["record_id"] = self.record_id
        self.modality_id = modality_id
        self["modality_id"] = self.modality_id

    def __call__(self):
        return self.dataset_id, self.case_id, self.record_id, self.modality_id


class Sampler(object):
    def __init__(self, identifiers, shuffle=False):
        self.identifiers = identifiers
        self.shuffle = shuffle
        self.randomize()

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self, idx):
        return self.identifiers[idx]

    def __iter__(self):
        return iter(self.identifiers)

    def __add__(self, other):
        assert self.shuffle == other.shuffle
        return Sampler(self.identifiers + other.identifiers, shuffle=self.shuffle)

    def randomize(self):
        if self.shuffle:
            random.shuffle(self.identifiers)


class MircSampler(Sampler):
    def __init__(self, mirc, mode="per_record", shuffle=False):
        if mode == "per_record":
            identifiers = [Identifier(mirc, dataset_id, case_id, record_id) for dataset_id in mirc for case_id in mirc[dataset_id] for record_id in mirc[dataset_id][case_id]]

        else:
            raise NotImplementedError

        super(MircSampler, self).__init__(identifiers, shuffle)
        # self.mirc = mirc
        # self.mode = mode


class Mirc(dict):
    def __init__(self):
        super(Mirc, self).__init__()

    def __add__(self, other):
        mirc = Mirc()
        for dataset_id in self:
            mirc.add(self[dataset_id])

        for dataset_id in other:
            mirc.add(other[dataset_id])

        return mirc

    def add(self, dataset):
        assert dataset.dataset_id not in self
        self[dataset.dataset_id] = dataset

    def mean_and_std(self, modality_id, n=None, clipping=(-np.inf, np.inf)):
        count, values = 0, []
        for dataset_id in self:
            for case_id in self[dataset_id]:
                for record_id in self[dataset_id][case_id]:
                    modality = np.clip(self[dataset_id][case_id][record_id][modality_id].load(), *clipping)
                    modality = modality[modality != clipping[0]]
                    modality = modality[modality != clipping[1]]
                    values += list(modality[~np.isnan(modality)])
                    count += 1
                    if count == n:
                        return np.mean(values, dtype=np.float64), np.std(values, dtype=np.float64)

        return np.mean(values), np.std(values)


class Dataset(dict):
    def __init__(self, dataset_id=None, dataset_dir=None):
        super(Dataset, self).__init__()
        self.dataset_id = dataset_id
        self.dataset_dir = dataset_dir

    def add(self, case):
        assert case.case_id not in self
        self[case.case_id] = case


class Case(dict):
    def __init__(self, case_id=None, case_dir=None):
        super(Case, self).__init__()
        self.case_id = case_id
        self.case_dir = case_dir

    def add(self, record):
        assert record.record_id not in self
        self[record.record_id] = record


class Record(dict):
    def __init__(self, record_id=None, record_dir=None):
        super(Record, self).__init__()
        self.record_id = record_id
        self.record_dir = record_dir

    def add(self, modality):
        assert modality.modality_id not in self
        self[modality.modality_id] = modality


class Modality(object):
    def __init__(self, modality_id, modality_dir=None):
        self.modality_id = modality_id
        self.modality_dir = modality_dir

    def load(self):
        raise NotImplementedError


class ArrayModality(Modality):
    def __init__(self, modality_id, array, affine=None):
        super(ArrayModality, self).__init__(modality_id)
        self.array = array
        self.affine = affine

    def load(self):
        return Sample(self.array, self.affine)


class NiftyModality(Modality):
    def __init__(self, modality_id, nifty):
        super(NiftyModality, self).__init__(modality_id)
        self.nifty = nifty

    def load(self):
        return Sample(self.nifty.get_fdata(caching="unchanged"), self.nifty.affine)


class NiftyFileModality(Modality):
    def __init__(self, modality_id, file_path):
        super(NiftyFileModality, self).__init__(modality_id, os.path.dirname(file_path))
        self.file_path = file_path

    def load(self):
        nii = nib.load(self.file_path)
        return Sample(nii.get_fdata(caching="unchanged"), nii.affine)
