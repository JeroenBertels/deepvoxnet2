import os
import numpy as np
import nibabel as nib
from deepvoxnet2.components.sample import Sample
from PIL import Image
from matplotlib import pyplot as plt


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

    def get_dataset_ids(self):
        return sorted([dataset_id for dataset_id in self])

    def get_case_ids(self):
        return sorted(set([case_id for dataset_id in self for case_id in self[dataset_id]]))

    def get_record_ids(self):
        return sorted(set([record_id for dataset_id in self for case_id in self[dataset_id] for record_id in self[dataset_id][case_id]]))

    def get_modality_ids(self):
        return sorted(set([modality_id for dataset_id in self for case_id in self[dataset_id] for record_id in self[dataset_id][case_id] for modality_id in self[dataset_id][case_id][record_id]]))

    def mean_and_std(self, modality_id, n=None, clipping=(-np.inf, np.inf), return_histogram=False):
        assert modality_id in self.get_modality_ids(), "The requested modality_id is not present in this Mirc object."
        count, values = 0, []
        for dataset_id in self:
            for case_id in self[dataset_id]:
                for record_id in self[dataset_id][case_id]:
                    if n is None or count < n:
                        modality = np.clip(self[dataset_id][case_id][record_id][modality_id].load(), *clipping)
                        modality = modality[modality != clipping[0]]
                        modality = modality[modality != clipping[1]]
                        values += list(modality[~np.isnan(modality)])
                        count += 1

        if return_histogram:
            return np.mean(values, dtype=np.float64), np.std(values, dtype=np.float64), np.histogram(values)

        else:
            return np.mean(values, dtype=np.float64), np.std(values, dtype=np.float64)

    def inspect(self, image_modality_ids=None, metadata_modality_ids=None, **kwargs):
        dataset_ids = self.get_dataset_ids()
        print(f"Total number of different dataset ids: {len(dataset_ids)}")
        case_ids = self.get_case_ids()
        print(f"Total number of different case ids: {len(case_ids)}")
        record_ids = self.get_record_ids()
        print(f"Total number of different record ids: {len(record_ids)}")
        modality_ids = self.get_modality_ids()
        print(f"Total number of different modality ids: {len(modality_ids)}")
        if image_modality_ids is not None:
            assert all([image_modality_id in modality_ids for image_modality_id in image_modality_ids]), "Some of the requested modality_ids are not present in this Mirc object."
            img_sizes = []
            voxel_sizes = []
            for dataset_id in dataset_ids:
                for case_id in self[dataset_id]:
                    for record_id in self[dataset_id][case_id]:
                        reference_modality = self[dataset_id][case_id][record_id][image_modality_ids[0]].load()
                        for image_modality_id in image_modality_ids:
                            modality = self[dataset_id][case_id][record_id][image_modality_id].load()
                            assert np.allclose(modality.affine, reference_modality.affine), f"Affine of {dataset_id}-{case_id}-{record_id}-{image_modality_id} is not equal to the reference from {image_modality_ids[0]}."
                            assert np.array_equal(reference_modality.shape[1:4], modality.shape[1:4])
                            img_sizes.append(modality.shape[1:4])
                            voxel_sizes.append([s.round(2) for s in np.linalg.norm(modality.affine[0][:3, :3], 2, axis=0)])

            fig, axs = plt.subplots(3, 3, figsize=(9, 9))
            for i in range(3):
                img_sizes_dim_i = list(zip(*img_sizes))[i]
                axs[0, i].hist(img_sizes_dim_i)
                axs[0, i].set_title(f"image size dim {i} (# voxels)")
                voxel_sizes_dim_i = list(zip(*voxel_sizes))[i]
                axs[1, i].hist(voxel_sizes_dim_i)
                axs[1, i].set_title(f"voxel size dim {i} (mm)")
                axs[2, i].hist([i * j for i, j in zip(img_sizes_dim_i, voxel_sizes_dim_i)])
                axs[2, i].set_title(f"image size dim {i} (mm)")

            plt.show()
            n_cols = min(len(image_modality_ids), 4)
            n_rows = int(np.ceil(len(image_modality_ids) / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            for i, image_modality_id in enumerate(image_modality_ids):
                mean, std, (counts, bins) = self.mean_and_std(image_modality_id, return_histogram=True, **kwargs)
                print(f"Mean and standard deviation of '{image_modality_id}': ({mean}, {std})")
                if n_rows == 1:
                    axs[i].bar((bins[:-1] + bins[1:]) / 2, counts)
                    axs[i].set_title(f"{image_modality_id} (n={np.sum(counts)})")

                else:
                    axs[i // 4, i % 4].bar((bins[:-1] + bins[1:]) / 2, counts)
                    axs[i // 4, i % 4].set_title(f"{image_modality_id} (n={np.sum(counts)})")

            plt.show()

        if metadata_modality_ids is not None:
            assert all([metadata_modality_id in modality_ids for metadata_modality_id in metadata_modality_ids]), "Some of the requested modality_ids are not present in this Mirc object."
            n_cols = min(len(metadata_modality_ids), 4)
            n_rows = int(np.ceil(len(metadata_modality_ids) / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
            for i, metadata_modality_id in enumerate(metadata_modality_ids):
                mean, std, (counts, bins) = self.mean_and_std(metadata_modality_id, return_histogram=True)
                print(f"Mean and standard deviation of '{metadata_modality_id}': ({mean}, {std})")
                if n_rows == 1:
                    axs[i].bar((bins[:-1] + bins[1:]) / 2, counts)
                    axs[i].set_title(f"{metadata_modality_id} (n={np.sum(counts)})")

                else:
                    axs[i // 4, i % 4].bar((bins[:-1] + bins[1:]) / 2, counts)
                    axs[i // 4, i % 4].set_title(f"{metadata_modality_id} (n={np.sum(counts)})")

            plt.show()


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


NiftiModality = NiftyModality


class NiftyFileModality(Modality):
    def __init__(self, modality_id, file_path):
        super(NiftyFileModality, self).__init__(modality_id, os.path.dirname(file_path))
        self.file_path = file_path

    def load(self):
        nii = nib.load(self.file_path)
        return Sample(nii.get_fdata(caching="unchanged"), nii.affine)


NiftiFileModality = NiftyFileModality


class ImageFileModality(Modality):
    def __init__(self, modality_id, file_path, **kwargs):  # check documentation of Image.convert for **kwargs: e.g. mode, which can be "1" (binary), "RGB" (color), "L" grayscale
        super(ImageFileModality, self).__init__(modality_id, os.path.dirname(file_path))
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        img = Image.open(self.file_path)
        if self.kwargs:
            img = img.convert(**self.kwargs)

        img = np.asarray(img)
        assert img.ndim in [2, 3], "A 2D image file can only have a maximum of three dimensions: x, y, color."
        return Sample(img if img.ndim == 2 else img[:, :, None, :])
