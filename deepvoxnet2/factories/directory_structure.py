import os
import numpy as np
import pandas as pd
from collections import Iterable
from deepvoxnet2.components.mirc import NiftiFileModality
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.components.sample import Sample


class Structure(object):
    def __init__(self, logs_dir=None, models_dir=None, history_path=None, train_images_output_dirs=None, val_images_output_dirs=None, test_images_output_dirs=None):
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.history_path = history_path
        self.train_images_output_dirs = train_images_output_dirs
        self.val_images_output_dirs = val_images_output_dirs
        self.test_images_output_dirs = test_images_output_dirs

    def reset(self):
        self.logs_dir = None
        self.models_dir = None
        self.history_path = None
        self.train_images_output_dirs = None
        self.val_images_output_dirs = None
        self.test_images_output_dirs = None

    def create(self):
        if self.logs_dir is not None and not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)

        if self.models_dir is not None and not os.path.isdir(self.models_dir):
            os.makedirs(self.models_dir)

        if self.history_path is not None and not os.path.isdir(os.path.dirname(self.history_path)):
            os.makedirs(os.path.dirname(self.history_path))

        if self.train_images_output_dirs is not None:
            for train_images_output_dir in self.train_images_output_dirs:
                if not os.path.isdir(train_images_output_dir):
                    os.makedirs(train_images_output_dir)

        if self.val_images_output_dirs is not None:
            for val_images_output_dir in self.val_images_output_dirs:
                if not os.path.isdir(val_images_output_dir):
                    os.makedirs(val_images_output_dir)

        if self.test_images_output_dirs is not None:
            for test_images_output_dir in self.test_images_output_dirs:
                if not os.path.isdir(test_images_output_dir):
                    os.makedirs(test_images_output_dir)

    def get_split_dirs(self, split):
        if split == "training" or split == "train":
            output_dirs = self.train_images_output_dirs

        elif split == "validation" or split == "val":
            output_dirs = self.val_images_output_dirs

        elif split == "testing" or split == "test":
            output_dirs = self.test_images_output_dirs

        else:
            raise ValueError("Split name unknown.")

        return output_dirs

    def update(self, *args, **kwargs):
        pass

    def predict(self, split, model_name, key, **kwargs):
        pass


class MircStructure(Structure):
    def __init__(self, base_dir, run_name, experiment_name, fold_i=None, round_i=None, training_identifiers=None, validation_identifiers=None, testing_identifiers=None, training_mirc=None, validation_mirc=None, testing_mirc=None):
        super(MircStructure, self).__init__()
        self.base_dir = base_dir
        self.run_name = run_name
        self.run_dir = None
        self.experiment_name = experiment_name
        self.experiment_dir = None
        self.round_i = None
        self.round_dir = None
        self.fold_i = None
        self.fold_dir = None
        self.training_identifiers = training_identifiers
        self.validation_identifiers = validation_identifiers
        self.testing_identifiers = testing_identifiers
        self.training_mirc = training_mirc
        self.validation_mirc = validation_mirc
        self.testing_mirc = testing_mirc
        self.update(fold_i, round_i)

    def update(self, fold_i=None, round_i=None):
        if fold_i is None:
            self.fold_i = fold_i
            self.round_i = round_i
            self.reset()

        else:
            self.run_dir = os.path.join(self.base_dir, self.run_name)
            print("This is {} run: {}".format("a new" if not os.path.isdir(self.run_dir) else "an existing", self.run_name))
            self.experiment_dir = os.path.join(self.run_dir, "Experiments", self.experiment_name)
            if os.path.isdir(self.experiment_dir):
                print("This is an existing experiment: {}".format(self.experiment_name))
                if round_i is None:
                    prev_rounds = [r for r in sorted(os.listdir(self.experiment_dir)) if r.startswith("Round_") and os.path.isdir(os.path.join(self.experiment_dir, r))]
                    round_i = len(prev_rounds)
                    prev_folds = [int(fold_i.split("_")[-1]) for fold_i in sorted(os.listdir(os.path.join(self.experiment_dir, prev_rounds[-1]))) if fold_i.startswith("Fold_") and os.path.isdir(os.path.join(self.experiment_dir, prev_rounds[-1], fold_i))]
                    if fold_i not in prev_folds:
                        round_i -= 1

            else:
                print("This is a new experiment: {}".format(self.experiment_name))
                assert round_i is None
                round_i = 0

            self.round_i = round_i
            self.round_dir = os.path.join(self.experiment_dir, "Round_{}".format(self.round_i))
            print("Starting Round: {}".format(self.round_i))
            self.fold_i = fold_i
            self.fold_dir = os.path.join(self.round_dir, "Fold_{}".format(self.fold_i))
            print("Starting Fold: {}".format(self.fold_i))
            self.logs_dir = os.path.join(self.fold_dir, "Logs")
            self.models_dir = os.path.join(self.fold_dir)
            self.history_path = os.path.join(self.fold_dir, "training_result.pkl")
            if self.training_mirc is not None and self.training_identifiers is None:
                self.training_identifiers = MircSampler(self.training_mirc).identifiers

            if self.training_identifiers is not None:
                self.train_images_output_dirs = [os.path.join(self.run_dir, identifier.case_id, identifier.record_id, "Training", "{}_Round_{}_Fold_{}".format(self.experiment_name, self.round_i, self.fold_i)) for identifier in self.training_identifiers]

            if self.validation_mirc is not None and self.validation_identifiers is None:
                self.validation_identifiers = MircSampler(self.validation_mirc).identifiers

            if self.validation_identifiers is not None:
                self.val_images_output_dirs = [os.path.join(self.run_dir, identifier.case_id, identifier.record_id, "Validation", "{}_Round_{}_Fold_{}".format(self.experiment_name, self.round_i, self.fold_i)) for identifier in self.validation_identifiers]

            if self.testing_mirc is not None and self.testing_identifiers is None:
                self.testing_identifiers = MircSampler(self.testing_mirc).identifiers

            if self.testing_identifiers is not None:
                self.test_images_output_dirs = [os.path.join(self.run_dir, identifier.case_id, identifier.record_id, "Testing", "{}_Round_{}_Fold_{}".format(self.experiment_name, self.round_i, self.fold_i)) for identifier in self.testing_identifiers]

    def get_split_identifiers(self, split):
        if split == "training" or split == "train":
            identifiers = self.training_identifiers

        elif split == "validation" or split == "val":
            identifiers = self.validation_identifiers

        elif split == "testing" or split == "test":
            identifiers = self.testing_identifiers

        else:
            raise ValueError("Split name unknown.")

        return identifiers

    def predict(self, split, model_name, key, fold_i=None, round_i=None, name_tag=None, **kwargs):
        prev_fold_i = self.fold_i
        if fold_i is None:
            fold_i = self.fold_i

        prev_round_i = self.round_i
        if round_i is None:
            round_i = self.round_i

        if fold_i is not None and round_i is not None:
            if not isinstance(fold_i, Iterable):
                fold_i = [fold_i]

            dvn_models = []
            for fold_i_ in fold_i:
                self.update(fold_i=fold_i_, round_i=round_i)
                dvn_models.append(DvnModel.load_model(os.path.join(self.models_dir, model_name)))

            self.update(fold_i="-".join([str(fold_i_) for fold_i_ in fold_i]), round_i=round_i)
            output_dirs = self.get_split_dirs(split)
            identifiers = self.get_split_identifiers(split)
            assert output_dirs is not None and identifiers is not None, "For this split there were no identifiers or there was no Mirc object specified."
            self.create()
            for identifier, output_dir in zip(identifiers, output_dirs):
                samples = [dvn_model.predict(key, [identifier])[0] for dvn_model in dvn_models]
                sample = [[Sample(np.mean([sample[i][j] for sample in samples], axis=0), np.mean([sample[i][j].affine for sample in samples], axis=0)) for j in range(len(samples[0][i]))] for i in range(len(samples[0]))]
                DvnModel.save_sample(key, sample, output_dir, name_tag=name_tag, **kwargs)

            self.update(fold_i=prev_fold_i, round_i=prev_round_i)

    def get_df(self, split, key, si=0, bi=0, name_tag=None, x_or_y_or_sample_weight="x"):
        output_dirs = self.get_split_dirs(split)
        identifiers = self.get_split_identifiers(split)
        assert output_dirs is not None and identifiers is not None, "For this split there were no identifiers or there was no Mirc object specified."
        indices = pd.MultiIndex.from_tuples([(identifier.dataset_id, identifier.case_id, identifier.record_id) for identifier in identifiers], names=["dataset_id", "case_id", "record_id"])
        columns = pd.MultiIndex.from_tuples([(self.run_name, self.experiment_name,)], names=["run_name", "experiment_name"])
        df = pd.DataFrame(index=indices, columns=columns)
        for identifier, output_dir in zip(identifiers, output_dirs):
            output_path = os.path.join(output_dir, "{}{}{}{}__{}.nii.gz".format(key, f"__s{si}", f"__b{bi}", "__" + name_tag if name_tag is not None else "", x_or_y_or_sample_weight))
            df.at[(identifier.dataset_id, identifier.case_id, identifier.record_id), (self.run_name, self.experiment_name)] = NiftiFileModality("_", output_path).load()

        return df
