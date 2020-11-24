import os


class Structure(object):
    def __init__(self, logs_dir=None, models_dir=None, history_path=None, train_images_output_dirs=None, val_images_output_dirs=None, test_images_output_dirs=None):
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.history_path = history_path
        self.train_images_output_dirs = train_images_output_dirs
        self.val_images_output_dirs = val_images_output_dirs
        self.test_images_output_dirs = test_images_output_dirs

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


class MircStructure(Structure):
    def __init__(self, base_dir, run_name, experiment_name, fold_i, round_i=None, training_identifiers=None, validation_identifiers=None, testing_identifiers=None):
        self.base_dir = base_dir
        self.run_name = run_name
        self.experiment_name = experiment_name
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
        super(MircStructure, self).__init__(
            logs_dir=os.path.join(self.fold_dir, "Logs"),
            models_dir=os.path.join(self.fold_dir),
            history_path=os.path.join(self.fold_dir, "training_result.pkl"),
            train_images_output_dirs=[os.path.join(self.run_dir, identifier.case_id, identifier.record_id, "Training", "{}_Round_{}_Fold_{}".format(self.experiment_name, self.round_i, self.fold_i)) for identifier in training_identifiers] if training_identifiers is not None else None,
            val_images_output_dirs=[os.path.join(self.run_dir, identifier.case_id, identifier.record_id, "Validation", "{}_Round_{}_Fold_{}".format(self.experiment_name, self.round_i, self.fold_i)) for identifier in validation_identifiers] if validation_identifiers is not None else None,
            test_images_output_dirs=[os.path.join(self.run_dir, identifier.case_id, identifier.record_id, "Testing", "{}_Round_{}_Fold_{}".format(self.experiment_name, self.round_i, self.fold_i)) for identifier in testing_identifiers] if testing_identifiers is not None else None
        )
