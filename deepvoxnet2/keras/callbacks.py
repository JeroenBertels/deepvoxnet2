import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, History, ReduceLROnPlateau, EarlyStopping


class DvnHistory(History):
    def __init__(self, logs_dir=None):
        super(DvnHistory, self).__init__()
        self.history = {"epoch": []}
        self.logs_path = None
        if logs_dir is not None:
            self.logs_path = os.path.join(logs_dir, "history.pkl")
            if os.path.isfile(self.logs_path):
                with open(self.logs_path, "rb") as f:
                    self.history = pickle.load(f)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["epoch"] = self.history["epoch"][:epoch] + [epoch]
        for k, v in logs.items():
            self.history[k] = self.history.get(k, [])[:epoch] + [v]

        self.model.history = self
        if self.logs_path is not None:
            with open(self.logs_path, "wb") as f:
                pickle.dump(self.history, f)


class LogsLogger(Callback):
    def __init__(self, logs_dir):
        super(LogsLogger, self).__init__()
        self.file_writer = tf.summary.create_file_writer(logs_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            with self.file_writer.as_default():
                for log in logs:
                    tf.summary.scalar(log, logs[log], step=epoch)


class MetricNameChanger(Callback):
    def __init__(self, training_key=None, validation_key=None):
        super(MetricNameChanger, self).__init__()
        self.training_key = training_key
        self.validation_key = validation_key

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for log in logs.copy():
                if log != "lr" and log != "epoch" and self.training_key is not None and not log.startswith("val_"):
                    log_name = f"{self.training_key}__" + log
                    for i, keras_model_output_name in enumerate(self.model.output_names):
                        if log.startswith(keras_model_output_name + "_"):
                            log_name = f"{self.training_key}__" + log[len(keras_model_output_name) + 1:]
                            if not log.endswith(f"__s{i}"):
                                log_name = log_name + f"__s{i}"

                    logs[log_name] = logs.pop(log)

                elif log != "lr" and log != "epoch" and self.validation_key is not None and log.startswith("val_"):
                    log_name = f"{self.validation_key}__" + log[4:]
                    for i, keras_model_output_name in enumerate(self.model.output_names):
                        if log[4:].startswith(keras_model_output_name + "_"):
                            log_name = f"{self.validation_key}__" + log[4 + len(keras_model_output_name) + 1:]
                            if not log.endswith(f"__s{i}"):
                                log_name = log_name + f"__s{i}"

                    logs[log_name] = logs.pop(log)


class DvnModelCheckpoint(Callback):
    def __init__(self, dvn_model, model_dir, freq, epoch_as_name_tag=False, save_keras_models=True):
        super(DvnModelCheckpoint, self).__init__()
        self.dvn_model = dvn_model
        self.model_dir = model_dir
        self.freq = freq
        self.epoch_as_name_tag = epoch_as_name_tag
        self.save_keras_models = save_keras_models

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            model_name = "dvn_model_{:05}".format(epoch) if self.epoch_as_name_tag else "dvn_model"
            self.dvn_model.save(os.path.join(self.model_dir, model_name), save_keras_models=self.save_keras_models)


class DvnModelEvaluator(Callback):
    def __init__(self, dvn_model, key, sampler, freq=1, epoch_as_name_tag=False, mode="last", output_dirs=None, name_tag=None, save_x=True, save_y=False, save_sample_weight=False, logs_dir=None):
        super(DvnModelEvaluator, self).__init__()
        self.dvn_model = dvn_model
        self.key = key
        self.sampler = sampler
        self.freq = freq
        self.epoch_as_name_tag = epoch_as_name_tag
        self.mode = mode
        self.output_dirs = output_dirs
        self.name_tag = name_tag
        self.save_x = save_x
        self.save_y = save_y
        self.save_sample_weight = save_sample_weight
        self.history = {}
        self.logs_path = None
        if logs_dir is not None:
            self.logs_path = os.path.join(logs_dir, f"{self.key}__history.pkl" if self.name_tag is None else f"{self.key}__history__{self.name_tag}.pkl")
            if os.path.isfile(self.logs_path):
                with open(self.logs_path, "rb") as f:
                    self.history = pickle.load(f)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            evaluations = self.dvn_model.evaluate(self.key, self.sampler, mode=self.mode, output_dirs=self.output_dirs, name_tag="{:05}".format(epoch) if self.epoch_as_name_tag else None, save_x=self.save_x, save_y=self.save_y, save_sample_weight=self.save_sample_weight)
            for metric_name in evaluations[0]:
                metric_name_ = metric_name if self.name_tag is None else metric_name + f"__{self.name_tag}"
                self.history[metric_name_] = self.history.get(metric_name_, [])[:(epoch + 1) // self.freq - 1] + [[evaluation[metric_name] for evaluation in evaluations]]

            if self.logs_path is not None:
                with open(self.logs_path, "wb") as f:
                    pickle.dump(self.history, f)

        for metric_name in self.history:
            if logs is None:
                logs = {}

            assert metric_name not in logs
            logs[metric_name] = np.mean(self.history[metric_name][-1])
