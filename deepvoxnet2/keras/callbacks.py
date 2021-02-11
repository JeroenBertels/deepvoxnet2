import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, History


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
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for log in logs.copy():
                if not (log.startswith("full_") or log.startswith("dvn_") or log.startswith("val_") or log.startswith("train_")):
                    logs["train_" + log] = logs.pop(log)


class DvnModelCheckpoint(Callback):
    def __init__(self, dvn_model, model_dir, freq, epoch_as_name_tag=False):
        super(DvnModelCheckpoint, self).__init__()
        self.dvn_model = dvn_model
        self.model_dir = model_dir
        self.freq = freq
        self.epoch_as_name_tag = epoch_as_name_tag

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and (epoch + 1) % self.freq == 0:
            model_name = "dvn_model_{:05}".format(epoch) if self.epoch_as_name_tag else "dvn_model"
            self.dvn_model.save(os.path.join(self.model_dir, model_name))


class DvnModelEvaluator(Callback):
    def __init__(self, dvn_model, sampler, key, output_dirs=None, freq=1, prediction_batch_size=None, epoch_as_name_tag=False, name_tag=None):
        super(DvnModelEvaluator, self).__init__()
        self.dvn_model = dvn_model
        self.sampler = sampler
        self.key = key
        self.output_dirs = output_dirs
        self.freq = freq
        self.prediction_batch_size = prediction_batch_size
        self.epoch_as_name_tag = epoch_as_name_tag
        self.name_tag = name_tag
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and (epoch + 1) % self.freq == 0:
            evaluations = self.dvn_model.evaluate_dvn(self.sampler, self.key, self.output_dirs, prediction_batch_size=self.prediction_batch_size, name_tag="{:05}".format(epoch) if self.epoch_as_name_tag else None)
            for metric_name in evaluations[0]:
                metric_name_ = metric_name if self.name_tag is None else metric_name + f"_{self.name_tag}"
                self.history[metric_name_] = self.history.get(metric_name_, []) + [[evaluation[metric_name] for evaluation in evaluations]]

        for metric_name in self.history:
            if logs is None:
                logs = {}

            assert metric_name not in logs
            logs[metric_name] = np.mean(self.history[metric_name][-1])
