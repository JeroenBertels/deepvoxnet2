import os
import time
import json
import pickle
import gc
import numpy as np
import nibabel as nib
import tensorflow as tf
from abc import ABC
from functools import partial
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.components.creator import Creator
from deepvoxnet2.components.sampler import Sampler, Identifier
from deepvoxnet2.components.transformers import KerasModel
from deepvoxnet2.keras.callbacks import MetricNameChanger, LogsLogger, DvnHistory
from deepvoxnet2.keras.losses import get_combined_loss


class TfDataset(tf.data.Dataset, ABC):
    def __new__(cls, creator, sampler=None, batch_size=None, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE, shuffle_samples=False, repeat=1):
        if sampler is None:
            sampler = Sampler([Identifier()])

        def _generator_fn(idx):
            identifier = sampler[idx]
            if idx == len(sampler) - 1:
                sampler.randomize()

            outputs = [output for output in Creator(creator.outputs).eval(identifier)]
            if len(outputs) > 0:
                outputs = [[Sample(np.concatenate([output[j][i] for output in outputs]), np.concatenate([output[j][i].affine for output in outputs])) for i in range(len(outputs[0][j]))] for j in range(len(outputs[0]))]

            else:
                outputs = [[np.full((1, output_shape[1] or 1, output_shape[2] or 1, output_shape[3] or 1, output_shape[4] or 1), 1234567890, dtype=np.float32) for output_shape in output_shapes] for output_shapes in creator.output_shapes]

            gc.collect()
            return [output_ for output in outputs for output_ in output]

        def _map_fn(idx):
            outputs = tf.py_function(_generator_fn, [idx], [tf.dtypes.float32 for output in creator.outputs for _ in output])
            output_shapes = [(None, output_shape[1], output_shape[2], output_shape[3], output_shape[4]) for output_shapes in creator.output_shapes for output_shape in output_shapes]  # batch size depends on how many samples the creator generates
            for output, output_shape in zip(outputs, output_shapes):
                output.set_shape(output_shape)

            return tuple([tuple([outputs.pop(0) for _ in range(len(creator.outputs[i]))]) for i in range(len(creator.outputs))])

        def _filter_fn(*x):
            return tf.math.reduce_any(tf.not_equal(x[0][0], 1234567890))

        dataset = tf.data.Dataset.from_tensor_slices(list(range(len(sampler)))).repeat(repeat)
        dataset = dataset.map(_map_fn, num_parallel_calls=num_parallel_calls, deterministic=True).filter(_filter_fn)
        if batch_size is not None:
            dataset = dataset.unbatch()

        if shuffle_samples:
            assert batch_size is not None, "When batch size is None, we assume the creator produced a batch and thus we are not going to shuffle the samples (which shuffles potentially across records)."
            dataset = dataset.shuffle(shuffle_samples)

        if batch_size is not None:
            dataset = dataset.batch(batch_size=batch_size)

        if prefetch_size != 0:
            dataset = dataset.prefetch(prefetch_size)

        return dataset


class DvnModel(object):
    def __init__(self, outputs):
        self.creator = Creator([connection for key in outputs for connection in outputs[key]])
        self.outputs = {}
        self.optimizer = {}
        self.losses = {}
        self.losses_weights = {}
        self.metrics = {}
        self.weighted_metrics = {}
        i = 0
        for key in outputs:
            self.outputs[key] = self.creator.outputs[i:i + len(outputs[key])]
            self.optimizer[key] = None
            self.losses[key] = []
            self.losses_weights[key] = []
            self.metrics[key] = []
            self.weighted_metrics[key] = []
            i += len(outputs[key])

    def compile(self, key, optimizer=None, losses=None, metrics=None, losses_weights=None, weighted_metrics=None):
        assert key in self.outputs, "There are no outputs available for this key."
        assert len(self.outputs[key]) >= 2, "Outputs must be in the format [x/y_, y, sample_weight] and for compile at least [x/y_, y] must be available."
        if losses is not None:
            assert isinstance(losses, list) and len(losses) == len(self.outputs[key][0]), "The losses must be given as a list of losses with length equal to the number of outputs (i.e. length of x/y_)."
            losses = [loss if isinstance(loss, list) else [loss] for loss in losses]

            if losses_weights is not None:
                assert isinstance(losses_weights, list) and len(losses_weights) == len(self.outputs[key][0]), "The losses_weights must be given as a list of losses_weights with length equal to the number of outputs and thus the number of losses  (i.e. length of x/y_)."
                losses_weights = [loss_weights if isinstance(loss_weights, list) else [loss_weights] for loss_weights in losses_weights]
                for loss, loss_weights in zip(losses, losses_weights):
                    assert len(loss) == len(loss_weights), "When a list of losses is given for a certain output and losses_weights are not None, you must specify a list of loss_weights per loss of the same length."

            else:
                losses_weights = [[1 / (len(losses) * len(loss)) for _ in loss] for loss in losses]

            self.losses[key] = []
            self.losses_weights[key] = []
            for i, (loss, loss_weights) in enumerate(zip(losses, losses_weights)):
                combined_loss = get_combined_loss(loss, loss_weights=loss_weights, custom_combined_loss_name=f"loss__s{i}")
                self.losses[key].append(combined_loss)
                self.losses_weights[key].append(1)

        else:
            self.losses[key] = [[] for _ in range(len(self.outputs[key][0]))]

        if metrics is not None:
            assert isinstance(metrics, list) and len(metrics) == len(self.outputs[key][0]), "The metrics must be given as a list of metrics lists with length equal to the number of outputs  (i.e. length of x/y_)."
            for i, metric in enumerate(metrics):
                self.metrics[key].append([])
                for metric_ in metric if isinstance(metric, list) else [metric]:
                    metric__ = partial(metric_)
                    metric__.__name__ = f"{metric_.__name__}__s{i}"
                    self.metrics[key][i].append(metric__)

        else:
            self.metrics[key] = [[] for _ in range(len(self.outputs[key][0]))]

        if weighted_metrics is not None:
            assert isinstance(weighted_metrics, list) and len(weighted_metrics) == len(self.outputs[key][0]), "The weighted_metrics must be given as a list of weighted_metrics lists with length equal to the number of outputs  (i.e. length of x/y_)."
            for i, weighted_metric in enumerate(weighted_metrics):
                self.weighted_metrics[key].append([])
                for weighted_metric_ in weighted_metric if isinstance(weighted_metric, list) else [weighted_metric]:
                    weighted_metric__ = partial(weighted_metric_)
                    weighted_metric__.__name__ = f"{weighted_metric_.__name__}__s{i}"
                    self.weighted_metrics[key][i].append(weighted_metric__)

        else:
            self.weighted_metrics[key] = [[] for _ in range(len(self.outputs[key][0]))]

        if optimizer is not None:
            assert isinstance(self.outputs[key][0].transformer, KerasModel), "When using compile with an optimizer specified the output and index 0 (i.e. x/y_) must be after a KerasTransformer."
            if isinstance(optimizer, str):
                optimizer = tf.keras.optimizers.get(optimizer)

            self.optimizer[key] = optimizer
            self.outputs[key][0].transformer.keras_model.compile(optimizer=self.optimizer[key], loss=self.losses[key], metrics=self.metrics[key], loss_weights=self.losses_weights[key], weighted_metrics=self.weighted_metrics[key])

    def fit(self, key, sampler, batch_size=1, epochs=1, callbacks=None, validation_sampler=None, validation_key=None, validation_freq=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE, shuffle_samples=False, verbose=1, logs_dir=None, initial_epoch=0, steps_per_epoch=None):
        assert key in self.outputs, "There are no outputs available for this key."
        assert len(self.outputs[key]) >= 2, "Outputs must be in the format [x/y_, y, sample_weight] and to use fit at least [x/y_, y] must be available."
        assert isinstance(self.outputs[key][0].transformer, KerasModel), "To use fit, x/y_ must be the output of a KerasModel transformer."
        assert self.optimizer[key] is not None and len(self.losses[key]) > 0, "To use fit, these outputs must have been compiled with an optimizer and losses first."
        fit_dataset = TfDataset(Creator([*self.outputs[key][0].transformer.connections[self.outputs[key][0].idx], *self.outputs[key][1:]]), sampler=sampler, batch_size=batch_size, num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size, shuffle_samples=shuffle_samples, repeat=1 if steps_per_epoch is None else None)
        validation_fit_dataset = None
        if validation_sampler is not None:
            assert validation_key in self.outputs, "There are no outputs available for this validation_key."
            assert len(self.outputs[validation_key]) >= 2, "Outputs must be in the format [x, y/y_, sample_weight] and to use fit at least [x, y/y_] must be available."
            assert isinstance(self.outputs[validation_key][0].transformer, KerasModel), "To use fit, x/y_ must be the output of a KerasModel transformer."
            assert self.outputs[key][0].transformer.keras_model is self.outputs[validation_key][0].transformer.keras_model, "The Keras model for training and validation must be the same."
            validation_fit_dataset = TfDataset(Creator([*self.outputs[validation_key][0].transformer.connections[self.outputs[validation_key][0].idx], *self.outputs[validation_key][1:]]), sampler=validation_sampler, batch_size=batch_size, num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size, shuffle_samples=shuffle_samples, repeat=1 if steps_per_epoch is None else None)

        if callbacks is None:
            callbacks = []

        callbacks.insert(0, MetricNameChanger(training_key=key, validation_key=validation_key))
        if logs_dir is not None:
            callbacks += [LogsLogger(logs_dir), DvnHistory(logs_dir)]

        return self.outputs[key][0].transformer.keras_model.fit(x=fit_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_fit_dataset, validation_freq=validation_freq, verbose=verbose, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch)

    def evaluate(self, key, sampler, mode="last", output_dirs=None, name_tag=None, save_x=True, save_y=False, save_sample_weight=False):
        assert mode in ["all", "last"], "Will we only keep the last generated output (i.e. last) or everything (i.e. all)?"
        assert key in self.outputs, "There are no outputs available for this key."
        assert len(self.outputs[key]) >= 2, "Outputs must be in the format [x/y_, y, sample_weight] and to use evaluate at least [x/y_, y] must be available."
        if output_dirs is not None:
            assert len(sampler) == len(output_dirs)

        evaluations = []
        for identifier_i, identifier in enumerate(sampler):
            start_time = time.time()
            samples = self.predict(key, Sampler([identifier]), mode=mode, output_dirs=[output_dirs[identifier_i]] if output_dirs is not None else output_dirs, name_tag=name_tag, save_x=save_x, save_y=save_y, save_sample_weight=save_sample_weight)[0]
            evaluation = {}
            total_loss_value = 0
            for i, loss in enumerate(self.losses[key]):
                loss_name = f"{key}__{loss.__name__}"
                evaluation[loss_name] = loss(samples[1][i], samples[0][i]).numpy().mean().item() if len(samples) == 2 else (loss(samples[1][i], samples[0][i]).numpy() * samples[2][i]).mean().item()
                total_loss_value += evaluation[loss_name] * self.losses_weights[key][i]

            evaluation[f"{key}__loss__combined"] = total_loss_value
            for i, metric in enumerate(self.metrics[key]):
                for metric_ in metric:
                    metric_name = f"{key}__{metric_.__name__}"
                    evaluation[metric_name] = metric_(samples[1][i], samples[0][i]).numpy().mean().item()

            if len(samples) == 3:
                for i, weighted_metric in enumerate(self.weighted_metrics[key]):
                    for weighted_metric_ in weighted_metric:
                        weighted_metric_name = f"{key}__weighted_{weighted_metric_.__name__}"
                        evaluation[weighted_metric_name] = (weighted_metric_(samples[1][i], samples[0][i]).numpy() * samples[2][i]).mean().item()

            evaluations.append(evaluation)
            print("Evaluated {} with {} in {:.0f} s: \n{}".format(identifier(), key, time.time() - start_time, json.dumps(evaluation, indent=2)))

        print("\nMean evaluation results: ")
        for metric_name in evaluations[0]:
            print("{}: {:.2f}".format(metric_name, np.mean([evaluation[metric_name] for evaluation in evaluations])))

        return evaluations

    def predict(self, key, sampler, mode="last", output_dirs=None, name_tag=None, save_x=True, save_y=False, save_sample_weight=False):
        assert mode in ["all", "last"], "Will we only keep the last generated output (i.e. last) or everything (i.e. all)?"
        assert key in self.outputs, "There are no outputs available for this key."
        assert len(self.outputs[key]) >= 1, "Outputs must be in the format [x, y, sample_weight] and to use predict at least [x] must be available."
        predictions = []
        for identifier_i, identifier in enumerate(sampler):
            start_time = time.time()
            samples = [sample for sample in Creator(self.outputs[key]).eval(identifier)][0 if mode == "all" else -1:]
            samples = [[Sample(np.concatenate([output[j][i] for output in samples]), np.concatenate([output[j][i].affine for output in samples])) for i in range(len(samples[0][j]))] for j in range(len(samples[0]))]
            if output_dirs is not None:
                self.save_sample(key, samples, output_dirs[identifier_i], name_tag=name_tag, save_x=save_x, save_y=save_y, save_sample_weight=save_sample_weight)

            predictions.append(samples)
            print("Predicted {} with {} in {:.0f} s.".format(sampler[identifier_i](), key, time.time() - start_time))

        return predictions

    @staticmethod
    def save_sample(key, sample, output_dir, name_tag=None, save_x=True, save_y=False, save_sample_weight=False):
        for i, prediction in enumerate(sample[0]):
            for j in range(len(prediction)):
                output_path = os.path.join(output_dir, "{}{}{}{}.nii.gz".format(key, f"__s{i}", f"__b{j}", "__" + name_tag if name_tag is not None else ""))
                if save_x:
                    nib.save(nib.Nifti1Image(prediction[j], prediction.affine[j]), output_path[:-7] + "__x" + ".nii.gz")

                if save_y and len(sample) > 1:
                    nib.save(nib.Nifti1Image(sample[1][i][j], sample[1][i].affine[j]), output_path[:-7] + "__y" + ".nii.gz")

                if save_sample_weight and len(sample) > 2:
                    nib.save(nib.Nifti1Image(sample[2][i][j], sample[2][i].affine[j]), output_path[:-7] + "__sample_weight" + ".nii.gz")

    def summary(self, only_active=True):
        self.creator.summary(only_active=only_active)

    def save(self, file_dir, save_keras_models=True):
        self.save_model(self, file_dir, save_keras_models=save_keras_models)

    @staticmethod
    def save_model(dvn_model, file_dir, save_keras_models=True):
        keras_models = dvn_model.creator.clear_keras_models()
        if save_keras_models:
            for name in keras_models:
                keras_model_dir = os.path.join(file_dir, "keras_models")
                if not os.path.isdir(keras_model_dir):
                    os.makedirs(keras_model_dir)

                keras_models[name].save(os.path.join(keras_model_dir, name))

        with open(os.path.join(file_dir, "dvn_model.pkl"), "wb") as f:
            pickle.dump(dvn_model, f)

        dvn_model.creator.set_keras_models(keras_models)

    @staticmethod
    def load_model(file_dir, load_keras_models=True):
        with open(os.path.join(file_dir, "dvn_model.pkl"), "rb") as f:
            dvn_model = pickle.load(f)

        if load_keras_models:
            custom_objects_dict = {}
            for custom_objects in [dvn_model.losses, dvn_model.metrics, dvn_model.weighted_metrics]:
                for key in custom_objects:
                    for custom_objects_ in custom_objects[key]:
                        for custom_object in custom_objects_ if isinstance(custom_objects_, list) else [custom_objects_]:
                            custom_objects_dict[custom_object.__name__] = custom_object

            keras_model_dir = os.path.join(file_dir, "keras_models")
            keras_models = dvn_model.creator.clear_keras_models()
            for name in keras_models:
                keras_models[name] = tf.keras.models.load_model(os.path.join(keras_model_dir, name), custom_objects=custom_objects_dict)

            dvn_model.creator.set_keras_models(keras_models)

        return dvn_model
