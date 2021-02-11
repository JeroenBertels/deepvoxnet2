import os
import time
import json
import copy
import pickle
import gc
import inspect
import numpy as np
import nibabel as nib
import tensorflow as tf
from abc import ABC
from functools import partial
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.components.creator import Creator
from deepvoxnet2.components.mirc import Sampler
from deepvoxnet2.components.transformers import KerasModel, _SampleInput


class TfDataset(tf.data.Dataset, ABC):
    def __new__(cls, sampler, creator, batch_size=None, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE, shuffle=False):
        def _generator(idx):
            outputs = [output for output in copy.deepcopy(creator).eval(sampler[idx])]
            outputs = [[Sample(np.concatenate([output[j][i] for output in outputs]), np.concatenate([output[j][i].affine for output in outputs])) for i in range(len(outputs[0][j]))] for j in range(len(outputs[0]))]
            gc.collect()
            return [output_ for output in outputs for output_ in output]

        def _map_fn(idx):
            output_shapes = [output_shape for output_connection in creator.outputs for output_shape in output_connection.shape]
            outputs = tf.py_function(_generator, [idx], [tf.dtypes.float32 for output in creator.outputs for _ in output])
            for output, output_shape in zip(outputs, output_shapes):
                output.set_shape(output_shape)

            return tuple([tuple([outputs.pop(0) for _ in range(len(creator.outputs[i]))]) for i in range(len(creator.outputs))])

        dataset = tf.data.Dataset.from_tensor_slices(list(range(len(sampler))))
        dataset = dataset.map(_map_fn, num_parallel_calls=num_parallel_calls, deterministic=shuffle)
        if batch_size is not None:
            dataset = dataset.unbatch().batch(batch_size=batch_size)

        if prefetch_size != 0:
            dataset = dataset.prefetch(prefetch_size)

        return dataset


class DvnModel(object):
    def __init__(self, outputs=None, dvn_outputs=None):
        self.outputs, self.inputs = {}, {}
        if outputs is not None:
            for key in outputs:
                assert isinstance(outputs[key], list)
                self.outputs[key] = Creator.deepcopy(outputs[key])
                keras_model_connections = [connection for connection in Creator.get_trace(self.outputs[key])[1] if isinstance(connection.transformer, KerasModel)]
                assert len(keras_model_connections) == 1
                self.inputs[key] = keras_model_connections[0].transformer.connections[keras_model_connections[0].idx][0]

        self.dvn_outputs, self.dvn_inputs = {}, {}
        if dvn_outputs is not None:
            for key in dvn_outputs:
                assert isinstance(dvn_outputs[key], list)
                self.dvn_outputs[key] = Creator.deepcopy(dvn_outputs[key])
                keras_model_connections = [connection for connection in Creator.get_trace(self.dvn_outputs[key])[1] if isinstance(connection.transformer, KerasModel)]
                assert len(keras_model_connections) == 1
                self.dvn_inputs[key] = keras_model_connections[0].transformer.connections[keras_model_connections[0].idx][0]

        self.keras_model = self.clear_keras_model()
        self.set_keras_model(self.keras_model)
        self.optimizer = None
        self.loss = None
        self.loss_weights = None
        self.metrics = None
        self.weighted_metrics = None

    def compile(self, optimizer, loss, metrics=None, loss_weights=None, weighted_metrics=None):
        self.optimizer = tf.keras.optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer
        self.loss = self.dress(self.keras_model, loss, mode="losses")
        self.loss_weights = self.dress(self.keras_model, loss_weights)
        for layer_name in self.loss:
            loss_weights = self.loss_weights.get(layer_name, [1] * len(self.loss[layer_name]))
            assert len(loss_weights) == len(self.loss[layer_name])
            combined_loss = partial(self.get_combined_loss, losses=self.loss[layer_name], loss_weights=loss_weights)
            combined_loss.__name__ = "loss_{}".format(layer_name)
            self.loss[layer_name] = [combined_loss]
            self.loss_weights[layer_name] = [1]

        self.metrics = self.dress(self.keras_model, metrics, mode="metrics")
        self.weighted_metrics = self.dress(self.keras_model, weighted_metrics, mode="metrics")
        self.keras_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights, weighted_metrics=self.weighted_metrics)

    def fit(self, sampler, training_key, batch_size=1, epochs=1, callbacks=None, validation_sampler=None, validation_key=None, validation_freq=1, num_parallel_calls=None, prefetch_size=0):
        assert len(self.outputs[training_key]) >= 2, "The requested training outputs are not appropriate to use fit."
        fit_dataset = TfDataset(sampler, Creator([self.inputs[training_key], *self.outputs[training_key][1:]]), batch_size=batch_size, num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size, shuffle=True)
        validation_fit_dataset = None
        if validation_sampler is not None:
            assert len(self.outputs[validation_key]) >= 2, "The requested validation outputs are not appropriate to use fit."
            validation_fit_dataset = TfDataset(validation_sampler, Creator([self.inputs[validation_key], *self.outputs[validation_key][1:]]), batch_size=None, num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size, shuffle=False)

        return self.keras_model.fit(x=fit_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_fit_dataset, validation_freq=validation_freq)

    def evaluate(self, sampler, key, num_parallel_calls=None, prefetch_size=0):
        assert len(self.outputs[key]) >= 2, "The requested outputs are not appropriate to use evaluate."
        return self.keras_model.evaluate(x=TfDataset(sampler, Creator([self.inputs[key], *self.outputs[key][1:]]), num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size), return_dict=True)

    def predict(self, sampler, key, num_parallel_calls=None, prefetch_size=0):
        return self.keras_model.predict(x=TfDataset(sampler, Creator(self.inputs[key]), num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size))

    def evaluate_dvn(self, sampler, key, output_dirs=None, prediction_batch_size=None, save_y=True, name_tag=None):
        assert len(self.dvn_outputs[key]) >= 2, "The requested full outputs are not appropriate to use full evaluate."
        evaluations = []
        for idx, identifier in enumerate(sampler):
            start_time = time.time()
            samples = self.predict_dvn(Sampler([identifier]), key, [output_dirs[idx]] if output_dirs is not None else None, prediction_batch_size, save_y, name_tag)[0]
            evaluation = {}
            total_loss_value = 0
            for layer_name in self.loss:
                layer_idx = self.keras_model.output_names.index(layer_name)
                for i, loss in enumerate(self.loss[layer_name]):
                    loss_name = "dvn_{}_{}{}".format(key, loss.__class__.__name__ if inspect.isclass(loss) else loss.__name__, "_" + layer_name if len(self.keras_model.output_names) > 1 else "")
                    evaluation[loss_name] = loss(samples[1][layer_idx], samples[0][layer_idx]).numpy().mean().item() if len(samples) == 2 else loss(samples[1][layer_idx], samples[0][layer_idx], sample_weight=samples[2][layer_idx]).numpy().mean().item()
                    total_loss_value += evaluation[loss_name] * (self.loss_weights[layer_name][i] if layer_name in self.loss_weights else 1)

            evaluation["dvn_{}_loss".format(key)] = total_loss_value
            for layer_name in self.metrics:
                layer_idx = self.keras_model.output_names.index(layer_name)
                for i, metric in enumerate(self.metrics[layer_name]):
                    metric_name = "dvn_{}_{}{}".format(key, metric.__class__.__name__ if inspect.isclass(metric) else metric.__name__, "_" + layer_name if len(self.keras_model.output_names) > 1 else "")
                    evaluation[metric_name] = metric(samples[1][layer_idx], samples[0][layer_idx]).numpy().mean().item()

            if len(samples) == 3:
                for layer_name in self.weighted_metrics:
                    layer_idx = self.keras_model.output_names.index(layer_name)
                    for i, weighted_metric in enumerate(self.weighted_metrics[layer_name]):
                        weighted_metric_name = "dvn_{}_weighted_{}{}".format(key, weighted_metric.__class__.__name__ if inspect.isclass(weighted_metric) else weighted_metric.__name__, "_" + layer_name if len(self.keras_model.output_names) > 1 else "")
                        evaluation[weighted_metric_name] = weighted_metric(samples[1][layer_idx], samples[0][layer_idx], sample_weight=samples[2][layer_idx]).numpy().mean().item()

            evaluations.append(evaluation)
            print("Evaluated {} with {} in {:.0f} s: \n{}".format(identifier(), key, time.time() - start_time, json.dumps(evaluation, indent=2)))

        print("\nMean evaluation results: ")
        for metric_name in evaluations[0]:
            print("{}: {:.2f}".format(metric_name, np.mean([evaluation[metric_name] for evaluation in evaluations])))

        return evaluations

    def predict_dvn(self, sampler, key, output_dirs=None, prediction_batch_size=None, save_y=False, name_tag=None):
        dvn_outputs = Creator.deepcopy(self.dvn_outputs[key])
        keras_model_connection = [connection for connection in Creator.get_trace(dvn_outputs)[1] if isinstance(connection.transformer, KerasModel)][0]
        predictions = []
        for identifier_i, identifier in enumerate(sampler):
            start_time = time.time()
            samples = [sample for sample in Creator(self.dvn_inputs[key]).eval(sampler[identifier_i])]
            samples = [[Sample(np.concatenate([output[j][i] for output in samples]), np.concatenate([output[j][i].affine for output in samples])) for i in range(len(samples[0][j]))] for j in range(len(samples[0]))]
            y = []
            batch_size = prediction_batch_size or len(samples[0][0])
            for sample_i in range(len(samples[0][0])):
                if sample_i % batch_size == 0:
                    y_batch_i = self.keras_model.predict([sample[sample_i:sample_i + batch_size] for sample in samples[0]])
                    y_batch_i = y_batch_i if isinstance(y_batch_i, list) else [y_batch_i]
                    for i, (y_, output_affine) in enumerate(zip(y_batch_i, keras_model_connection.transformer.output_affines)):
                        if output_affine is None:
                            output_affine = Sample.update_affine(translation=[-(out_shape // 2) + (in_shape // 2) for in_shape, out_shape in zip(samples[0][0].shape[1:4], y_.shape[1:4])])

                        y_batch_i[i] = Sample(y_, Sample.update_affine(samples[0][0].affine[sample_i:sample_i + batch_size], transformation_matrix=output_affine))

                    y.append(y_batch_i)

            keras_model_connection.transformer = _SampleInput([Sample(np.concatenate([y_batch_i[i] for y_batch_i in y]), np.concatenate([y_batch_i[i].affine for y_batch_i in y])) for i in range(len(y[0]))])
            keras_model_connection.idx = 0
            samples = [sample for sample in Creator(dvn_outputs).eval(sampler[identifier_i])]
            samples = [[Sample(np.concatenate([output[j][i] for output in samples]), np.concatenate([output[j][i].affine for output in samples])) for i in range(len(samples[0][j]))] for j in range(len(samples[0]))]
            if output_dirs is not None:
                for i, (prediction, layer_name) in enumerate(zip(samples[0], self.keras_model.output_names)):
                    output_path = os.path.join(output_dirs[identifier_i], "dvn_{}{}{}.nii.gz".format(key, "_" + layer_name if len(self.keras_model.output_names) > 1 else "", "_" + name_tag if name_tag is not None else ""))
                    nib.save(nib.Nifti1Image(prediction[0], prediction.affine[0]), output_path)
                    if save_y:
                        nib.save(nib.Nifti1Image(samples[1][i][0], samples[1][i].affine[0]), output_path[:-7] + "_y" + ".nii.gz")

            predictions.append(samples)
            print("Predicted {} with {} in {:.0f} s.".format(sampler[identifier_i](), key, time.time() - start_time))

        return predictions

    def get_custom_objects(self):
        custom_objects = {}
        for custom_object in [self.loss, self.metrics, self.weighted_metrics]:
            if custom_object is not None and not isinstance(custom_object, str):
                for layer_name in custom_object:
                    for custom_object_ in custom_object[layer_name]:
                        if not isinstance(custom_object_, str) and custom_object_.__name__ not in custom_objects:
                            custom_objects[custom_object_.__class__.__name__ if inspect.isclass(custom_object_) else custom_object_.__name__] = custom_object_

        return custom_objects

    @staticmethod
    def get_combined_loss(y_true, y_pred, sample_weight=None, losses=None, loss_weights=None):
        loss_value = 0
        for loss, loss_weight in zip(losses, loss_weights):
            loss_value += loss(y_true, y_pred, sample_weight=sample_weight) * loss_weight

        return loss_value

    def clear_keras_model(self):
        clear_state = False
        for output_connections in [self.outputs[key] for key in self.outputs] + [self.dvn_outputs[key] for key in self.dvn_outputs]:
            transformers, connections = Creator.get_trace(output_connections)
            for transformer in transformers:
                if isinstance(transformer, KerasModel):
                    if not clear_state:
                        self.keras_model = transformer.keras_model
                        clear_state = True

                    assert self.keras_model is transformer.keras_model
                    transformer.keras_model = None

        return self.keras_model

    def set_keras_model(self, keras_model):
        self.keras_model = keras_model
        for output_connections in [self.outputs[key] for key in self.outputs] + [self.dvn_outputs[key] for key in self.dvn_outputs]:
            transformers, connections = Creator.get_trace(output_connections)
            for transformer in transformers:
                if isinstance(transformer, KerasModel):
                    transformer.keras_model = self.keras_model

    def save(self, file_dir):
        self.save_model(self, file_dir)

    @staticmethod
    def save_model(dvn_model, file_dir):
        keras_model = dvn_model.clear_keras_model()
        keras_model.save(os.path.join(file_dir, "keras_model"))
        dvn_model.keras_model = None
        with open(os.path.join(file_dir, "dvn_model.pkl"), "wb") as f:
            pickle.dump(dvn_model, f)

        dvn_model.set_keras_model(keras_model)

    @staticmethod
    def load_model(file_dir):
        with open(os.path.join(file_dir, "dvn_model.pkl"), "rb") as f:
            dvn_model = pickle.load(f)

        dvn_model.keras_model = tf.keras.models.load_model(os.path.join(file_dir, "keras_model"), custom_objects=dvn_model.get_custom_objects())
        return dvn_model

    @staticmethod
    def dress(keras_model, arg, mode=None):
        def _mode_fn(_arg):
            if mode is None:
                return _arg

            elif mode == "losses":
                return tf.keras.losses.get(_arg) if isinstance(_arg, str) else _arg

            elif mode == "metrics":
                return tf.keras.metrics.get(_arg) if isinstance(_arg, str) else _arg

            else:
                raise NotImplementedError

        if not isinstance(arg, dict):
            if arg is not None:
                arg = {keras_model.output_names[0]: [_mode_fn(arg_) for arg_ in (arg if isinstance(arg, list) else [arg])]}

            else:
                arg = {}

        else:
            for key in arg:
                assert key in keras_model.output_names
                arg[key] = [_mode_fn(arg_) for arg_ in (arg[key] if isinstance(arg[key], list) else [arg[key]])]

        return arg
