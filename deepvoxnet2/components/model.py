import os
import time
import json
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
from deepvoxnet2.components.mirc import Sampler, Identifier
from deepvoxnet2.components.transformers import KerasModel, _SampleInput


class TfDataset(tf.data.Dataset, ABC):
    def __new__(cls, creator, sampler=None, batch_size=None, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE, shuffle=False):
        if sampler is None:
            sampler = Sampler([Identifier()])

        def _generator(idx):
            outputs = [output for output in Creator(creator.outputs).eval(sampler[idx])]
            outputs = [[Sample(np.concatenate([output[j][i] for output in outputs]), np.concatenate([output[j][i].affine for output in outputs])) for i in range(len(outputs[0][j]))] for j in range(len(outputs[0]))]
            gc.collect()
            return [output_ for output in outputs for output_ in output]

        def _map_fn(idx):
            outputs = tf.py_function(_generator, [idx], [tf.dtypes.float32 for output in creator.outputs for _ in output])
            output_shapes = [output_shape for output_shapes in creator.output_shapes for output_shape in output_shapes]
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
    def __init__(self, creators=None):
        transformers = {}
        for key in creators:
            for transformer in creators[key].transformers:
                if transformer.name not in transformers:
                    transformers[transformer.name] = transformer

                else:
                    assert transformers[transformer.name] is transformer, "In a DvnModel all transformers must have a unique name. (You can do this by initializing the names by e.g. making first a creator with all the output connections and then using creator.outputs and use those outputs to make the separate creators.)"

        self.creators = creators
        self.optimizer = {}
        self.loss = {}
        self.loss_weights = {}
        self.metrics = {}
        self.weighted_metrics = {}

    def compile(self, key, optimizer, loss, metrics=None, loss_weights=None, weighted_metrics=None):
        assert key in self.creators[key], "There is no creator available for this key."
        assert len(self.creators[key].outputs) >= 2, "Creators must output [x, y, sample_weight] and for compile at least [x, y]."
        assert isinstance(self.creators[key].outputs[0].transformer, KerasModel), "To use compile, x must be the output of a KerasModel transformer."
        self.optimizer[key] = tf.keras.optimizers.get(optimizer) if isinstance(optimizer, str) else optimizer
        self.loss[key] = self.dress(self.creators[key].outputs[0].transformer.keras_model, loss, mode="losses")
        self.loss_weights[key] = self.dress(self.creators[key].outputs[0].transformer.keras_model, loss_weights)
        for layer_name in self.loss[key]:
            loss_weights = self.loss_weights[key].get(layer_name, [1] * len(self.loss[key][layer_name]))
            assert len(loss_weights) == len(self.loss[key][layer_name])
            combined_loss = partial(self.get_combined_loss, losses=self.loss[key][layer_name], loss_weights=loss_weights)
            combined_loss.__name__ = "loss_{}".format(layer_name)
            self.loss[key][layer_name] = [combined_loss]
            self.loss_weights[key][layer_name] = [1]

        self.metrics[key] = self.dress(self.creators[key].outputs[0].transformer.keras_model, metrics, mode="metrics")
        self.weighted_metrics[key] = self.dress(self.creators[key].outputs[0].transformer.keras_model, weighted_metrics, mode="metrics")
        self.creators[key].outputs[0].transformer.keras_model.compile(optimizer=self.optimizer[key], loss=self.loss[key], metrics=self.metrics[key], loss_weights=self.loss_weights[key], weighted_metrics=self.weighted_metrics[key])

    def fit(self, key, sampler, batch_size=1, epochs=1, callbacks=None, validation_sampler=None, validation_key=None, validation_freq=1, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE):
        assert key in self.creators, "There is no creator available for this key."
        assert len(self.creators[key].outputs) >= 2, "Creators must output [x, y, sample_weight] and to use fit at least [x, y]."
        assert isinstance(self.creators[key].outputs[0].transformer, KerasModel), "To use fit, x must be the output of a KerasModel transformer."
        x = self.creators[key].outputs[0]
        fit_dataset = TfDataset(Creator([x.transfomer.connections[x.idx], *self.creators[key].outputs[1:]]), sampler=sampler, batch_size=batch_size, num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size, shuffle=True)
        validation_fit_dataset = None
        if validation_sampler is not None:
            assert validation_key in self.creators, "There is no creator available for this validation_key."
            assert len(self.creators[validation_key].outputs) >= 2, "Creators must output [x, y, sample_weight] and to use fit at least [x, y]."
            assert isinstance(self.creators[validation_key].outputs[0].transformer, KerasModel), "To use fit, x must be the output of a KerasModel transformer."
            assert self.creators[key].outputs[0].transformer.keras_model is self.creators[validation_key].outputs[0].transformer.keras_model, "The Keras model for training and validation must be the same."
            validation_x = self.creators[validation_key].outputs[0]
            validation_fit_dataset = TfDataset(Creator([validation_x.transfomer.connections[validation_x.idx], *self.creators[validation_key].outputs[1:]]), sampler=validation_sampler, batch_size=None, num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size, shuffle=False)

        return self.keras_model.fit(x=fit_dataset, epochs=epochs, callbacks=callbacks, validation_data=validation_fit_dataset, validation_freq=validation_freq)

    def evaluate(self, sampler, key, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE):
        assert len(self.outputs[key]) >= 2, "The requested outputs are not appropriate to use evaluate."
        return self.keras_model.evaluate(x=TfDataset(sampler, Creator([self.inputs[key], *self.outputs[key][1:]]), num_parallel_calls=num_parallel_calls, prefetch_size=prefetch_size), return_dict=True)

    def predict(self, sampler, key, num_parallel_calls=tf.data.experimental.AUTOTUNE, prefetch_size=tf.data.experimental.AUTOTUNE):
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
        keras_model_connection = [connection for connection in Creator.trace(dvn_outputs)[1] if isinstance(connection.transformer, KerasModel)][0]
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

    def save(self, file_dir):
        self.save_model(self, file_dir)

    @staticmethod
    def save_model(dvn_model, file_dir):
        unique_keras_models = {}
        keras_model = dvn_model.keras_model
        dvn_model.keras_model = None
        creator_keras_models = {}
        for key in dvn_model.creator:
            creator_keras_models[key] = dvn_model.creator[key].clear_keras_model()

        unique_keras_models = {}
        for key in


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
