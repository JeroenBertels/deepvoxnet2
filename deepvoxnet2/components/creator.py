import uuid
import copy
from deepvoxnet2.components.transformers import _Input, KerasModel


class Creator(object):
    def __init__(self, outputs):
        outputs = outputs if isinstance(outputs, list) else [outputs]
        self.outputs = Creator.deepcopy(outputs)
        self.all_transformers, self.all_connections = self.get_trace(self.outputs)
        for transformer in self.all_transformers:
            transformer.active_indices = []

        self.active_transformers, self.active_connections = self.get_trace(self.outputs, set_active_indices=True)
        self.active_input_transformers = [transformer for transformer in self.active_transformers if isinstance(transformer, _Input)]

    def eval(self, identifier=None):
        self.reset_transformers(self.active_transformers)
        for transformer in self.active_input_transformers:
            transformer.input_transformers = self.active_input_transformers
            transformer.load(identifier)

        while True:
            try:
                sample_id = uuid.uuid4()
                for output_connection in self.outputs:
                    output_connection.eval(sample_id)

            except RuntimeError:
                break

            yield copy.deepcopy([output.get() for output in self.outputs])

    @staticmethod
    def reset_transformers(transformers):
        for transformer in transformers:
            transformer.n_ = 0
            transformer.sample_id = None
            transformer.generator = None
            if isinstance(transformer, _Input):
                transformer.n_resets = 0
                transformer.input_transformers = None

    @staticmethod
    def get_trace(output_connections, set_active_indices=False):
        traced_transformers = []
        traced_connections = []
        connections = [output_connection for output_connection in output_connections]
        while len(connections) > 0:
            connection = connections.pop(0)
            if connection not in traced_connections:
                traced_connections.append(connection)
                if connection.transformer not in traced_transformers:
                    traced_transformers.append(connection.transformer)

                if set_active_indices and connection.idx not in connection.transformer.active_indices:
                    connection.transformer.active_indices.append(connection.idx)

                for connection__ in connection.transformer.extra_connections:
                    if connection__ not in traced_connections and connection__ not in connections:
                        connections.append(connection__)

                for idx, connection_ in enumerate(connection.transformer.connections):
                    for connection__ in connection_:
                        if connection__ not in traced_connections and connection__ not in connections and (not set_active_indices or idx == connection.idx):
                            connections.append(connection__)

        return traced_transformers, traced_connections

    @staticmethod
    def deepcopy(output_connections):
        keras_models = {}
        transformers, _ = Creator.get_trace(output_connections)
        Creator.reset_transformers(transformers)
        for transformer in transformers:
            if isinstance(transformer, KerasModel):
                keras_model = transformer.keras_model
                transformer.keras_model = uuid.uuid4()
                keras_models[transformer.keras_model] = keras_model

        output_connections_ = copy.deepcopy(output_connections)
        transformers_, _ = Creator.get_trace(output_connections_)
        for transformer in transformers + transformers_:
            if isinstance(transformer, KerasModel):
                transformer.keras_model = keras_models[transformer.keras_model]

        return output_connections_

    def unique_transformer_names(self):
        trans_ids = {}
        nb_transformers_present = {}
        for at in reversed(self.active_transformers):
            at_name = type(at).__name__
            mod_names = ''
            if 'MircInput' in at_name:
                for mn in at.modality_ids:
                    mod_names = mod_names + mn + ', '
                mod_names = mod_names[:-2]
            if at_name not in nb_transformers_present.keys():
                nb_transformers_present[at_name] = 1
            else:
                nb_transformers_present[at_name] = nb_transformers_present[at_name] + 1
            final_name = at_name + '_{}'.format(nb_transformers_present[at_name] - 1)
            if mod_names != '':
                final_name = final_name + ' (' + mod_names + ')'
            trans_ids[id(at)] = final_name
        return trans_ids

    def summary(self):
        done_transformers = []
        print('_' * 200)
        print('{:<30} {:<5} {:<50} {:<5} {:<5} {:<100}'.format('Transformer', '#I', 'Input(s)', '#O', 'n', 'Output shape(s)'))
        print('=' * 200)
        trans_id_names = self.unique_transformer_names()
        output_shape_per_id = {}
        for ac in reversed(self.active_connections):
            ac_transformer_id = id(ac.transformer)
            if ac_transformer_id in done_transformers:
                continue
            else:
                done_transformers.append(ac_transformer_id)
                # get the unique name
                trans_name = trans_id_names[ac_transformer_id]
                # get the number of outputs
                nb_outputs = len(ac.transformer.active_indices)
                # get the names of the inputs
                active_inputs = ''
                active_inputs_ids = []
                nb_inputs = 0
                if len(ac.transformer.connections) != 0:
                    for ai in ac.transformer.active_indices:
                        for li in ac.transformer.connections[ai]:
                            nb_inputs = nb_inputs + 1
                            if trans_id_names[id(li.transformer)] not in active_inputs:  # only add the active input if not yet present in the string
                                active_inputs = trans_id_names[id(li.transformer)] + ', ' + active_inputs
                                active_inputs_ids.append(id(li.transformer))
                    active_inputs = active_inputs[:-2]
                else:
                    active_inputs = ''
                # get the output shapes
                output_shapes = []
                if not isinstance(ac.transformer.output_shape[0], tuple):
                    output_shapes.append(ac.transformer.output_shape)
                else:
                    for os in ac.transformer.output_shape:
                        output_shapes.append(os)
                # get the number of samples generated by the transformer
                n = ac.transformer.n if ac.transformer.n is not None else ''

                # use information of the input layer output shape
                if len(active_inputs_ids) != 0:
                    output_shape_of_input = output_shape_per_id[active_inputs_ids[0]] if active_inputs_ids[0] in output_shape_per_id.keys() else [(
                        None, None, None, None, None)]
                else:
                    output_shape_of_input = [(None, None, None, None, None)]

                if 'Group' in trans_name:
                    final_output_shape = [output_shape_per_id[aiid][0] for aiid in active_inputs_ids[::-1]]
                elif 'KerasModel' in trans_name:
                    final_output_shape = output_shapes
                else:
                    final_output_shape = tuple()
                    for os, osoi in zip(output_shapes[0], output_shape_of_input[0]):
                        if os is not None:
                            final_output_shape = final_output_shape + ((os),)
                        else:
                            final_output_shape = final_output_shape + ((osoi),)
                    final_output_shape = [final_output_shape]

                output_shape_per_id[ac_transformer_id] = final_output_shape
                final_output_shape = [final_output_shape]

                # print everything
                first_line = True
                for os in final_output_shape:
                    if first_line:
                        first_line = False
                        print('{:<30} {:<5} {:<50} {:<5} {:<5} {:<100}'.format(trans_name, nb_inputs, active_inputs, nb_outputs, n, str(os)))
                    else:
                        print('{:<30} {:<5} {:<50} {:<5} {:<5} {:<100}'.format('', '', '', '', '', str(os), ))
                print('_' * 200)
        print('')
