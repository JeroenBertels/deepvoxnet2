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

    def eval(self, identifier=None):
        self.reset_transformers(self.active_transformers)
        for transformer in self.active_transformers:
            if isinstance(transformer, _Input):
                transformer.load(identifier)

        while True:
            sample_id = uuid.uuid4()
            for output_connection in self.outputs:
                output_connection.eval(sample_id)

            if all([transformer.n_resets > transformer.n for transformer in self.active_transformers if isinstance(transformer, _Input)]):
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
