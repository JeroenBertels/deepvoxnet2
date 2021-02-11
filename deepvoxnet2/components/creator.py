import uuid
import copy
from deepvoxnet2.components.transformers import _Input, KerasModel


class Creator(object):
    def __init__(self, outputs):
        outputs = outputs if isinstance(outputs, list) else [outputs]
        self.outputs = Creator.deepcopy(outputs)
        self.active_transformers, self.active_connections = self.trace(self.outputs, set_active_indices=True, reset_transformers=True)
        self.active_input_transformers = [transformer for transformer in self.active_transformers if isinstance(transformer, _Input)]

    def eval(self, identifier=None):
        for transformer in self.active_transformers:
            Creator.reset_transformer(transformer)
            if transformer in self.active_input_transformers:
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
    def reset_transformer(transformer, clear_active_indices=False):
        transformer.n_ = 0
        transformer.sample_id = None
        transformer.generator = None
        if isinstance(transformer, _Input):
            transformer.n_resets = 0
            transformer.input_transformers = None

        if clear_active_indices:
            transformer.active_indices = []

        return transformer

    @staticmethod
    def trace(output_connections, set_active_indices=False, reset_transformers=False, set_names=False):
        traced_transformers = []
        traced_connections = []
        connections = [output_connection for output_connection in output_connections]
        while len(connections) > 0:
            connection = connections.pop(0)
            if connection not in traced_connections:
                traced_connections.append(connection)
                if connection.transformer not in traced_transformers:
                    if reset_transformers:
                        Creator.reset_transformer(connection.transformer, clear_active_indices=set_active_indices)
                    
                    if set_names:
                        if connection.transformer.name is None:
                            connection.transformer.name = "{}_{}".format(connection.transformer.__name__, len([traced_transformer for traced_transformer in traced_transformers if traced_transformer.__name__ == connection.transformer.__name__]))
                        
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
        transformers, _ = Creator.trace(output_connections)
        for transformer in transformers:
            # Creator.reset_transformer(transformer)
            if isinstance(transformer, KerasModel):
                keras_model = transformer.keras_model
                transformer.keras_model = uuid.uuid4()
                keras_models[transformer.keras_model] = keras_model

        output_connections = copy.deepcopy(output_connections)
        transformers_, _ = Creator.trace(output_connections)
        for transformer in transformers + transformers_:
            if isinstance(transformer, KerasModel):
                transformer.keras_model = keras_models[transformer.keras_model]

        return output_connections

    def summary(self):
        print('_' * 200)
        print('{:<50} {:<50} {:<50} {:<50}'.format('Transformer (type) (n)', '#I/O', 'Output Shape', 'Connected to'))
        print('=' * 200)
        for transformer in self.active_transformers:
            for i, idx in enumerate(transformer.active_indices):
                if len(transformer.connections[idx]) == 0:
                    print('{:<50} {:<50} {:<50} {:<50}'.format("" if i > 0 else f"{transformer.name} ({type(transformer)}) ({transformer.n})", idx, transformer.output_shapes[idx], ""))

                else:
                    for idx_, connection in enumerate(transformer.connections[idx]):
                        print('{:<50} {:<50} {:<50} {:<50}'.format("" if i > 0 else f"{transformer.name} ({type(transformer)}) ({transformer.n})", idx, transformer.output_shapes[idx], f"{connection.transformer.name}[{connection.idx}]"))

            print('_' * 200)

        print('')
