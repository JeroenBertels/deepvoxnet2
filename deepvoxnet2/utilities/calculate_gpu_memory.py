"""
    From: https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
"""


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__

        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)

        single_layer_mem = 1
        for s in l.output_shape:
            if isinstance(s, tuple):
                for si in s:
                    if si is None:
                        continue

                    single_layer_mem *= si

            else:
                if s is None:
                    continue

                single_layer_mem *= s

        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0

    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes
