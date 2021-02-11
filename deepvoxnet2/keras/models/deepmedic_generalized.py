"""
Largely based on:
Kamnitsas, K., Ledig, C., & Newcombe, V. F. J. (2017). Efficient Multi-Scale 3D CNN with fully connected CRF for Accurate Brain Lesion Segmentation. Medical Image Analysis, 36, 61–78.
"""

import numpy as np
from deepvoxnet2.utilities.calculate_gpu_memory import get_model_memory_usage


def create_generalized_deepmedic_model(
        number_input_features_per_pathway=(1, 1),
        subsample_factors_per_pathway=(
            (1, 1, 1),
            (3, 3, 3)
        ),
        kernel_sizes_per_pathway=(
            ((3, 3, 1), ) * 5 + ((3, 3, 3), ) * 5,
            ((3, 3, 1), ) * 5 + ((3, 3, 3), ) * 5
        ),
        number_features_per_pathway=(
            (32, ) * 5 + (48, ) * 5,
            (32, ) * 5 + (48, ) * 5
        ),
        kernel_sizes_common_pathway=((1, 1, 1), ) * 3,
        number_features_common_pathway=(150, 150, 1),
        dropout_common_pathway=(0, 0.5, 0.5),
        output_size=(22, 15, 9),
        metadata_sizes=None,
        metadata_number_features=None,
        metadata_dropout=None,
        metadata_at_common_pathway_layer=None,
        padding='valid',
        pooling='avg',  # Not used yet; set to 'avg' to allow use with dense_connection=<int>
        upsampling='copy',
        activation='prelu',
        activation_final_layer='sigmoid',
        kernel_initializer='he_normal',
        batch_normalization=False,
        batch_normalization_on_input=False,
        instance_normalization=False,
        instance_normalization_on_input=False,
        relaxed_normalization_scheme=False,  # Every <int> layers we request normalization
        mask_output=False,
        residual_connections=False,  # We group <int> layers into a residual block
        dense_connections=False,  # We group <int> layers into a densely connected block (<int> - 1 layers have dense connections and there is always 1 transition layer in between)
        add_extra_dimension=False,
        l1_reg=0.0,
        l2_reg=0.0,
        verbose=True,
        input_interpolation="nearest"):
    from tensorflow.keras.layers import Input, Dropout, MaxPooling3D, Concatenate, Multiply, Add, Reshape, Conv3DTranspose, AveragePooling3D, Conv3D, UpSampling3D, Cropping3D, LeakyReLU, PReLU, BatchNormalization
    from tensorflow.keras import regularizers
    from tensorflow.keras import backend as K
    from tensorflow_addons.layers import InstanceNormalization
    from tensorflow.keras import Model

    # Define some in-house functions
    def normalization_function():
        if batch_normalization_on_input or batch_normalization:
            normalization_function_ = BatchNormalization()

        elif instance_normalization_on_input or instance_normalization:
            normalization_function_ = InstanceNormalization()

        else:
            raise NotImplementedError

        return normalization_function_

    def introduce_metadata(path, i):
        metadata_input_ = metadata_path = Input(shape=(1, 1, 1, metadata_sizes[i]))
        inputs.append(metadata_input_)
        for j, (m_n_f, m_d) in enumerate(zip(metadata_number_features[i], metadata_dropout[i])):
            if m_d:
                metadata_path = Dropout(m_d)(metadata_path)
            metadata_path = Conv3D(filters=m_n_f, kernel_size=(1, 1, 1), padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(metadata_path)
            metadata_path = activation_function("m{}_activation{}".format(i, j))(metadata_path)

        # When the metadata is inserted, every voxel most likely corresponds to an output.
        # Currently I cannot think of a good reason it wouldn't be the case, hence the hard assert.
        path_size = K.int_shape(path)[1:4]
        assert tuple(output_size) == tuple(path_size)
        metadata_path = UpSampling3D(tuple([int(s) for s in path_size]))(metadata_path)
        return Concatenate(axis=-1)([path, metadata_path])

    def activation_function(name):
        if activation == "relu":
            activation_function_ = LeakyReLU(alpha=0, name=name)

        elif activation == "lrelu":
            activation_function_ = LeakyReLU(alpha=0.01, name=name)

        elif activation == "prelu":
            activation_function_ = PReLU(shared_axes=[1, 2, 3], name=name)

        elif activation == "linear":
            def activation_function_(path):
                return path

        else:
            raise NotImplementedError

        return activation_function_

    def pooling_function(pool_size):
        if pooling == "max":
            pooling_function_ = MaxPooling3D(pool_size, pool_size)

        elif pooling == "avg":
            pooling_function_ = AveragePooling3D(pool_size, pool_size)

        else:
            raise NotImplementedError

        return pooling_function_

    def upsampling_function(upsample_size):
        if upsampling == "copy":
            upsampling_function_ = UpSampling3D(upsample_size)

        elif upsampling == "linear":
            def upsampling_function_(path):
                path = UpSampling3D(upsample_size)(path)
                path = AveragePooling3D(upsample_size, strides=(1, 1, 1), padding='valid')(path)
                return path

        elif upsampling == "conv":
            def upsampling_function_(path):
                path = Conv3DTranspose(K.int_shape(path)[-1], upsample_size, upsample_size)(path)
                return path

        else:
            raise NotImplementedError

        return upsampling_function_

    # Define some in-house variables
    nb_pathways = len(subsample_factors_per_pathway)
    supported_activations = ["relu", "lrelu", "prelu", "linear"]
    supported_poolings = ["max", "avg"]
    supported_upsamplings = ["copy", "linear", "conv"]
    supported_paddings = ["valid", "same"]

    #Do some sanity checks
    if not len(number_input_features_per_pathway) == len(kernel_sizes_per_pathway) == len(number_features_per_pathway) == nb_pathways:
        raise ValueError("Inconsistent number of pathways.")

    for p in range(nb_pathways):
        if not len(kernel_sizes_per_pathway[p]) == len(number_features_per_pathway[p]):
            raise ValueError("Inconsistent depth of pathway #{}.".format(p))

        for ssf in subsample_factors_per_pathway[p]:
            if ssf % 2 != 1:
                raise ValueError("Subsample factors must be odd.")

    for k_s_p_p, n_f_p_p in zip(kernel_sizes_per_pathway, number_features_per_pathway):
        if not len(k_s_p_p) == len(n_f_p_p):
            raise ValueError("Each kernel size in each element from kernel_sizes_per_pathway must correspond with a number of features in each element of number_features_per_pathway.")

    if not len(kernel_sizes_common_pathway) == len(dropout_common_pathway) == len(number_features_common_pathway):
        raise ValueError("Inconsistent depth of common pathway.")

    if metadata_sizes in [None, []]:
        metadata_sizes = []
        if metadata_number_features in [None, []]:
            metadata_number_features = []

        else:
            raise ValueError("Invalid value for metadata_number_features when there is no metadata")

        if metadata_dropout in [None, []]:
            metadata_dropout = []

        else:
            raise ValueError("Invalid value for metadata_dropout when there is no metadata")

        if metadata_at_common_pathway_layer in [None, []]:
            metadata_at_common_pathway_layer = []

        else:
            raise ValueError("Invalid value for metadata_at_common_pathway_layer when there is no metadata")

    else:
        if not len(metadata_sizes) == len(metadata_dropout) == len(metadata_number_features) == len(metadata_at_common_pathway_layer):
            raise ValueError("Inconsistent depth of metadata pathway.")

    if residual_connections and dense_connections:
        raise ValueError("Residual connections and Dense connections should not be used together.")

    if dense_connections and not pooling == "avg":
        raise ValueError("According to Huang et al. a densely connected network should have average pooling.")

    if activation not in supported_activations:
        raise ValueError("The chosen activation is not supported.")

    if pooling not in supported_poolings:
        raise ValueError("The chosen pooling is not supported.")

    if upsampling not in supported_upsamplings:
        raise ValueError("The chosen upsampling is not supported.")

    if padding not in supported_paddings:
        raise ValueError("The chosen padding is not supported.")

    if (batch_normalization_on_input or batch_normalization) and (instance_normalization_on_input or instance_normalization):
        raise ValueError("You have to choose between batch or instance normalization.")

    if relaxed_normalization_scheme and not (batch_normalization or instance_normalization):
        raise ValueError("The relaxed normalization scheme can only be used if you also do (batch or instance) normalization.")

    # Calculate the field of view
    field_of_views = []
    input_sizes = []
    for p, (s_f_p_p, k_s_p_p) in enumerate(zip(subsample_factors_per_pathway, kernel_sizes_per_pathway)):
        field_of_view = np.ones(3, dtype=int)
        if input_interpolation == "mean":
            field_of_view *= np.array(s_f_p_p)

        for k_s in k_s_p_p:
            field_of_view += (np.array(k_s) - 1) * s_f_p_p

        for k_s in kernel_sizes_common_pathway:
            field_of_view += np.array(k_s) - 1

        field_of_views.append(list(field_of_view))
        input_sizes.append(list(field_of_view - 1 + output_size))

    output_size = list(output_size)
    if verbose:
        for p in range(nb_pathways):
            print("\nfield of view for pathway {}:\t{}\t(theoretical (less meaningful if padding='same' and output size is small))".format(p, field_of_views[p]))

        print("output size:\t{}\t(user defined)".format(output_size))
        for p in range(nb_pathways):
            print("input size for pathway {}:\t{}\t(inferred with theoretical field of view (less meaningful if padding='same'))".format(p, input_sizes[p]))

    # What are the possible input and output sizes?
    input_sizes = output_sizes = np.stack([np.arange(150)] * 3, axis=-1)
    for k_s in reversed(kernel_sizes_common_pathway):
        input_sizes = input_sizes + (np.array(k_s) - 1 if padding == 'valid' else 0)

    input_sizes_per_pathway = []
    sizes_per_pathway_after_upsampling = []
    for p, (s_f_p_p, k_s_p_p) in enumerate(zip(subsample_factors_per_pathway, kernel_sizes_per_pathway)):
        sizes_p_before_upsampling = np.ceil((input_sizes - (1 if upsampling == "linear" else 0)) / s_f_p_p) + (1 if upsampling == "linear" else 0)
        sizes_p_after_upsampling = (sizes_p_before_upsampling - (1 if upsampling == "linear" else 0)) * s_f_p_p + (1 if upsampling == "linear" else 0)
        input_sizes_p = sizes_p_before_upsampling
        for k_s in k_s_p_p:
            input_sizes_p = input_sizes_p + (np.array(k_s) - 1 if padding == 'valid' else 0)

        input_sizes_per_pathway.append(input_sizes_p)
        sizes_per_pathway_after_upsampling.append(sizes_p_after_upsampling)
        if p > 0:
            output_sizes[(sizes_p_after_upsampling - sizes_per_pathway_after_upsampling[0]) % 2 > 0] = 0

    possible_sizes_per_pathway_after_upsampling = [[list(sizes_per_pathway_after_upsampling[p][output_sizes[:, i] > 0, i].astype(int)) for i in range(3)] for p in range(nb_pathways)]
    possible_input_sizes_per_pathway = [[list(input_sizes_per_pathway[p][output_sizes[:, i] > 0, i].astype(int)) for i in range(3)] for p in range(nb_pathways)]
    possible_output_sizes = [list(output_sizes[output_sizes[:, i] > 0, i].astype('int')) for i in range(3)]
    if verbose and not all([o_s in p_o_s for o_s, p_o_s in zip(output_size, possible_output_sizes)]):
        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*possible_output_sizes))
        for p in range(nb_pathways):
            print("\npossible input sizes for pathway {} (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(p, *possible_input_sizes_per_pathway[p]))

        raise ValueError("The user defined output_size is not possible. Please choose from list above.")

    else:
        input_sizes = [[int(possible_input_sizes_per_pathway[p][i][possible_output_sizes[i].index(o_s)]) for i, o_s in enumerate(output_size)] for p in range(nb_pathways)]
        sizes_after_upsampling = [[int(possible_sizes_per_pathway_after_upsampling[p][i][possible_output_sizes[i].index(o_s)]) for i, o_s in enumerate(output_size)] for p in range(nb_pathways)]
        for p in range(nb_pathways):
            print("input size for pathway {}:\t{}\t(true input size of the network)".format(p, input_sizes[p]))

        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*possible_output_sizes))
        for p in range(nb_pathways):
            print("\npossible input sizes for pathway {} (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(p, *possible_input_sizes_per_pathway[p]))

    # Construct model
    inputs = []
    paths = []

    #1. Construct parallel pathways
    for p in range(nb_pathways):
        input_ = path = Input(shape=tuple(input_sizes[p]) + (number_input_features_per_pathway[p], ), name="p{}_input".format(p))
        inputs.append(input_)
        for i, (k_s_p_p, n_f_p_p) in enumerate(zip(((), ) + kernel_sizes_per_pathway[p], ((), ) + number_features_per_pathway[p])):
            if i != 0:
                path = Conv3D(filters=n_f_p_p, kernel_size=k_s_p_p, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(path)

            if dense_connections:
                if i % dense_connections != 0:
                    shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(shortcut)[1:-1], K.int_shape(path)[1:-1])])(shortcut)
                    path = Concatenate(axis=-1)([path, shortcut])

                if i + 1 != len(kernel_sizes_per_pathway[p]):
                    shortcut = path

            if residual_connections and i % residual_connections == 0:
                if i != 0:
                    shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(shortcut)[1:-1], K.int_shape(path)[1:-1])])(shortcut)
                    if K.int_shape(path)[-1] != K.int_shape(shortcut)[-1]:
                        shortcut = Conv3D(filters=K.int_shape(path)[-1], kernel_size=(1, 1, 1), padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(shortcut)

                    path = Add()([path, shortcut])

                if i + 1 != len(kernel_sizes_per_pathway[p]):
                    shortcut = path

            if not relaxed_normalization_scheme or i % relaxed_normalization_scheme == 0:
                if (i == 0 and (batch_normalization_on_input or instance_normalization_on_input)) or (i != 0 and (batch_normalization or instance_normalization)):
                    path = normalization_function()(path)

            if i != 0:
                path = activation_function("p{}_activation{}".format(p, i - 1))(path)

        path = upsampling_function(subsample_factors_per_pathway[p])(path)
        if p > 0:
            path = Cropping3D([int((l - r) / 2) for l, r in zip(sizes_after_upsampling[p], sizes_after_upsampling[0])])(path)

        paths.append(path)

    # 2. Construct common pathway
    path = Concatenate(axis=-1)(paths) if len(paths) > 1 else paths[0]
    for i, (n_f_c_p, k_s_c_p, d_c_p) in enumerate(zip(number_features_common_pathway, kernel_sizes_common_pathway, dropout_common_pathway)):
        for j, m_a_c_p_l in enumerate(metadata_at_common_pathway_layer):
            if m_a_c_p_l == i:
                path = introduce_metadata(path, j)

        if d_c_p:
            path = Dropout(d_c_p)(path)

        path = Conv3D(filters=n_f_c_p, kernel_size=k_s_c_p, activation=activation_final_layer if i + 1 == len(number_features_common_pathway) else None, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(path)
        if i + 1 < len(number_features_common_pathway):
            if batch_normalization or instance_normalization:
                path = normalization_function()(path)

            path = activation_function("c_activation{}".format(i))(path)

    # 3. Mask the output (optionally)
    if mask_output:
        mask_input_ = mask_path = Input(shape=tuple(output_size) + (K.int_shape(path)[-1],))
        inputs.append(mask_input_)
        path = Multiply()([path, mask_path])

    # 4. For example: Correct for segment sampling changes to P(X|Y) --> this adds an extra dimension because the correction is done inside loss function and weights are given with y_creator in extra dimension (can only be done for binary like this)
    if add_extra_dimension:
        path = Reshape(K.int_shape(path)[1:] + (1,))(path)

    model = Model(inputs=inputs, outputs=[path])

    # Final sanity check: were our calculations correct?
    if verbose:
        print("\nNetwork summary:")
        print(model.summary())
        model_input_shape = model.input_shape
        if not isinstance(model_input_shape, list):
            model_input_shape = [model_input_shape]

        for p in range(nb_pathways):
            assert list(model_input_shape[p][1:-1]) == input_sizes[p]

        print('With a batch size of {} this model needs {} GB on the GPU.'.format(1, get_model_memory_usage(1, model)))

    return model


def create_simplified_deepmedic_model(**kwargs):
    from tensorflow.keras.layers import Input, AveragePooling3D, Cropping3D
    from tensorflow.keras import backend as K
    from tensorflow.keras import Model

    assert np.all([n_i_f_p_p == kwargs["number_input_features_per_pathway"][0] for n_i_f_p_p in kwargs["number_input_features_per_pathway"]])
    if "input_interpolation" in kwargs:
        assert kwargs["input_interpolation"] == "mean"

    else:
        kwargs["input_interpolation"] = "mean"

    model = create_generalized_deepmedic_model(**kwargs)
    input_size = np.max([[s_f_p_p[i] * K.int_shape(input_path)[i + 1] for i in range(3)] for s_f_p_p, input_path in zip(kwargs["subsample_factors_per_pathway"], model.inputs)], axis=0)
    input = Input(tuple(input_size) + (kwargs["number_input_features_per_pathway"][0],))
    paths = []
    for s_f_p_p, input_path in zip(kwargs["subsample_factors_per_pathway"], model.inputs):
        crop_size = [input_size[i] - s_f_p_p[i] * K.int_shape(input_path)[i + 1] for i in range(3)]
        path = Cropping3D(tuple([(c_z // 2, c_z - c_z // 2) for c_z in crop_size]))(input)
        path = AveragePooling3D(tuple(s_f_p_p))(path)
        paths.append(path)

    inputs = [input] + model.inputs[len(kwargs["subsample_factors_per_pathway"]):]
    outputs = model(paths + model.inputs[len(kwargs["subsample_factors_per_pathway"]):])
    model = Model(inputs=inputs, outputs=outputs if isinstance(outputs, list) else [outputs])
    print("\nNetwork summary of simplified DeepMedic model:")
    print(model.summary())
    return model


if __name__ == "__main__":
    m = create_generalized_deepmedic_model()
    print("\nOkay, this is a valid network architecture.")
