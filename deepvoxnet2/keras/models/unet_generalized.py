"""
Largely based on:
Isensee F., Kickingereder P., Wick W., Bendszus M., Maier-Hein K.H. (2019) No New-Net. In: Crimi A., Bakas S., Kuijf H., Keyvan F., Reyes M., van Walsum T. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2018. Lecture Notes in Computer Science, vol 11384. Springer, Cham. https://doi.org/10.1007/978-3-030-11726-9_21
"""

import numpy as np


def create_generalized_unet_model(
        number_input_features=4,
        subsample_factors_per_pathway=(
                (1, 1, 1),
                (2, 2, 2),
                (4, 4, 4),
                (8, 8, 8),
                (16, 16, 16)
        ),
        kernel_sizes_per_pathway=(
                (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
                (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
                (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
                (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
                (((3, 3, 3), (3, 3, 3)), ())
        ),
        number_features_per_pathway=(
                ((30, 30), (30, 30)),
                ((60, 60), (60, 30)),
                ((120, 120), (120, 60)),
                ((240, 240), (240, 120)),
                ((480, 240), ())
        ),
        kernel_sizes_common_pathway=((1, 1, 1),) * 1,
        number_features_common_pathway=(1,),
        dropout_common_pathway=(0,),
        output_size=(116, 116, 116),
        metadata_sizes=None,
        metadata_number_features=None,
        metadata_dropout=None,
        metadata_at_common_pathway_layer=None,
        padding='valid',
        pooling='max',
        upsampling='copy',
        activation='lrelu',
        activation_final_layer='sigmoid',
        kernel_initializer='he_normal',
        batch_normalization=False,
        batch_normalization_on_input=False,
        instance_normalization=True,
        instance_normalization_on_input=False,
        relaxed_normalization_scheme=False,
        mask_output=False,
        residual_connections=False,
        dense_connections=False,
        add_extra_dimension=False,
        l1_reg=0.0,
        l2_reg=1e-5,
        verbose=True):
    from tensorflow.keras.layers import Input, Dropout, MaxPooling3D, Concatenate, Multiply, Add, Reshape, AveragePooling3D, Conv3D, UpSampling3D, Cropping3D, LeakyReLU, PReLU, BatchNormalization, Conv3DTranspose
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
            metadata_path = Dropout(m_d)(metadata_path)
            metadata_path = Conv3D(filters=m_n_f, kernel_size=(1, 1, 1), padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(metadata_path)
            if j < len(metadata_number_features[i]) - 1:
                metadata_path = activation_function()(metadata_path)

        metadata_path = UpSampling3D(tuple([int(s) for s in K.int_shape(path)[1:4]]))(metadata_path)
        return Concatenate(axis=-1)([path, metadata_path])

    def activation_function():
        if activation == "relu":
            activation_function_ = LeakyReLU(alpha=0)

        elif activation == "lrelu":
            activation_function_ = LeakyReLU(alpha=0.01)

        elif activation == "prelu":
            activation_function_ = PReLU(shared_axes=[1, 2, 3])

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

    # Do some sanity checks
    if not len(kernel_sizes_per_pathway) == len(number_features_per_pathway) == nb_pathways:
        raise ValueError("Inconsistent number of pathways.")

    for p in range(nb_pathways - 1):
        if not [f_next % f for f, f_next in zip(subsample_factors_per_pathway[p], subsample_factors_per_pathway[p + 1])] == [0, 0, 0]:
            raise ValueError("Each subsample factor must be a integer factor of the subsample factor of the previous pathway.")

    for k_s_p_p, n_f_p_p in zip(kernel_sizes_per_pathway, number_features_per_pathway):
        if not len(k_s_p_p) == len(n_f_p_p) == 2:
            raise ValueError("Each element in kernel_sizes_per_pathway and number_features_per_pathway must be a list of two elements, giving information about the left (downwards) and right (upwards) paths of the U-Net.")

        for k_s, n_f in zip(k_s_p_p, n_f_p_p):
            if not len(k_s) == len(n_f):
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

    if dense_connections and pooling != "avg":
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
    field_of_view = np.ones(3, dtype=int)
    for s_f_p_p, k_s_p_p in zip(subsample_factors_per_pathway, kernel_sizes_per_pathway):
        for k_s in k_s_p_p[0]:
            field_of_view += (np.array(k_s) - 1) * s_f_p_p

    for s_f_p_p, k_s_p_p in reversed(list(zip(subsample_factors_per_pathway, kernel_sizes_per_pathway))):
        for k_s in k_s_p_p[1]:
            field_of_view += (np.array(k_s) - 1) * s_f_p_p

    for k_s in kernel_sizes_common_pathway:
        field_of_view += (np.array(k_s) - 1) * subsample_factors_per_pathway[0]

    input_size = list(field_of_view - 1 + output_size)
    field_of_view = list(field_of_view)
    output_size = list(output_size)
    if verbose:
        print("\nfield of view:\t{}\t(theoretical)".format(field_of_view))
        print("output size:\t{}\t(user defined)".format(output_size))
        print("input size:\t{}\t(inferred with theoretical field of view (less meaningful if padding='same'))".format(input_size))

    # What are the possible input and output sizes?
    path_left_output_positions = []
    path_right_input_positions = []
    path_left_output_sizes = []
    path_right_input_sizes = []
    input_sizes = output_sizes = np.stack([np.arange(500)] * 3, axis=-1)
    prev_s_f_p_p = np.ones(3)
    prev_positions = np.zeros(3)
    for i, (s_f_p_p, k_s_p_p) in enumerate(zip(((1, 1, 1),) + subsample_factors_per_pathway, (((), ()),) + kernel_sizes_per_pathway)):
        s_f = s_f_p_p // prev_s_f_p_p
        prev_s_f_p_p = np.array(s_f_p_p)
        prev_positions = np.abs(prev_positions - (1 - (s_f_p_p * (s_f - 1) + 1) % 2))
        output_sizes = (output_sizes // s_f) * ((output_sizes % s_f) == 0)
        for k_s in k_s_p_p[0]:
            output_sizes -= np.array(k_s) - 1 if padding == 'valid' else 0
            prev_positions = np.abs(prev_positions - (1 - (s_f_p_p * (np.array(k_s) - 1) + 1) % 2))

        path_left_output_positions.append(prev_positions)
        path_left_output_sizes.append(output_sizes)

    prev_s_f_p_p = np.array(subsample_factors_per_pathway[-1])
    for i, (s_f_p_p, k_s_p_p) in reversed(list(enumerate(zip(((1, 1, 1), (1, 1, 1)) + subsample_factors_per_pathway[:-1], (((), ()),) + kernel_sizes_per_pathway)))):
        path_right_input_positions.append(prev_positions)
        path_right_input_sizes.append(output_sizes)
        output_sizes = output_sizes - np.abs(path_left_output_positions[i] - prev_positions)
        output_sizes *= ((path_left_output_sizes[i] - output_sizes) % 2) == 0
        prev_positions = path_left_output_positions[i]
        for k_s in k_s_p_p[1]:
            output_sizes -= (np.array(k_s) - 1) if padding == 'valid' else 0
            prev_positions = np.abs(prev_positions - (1 - (s_f_p_p * (np.array(k_s) - 1) + 1) % 2))

        s_f = prev_s_f_p_p // s_f_p_p
        prev_s_f_p_p = np.array(s_f_p_p)
        output_sizes *= s_f
        output_sizes = output_sizes - s_f + 1 if upsampling == "linear" else output_sizes

    for k_s in kernel_sizes_common_pathway:
        output_sizes -= (np.array(k_s) - 1) if padding == 'valid' else 0

    path_right_input_positions.reverse()
    possible_input_sizes = [list(input_sizes[output_sizes[:, i] > 0, i]) for i in range(3)]
    possible_output_sizes = [list(output_sizes[output_sizes[:, i] > 0, i].astype('int')) for i in range(3)]
    if not [o_s in p_o_s for o_s, p_o_s in zip(output_size, possible_output_sizes)] == [True] * 3:
        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*possible_output_sizes))
        print("\npossible input sizes (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(*possible_input_sizes))
        raise ValueError("The user defined output_size is not possible. Please choose from list above.")

    elif verbose:
        input_size = [possible_input_sizes[i][possible_output_sizes[i].index(o_s)] for i, o_s in enumerate(output_size)]
        print("input size:\t{}\t(true input size of the network)".format(input_size))
        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*possible_output_sizes))
        print("\npossible input sizes (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(*possible_input_sizes))

    # Construct model
    inputs = []
    paths = []

    # 1. Construct U part
    input_ = path = Input(shape=list(input_size) + [number_input_features])
    inputs.append(input_)
    path_left_output_paths = []
    path_right_output_paths = []
    # Downwards
    prev_s_f_p_p = np.ones(3)
    for i, (s_f_p_p, k_s_p_p, n_f_p_p) in enumerate(zip(((1, 1, 1),) + subsample_factors_per_pathway, (((), ()),) + kernel_sizes_per_pathway, (((), ()),) + number_features_per_pathway)):
        s_f = s_f_p_p // prev_s_f_p_p
        prev_s_f_p_p = np.array(s_f_p_p)
        path = pooling_function(s_f)(path)
        if i > 0:
            if i == 1 and (batch_normalization_on_input or instance_normalization_on_input):
                path = normalization_function()(path)

            for j, (k_s, n_f) in enumerate(zip(k_s_p_p[0], n_f_p_p[0])):
                if j == 0 and (residual_connections or dense_connections):
                    shortcut = path

                elif (0 < j < len(k_s_p_p[0]) - 1) and dense_connections:
                    shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(shortcut)[1:-1], K.int_shape(path)[1:-1])])(shortcut)
                    shortcut = Concatenate(axis=-1)([path, shortcut])
                    path = shortcut

                if ((i > 1 or j > 0) and (batch_normalization or instance_normalization)) and (not relaxed_normalization_scheme or j == len(k_s_p_p[0]) - 1):
                    path = normalization_function()(path)

                if i > 1 or j > 0:
                    path = activation_function()(path)

                path = Conv3D(filters=n_f, kernel_size=k_s, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg), name="{}_{}".format(i, j))(path)
                if j == len(k_s_p_p[0]) - 1 and residual_connections:
                    shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(shortcut)[1:-1], K.int_shape(path)[1:-1])])(shortcut)
                    if K.int_shape(path)[-1] != K.int_shape(shortcut)[-1]:
                        shortcut = Conv3D(filters=K.int_shape(path)[-1], kernel_size=(1, 1, 1), padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(shortcut)

                    path = Add()([path, shortcut])
        path_left_output_paths.append(path)
        paths.append(path)

        # Metadata
    for i, m_a_c_p_l in enumerate(metadata_at_common_pathway_layer):
        if m_a_c_p_l == "x":
            path = introduce_metadata(path, i)

        # Upwards
    prev_s_f_p_p = np.array(subsample_factors_per_pathway[-1])
    for i, (s_f_p_p, k_s_p_p, n_f_p_p) in reversed(list(enumerate(zip(((1, 1, 1), (1, 1, 1)) + subsample_factors_per_pathway[:-1], (((), ()),) + kernel_sizes_per_pathway, (((), ()),) + number_features_per_pathway)))):
        path = AveragePooling3D(np.abs(path_left_output_positions[i] - path_right_input_positions[i]) + 1, strides=(1, 1, 1))(path)
        if i > 0:
            if i < nb_pathways:
                path_left = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(path_left_output_paths[i])[1:-1], K.int_shape(path)[1:-1])])(path_left_output_paths[i])
                path = Concatenate(axis=-1)([path_left, path])

            for j, (k_s, n_f) in enumerate(zip(k_s_p_p[1], n_f_p_p[1])):
                if j == 0 and (residual_connections or dense_connections):
                    shortcut = path

                elif (0 < j < len(k_s_p_p[1]) - 1) and dense_connections:
                    shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(shortcut)[1:-1], K.int_shape(path)[1:-1])])(shortcut)
                    shortcut = Concatenate(axis=-1)([path, shortcut])
                    path = shortcut

                if (batch_normalization or instance_normalization) and (not relaxed_normalization_scheme or j == len(k_s_p_p[1]) - 1):
                    path = normalization_function()(path)

                path = activation_function()(path)
                path = Conv3D(filters=n_f, kernel_size=k_s, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(path)
                if j == len(k_s_p_p[1]) - 1 and residual_connections:
                    shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(K.int_shape(shortcut)[1:-1], K.int_shape(path)[1:-1])])(shortcut)
                    if K.int_shape(path)[-1] != K.int_shape(shortcut)[-1]:
                        shortcut = Conv3D(filters=K.int_shape(path)[-1], kernel_size=(1, 1, 1), padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(shortcut)

                    path = Add()([path, shortcut])

        path_right_output_paths.append(path)
        s_f = prev_s_f_p_p // s_f_p_p
        prev_s_f_p_p = np.array(s_f_p_p)
        path = upsampling_function(s_f)(path)
        paths.append(path)

    # 2. Construct common pathway
    for i, (n_f_c_p, k_s_c_p, d_c_p) in enumerate(zip(number_features_common_pathway, kernel_sizes_common_pathway, dropout_common_pathway)):
        for j, m_a_c_p_l in enumerate(metadata_at_common_pathway_layer):
            if m_a_c_p_l == i:
                path = introduce_metadata(path, j)

        if (batch_normalization or instance_normalization) and not relaxed_normalization_scheme:
            path = normalization_function()(path)

        path = activation_function()(path)
        if d_c_p:
            path = Dropout(d_c_p)(path)

        path = Conv3D(filters=n_f_c_p, kernel_size=k_s_c_p, activation=activation_final_layer if i + 1 == len(number_features_common_pathway) else None, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(path)

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

        assert list(model_input_shape[0][1:-1]) == input_size

    return model


if __name__ == "__main__":
    m = create_generalized_unet_model()
    print("\nOkay, this is a valid network architecture.")
