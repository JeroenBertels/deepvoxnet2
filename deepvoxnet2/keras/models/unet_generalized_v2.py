"""
Largely based on:
Isensee F., Kickingereder P., Wick W., Bendszus M., Maier-Hein K.H. (2019) No New-Net. In: Crimi A., Bakas S., Kuijf H., Keyvan F., Reyes M., van Walsum T. (eds) Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries. BrainLes 2018. Lecture Notes in Computer Science, vol 11384. Springer, Cham. https://doi.org/10.1007/978-3-030-11726-9_21

Compared to unet_generalized.py the default arguments now lead to the implementation from Isensee 2019 (apart from copu upsampling instead of linear (doesnt really matter when followed with additional 3x3x3 conv layers)).
And further:
- possible to have a siam network by using number_siam_pathways > 1
- possible to have multi-output network by using the extra_output... arguments
- possible to use dynamic input shapes so that you can use a larger patch size during inference
"""

import numpy as np
from deepvoxnet2.utilities.calculate_gpu_memory import get_model_memory_usage


def create_generalized_unet_v2_model(
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
        output_size=(128, 128, 128),
        metadata_sizes=None,
        metadata_number_features=None,
        metadata_dropout=None,
        metadata_at_common_pathway_layer=None,
        padding='same',
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
        verbose=True,
        input_interpolation="nearest",
        number_siam_pathways=1,
        extra_output_kernel_sizes=None,
        extra_output_number_features=None,
        extra_output_dropout=None,
        extra_output_at_common_pathway_layer=None,
        extra_output_activation_final_layer=None,
        dynamic_input_shapes=False):
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
        if dynamic_input_shapes:
            metadata_input_ = metadata_path = Input(shape=(None, None, None, metadata_sizes[i][-1] if isinstance(metadata_sizes[i], tuple) else metadata_sizes[i]))

        else:
            metadata_input_ = metadata_path = Input(shape=metadata_sizes[i] if isinstance(metadata_sizes[i], tuple) else (1, 1, 1, metadata_sizes[i]))

        metadata_inputs[i] = metadata_input_
        for j, (m_n_f, m_d) in enumerate(zip(metadata_number_features[i], metadata_dropout[i])):
            metadata_path = Dropout(m_d)(metadata_path)
            metadata_path = Conv3D(filters=m_n_f, kernel_size=(1, 1, 1), padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg))(metadata_path)
            metadata_path = activation_function("m{}_activation{}".format(i, j))(metadata_path)

        if not isinstance(metadata_sizes[i], tuple):
            broadcast_shape = K.concatenate([K.shape(path)[:-1], [metadata_sizes[i]]])
            metadata_path = K.tile(metadata_path, broadcast_shape)

        return Concatenate(axis=-1)([path, metadata_path])

    def retrieve_extra_output(path, i):
        extra_output_path = path
        for j, (e_o_k_s, e_o_n_f, e_o_d) in enumerate(zip(extra_output_kernel_sizes[i], extra_output_number_features[i], extra_output_dropout[i])):
            extra_output_path = Dropout(e_o_d)(extra_output_path)
            extra_output_path = Conv3D(filters=e_o_n_f, kernel_size=e_o_k_s, activation=extra_output_activation_final_layer[i] if j + 1 == len(extra_output_number_features[i]) else None, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg), name=f"s{i + 1}" if j + 1 == len(extra_output_number_features[i]) else None)(extra_output_path)
            if j + 1 < len(extra_output_number_features[i]):
                if (batch_normalization or instance_normalization) and not relaxed_normalization_scheme:
                    extra_output_path = normalization_function()(extra_output_path)

                extra_output_path = activation_function("eo{}_activation{}".format(i, j))(extra_output_path)

        return extra_output_path

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

    if extra_output_number_features in [None, [], ()]:
        extra_output_number_features = ()
        if extra_output_kernel_sizes in [None, [], ()]:
            extra_output_kernel_sizes = ()

        else:
            raise ValueError("Invalid value for extra_output_kernel_sizes when there is no extra_output")

        if extra_output_dropout in [None, [], ()]:
            extra_output_dropout = ()

        else:
            raise ValueError("Invalid value for extra_output_dropout when there is no extra_output")

        if extra_output_at_common_pathway_layer in [None, [], ()]:
            extra_output_at_common_pathway_layer = ()

        else:
            raise ValueError("Invalid value for extra_output_at_common_pathway_layer when there is no extra_output")

        if extra_output_activation_final_layer in [None, [], ()]:
            extra_output_activation_final_layer = ()

        else:
            raise ValueError("Invalid value for extra_output_activation_final_layer when there is no extra_output")

    if not len(extra_output_activation_final_layer) == len(extra_output_dropout) == len(extra_output_number_features) == len(extra_output_kernel_sizes) == len(extra_output_at_common_pathway_layer):
        raise ValueError("Inconsistent depth of extra_output pathway.")

    if residual_connections and dense_connections:
        raise ValueError("Residual connections and Dense connections should not be used together.")

    if dynamic_input_shapes and (residual_connections or dense_connections):
        raise NotImplementedError("Currently using residual or dense connections are not supported when using dynamic input shapes!")

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
    if input_interpolation == "mean":
        field_of_view *= np.array(subsample_factors_per_pathway[0])

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
    possible_path_left_output_sizes, possible_path_right_input_sizes, possible_crops = [], [], []
    for i in range(3):
        possible_path_left_output_sizes_, possible_path_right_input_sizes_, possible_crops_ = [], [], []
        for j, o_s in enumerate(output_sizes[:, i]):
            if o_s > 0:
                possible_path_left_output_sizes_.append([path_left_output_sizes_[j, i] for path_left_output_sizes_ in path_left_output_sizes])
                possible_path_right_input_sizes_.append([path_right_input_sizes_[j, i] for path_right_input_sizes_ in reversed(path_right_input_sizes)])
                possible_crops_.append([int((pplos - ppris) / 2) for pplos, ppris in zip(possible_path_left_output_sizes_[-1], possible_path_right_input_sizes_[-1])])

        possible_path_left_output_sizes.append(possible_path_left_output_sizes_)
        possible_path_right_input_sizes.append(possible_path_right_input_sizes_)
        possible_crops.append(possible_crops_)

    if not [o_s in p_o_s for o_s, p_o_s in zip(output_size, possible_output_sizes)] == [True] * 3:
        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*possible_output_sizes))
        print("\npossible input sizes (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(*possible_input_sizes))
        raise ValueError("The user defined output_size is not possible. Please choose from list above.")

    elif verbose:
        input_size = [possible_input_sizes[i][possible_output_sizes[i].index(o_s)] for i, o_s in enumerate(output_size)]
        print("input size:\t{}\t(true input size of the network)".format(input_size))
        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*possible_output_sizes))
        print("\npossible input sizes (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(*possible_input_sizes))
        crops = [possible_crops[i][possible_output_sizes[i].index(output_size[i])] for i in range(3)]
        indices_with_identical_cropping = []
        for i in range(3):
            indices_with_identical_cropping_ = []
            for j, crops_ in enumerate(possible_crops[i]):
                if all([crop == crop_ for crop, crop_ in zip(crops[i], crops_)]):
                    indices_with_identical_cropping_.append(j)

            indices_with_identical_cropping.append(indices_with_identical_cropping_)

        print("\npossible output sizes when using dynamic shapes (based on output size {}):\nx: {}\ny: {}\nz: {}".format(output_size, *[[possible_output_sizes[i][j] for j in indices_with_identical_cropping[i]] for i in range(3)]))
        print("\npossible input sizes when using dynamic shapes (based on output size {}):\nx: {}\ny: {}\nz: {}".format(output_size, *[[possible_input_sizes[i][j] for j in indices_with_identical_cropping[i]] for i in range(3)]))

    # Construct model
    inputs = []
    paths = []
    metadata_inputs = [None] * len(metadata_at_common_pathway_layer)

    # 1. Construct U part
    if dynamic_input_shapes:
        input_ = path = Input(shape=(None, None, None, number_input_features), name="siam0_input")

    else:
        input_ = path = Input(shape=list(input_size) + [number_input_features], name="siam0_input")

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
                    path = activation_function("down_p{}_activation{}".format(i, j))(path)

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
            assert number_siam_pathways == 1, "Insertion of metadata at 'x' is currently not supported with the use of siam pathways."
            path = introduce_metadata(path, i)

        # Upwards
    prev_s_f_p_p = np.array(subsample_factors_per_pathway[-1])
    for i, (s_f_p_p, k_s_p_p, n_f_p_p) in reversed(list(enumerate(zip(((1, 1, 1), (1, 1, 1)) + subsample_factors_per_pathway[:-1], (((), ()),) + kernel_sizes_per_pathway, (((), ()),) + number_features_per_pathway)))):
        path = AveragePooling3D(np.abs(path_left_output_positions[i] - path_right_input_positions[i]) + 1, strides=(1, 1, 1))(path)
        if i > 0:
            if i < nb_pathways:
                path_left = Cropping3D([crops_[i] for crops_ in crops])(path_left_output_paths[i])
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

                path = activation_function("up_p{}_activation{}".format(i, j))(path)
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

    # SIAMESE networks
    if number_siam_pathways > 1:
        outputs = [path]
        siam_model = Model(inputs=inputs, outputs=path)
        for i in range(number_siam_pathways - 1):
            inputs_ = []
            if dynamic_input_shapes:
                input__ = Input(shape=(None, None, None, number_input_features), name=f"siam{i + 1}_input")

            else:
                input_ = Input(shape=list(input_size) + [number_input_features], name=f"siam{i + 1}_input")

            inputs_.append(input_)
            inputs.append(input_)
            for j, m_a_c_p_l in enumerate(metadata_at_common_pathway_layer):
                if m_a_c_p_l == "x":
                    meta_input_ = Input(shape=(1, 1, 1, metadata_sizes[j]), name=f"siam{i + 1}_meta{j + 1}")
                    inputs_.append(meta_input_)
                    inputs.append(meta_input_)

            outputs.append(siam_model(inputs_ if len(inputs_) > 1 else inputs_[0]))

        path = Concatenate(axis=-1)(outputs)

    outputs = [None] * len(extra_output_at_common_pathway_layer)
    # 2. Construct common pathway
    for i, (n_f_c_p, k_s_c_p, d_c_p) in enumerate(zip(number_features_common_pathway, kernel_sizes_common_pathway, dropout_common_pathway)):
        if (batch_normalization or instance_normalization) and (i == 0 or not relaxed_normalization_scheme):
            path = normalization_function()(path)

        path = activation_function("c_activation{}".format(i))(path)
        for j, m_a_c_p_l in enumerate(metadata_at_common_pathway_layer):
            if m_a_c_p_l == i:
                path = introduce_metadata(path, j)

        for j, e_o_a_c_p_l in enumerate(extra_output_at_common_pathway_layer):
            if e_o_a_c_p_l == i:
                outputs[j] = retrieve_extra_output(path, j)

        if d_c_p:
            path = Dropout(d_c_p)(path)

        path = Conv3D(filters=n_f_c_p, kernel_size=k_s_c_p, activation=activation_final_layer if i + 1 == len(number_features_common_pathway) else None, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg), name="s0" if i + 1 == len(number_features_common_pathway) else None)(path)

    inputs = inputs + metadata_inputs
    outputs.insert(0, path)
    # 3. Mask the output (optionally)
    if mask_output:
        if dynamic_input_shapes:
            mask_input_ = mask_path = Input(shape=(None, None, None, K.int_shape(path)[-1]))

        else:
            mask_input_ = mask_path = Input(shape=tuple(output_size) + (K.int_shape(path)[-1],))

        inputs.append(mask_input_)
        for i, output in outputs:
            outputs[i] = Multiply()([output, mask_path])

    # 4. For example: Correct for segment sampling changes to P(X|Y) --> this adds an extra dimension because the correction is done inside loss function and weights are given with y_creator in extra dimension (can only be done for binary like this)
    if add_extra_dimension:
        for i, output in outputs:
            outputs[i] = K.expand_dims(output, -1)

    model = Model(inputs=inputs, outputs=outputs)

    # Final sanity check: were our calculations correct?
    if verbose:
        print("\nNetwork summary:")
        print(model.summary())
        model_input_shape = model.input_shape
        if not isinstance(model_input_shape, list):
            model_input_shape = [model_input_shape]

        if not dynamic_input_shapes:
            assert list(model_input_shape[0][1:-1]) == input_size
            print('With a batch size of {} this model needs {} GB on the GPU.'.format(1, get_model_memory_usage(1, model)))

        else:
            print("Since you are using dynamic_input_shapes=True we cannot calculate the memory usage, nor can we truely check the correctness of the shapes...")

    return model


if __name__ == "__main__":
    m = create_generalized_unet_siam_extra_model(
        padding="valid",
        output_size=(116, 116, 100)
        # number_siam_pathways=2,
        # extra_output_kernel_sizes=(((1, 1, 1),)),
        # extra_output_number_features=((1,),),
        # extra_output_dropout=((0,),),
        # extra_output_at_common_pathway_layer=(0,),
        # extra_output_activation_final_layer=("sigmoid",)
    )
    print("\nOkay, this is a valid network architecture.")
    print(m.outputs)
    print(m.output_names)
