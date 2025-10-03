import keras
import numpy as np

from keras import Model
from keras import regularizers, activations
from keras.layers import Input, Activation, Dropout, MaxPooling3D, Concatenate, Multiply, Add, Reshape, AveragePooling3D, Conv3D, UpSampling3D, Cropping3D, LeakyReLU, PReLU, BatchNormalization, Conv3DTranspose, GroupNormalization, MultiHeadAttention, LayerNormalization, Embedding, Dense


class UnetGeneralized(object):
    def __init__(self,
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
        input_interpolation="nearest",
        number_siam_pathways=1,
        extra_output_kernel_sizes=None,
        extra_output_number_features=None,
        extra_output_dropout=None,
        extra_output_at_common_pathway_layer=None,
        extra_output_activation_final_layer=None,
        dynamic_input_shapes=False,
        do_skips=True,
        bottleneck=None,
        replace_instance_normalization_with_group_norm=None,
        deep_supervision_pathways=None,
        transformer_pathways=None):
        
        self.number_input_features = number_input_features
        self.subsample_factors_per_pathway = subsample_factors_per_pathway
        self.kernel_sizes_per_pathway = kernel_sizes_per_pathway
        self.number_features_per_pathway = number_features_per_pathway
        self.kernel_sizes_common_pathway = kernel_sizes_common_pathway
        self.number_features_common_pathway = number_features_common_pathway
        self.dropout_common_pathway = dropout_common_pathway
        self.output_size = output_size
        self.metadata_sizes = metadata_sizes
        self.metadata_number_features = metadata_number_features
        self.metadata_dropout = metadata_dropout
        self.metadata_at_common_pathway_layer = metadata_at_common_pathway_layer
        self.padding = padding
        self.pooling = pooling
        self.upsampling = upsampling
        self.activation = activation
        self.activation_final_layer = activation_final_layer
        self.kernel_initializer = kernel_initializer
        self.batch_normalization = batch_normalization
        self.batch_normalization_on_input = batch_normalization_on_input
        self.instance_normalization = instance_normalization
        self.instance_normalization_on_input = instance_normalization_on_input
        self.relaxed_normalization_scheme = relaxed_normalization_scheme
        self.mask_output = mask_output
        self.residual_connections = residual_connections
        self.dense_connections = dense_connections
        self.add_extra_dimension = add_extra_dimension
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.input_interpolation = input_interpolation
        self.number_siam_pathways = number_siam_pathways
        self.extra_output_kernel_sizes = extra_output_kernel_sizes
        self.extra_output_number_features = extra_output_number_features
        self.extra_output_dropout = extra_output_dropout
        self.extra_output_at_common_pathway_layer = extra_output_at_common_pathway_layer
        self.extra_output_activation_final_layer = extra_output_activation_final_layer
        self.dynamic_input_shapes = dynamic_input_shapes
        self.do_skips = do_skips
        self.bottleneck = bottleneck
        self.replace_instance_normalization_with_group_norm = replace_instance_normalization_with_group_norm
        self.deep_supervision_pathways = deep_supervision_pathways or ()
        self.transformer_pathways = transformer_pathways or ()
        
        # initializing
        self.nb_pathways = len(subsample_factors_per_pathway)
        self.supported_activations = ["relu", "lrelu", "prelu", "linear", "gelu"]
        self.supported_poolings = ["max", "avg", "conv"]
        self.supported_upsamplings = ["copy", "linear", "conv"]
        self.supported_paddings = ["valid", "same"]
        self.do_sanity_checks_and_format_variables()
        self.set_field_of_view_and_related_stuff()
        self.inputs = []
        self.paths = []
        self.metadata_inputs = [None] * len(self.metadata_at_common_pathway_layer)
        self.outputs = [None] * len(self.extra_output_at_common_pathway_layer)
        self.model = None

    def normalization_function(self, path=None):
        if self.batch_normalization_on_input or self.batch_normalization:
            if self.replace_instance_normalization_with_group_norm and path is not None and path.shape[-1] % self.replace_instance_normalization_with_group_norm == 0:
                def normalization_function_(path):
                    _, x, y, z, c = path.shape
                    if c > self.replace_instance_normalization_with_group_norm:
                        path = Reshape((x, y, z * (c // self.replace_instance_normalization_with_group_norm), self.replace_instance_normalization_with_group_norm))(path)
                        path = BatchNormalization()(path)
                        return Reshape((x, y, z, c))(path)

                    else:
                        return BatchNormalization()(path)

            else:
                normalization_function_ = BatchNormalization()

        elif self.instance_normalization_on_input or self.instance_normalization:
            if self.replace_instance_normalization_with_group_norm:
                if self.replace_instance_normalization_with_group_norm == 1:
                    normalization_function_ = LayerNormalization()
                
                else:
                    normalization_function_ = GroupNormalization(groups=self.replace_instance_normalization_with_group_norm)
            
            else:
                normalization_function_ = GroupNormalization(groups=-1)

        else:
            raise NotImplementedError

        return normalization_function_
    
    def introduce_metadata(self, path, i):
        if self.dynamic_input_shapes:
            metadata_input_ = metadata_path = Input(shape=(None, None, None, self.metadata_sizes[i][-1] if isinstance(self.metadata_sizes[i], tuple) else self.metadata_sizes[i]))

        else:
            metadata_input_ = metadata_path = Input(shape=self.metadata_sizes[i] if isinstance(self.metadata_sizes[i], tuple) else (1, 1, 1, self.metadata_sizes[i]))

        self.metadata_inputs[i] = metadata_input_
        for j, (m_n_f, m_d) in enumerate(zip(self.metadata_number_features[i], self.metadata_dropout[i])):
            metadata_path = Dropout(m_d)(metadata_path)
            metadata_path = Conv3D(filters=m_n_f, kernel_size=(1, 1, 1), padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(metadata_path)
            metadata_path = self.activation_function("m{}_activation{}".format(i, j))(metadata_path)

        if not isinstance(self.metadata_sizes[i], tuple):
            broadcast_shape = keras.ops.concatenate([keras.ops.ones((1,), dtype="int32"), path.shape[1:-1], keras.ops.ones((1,), dtype="int32")])
            metadata_path = keras.ops.tile(metadata_path, broadcast_shape)

        return Concatenate(axis=-1)([path, metadata_path])
    
    def retrieve_extra_output(self, path, i):
        assert self.deep_supervision_pathways in [None, (0,), ()], "For now, no extra outputs allowed when using deep supervision."
        extra_output_path = path
        for j, (e_o_k_s, e_o_n_f, e_o_d) in enumerate(zip(self.extra_output_kernel_sizes[i], self.extra_output_number_features[i], self.extra_output_dropout[i])):
            extra_output_path = Dropout(e_o_d)(extra_output_path)
            extra_output_path = Conv3D(filters=e_o_n_f, kernel_size=e_o_k_s, activation=self.extra_output_activation_final_layer[i] if j + 1 == len(self.extra_output_number_features[i]) else None, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg) if j + 1 != len(self.extra_output_number_features[i]) else None, name=f"s{i + 1}" if j + 1 == len(self.extra_output_number_features[i]) else None)(extra_output_path)
            if j + 1 < len(self.extra_output_number_features[i]):
                if (self.batch_normalization or self.instance_normalization) and not self.relaxed_normalization_scheme:
                    extra_output_path = self.normalization_function(extra_output_path)(extra_output_path)

                extra_output_path = self.activation_function("eo{}_activation{}".format(i, j))(extra_output_path)

        return extra_output_path
    
    def activation_function(self, name):
        if self.activation == "relu":
            activation_function_ = LeakyReLU(alpha=0, name=name)

        elif self.activation == "lrelu":
            activation_function_ = LeakyReLU(alpha=0.01, name=name)

        elif self.activation == "prelu":
            activation_function_ = PReLU(shared_axes=[1, 2, 3], name=name)

        elif self.activation == "gelu":
            activation_function_ = Activation(activations.gelu, name=name)

        elif self.activation == "linear":
            def activation_function_(path):
                return path

        else:
            raise NotImplementedError

        return activation_function_
    
    def pooling_function(self, pool_size, i):
        if all([int(p_s) == 1 for p_s in pool_size]):
            def pooling_function_(path):
                return path
        
        else:
            if self.pooling == "max":
                pooling_function_ = MaxPooling3D(pool_size, pool_size)

            elif self.pooling == "avg":
                pooling_function_ = AveragePooling3D(pool_size, pool_size)

            elif self.pooling == "conv":
                pool_size = [int(p_s) for p_s in pool_size]
                def pooling_function_(path):
                    if i < 2:
                        return AveragePooling3D(pool_size, pool_size)(path)

                    else:
                        path = self.normalization_function(path)(path)
                        path = self.activation_function("pool_p{}_activation".format(i))(path)
                        return Conv3D(filters=path.shape[-1], kernel_size=pool_size, strides=pool_size, padding="valid", kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(path)
                
            else:
                raise NotImplementedError

        return pooling_function_

    def upsampling_function(self, upsample_size, i):
        if all([int(u_s) == 1 for u_s in upsample_size]):
            def upsampling_function_(path):
                return path
        
        else:
            if self.upsampling == "copy":
                upsampling_function_ = UpSampling3D(upsample_size)

            elif self.upsampling == "linear":
                def upsampling_function_(path):
                    path = UpSampling3D(upsample_size)(path)
                    return AveragePooling3D(upsample_size, strides=(1, 1, 1), padding='valid')(path)

            elif self.upsampling == "conv":
                def upsampling_function_(path):
                    if i < 2:
                        path = UpSampling3D(upsample_size)(path)
                        return AveragePooling3D(upsample_size, strides=(1, 1, 1), padding='valid')(path)
                    
                    else:
                        path = self.normalization_function(path)(path)
                        path = self.activation_function("upsample_p{}_activation".format(i))(path)
                        return Conv3DTranspose(path.shape[-1], upsample_size, upsample_size, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(path)

            else:
                raise NotImplementedError

        return upsampling_function_
    
    def do_sanity_checks_and_format_variables(self):
        if not len(self.kernel_sizes_per_pathway) == len(self.number_features_per_pathway) == self.nb_pathways:
            raise ValueError("Inconsistent number of pathways.")

        for p in range(self.nb_pathways - 1):
            if not [f_next % f for f, f_next in zip(self.subsample_factors_per_pathway[p], self.subsample_factors_per_pathway[p + 1])] == [0, 0, 0]:
                raise ValueError("Each subsample factor must be a integer factor of the subsample factor of the previous pathway.")

        for k_s_p_p, n_f_p_p in zip(self.kernel_sizes_per_pathway, self.number_features_per_pathway):
            if not len(k_s_p_p) == len(n_f_p_p) == 2:
                raise ValueError("Each element in kernel_sizes_per_pathway and number_features_per_pathway must be a list of two elements, giving information about the left (downwards) and right (upwards) paths of the U-Net.")

            for k_s, n_f in zip(k_s_p_p, n_f_p_p):
                if not len(k_s) == len(n_f):
                    raise ValueError("Each kernel size in each element from kernel_sizes_per_pathway must correspond with a number of features in each element of number_features_per_pathway.")

        if not len(self.kernel_sizes_common_pathway) == len(self.dropout_common_pathway) == len(self.number_features_common_pathway):
            raise ValueError("Inconsistent depth of common pathway.")

        if self.metadata_sizes in [None, []]:
            self.metadata_sizes = []
            if self.metadata_number_features in [None, []]:
                self.metadata_number_features = []

            else:
                raise ValueError("Invalid value for metadata_number_features when there is no metadata")

            if self.metadata_dropout in [None, []]:
                self.metadata_dropout = []

            else:
                raise ValueError("Invalid value for metadata_dropout when there is no metadata")

            if self.metadata_at_common_pathway_layer in [None, []]:
                self.metadata_at_common_pathway_layer = []

            else:
                raise ValueError("Invalid value for metadata_at_common_pathway_layer when there is no metadata")

        else:
            if not len(self.metadata_sizes) == len(self.metadata_dropout) == len(self.metadata_number_features) == len(self.metadata_at_common_pathway_layer):
                raise ValueError("Inconsistent depth of metadata pathway.")

        if self.extra_output_number_features in [None, [], ()]:
            self.extra_output_number_features = ()
            if self.extra_output_kernel_sizes in [None, [], ()]:
                self.extra_output_kernel_sizes = ()

            else:
                raise ValueError("Invalid value for extra_output_kernel_sizes when there is no extra_output")

            if self.extra_output_dropout in [None, [], ()]:
                self.extra_output_dropout = ()

            else:
                raise ValueError("Invalid value for extra_output_dropout when there is no extra_output")

            if self.extra_output_at_common_pathway_layer in [None, [], ()]:
                self.extra_output_at_common_pathway_layer = ()

            else:
                raise ValueError("Invalid value for extra_output_at_common_pathway_layer when there is no extra_output")

            if self.extra_output_activation_final_layer in [None, [], ()]:
                self.extra_output_activation_final_layer = ()

            else:
                raise ValueError("Invalid value for extra_output_activation_final_layer when there is no extra_output")

        if not len(self.extra_output_activation_final_layer) == len(self.extra_output_dropout) == len(self.extra_output_number_features) == len(self.extra_output_kernel_sizes) == len(self.extra_output_at_common_pathway_layer):
            raise ValueError("Inconsistent depth of extra_output pathway.")

        if self.residual_connections and self.dense_connections:
            raise ValueError("Residual connections and Dense connections should not be used together.")

        if self.dynamic_input_shapes and (self.residual_connections or self.dense_connections):
            raise NotImplementedError("Currently using residual or dense connections are not supported when using dynamic input shapes!")

        if self.dense_connections and self.pooling != "avg":
            raise ValueError("According to Huang et al. a densely connected network should have average pooling.")

        if self.activation not in self.supported_activations:
            raise ValueError("The chosen activation is not supported.")

        if self.pooling not in self.supported_poolings:
            raise ValueError("The chosen pooling is not supported.")

        if self.upsampling not in self.supported_upsamplings:
            raise ValueError("The chosen upsampling is not supported.")

        if self.padding not in self.supported_paddings:
            raise ValueError("The chosen padding is not supported.")

        if (self.batch_normalization_on_input or self.batch_normalization) and (self.instance_normalization_on_input or self.instance_normalization):
            raise ValueError("You have to choose between batch or instance normalization.")

        if self.relaxed_normalization_scheme and not (self.batch_normalization or self.instance_normalization):
            raise ValueError("The relaxed normalization scheme can only be used if you also do (batch or instance) normalization.")

    def set_field_of_view_and_related_stuff(self):
        self.field_of_view = np.ones(3, dtype=int)
        if self.input_interpolation == "mean":
            self.field_of_view *= np.array(self.subsample_factors_per_pathway[0])

        for s_f_p_p, k_s_p_p in zip(self.subsample_factors_per_pathway, self.kernel_sizes_per_pathway):
            for k_s in k_s_p_p[0]:
                self.field_of_view += (np.array(k_s) - 1) * s_f_p_p

        for s_f_p_p, k_s_p_p in reversed(list(zip(self.subsample_factors_per_pathway, self.kernel_sizes_per_pathway))):
            for k_s in k_s_p_p[1]:
                self.field_of_view += (np.array(k_s) - 1) * s_f_p_p

        for k_s in self.kernel_sizes_common_pathway:
            self.field_of_view += (np.array(k_s) - 1) * self.subsample_factors_per_pathway[0]

        self.input_size = list(self.field_of_view - 1 + self.output_size)
        self.field_of_view = list(self.field_of_view)
        self.output_size = list(self.output_size)
        print("\nfield of view:\t{}\t(theoretical)".format(self.field_of_view))
        print("output size:\t{}\t(user defined)".format(self.output_size))
        print("input size:\t{}\t(inferred with theoretical field of view (less meaningful if padding='same'))".format(self.input_size))

        # What are the possible input and output sizes?
        self.path_left_output_positions = []
        self.path_right_input_positions = []
        self.path_left_output_sizes = []
        self.path_right_input_sizes = []
        self.input_sizes = self.output_sizes = np.stack([np.arange(500)] * 3, axis=-1)
        prev_s_f_p_p = np.ones(3)
        prev_positions = np.zeros(3)
        for i, (s_f_p_p, k_s_p_p) in enumerate(zip(((1, 1, 1),) + self.subsample_factors_per_pathway, (((), ()),) + self.kernel_sizes_per_pathway)):
            s_f = s_f_p_p // prev_s_f_p_p
            prev_s_f_p_p = np.array(s_f_p_p)
            prev_positions = np.abs(prev_positions - (1 - (s_f_p_p * (s_f - 1) + 1) % 2))
            self.output_sizes = (self.output_sizes // s_f) * ((self.output_sizes % s_f) == 0)
            for k_s in k_s_p_p[0]:
                self.output_sizes -= np.array(k_s) - 1 if self.padding == 'valid' else 0
                prev_positions = np.abs(prev_positions - (1 - (s_f_p_p * (np.array(k_s) - 1) + 1) % 2))

            self.path_left_output_positions.append(prev_positions)
            self.path_left_output_sizes.append(self.output_sizes)

        prev_s_f_p_p = np.array(self.subsample_factors_per_pathway[-1])
        for i, (s_f_p_p, k_s_p_p) in reversed(list(enumerate(zip(((1, 1, 1), (1, 1, 1)) + self.subsample_factors_per_pathway[:-1], (((), ()),) + self.kernel_sizes_per_pathway)))):
            self.path_right_input_positions.append(prev_positions)
            self.path_right_input_sizes.append(self.output_sizes)
            self.output_sizes = self.output_sizes - np.abs(self.path_left_output_positions[i] - prev_positions)
            if self.do_skips:
                self.output_sizes *= ((self.path_left_output_sizes[i] - self.output_sizes) % 2) == 0

            prev_positions = self.path_left_output_positions[i]
            for k_s in k_s_p_p[1]:
                self.output_sizes -= (np.array(k_s) - 1) if self.padding == 'valid' else 0
                prev_positions = np.abs(prev_positions - (1 - (s_f_p_p * (np.array(k_s) - 1) + 1) % 2))

            s_f = prev_s_f_p_p // s_f_p_p
            prev_s_f_p_p = np.array(s_f_p_p)
            self.output_sizes *= s_f
            self.output_sizes = self.output_sizes - s_f + 1 if self.upsampling == "linear" else self.output_sizes

        for k_s in self.kernel_sizes_common_pathway:
            self.output_sizes -= (np.array(k_s) - 1) if self.padding == 'valid' else 0

        self.path_right_input_positions.reverse()
        self.possible_input_sizes = [list(self.input_sizes[self.output_sizes[:, i] > 0, i]) for i in range(3)]
        self.possible_output_sizes = [list(self.output_sizes[self.output_sizes[:, i] > 0, i].astype('int')) for i in range(3)]
        self.possible_path_left_output_sizes, self.possible_path_right_input_sizes, self.possible_crops = [], [], []
        for i in range(3):
            possible_path_left_output_sizes_, possible_path_right_input_sizes_, possible_crops_ = [], [], []
            for j, o_s in enumerate(self.output_sizes[:, i]):
                if o_s > 0:
                    possible_path_left_output_sizes_.append([path_left_output_sizes_[j, i] for path_left_output_sizes_ in self.path_left_output_sizes])
                    possible_path_right_input_sizes_.append([path_right_input_sizes_[j, i] for path_right_input_sizes_ in reversed(self.path_right_input_sizes)])
                    possible_crops_.append([int((pplos - ppris) / 2) for pplos, ppris in zip(possible_path_left_output_sizes_[-1], possible_path_right_input_sizes_[-1])])

            self.possible_path_left_output_sizes.append(possible_path_left_output_sizes_)
            self.possible_path_right_input_sizes.append(possible_path_right_input_sizes_)
            self.possible_crops.append(possible_crops_)

        if not [o_s in p_o_s for o_s, p_o_s in zip(self.output_size, self.possible_output_sizes)] == [True] * 3:
            print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*self.possible_output_sizes))
            print("\npossible input sizes (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(*self.possible_input_sizes))
            raise ValueError("The user defined output_size is not possible. Please choose from list above.")

        self.input_size = [self.possible_input_sizes[i][self.possible_output_sizes[i].index(o_s)].item() for i, o_s in enumerate(self.output_size)]
        print("input size:\t{}\t(true input size of the network)".format(self.input_size))
        print("\npossible output sizes:\nx: {}\ny: {}\nz: {}".format(*self.possible_output_sizes))
        print("\npossible input sizes (corresponding with the possible output sizes):\nx: {}\ny: {}\nz: {}".format(*self.possible_input_sizes))
        self.crops = [self.possible_crops[i][self.possible_output_sizes[i].index(self.output_size[i])] for i in range(3)]
        indices_with_identical_cropping = []
        for i in range(3):
            indices_with_identical_cropping_ = []
            for j, crops_ in enumerate(self.possible_crops[i]):
                if all([crop == crop_ for crop, crop_ in zip(self.crops[i], crops_)]):
                    indices_with_identical_cropping_.append(j)

            indices_with_identical_cropping.append(indices_with_identical_cropping_)

        print("\npossible output sizes when using dynamic shapes (based on output size {}):\nx: {}\ny: {}\nz: {}".format(self.output_size, *[[self.possible_output_sizes[i][j] for j in indices_with_identical_cropping[i]] for i in range(3)]))
        print("\npossible input sizes when using dynamic shapes (based on output size {}):\nx: {}\ny: {}\nz: {}".format(self.output_size, *[[self.possible_input_sizes[i][j] for j in indices_with_identical_cropping[i]] for i in range(3)]))
    
    def multi_layer_perceptron(self, p, hidden_units=(None, None), dropout_rate=0.1, pathway_index=0, mlp_index=0, expansion_factor=2):
        assert len(p.shape) == 3
        hidden_units = [u or p.shape[-1] * (1 if i + 1 == len(hidden_units) else expansion_factor) for i, u in enumerate(hidden_units)]
        for i, units in enumerate(hidden_units):
            p = Dense(units, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(p)
            if i < len(hidden_units) - 1: 
                p = self.normalization_function()(p)
                p = self.activation_function(f"p{pathway_index}_mlp{mlp_index}_activation{i}")(p)
                p = Dropout(dropout_rate)(p)

        return p

    def transformer_block(self, p, nb_transformer_layers=2, nb_heads=4, mlp_units=(None, None), dropout_rate=0.1, pathway_index=0, mlp_expansion_factor=2):
        _, x, y, z, f = p.shape
        p = Reshape((x * y * z, f))(p)
        # positions = keras.ops.arange(x * y * z)
        # positions = keras.ops.tile(positions, keras.ops.concatenate([p.shape[0], keras.ops.ones((1,), dtype="int32")]))
        positions = keras.ops.arange(x * y * z)
        positions = keras.ops.expand_dims(positions, axis=0)
        embedding = Embedding(x * y * z, f)(positions)
        p = Add()([p, embedding])
        assert f % nb_heads == 0
        for i in range(nb_transformer_layers):
            # p_ = LayerNormalization()(p)
            p_ = self.normalization_function()(p)
            p_ = MultiHeadAttention(nb_heads, f // nb_heads, dropout=dropout_rate, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(p_, p_)
            p = Add()([p, p_])
            # p_ = LayerNormalization()(p)
            p_ = self.normalization_function()(p)
            p_ = self.multi_layer_perceptron(p_, mlp_units, dropout_rate=dropout_rate, pathway_index=pathway_index, mlp_index=i, expansion_factor=mlp_expansion_factor)
            p = Add()([p, p_])
        
        p = Reshape((x, y, z, f))(p)
        return p

    def set_model(self):
        print(self.input_size)
        # 1. Construct U part
        if self.dynamic_input_shapes:
            input_ = path = Input(shape=(None, None, None, self.number_input_features), name="siam0_input")

        else:
            input_ = path = Input(shape=list(self.input_size) + [self.number_input_features], name="siam0_input")

        self.inputs.append(input_)
        self.path_left_output_paths = []
        self.path_right_output_paths = []
        # Downwards
        prev_s_f_p_p = np.ones(3)
        for i, (s_f_p_p, k_s_p_p, n_f_p_p) in enumerate(zip(((1, 1, 1),) + self.subsample_factors_per_pathway, (((), ()),) + self.kernel_sizes_per_pathway, (((), ()),) + self.number_features_per_pathway)):
            s_f = s_f_p_p // prev_s_f_p_p
            prev_s_f_p_p = np.array(s_f_p_p)
            path = self.pooling_function(s_f, i)(path)
            if i > 0:
                if i == 1 and (self.batch_normalization_on_input or self.instance_normalization_on_input):
                    path = self.normalization_function(path)(path)

                for j, (k_s, n_f) in enumerate(zip(k_s_p_p[0], n_f_p_p[0])):
                    if j == 0 and (self.residual_connections or self.dense_connections):
                        shortcut = path

                    elif (0 < j < len(k_s_p_p[0]) - 1) and self.dense_connections:
                        if any([int((l - r) / 2) != 0 for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])]):
                            shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])])(shortcut)

                        shortcut = Concatenate(axis=-1)([path, shortcut])
                        path = shortcut

                    if ((i > 1 or j > 0) and (self.batch_normalization or self.instance_normalization)) and (not self.relaxed_normalization_scheme or j == len(k_s_p_p[0]) - 1):
                        path = self.normalization_function(path)(path)

                    if i > 1 or j > 0:
                        path = self.activation_function("down_p{}_activation{}".format(i, j))(path)

                    path = Conv3D(filters=n_f, kernel_size=k_s, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg), name="{}_{}".format(i, j))(path)
                    if j == len(k_s_p_p[0]) - 1 and self.residual_connections:
                        if any([int((l - r) / 2) != 0 for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])]):
                            shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])])(shortcut)

                        if path.shape[-1] != shortcut.shape[-1]:
                            shortcut = Conv3D(filters=path.shape[-1], kernel_size=(1, 1, 1), padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(shortcut)

                        path = Add()([path, shortcut])
            self.path_left_output_paths.append(path)
            self.paths.append(path)

        # Metadata
        for i, m_a_c_p_l in enumerate(self.metadata_at_common_pathway_layer):
            if m_a_c_p_l == "x":
                assert self.number_siam_pathways == 1, "Insertion of metadata at 'x' is currently not supported with the use of siam pathways."
                path = self.introduce_metadata(path, i)

        # Upwards
        prev_s_f_p_p = np.array(self.subsample_factors_per_pathway[-1])
        for i, (s_f_p_p, k_s_p_p, n_f_p_p) in reversed(list(enumerate(zip(((1, 1, 1), (1, 1, 1)) + self.subsample_factors_per_pathway[:-1], (((), ()),) + self.kernel_sizes_per_pathway, (((), ()),) + self.number_features_per_pathway)))):
            if np.any(np.abs(self.path_left_output_positions[i] - self.path_right_input_positions[i]) != 0):
                path = AveragePooling3D(np.abs(self.path_left_output_positions[i] - self.path_right_input_positions[i]) + 1, strides=(1, 1, 1))(path)

            if i > 0:
                # begin bottleneck code
                if i == self.nb_pathways and ("x" in self.extra_output_at_common_pathway_layer or self.bottleneck):
                    path_ = path
                    if self.batch_normalization or self.instance_normalization:
                        path = self.normalization_function(path)(path)

                    path = self.activation_function("pre_bottleneck_activation")(path)
                    shape_before_bottleneck = path.shape
                    path = Reshape((1, 1, 1, -1))(path)
                    if self.bottleneck and not isinstance(self.bottleneck, bool):
                        path = Conv3D(filters=self.bottleneck, kernel_size=(1, 1, 1), padding="valid", kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(path)
                        path = self.normalization_function(path)(path)
                        path = self.activation_function("bottleneck_activation")(path)
                    
                    for k, e_o_a_c_p_l in enumerate(self.extra_output_at_common_pathway_layer):
                        if e_o_a_c_p_l == "x":
                            assert self.number_siam_pathways == 1, "Extra outputs at 'x' are currently not supported with the use of siam pathways."
                            self.outputs[k] = self.retrieve_extra_output(path, k)

                    if self.bottleneck and not isinstance(self.bottleneck, bool):
                        path = Conv3D(filters=np.prod(shape_before_bottleneck[1:]), kernel_size=(1, 1, 1), padding="valid", kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(path)
                        path = Reshape(shape_before_bottleneck[1:])(path)
                    
                    else:
                        path = path_
                # end bottleneck code

                if i < self.nb_pathways and self.do_skips:
                    if any([crops_[i] != 0 for crops_ in self.crops]):
                        path_left = Cropping3D([crops_[i] for crops_ in self.crops])(self.path_left_output_paths[i])

                    else:
                        path_left = self.path_left_output_paths[i]
                    
                    if i - 1 in self.transformer_pathways:
                        path_left = self.transformer_block(path_left, pathway_index=i - 1)

                    path = Concatenate(axis=-1)([path_left, path])

                elif i == self.nb_pathways and i - 1 in self.transformer_pathways:
                    path = self.transformer_block(path, pathway_index=i - 1)

                for j, (k_s, n_f) in enumerate(zip(k_s_p_p[1], n_f_p_p[1])):
                    if j == 0 and (self.residual_connections or self.dense_connections):
                        shortcut = path

                    elif (0 < j < len(k_s_p_p[1]) - 1) and self.dense_connections:
                        if any([int((l - r) / 2) != 0 for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])]):
                            shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])])(shortcut)

                        shortcut = Concatenate(axis=-1)([path, shortcut])
                        path = shortcut

                    if (self.batch_normalization or self.instance_normalization) and (not self.relaxed_normalization_scheme or j == len(k_s_p_p[1]) - 1):
                        path = self.normalization_function(path)(path)

                    path = self.activation_function("up_p{}_activation{}".format(i, j))(path)
                    path = Conv3D(filters=n_f, kernel_size=k_s, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(path)
                    if j == len(k_s_p_p[1]) - 1 and self.residual_connections:
                        if any([int((l - r) / 2) != 0 for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])]):
                            shortcut = Cropping3D([int((l - r) / 2) for l, r in zip(shortcut.shape[1:-1], path.shape[1:-1])])(shortcut)

                        if path.shape[-1] != shortcut.shape[-1]:
                            shortcut = Conv3D(filters=path.shape[-1], kernel_size=(1, 1, 1), padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg))(shortcut)

                        path = Add()([path, shortcut])

            self.path_right_output_paths.append(path)
            s_f = prev_s_f_p_p // s_f_p_p
            prev_s_f_p_p = np.array(s_f_p_p)
            path = self.upsampling_function(s_f, i)(path)
            self.paths.append(path)

        # SIAMESE networks
        if self.number_siam_pathways > 1:
            assert self.deep_supervision_pathways in [None, (0,), ()], "For now, no siamese pathways allowed when using deep supervision."
            self.outputs = [path]
            siam_model = Model(inputs=self.inputs, outputs=path)
            for i in range(self.number_siam_pathways - 1):
                inputs_ = []
                if self.dynamic_input_shapes:
                    input_ = Input(shape=(None, None, None, self.number_input_features), name=f"siam{i + 1}_input")

                else:
                    input_ = Input(shape=list(self.input_size) + [self.number_input_features], name=f"siam{i + 1}_input")

                inputs_.append(input_)
                self.inputs.append(input_)
                for j, m_a_c_p_l in enumerate(self.metadata_at_common_pathway_layer):
                    if m_a_c_p_l == "x":
                        meta_input_ = Input(shape=(1, 1, 1, self.metadata_sizes[j]), name=f"siam{i + 1}_meta{j + 1}")
                        inputs_.append(meta_input_)
                        self.inputs.append(meta_input_)

                self.outputs.append(siam_model(inputs_ if len(inputs_) > 1 else inputs_[0]))

            path = Concatenate(axis=-1)(self.outputs)

        # 2. Construct common pathway
        deep_supervision_indices = sorted(set(self.deep_supervision_pathways + (0,)), reverse=True)
        for ds_i in deep_supervision_indices:
            path = self.path_right_output_paths[-2 - ds_i]
            for i, (n_f_c_p, k_s_c_p, d_c_p) in enumerate(zip(self.number_features_common_pathway, self.kernel_sizes_common_pathway, self.dropout_common_pathway)):
                if (self.batch_normalization or self.instance_normalization) and (i == 0 or not self.relaxed_normalization_scheme):
                    path = self.normalization_function(path)(path)

                path = self.activation_function("c_activation{}_{}".format(i, ds_i))(path)
                for j, m_a_c_p_l in enumerate(self.metadata_at_common_pathway_layer):
                    if m_a_c_p_l == i:
                        path = self.introduce_metadata(path, j)

                for j, e_o_a_c_p_l in enumerate(self.extra_output_at_common_pathway_layer):
                    if e_o_a_c_p_l == i:
                        self.outputs[j] = self.retrieve_extra_output(path, j)

                if d_c_p:
                    path = Dropout(d_c_p)(path)

                path = Conv3D(filters=n_f_c_p, kernel_size=k_s_c_p, activation=self.activation_final_layer if i + 1 == len(self.number_features_common_pathway) else None, padding=self.padding, kernel_initializer=self.kernel_initializer, kernel_regularizer=regularizers.l1_l2(self.l1_reg, self.l2_reg) if i + 1 != len(self.number_features_common_pathway) else None, name=f"s{ds_i}" if i + 1 == len(self.number_features_common_pathway) else None)(path)

            self.outputs.insert(0, path)

        self.inputs = self.inputs + self.metadata_inputs
        
        # 3. Mask the output (optionally)
        if self.mask_output:
            assert self.deep_supervision in [None, (0,), ()], "For now, no masking of output allowed when using deep supervision."
            if self.dynamic_input_shapes:
                mask_input_ = mask_path = Input(shape=(None, None, None, path.shape[-1]))

            else:
                mask_input_ = mask_path = Input(shape=tuple(self.output_size) + (path.shape[-1],))

            self.inputs.append(mask_input_)
            for i, output in self.outputs:
                self.outputs[i] = Multiply()([output, mask_path])

        # 4. For example: Correct for segment sampling changes to P(X|Y) --> this adds an extra dimension because the correction is done inside loss function and weights are given with y_creator in extra dimension (can only be done for binary like this)
        if self.add_extra_dimension:
            assert self.deep_supervision in [None, (0,), ()], "For now, no masking of output allowed when using deep supervision."
            for i, output in self.outputs:
                self.outputs[i] = keras.ops.expand_dims(output, -1)

        self.model = Model(inputs=self.inputs, outputs=self.outputs)

        # Final sanity check: were our calculations correct?
        print("\nNetwork summary:")
        print(self.model.summary())
        model_input_shape = self.model.input_shape
        if not isinstance(model_input_shape, list):
            model_input_shape = [model_input_shape]

        if not self.dynamic_input_shapes:
            assert list(model_input_shape[0][1:-1]) == self.input_size

        return self.model
    
    @staticmethod
    def create_model(**unet_generalized_kwargs):
        return UnetGeneralized(**unet_generalized_kwargs).set_model()


if __name__ == "__main__":
    from keras.utils import plot_model

    m = UnetGeneralized.create_model(
        number_input_features=5,
        padding="same",
        output_size=(189, 189, 189),
        pooling="max",
        upsampling="copy",
        do_skips=True,
        bottleneck=True,
        residual_connections=True,
        batch_normalization=False,
        instance_normalization=True,
        replace_instance_normalization_with_group_norm=16,
        subsample_factors_per_pathway=(
                (1, 1, 1),
                (3, 3, 3),
                (9, 9, 9),
                (27, 27, 27)
        ),
        kernel_sizes_per_pathway=(
                (((3, 3, 3),) * 2, ((3, 3, 3),) * 2),
                (((3, 3, 3),) * 2, ((3, 3, 3),) * 2),
                (((3, 3, 3),) * 2, ((3, 3, 3),) * 2),
                (((3, 3, 3),) * 5, ((3, 3, 3),) * 5),
        ),
        number_features_per_pathway=(
                ((32, 32), (32, 32)),
                ((64, 64), (64, 64)),
                ((128, 128), (128, 128)),
                ((256, 128, 64, 32, 16), (16, 32, 64, 128, 256))
        ),
        kernel_sizes_common_pathway=((1, 1, 1),),
        number_features_common_pathway=(1,),
        dropout_common_pathway=(0.1,),
        activation_final_layer='linear',
        # extra_output_kernel_sizes=(((1, 1, 1),), ((1, 1, 1),) * 4),
        # extra_output_number_features=((2,), (2048, 1024, 512, 12)),
        # extra_output_dropout=((0.1,), (0.1,) * 4),
        # extra_output_at_common_pathway_layer=(0, "x"),
        # extra_output_activation_final_layer=("sigmoid", "linear"),
        deep_supervision_pathways=(0, 1, 2, 3),
        transformer_pathways=(2, 3)
    )
    print("\nOkay, this is a valid network architecture.")
    print(m.outputs)
    print(m.output_names)
    plot_model(m, "/home/jbertels/model.png", show_shapes=True)  # apt install graphviz
