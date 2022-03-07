import os
import pickle
from demos.create_dataset import create_dataset
from deepvoxnet2 import DEMO_DIR
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.keras.models.unet_generalized_v2 import create_generalized_unet_v2_model
from deepvoxnet2.components.transformers import NormalizeIndividual, AffineDeformation, MircInput, ElasticDeformation, GridCrop, Flip, Put, KerasModel, Buffer, IntensityTransform
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.components.creator import Creator
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.losses import get_loss
from deepvoxnet2.keras.metrics import get_metric
from deepvoxnet2.keras.callbacks import DvnModelEvaluator, ReduceLROnPlateau, EarlyStopping, DvnModelCheckpoint
from deepvoxnet2.factories.directory_structure import MircStructure


def train(run_name, experiment_name, fold_i=0):
    train_data = Mirc()
    train_data.add(create_dataset("train"))
    val_data = Mirc(create_dataset("val"))  # note the two different ways of adding datasets to a Mirc object :-)

    # for many training pipelines it's good practice to normalize the data; look how easy it is to get the mean and standard deviation of a certain modality when grouped in a Mirc object
    # this calculation (when no n argument is specified) often takes some time so you can better comment out with calculated value after the first run (or specify n)
    # note that we not use it here as we will use the NormalizeIndividual (instead of Normalize) Transformer later
    flair_mean, flair_std = train_data.mean_and_std("flair", n=2)
    t1_mean, t1_std = train_data.mean_and_std("t1", n=2)

    # building mirc samplers: here we create the sampler that randomly samples a record out of train_data or val_data
    # --> depending on what objects these samplers return, you must choose an appropriate Input later when building your Dvn network/creator
    # more specifically, a doing my_sampler[i] will return a "Identifier" object; this can be anything: a numpy array, a pointer to a record in a Mirc object, ... (just make sure you use the corresponding Input later (e.g. MircInput, SampleInput, ...)
    train_sampler = MircSampler(train_data, shuffle=True)
    val_sampler = MircSampler(val_data)

    # let's create a keras model and put it in a Transformer layer to be used in our Dvn network (see "create_samples.py" for other examples on Transformers and sample creation)
    # have a look at the print-outs when this model is created; you'll see some interesting properties like # parameters, field of view, output/input sizes, etc.
    keras_model = create_generalized_unet_v2_model(
        number_input_features=2,
        subsample_factors_per_pathway=(
            (1, 1, 1),
            (3, 3, 3),
            (9, 9, 9)
        ),
        kernel_sizes_per_pathway=(
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ((3, 3, 3), (3, 3, 3))),
            (((3, 3, 3), (3, 3, 3)), ())
        ),
        number_features_per_pathway=(
            ((30, 30), (30, 30)),
            ((60, 60), (60, 30)),
            ((120, 60), ())
        ),
        output_size=(63, 63, 63),
        dynamic_input_shapes=True  # not necessary; though this makes sure we can actually input any patch size to the network that is compatible with the specified patch size (here 63, 63, 63)
    )
    keras_model_transformer = KerasModel(keras_model)

    # similar to the demo on sample creation, let's make our processing network(s) (keep in mind that the following network could be made in different ways; we just show one way)
    # inputs (we have samplers that sample identifier objects, so we can use MircInputs here; they now what to do with the sampled identifier objects (have a look at their load method)
    # specifying the shapes is not necessary; however like this you could check the shapes (input None for a dimension if unknown
    x_input = NormalizeIndividual(ignore_value=0)(MircInput(["flair-t1"], output_shapes=[(1, 240, 240, 155, 2)]))
    y_input = MircInput(["whole_tumor"], output_shapes=[(1, 240, 240, 155, 1)])

    # used for training and is on the level of patches
    x_path, y_path = AffineDeformation(x_input, translation_window_width=(5, 5, 5), rotation_window_width=(3.14 / 10, 0, 0), width_as_std=True)(x_input, y_input)
    x_path, y_path = ElasticDeformation(x_path, shift=(1, 1, 1))(x_path, y_path)
    x_path, y_path = GridCrop(x_path, (63, 63, 63), n=8, nonzero=True)(x_path, y_path)  # x_path is used as a reference volume to determine the coordinate around which to crop (here also constrained to nonzero flair voxels)
    x_path = IntensityTransform(std_shift=0.1, std_scale=0.1)(x_path)
    x_path, y_path = Flip((0.5, 0, 0))(x_path, y_path)
    x_train = keras_model_transformer(x_path)
    y_train = y_path

    # used for validation and is on the level of patches: note that here we take larger patches to speedup prediction
    x_path, y_path = GridCrop(x_input, (117, 117, 117), nonzero=True)(x_input, y_input)  # notice that there is no n specified --> this will sample the complete grid (n=None is default); actually we could have used Crop (a simple center crop here as well)
    x_val = keras_model_transformer(x_path)
    y_val = y_path

    # used for validation of the full images and thus is on the level of the input
    x_path = Buffer()(x_val)  # first stack all patch predictions
    x_full_val = Put(x_input)(x_path)  # put all patches back into x_input space
    y_full_val = y_input

    # you can use Creator.summary() method to visualize your designed architecture
    # when constructing your pathway, you can also name your transformers. If you don't do this, the creator will name the transformers.
    # ff you want to have unique transformer names for your entire network, you can first make one large creator and afterwards make the individual creators
    # This step is not necessary however. When you make a DvnModel later there will be built one large creator inside and thus names are given automagically.
    creator = Creator([x_train, y_train, x_val, y_val, x_full_val, y_full_val])
    x_train, y_train, x_val, y_val, x_full_val, y_full_val = creator.outputs
    creator.summary()

    # we make a DvnModel, which allows to give [x], [x, y] or [x, y, sample_weight] as "outputs". Here, the x's must be after the keras_model_transformer, thus referring to the predicted y.
    # what is a DvnModel? Similar to a keras model, but including the processing pipeline. If you inspect the DvnModel code you'll see it has fit, evaluate and predict methods
    # to apply a method, one needs to choose which configuration (keys; see below)
    dvn_model = DvnModel(
        outputs={
            "train": [x_train, y_train],
            "val": [x_val, y_val],
            "full_val": [x_full_val, y_full_val],
            "full_test": [x_full_val]  # we also add this since sometimes we won't have a ground truth (y_input) at test time so then we only want to produce a prediction
        }
    )

    # similar to keras, we can compile the model
    # here lists (of lists) can be used to apply different losses/metrics to different outputs of the network
    # the losses can also be lists of lists if you want a linear combination of losses to be applied to a certain output (when no weights specified, everything is weighted uniformly)
    cross_entropy = get_loss("cross_entropy")
    soft_dice = get_loss("dice_loss", reduce_along_features=True, reduce_along_batch=True)
    dice_score = get_metric("dice_coefficient", threshold=0.5)
    accuracy = get_metric("accuracy", threshold=0.5)
    abs_vol_diff = get_metric("absolute_volume_error", voxel_volume=0.001)  # in ml (voxels are 1 x 1 x 1 mm)
    dvn_model.compile("train", optimizer=Adam(lr=1e-3), losses=[[cross_entropy, soft_dice]], metrics=[[dice_score, accuracy, abs_vol_diff]])
    # although the following outputs are not used for fitting your model, you must compile it as well to know what metrics to calculate
    dvn_model.compile("val", losses=[[cross_entropy, soft_dice]], metrics=[[dice_score, accuracy, abs_vol_diff]])
    dvn_model.compile("full_val", losses=[[cross_entropy, soft_dice]], metrics=[[dice_score, accuracy, abs_vol_diff]])

    # typically one needs to organize everything on their local disks, e.g. a dir to save (intermediate) models, a logs directory to save intermediate log files to view via e.g. Tensorboad, some output dirs to save (intermediate) predictions, etc.
    # we have provided one way of doing so under deepvoxnet2/factories/directory_structure based on the samplers you created
    output_structure = MircStructure(
        base_dir=os.path.join(DEMO_DIR, "brats_2018", "runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=None,  # when None a new round will be created
        training_mirc=train_data,
        validation_mirc=val_data
    )
    output_structure.create()  # only now the non-existing output dirs are created

    # also similar to keras, we define some callbacks (note the ordering! otherwise ReduceLROnPlateau would have no access to full_val__dice_coefficient__s0)
    callbacks = [
        DvnModelEvaluator(dvn_model, "full_val", val_sampler, freq=5, output_dirs=output_structure.val_images_output_dirs),  # watch out, here you do need to make sure that the order of the val_sampler (shuffle=False by default) is te same as the order of your output_dirs...
        DvnModelCheckpoint(dvn_model, output_structure.models_dir, freq=10),  # every 10 epochs the model will be saved (for e.g. parallel offline use for testing some things)
        ReduceLROnPlateau("full_val__dice_coefficient__s0", factor=0.2, patience=30, mode="max"),  # try to understand deepvoxnet's naming convention of metrics and losses (see Tensorboard)
        EarlyStopping("full_val__dice_coefficient__s0", patience=60, mode="max")
    ]

    # let's train :-) Note that if we specify a logs_dir you will be able to watch tensorboard... (no need to give the tensorboard callback in the list of callbacks since it is tricky where to put it)
    history = dvn_model.fit(
        "train",
        train_sampler,
        batch_size=8,
        callbacks=callbacks,
        epochs=10000,
        logs_dir=output_structure.logs_dir,
        steps_per_epoch=128,
        shuffle_samples=64,
        prefetch_size=64,
        num_parallel_calls=4
    )
    with open(output_structure.history_path, "wb") as f:
        pickle.dump(history.history, f)

    # get final predictions, metrics and save model
    dvn_model.evaluate("full_val", val_sampler, output_dirs=output_structure.val_images_output_dirs)
    dvn_model.save(os.path.join(output_structure.models_dir, "dvn_model_final"))


def evaluate(run_name, experiment_name, fold_i=0, round_i=0):
    test_data = Mirc()
    test_data.add(create_dataset("test"))
    test_sampler = MircSampler(test_data)
    output_structure = MircStructure(
        base_dir=os.path.join(DEMO_DIR, "brats_2018", "runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=round_i,
        testing_mirc=test_data
    )
    output_structure.create()
    dvn_model = DvnModel.load_model(os.path.join(output_structure.models_dir, "dvn_model_final"))
    evaluations = dvn_model.evaluate("full_val", test_sampler, output_dirs=output_structure.test_images_output_dirs)
    print(evaluations)


def predict(run_name, experiment_name, fold_i=0, round_i=0):
    test_data = Mirc()
    test_data.add(create_dataset("test"))
    test_sampler = MircSampler(test_data)
    output_structure = MircStructure(
        base_dir=os.path.join(DEMO_DIR, "brats_2018", "runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=round_i,
        testing_mirc=test_data
    )
    dvn_model = DvnModel.load_model(os.path.join(output_structure.models_dir, "dvn_model_final"))
    predictions = dvn_model.predict("full_test", test_sampler)
    # or you could simply use baked in functions in your ProjectStructure class, e.g.:
    # output_structure.predict("test", "dvn_model_final", "full_test", fold_i=0, round_i=0, save_x=False)


if __name__ == '__main__':
    train("unet", "adam_ce-sd")
    evaluate("unet", "adam_ce-sd", round_i=0)  # when we do have a GT
    predict("unet", "adam_ce-sd", round_i=0)  # when we not have a GT
