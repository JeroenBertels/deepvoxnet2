import os
import pickle
from demos.create_dataset import create_dataset
from deepvoxnet2 import DEMO_DIR
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.components.sampler import MircSampler
from deepvoxnet2.keras.models.unet_generalized import create_generalized_unet_model
from deepvoxnet2.components.transformers import Normalize, AffineDeformation, MircInput, ElasticDeformation, GridCrop, Flip, Put, KerasModel, RandomCrop, Group, Concat, Multiply, Threshold, Buffer
from deepvoxnet2.components.model import DvnModel
from deepvoxnet2.components.creator import Creator
from deepvoxnet2.keras.optimizers import Adam
from deepvoxnet2.keras.losses import binary_dice_loss, binary_crossentropy
from deepvoxnet2.keras.metrics import binary_dice_score, binary_accuracy, binary_true_volume, binary_pred_volume, binary_volume_difference, binary_abs_volume_difference, binary_true_positives, binary_true_negatives, binary_false_positives, binary_false_negatives
from deepvoxnet2.keras.callbacks import LogsLogger, DvnModelEvaluator, MetricNameChanger, LearningRateScheduler, DvnModelCheckpoint
from deepvoxnet2.factories.directory_structure import MircStructure


def train(run_name, experiment_name, fold_i=0):
    train_data = Mirc()
    train_data.add(create_dataset("train"))
    val_data = Mirc()
    val_data.add(create_dataset("val"))

    # for many training pipelines it's good practice to normalize the data; look how easy it is to get the mean and standard deviation of a certain modality when grouped in a Mirc object
    # this calculation (when no n argument is specified) often takes some time so you can better comment out with calculated value after the first run (or specify n)
    flair_mean, flair_std = train_data.mean_and_std("flair", n=2)
    t1_mean, t1_std = train_data.mean_and_std("t1", n=2)

    # building mirc samplers: here the sampler randomly samples a record out of train_data or val_data --> depending on what objects these samplers return, you must choose an appropriate Input later on when building your Dvn network/creator
    train_sampler = MircSampler(train_data, shuffle=True)
    val_sampler = MircSampler(val_data, shuffle=False)

    # let's create a keras model and put it in a Transformer layer to be used in our Dvn network (see "create_samples.py" for other examples on Transformers and sample creation)
    # have a look at the print-outs when this model is created; you'll see some interesting properties like # parameters, field of view, output/input sizes, etc.
    keras_model = create_generalized_unet_model(
        number_input_features=2,
        output_size=(112, 112, 96),
        padding="same"
    )
    keras_model_transformer = KerasModel(keras_model)

    # similar to the demo on sample creation, let's make our processing network(s) (keep in mind that the following network could be made in different ways; we just show one way)
    # inputs (we have samplers that sample identifier objects, so we can use MircInputs here; they now what to do with the sampled identifier objects (have a look at their load method)
    x_input_0 = MircInput(["flair"], output_shapes=[(1, None, None, None, 1)])
    x_input_1 = MircInput(["t1"], output_shapes=[(1, None, None, None, 1)])
    y_input = MircInput(["whole_tumor"], output_shapes=[(1, None, None, None, 1)], n=None)

    # used for training and is on the level of patches
    x_path_0, x_path_1, y_path = AffineDeformation(x_input_0, translation_window_width=(20, 20, 20), rotation_window_width=(3.14 / 10, 0, 0))(x_input_0, x_input_1, y_input)
    x_path_0, x_path_1, y_path = ElasticDeformation(x_path_0, shift=(1, 1, 1))(x_path_0, x_path_1, y_path)
    x_path_0, x_path_1, y_path = GridCrop(x_path_0, (112, 112, 96), n=1, nonzero=True)(x_path_0, x_path_1, y_path)  # x_path_0 is used as a reference volume to determine the coordinate around which to crop (here also constrained to nonzero flair voxels)
    x_path_0 = Normalize(-flair_mean, 1 / flair_std)(x_path_0)
    x_path_1 = Normalize(-t1_mean, 1 / t1_std)(x_path_1)
    x_path = Concat()([x_path_0, x_path_1])
    x_path, y_path = Flip((0.5, 0.5, 0))(x_path, y_path)
    x_train = keras_model_transformer(x_path)
    y_train = y_path

    # used for validation and is on the level of patches
    x_path_0, x_path_1, y_path = GridCrop(x_input_0, (112, 112, 96), nonzero=True)(x_input_0, x_input_1, y_input)  # notice that there is no n specified --> this will sample the complete grid
    x_path_0 = Normalize(-flair_mean, 1 / flair_std)(x_path_0)
    x_path_1 = Normalize(-t1_mean, 1 / t1_std)(x_path_1)
    x_path = Concat()([x_path_0, x_path_1])
    x_val = keras_model_transformer(x_path)
    y_val = y_path

    # used for validation of the full images and thus is on the level of the input
    x_path = Multiply()([x_val, Threshold(-flair_mean / flair_std)(x_path_0)])
    x_path = Buffer()(x_path)
    x_full_val = Put(y_input)(x_path)  # x_val is on the patch level and the put transformers brings the patch back to the reference space; have a look why y_input is used (with n=None) and think about why this is
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
            "full_val": [x_full_val, y_full_val]
        }
    )

    # similar to keras, we can compile the model
    # here lists (of lists) can be used to apply different losses/metrics to different outputs of the network
    # the losses can also be lists of lists if you want a linear combination of losses to be applied to a certain output (when no weights specified, everything is weighted uniformly)
    dvn_model.compile("train", optimizer=Adam(lr=1e-3), losses=[[binary_crossentropy, binary_dice_loss]], metrics=[[binary_crossentropy, binary_dice_loss, binary_dice_score, binary_accuracy, binary_true_volume, binary_pred_volume, binary_volume_difference, binary_abs_volume_difference, binary_true_positives, binary_true_negatives, binary_false_positives, binary_false_negatives]])
    # although the following outputs are not used for fitting your model, you must compile it as well to know what metrics to calculate
    dvn_model.compile("val", losses=[[binary_crossentropy, binary_dice_loss]], metrics=[[binary_crossentropy, binary_dice_loss, binary_dice_score, binary_accuracy, binary_true_volume, binary_pred_volume, binary_volume_difference, binary_abs_volume_difference, binary_true_positives, binary_true_negatives, binary_false_positives, binary_false_negatives]])
    dvn_model.compile("full_val", losses=[[binary_crossentropy, binary_dice_loss]], metrics=[[binary_crossentropy, binary_dice_loss, binary_dice_score, binary_accuracy, binary_true_volume, binary_pred_volume, binary_volume_difference, binary_abs_volume_difference, binary_true_positives, binary_true_negatives, binary_false_positives, binary_false_negatives]])

    # typically one needs to organize everything on their local disks, e.g. a dir to save (intermediate) models, a logs directory to save intermediate log files to view via e.g. Tensorboad, some output dirs to save (intermediate) predictions, etc.
    # we have provided one way of doing so under deepvoxnet2/factories/directory_structure based on the samplers you created
    output_structure = MircStructure(
        base_dir=os.path.join(DEMO_DIR, "brats_2018", "runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=None,  # when None a new round will be created (only possible when the experiment dir already exists)
        training_identifiers=train_sampler.identifiers,
        validation_identifiers=val_sampler.identifiers
    )
    output_structure.create()  # only now the non-existing output dirs are created

    # also similar to keras, we define some callbacks
    callbacks = [
        LearningRateScheduler(lambda epoch, lr: ([1e-3] * 300 + [1e-4] * 150 + [1e-5] * 75)[epoch]),
        DvnModelEvaluator(dvn_model, "full_val", val_sampler, freq=25, output_dirs=output_structure.val_images_output_dirs, name_tag=None, save_x=True, save_y=False, save_sample_weight=False),
        LogsLogger(output_structure.logs_dir),
        DvnModelCheckpoint(dvn_model, output_structure.models_dir, freq=25)  # every 25 epochs the model will be saved (for e.g. parallel offline use for testing some things)
    ]

    # let's train :-)
    history = dvn_model.fit("train", train_sampler, batch_size=2, validation_key="val", validation_sampler=val_sampler, callbacks=callbacks, epochs=525)  # ideally choose the batch size as a whole multiple of the number of samples your processing pipeline produces
    with open(output_structure.history_path, "wb") as f:
        pickle.dump(history.history, f)

    dvn_model.save(os.path.join(output_structure.models_dir, "dvn_model_final"))


def evaluate(run_name, experiment_name, fold_i=0, round_i=0):
    val_data = Mirc()
    val_data.add(create_dataset("val"))
    val_sampler = MircSampler(val_data, shuffle=False)
    output_structure = MircStructure(
        base_dir=os.path.join(DEMO_DIR, "brats_2018", "runs"),
        run_name=run_name,
        experiment_name=experiment_name,
        fold_i=fold_i,
        round_i=round_i,
        validation_identifiers=val_sampler.identifiers
    )
    dvn_model = DvnModel.load_model(os.path.join(output_structure.models_dir, "dvn_model_final"))
    evaluations = dvn_model.evaluate("full_val", val_sampler, output_dirs=output_structure.val_images_output_dirs)
    print(evaluations)


if __name__ == '__main__':
    train("unet", "adam_ce-sd")
    evaluate("unet", "adam_ce-sd", round_i=0)
