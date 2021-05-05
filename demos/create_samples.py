import numpy as np
from matplotlib import pyplot as plt
from deepvoxnet2.components.creator import Creator
from deepvoxnet2.components.sample import Sample
from deepvoxnet2.components.transformers import AffineDeformation, Put, RandomCrop, SampleInput, GridCrop, Crop, Flip, Group, Buffer, Split


def create_samples():
    # Let's define a toy example
    x = np.zeros((100, 100))
    x[25:50, 25:50] = 1
    x[25:50, 50:75] = 2
    y = np.zeros((100, 100))
    y[25:50, 25:75] = 1
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x, vmin=0, vmax=2)
    axs[1].imshow(y, vmin=0, vmax=2)
    plt.show()

    # inputs
    x_input = x_path = SampleInput(Sample(x))  # Sample converts x into a 5D array (batch, x, y, z, features) with an affine attribute (here just np.eye)
    y_input = y_path = SampleInput(Sample(y), n=None)

    # processing
    x_path, y_path = AffineDeformation(x_path, translation_window_width=(10, 10, 0), rotation_window_width=(3.14 / 10, 0, 0))(x_path, y_path)
    x_path_intermediate = x_path
    x_path, y_path = RandomCrop(x_path, (27, 27, 1), n=16, nonzero=True)(x_path, y_path)
    x_path, y_path = Flip((0.5, 0, 0))(x_path, y_path)
    y_path = Crop(y_path, (21, 21, 1))(y_path)

    # outputs
    x_output = Put(y_input, caching=False)(x_path)
    y_output = y_input

    # Let's visualize some x samples at the end of the processing pathway:
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        x_sample = x_path.eval()[0]
        x_sample_intermediate = x_path_intermediate[0]
        axs[i, 0].imshow(x, vmin=0, vmax=2)
        axs[i, 1].imshow(x_sample_intermediate[0, :, :, 0, 0], vmin=0, vmax=2)
        axs[i, 2].imshow(x_sample[0, :, :, 0, 0], vmin=0, vmax=2)

    plt.show()

    # It was clear that the original x is affine transformed and then multiple patches are taken
    # Suppose we also want to put the drawn sample back onto the original image space (to see what happens if we would instantiate the Put transformer with caching=True, see end of this demo ;-))
    fig, axs = plt.subplots(1, 4)
    axs[0].imshow(x, vmin=0, vmax=2)
    for i in range(3):
        x_sample = x_output.eval()[0]
        axs[i + 1].imshow(x_sample[0, :, :, 0, 0], vmin=0, vmax=2)

    plt.show()

    # Suppose we want to produce x and y samples, we must make sure that all transformers are updated (and not only the ones "below" x_path
    # There are two options:
    # Option 1: Just make sure they are grouped in one path and evaluate that path
    samples_path = Group()([x_path, y_path])
    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(x, vmin=0, vmax=2)
    axs[1, 0].imshow(y, vmin=0, vmax=2)
    for i in range(3):
        x_sample, y_sample = samples_path.eval()
        axs[0, i + 1].imshow(x_sample[0, :, :, 0, 0], vmin=0, vmax=2)
        axs[1, i + 1].imshow(y_sample[0, :, :, 0, 0], vmin=0, vmax=2)

    plt.show()

    # Option 2: Make a higher abstract Creator network and evaluate that network (this has the advantage that it only evaluates the nodes that are relevant and thus is more efficient)
    creator = Creator([x_path, y_path])
    creator.summary()
    fig, axs = plt.subplots(2, 4)
    axs[0, 0].imshow(x, vmin=0, vmax=2)
    axs[1, 0].imshow(y, vmin=0, vmax=2)
    for i, output in enumerate(creator.eval()):
        x_sample, y_sample = output
        axs[0, i + 1].imshow(x_sample[0][0, :, :, 0, 0], vmin=0, vmax=2)
        axs[1, i + 1].imshow(y_sample[0][0, :, :, 0, 0], vmin=0, vmax=2)
        if i == 2:
            break

    plt.show()

    # Finally, a Creator has notion of when your network has ran out of samples. As you can see, we ask 16 samples from the RandomCrop transformer (all the rest generate a default of 1 sample).
    # This means the creator should create us 16 samples
    creator = Creator([x_output, y_output])
    creator.summary()
    fig, axs = plt.subplots(4, 4)
    count = 0
    for i, output in enumerate(creator.eval()):
        x_sample, y_sample = output
        axs[i // 4, i % 4].imshow(x_sample[0][0, :, :, 0, 0], vmin=0, vmax=2)
        count += 1

    plt.show()
    assert count == 16

    # Suppose we use a GridCrop instead of a RandomCrop with no n specified --> The GridCrop will create a number of samples as to complete the entire grid.
    # inputs
    x_input = x_path = SampleInput(Sample(x))
    y_input = y_path = SampleInput(Sample(y), n=None)
    # processing
    x_path, y_path = AffineDeformation(x_path, translation_window_width=(10, 10, 0), rotation_window_width=(0, 0, 0))(x_path, y_path)
    x_path, y_path = GridCrop(x_path, (27, 27, 1), strides=(21, 21, 1), nonzero=True)(x_path, y_path)
    x_path, y_path = Flip((0.5, 0, 0))(x_path, y_path)
    y_path = Crop(y_path, (21, 21, 1))(x_path)
    # outputs
    # Try to understand completely what the Buffer does! Experiment a bit with different buffer_sizes (also try None) and drop_remainder=True/False. It basically batches the samples that arrive here.
    # Just like with any other transformer you can hang multiple paths to the Buffer if they have to be buffered in the same way
    x_path, y_path = Buffer(buffer_size=3, drop_remainder=False)(x_path, y_path)
    x_output = Put(y_input)(x_path)
    y_output = y_input
    creator = Creator([x_output, y_output])
    creator.summary()
    fig, axs = plt.subplots(4, 4)
    count = 0
    for i, output in enumerate(creator.eval()):
        if i == 0:
            creator.write_transformer_outputs("/Users/jberte3/Desktop/deepvoxnet2")

        x_sample, y_sample = output
        axs[i // 4, i % 4].imshow(x_sample[0][0, :, :, 0, 0], vmin=0, vmax=2)
        count += 1

    plt.show()
    print("There were {} samples from the grid cropper.".format(count))

    # Have a look at how gradually the entire image got put back for the random and grid crop even though for both cases the samples came from an affine transformed input! :-)


if __name__ == '__main__':
    create_samples()
