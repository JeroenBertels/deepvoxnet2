# DeepVoxNet2
DeepVoxNet2 (DVN2) is a Python library to make it easier to implement deep learning pipelines for medical applications using convolutional neural networks (CNNs).
It is lightly based on the private [DeepVoxNet](https://github.com/JeroenBertels/deepvoxnet) library.
In essence, the library can be used as an add-on to Tensorflow/Keras, Pytorch or any other deep learning framework in Python.
Currently, the use with Tensorflow/Keras is simplest, with readily available CNN architectures, metrics, losses and the DvnModel class to group your entire pipeline and, e.g., bypass the use of Keras' fit function, etc.

DVN2 provides:
- Utility functions such as resampling, registration, Dicom loading, etc.
- Objects for data organization (Mirc, Dataset, Case, Record, Modality).
- Objects for data sampling (Sampler).
- Objects for building pre- to postprocessing pipelines (Transformer, Creator) that keep track of the spatial origin of the data inherently and that you build just like you work in Keras.

## Installation
The library can be used as a Python package that can be added to your active Python 3.9 environment via: 
- First downloading/cloning/forking a specific version of the repository to your local machine and then via: 
```
pip install -e "/path/to/deepvoxnet2[sitk]"
```
- Installing a specific version directly from GitHub via:
```
pip install "git+https://github.com/JeroenBertels/deepvoxnet2@deepvoxnet2-2.13.21#egg=deepvoxnet2[sitk]"
```
- Installing a specific version [directly from PyPI](https://pypi.org/project/deepvoxnet2/) via (only for official releases):
```
pip install "deepvoxnet2[sitk]==2.13.21"
```
To upgrade your installation using the first method just download another version and repeat the process or ```git pull``` another version if possible. When using the second or third method simply repeat the command but add the ```--upgrade``` flag.
The ```[sitk]``` flag will install the SimpleITK and SimpleElastix software packages, but this is optional (for wider compatibility).

Additionally, of the official releases there are also Docker containers [available on DockerHub](https://hub.docker.com/repository/docker/jeroenbertels/deepvoxnet2). These can be ran via:
- Docker:
```
docker run --rm -it --gpus="device=0" -v /path/on/local/machine/a:/path/in/container/a -v /path/on/local/machine/b:/path/in/container/b jeroenbertels/deepvoxnet2:latest
```
- Singularity:
```
cd /path/to/pulled/images
singularity pull docker://jeroenbertels/deepvoxnet2:latest
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity run -B /path/on/local/machine/a:/path/in/container/a,/path/on/local/machine/b:/path/in/container/b  --cleanenv --nv deepvoxnet2_latest.sif
```

## Tutorials
A Jupyter Notebook-style tutorial can be found [here](https://github.com/JeroenBertels/deepvoxnet2/blob/main/demos/deepvoxnet2.ipynb), which guides you through some of the basic design ideas behind DeepVoxNet2.

Other real-world examples are:
- A notebook with all [experiments and code](https://github.com/JeroenBertels/dicegrad/blob/main/BRATS/howto.ipynb) accompanying [this article](https://arxiv.org/abs/2207.09521) about the effect of $\Phi$ and $\epsilon$ when using the Dice loss in tasks with missing or empty labels.

[//]: # (## Acknowledgements)

[//]: # (Jeroen Bertels is part of [NEXIS]&#40;https://www.nexis-project.eu&#41;, a project that has received funding from the European Union's Horizon 2020 Research and Innovation Programme.)
## Cite as
Bertels, J., Robben, D., Lemmens, R., & Vandermeulen, D. (2022). DeepVoxNet2: Yet another CNN framework. ArXiv, 1–15. [http://arxiv.org/abs/2211.09569](http://arxiv.org/abs/2211.09569)