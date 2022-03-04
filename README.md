# DeepVoxNet2
Deep learning processing framework for Keras.
Partly based on [DeepVoxNet](https://github.com/JeroenBertels/deepvoxnet).

## Get started
This package can be added to your python 3.9 environment via: 
- First cloning/downloading the repository and then via: 
```
pip install -e /path/to/deepvoxnet2
```
- Installing it directly from Github via:
```
pip install git+https://github.com/JeroenBertels/deepvoxnet2
```
To upgrade your installation using the first method just download the latest version and repeat the process or ```git pull``` the new version. When using the second method simply repeat the command but add the ```--upgrade``` flag. You can also install/revert to a specific version; in that case append ```@version_tag``` (e.g. @deepvoxnet-2.10.23). 

Some functions require the SimpleITK and SimpleElastix software to be installed. To install these packages also, please append the paths in the above commands with ```[sitk]```.
## Acknowledgements
Jeroen Bertels is part of [NEXIS](https://www.nexis-project.eu), a project that has received funding from the European Union's Horizon 2020 Research and Innovation Programme.
