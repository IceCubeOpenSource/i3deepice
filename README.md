# DeepIceLearning IceTray Module

This repository provides and IceTray module for the classification & reconstruction of events in IceCube using Deep Neural Networks. The training and testing is done using the software package DeepIceLearning (`https://github.com/tglauch/DeepIceLearning`). This module is, however, fully self-contained, i.e. no additonal software from the main package is needed. External software requirements include:
- IceCube Software
- Tensorflow (CPU or GPU Version)
- Keras
- numpy, matplotlib

The IceCube public repository provides a docker/singularity contrainer with all the prerequisite already included (see section "Running on a GPU").

# 1. Installation & Usage

The package can be simply installed by running `pip install setup.py`. As an alternative add the path to the tool's main folder to your `$PYTHONPATH`. 

All the functionality is provided by the main script `i3module.py`. There are two ways of running the software after loading the icecube environment (and potential GPU libraries):

1. As a part of an icetray module by importing it as `from i3deepice.i3module import DeepLearningClassifier`
2. Directly: `python i3module.py --files /path/to/some/i3/files.i3 `

(Note that for the first case it's required that the location of the module is in your `PYTHONPATH` environment variable)

An example on how the tool can be used inside an icetray module is given in `./examples/`. In the same folder you also find another `README` with specific information on the usage with singularity on IceCube's NPX cluster.


# 2 Running on a GPU

Running the model on a GPU is around 20-30 times quicker then on a CPU (<100ms/event instead of ~2s/event on one core). For running the module on a GPU it is required that you have a list of Nvidia drivers & libraries available on your machine. 
This includes:
  - Nvidia GPU drivers
  - CUDA Toolkit
  - cuDNN
  
As mentioned above there is, however, a singularity/docker container which has a working version of the libraries. Installing the container on IceCube's NPX is fairly simple with the following commands:
  - `export SINGULARITY_CACHEDIR=/some/location/with/enough/quota`
  - `export SINGULARITY_TEMPDIR=/another/location/with/enough/quota`
  - `singularity pull docker://icecube/icetray:combo-stable-tensorflow.1.13.2-ubuntu18.04`
 The only thing that is additionally required in this cases are the Nvidia GPU drivers (see `https://www.tensorflow.org/install/docker` for reference). For an example of how the singularity command could look like consult the environment script in `./examples/singularity_env.sh`.
