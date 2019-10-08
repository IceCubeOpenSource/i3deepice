# DeepIceLearning IceTray Module

This repository provides and IceTray module for the Classification & Reconstruction of Events in IceCube using Deep Neural Networks. The training and testing is done using the software package DeepIceLearning (`https://github.com/tglauch/DeepIceLearning`). This module is fully self-contained, i.e. no additonal software from the main package is needed. External software requirements include:
- IceCube Software
- Tensorflow (CPU or GPU Version)
- Keras
- numpy, matplotlib

The IceCube public repository provides a docker/singularity contrainer with all the prerequisite already included `https://github.com/WIPACrepo/docker-icecube-icetray`.

All the functionality is provided over the main script `i3module.py`. There are essentially two ways of running it, after the icecube environment (and potential GPU libraries) are initialized.

1. Directly: `python i3module.py --files /path/to/some/i3/files.i3 `
2. As a part of an icetray module by importing it, i.e. `from i3module import DeepLearningClassifier`

Note that for the second cases it's required that the location of the module is in your `PYTHONPATH` envivornment variable

An example on how the tool can be used inside an icetray module is given in `./examples/`. In the same folder you also find an additional `README` with specific information of the usage with singularity on IceCube's NPX cluster


# 1 Running on a GPU

Running the model on a GPU is around 20-30 times quicker then on a CPU (~100ms/event instead of ~2s/event). For running the module on a GPU it is required that you have all the required Nvidia drivers avaialble on your machine. 
This includes:
  - Nvidia GPU drivers
  - CUDA Toolkit
  - cuDNN
  
As mentioned above there is a singularity/docker container which has a working version of the libraries. The only thing that is needed in this cases are the Nvidia GPU drivers (see `https://www.tensorflow.org/install/docker` for reference). For an example of how the singularity command could look like consult the environment script in `./examples/singularity_env.sh`.
