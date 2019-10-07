# DeepIceLearning IceTray Module

This repository provides and IceTray module for the Classification & Reconstruction of Events in IceCube using Deep Neural Networks. The training and testing is done using the software package DeepIceLearning (`https://github.com/tglauch/DeepIceLearning`). This module is fully self-contained, i.e. no additonal software from the main package is needed. External software requirements include:
- IceCube Software
- Tensorflow (CPU or GPU Version)
- Keras
- numpy, matplotlib

The IceCube public repository provides a docker/singularity contrainer with all the prerequisite already included `https://github.com/WIPACrepo/docker-icecube-icetray`.

All the functionality is provided over the main script `i3module.py`. There are essentially two ways of running it, given that the environments are loaded correctly.

1. Directly: `python i3module.py --files /path/to/some/i3/files.i3
2. As a part of an icetray module by importing it, i.e. `from i3module import DeepLearningClassifier`
