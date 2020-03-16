# DeepIceLearning IceTray Module

This repository provides and IceTray module for the classification & reconstruction of events in IceCube using Deep Neural Networks. The training and testing is done using the software package DeepIceLearning (`https://github.com/tglauch/DeepIceLearning`). This module is, however, fully self-contained, i.e. no additonal software from the main package is needed. External software requirements include:
- IceCube Software
- Tensorflow (CPU or GPU Version)
- Keras
- numpy, matplotlib

The IceCube public repository provides a docker/singularity contrainer with all the prerequisite already included (see section "Running on a GPU").

# 1. Installation & Usage

The package can be simply installed by running `python setup.py install`. As an alternative add the path to the tool's main folder to your `$PYTHONPATH` (this might especially the easier option if you work on Icecube's NPX cluster). 

All the functionality is provided by the main script `i3module.py`. There are two ways of running the software after loading the icecube environment (and potential GPU libraries):

1. As a part of an icetray module by importing it as `from i3deepice.i3module import DeepLearningModule`
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

```
  export SINGULARITY_CACHEDIR=/data/user/your_name/cache
  export SINGULARITY_TEMPDIR=/data/user/your_name/temp
  mkdir $SINGULARITY_CACHEDIR $SINGULARITY_TEMPDIR
  singularity pull docker://icecube/icetray:combo-stable-tensorflow.1.13.2-ubuntu18.04
  ```
 
For an example on how to run a script in a singularity container consult the environment script in `./examples/singularity_env.sh`. Before running the container make sure you have a clean environment, i.e. no previously loaded icecube software environment in your `$PATH`


# 3 Integration into IceTray

Using the module is fairly simple. Just import it and add it to the IceTray as usual. By setting calib_errata, bad_dom_list, saturation_windows, bright_doms keys for pulse cleaning are defined. If not needed simply remove or set to 'None'. Note that especially for classification purposes it is currently not recommended to use the pulse cleaning.

```
from i3deepice.i3module import DeepLearningModule
tray.AddModule(DeepLearningModule, 'dnn_classification',
                batch_size=128,
                cpu_cores=1,
                gpu_cores=1,
                model='classification',
                pulsemap='InIceDSTPulses',
#                calib_errata='CalibrationErrata',
#                bad_dom_list='BadDomsList',
#                saturation_windows='SaturationWindows',
#                bright_doms='BrightDOMs',
                save_as='TUM_classification')
```

Currenly there are two models available.

  - `classification` predicts the event topology of the event, i.e. one of skimming, starting cascade, through-going track, starting track or stopping track
  
  - `mu_energy_reco_full_range` predicts the log_10 of the energy on detector entry for up-going muons
  
  
The pulsemap should be chosen in accordance with the respective task. In general prediction is more accurate on single events, i.e. splitted pulsemaps, but there is a certain stabillity also against coincidences.
