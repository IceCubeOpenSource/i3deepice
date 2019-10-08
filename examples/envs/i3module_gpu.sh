#!/bin/bash

echo $PATH
echo $HOSTNAME
export KERAS_BACKEND="tensorflow"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/
export HDF5_USE_FILE_LOCKING='FALSE'
export TMPDIR=./tmp
export SINGULARITY_TMPDIR=./tmp
export SINGULARITY_CACHEDIR=./cache
export DNN_BASE=/data/user/tglauch/DeepIceLearning/I3Module/
singularity exec --nv -B /home/tglauch/:/home/tglauch/ -B /mnt/lfs3/user/:/data/user/ -B /mnt/lfs6/ana/:/data/ana/ -B
/mnt/lfs6/sim/:/data/sim/ /data/user/tglauch/icetray_combo-stable-tensorflow.1.13.2-ubuntu18.04.sif
/usr/local/icetray/env-shell.sh python "$DNN_BASE/i3module.py" $@
