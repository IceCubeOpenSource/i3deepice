#!/bin/bash

echo $PATH
echo $HOSTNAME
export KERAS_BACKEND="tensorflow"
export HDF5_USE_FILE_LOCKING='FALSE'
if [ ! -e /usr/local/cuda/bin/ ]; then
    echo "Running on CPU!"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/compat/
else
   echo "Running on GPU!"
fi
singularity exec --nv -B /home/$USER/:/home/$USER/ -B /mnt/lfs7/user/:/data/user/ -B /mnt/lfs6/ana/:/data/ana/ -B /mnt/lfs6/sim/:/data/sim/ /data/user/tglauch/icetray_combo-stable-tensorflow.1.13.2-ubuntu18.04.sif /usr/local/icetray/env-shell.sh python $@
