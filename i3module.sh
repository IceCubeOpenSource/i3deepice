#!/bin/bash

echo $PATH
echo $HOSTNAME
echo "$(</usr/local/cuda/version.txt)"
PY_ENV=/home/tglauch/virtualenvs/tf_env3/
#PY_ENV=/home/tglauch/tf_env_gpu/
IC_ENV=`/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
#PY_ENV=/home/tglauch/venv_new/
#META_PROJECT=/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/icerec/V05-02-04/env-shell.sh
META_PROJECT=/home/tglauch/i3/combo/build/env-shell.sh

echo $HOSTNAME
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
SDIR="$(dirname "$DIR")"
eval $IC_ENV
export HDF5_USE_FILE_LOCKING='FALSE'
export KERAS_BACKEND="tensorflow"
source $META_PROJECT python "$SDIR/I3Module/i3module.py" $@



#META_PROJECT=/usr/local/icetray/env-shell.sh
#source $META_PROJECT python "/data/user/tglauch/DeepIceLearning/I3Module/i3module.py" $@

#python "$SDIR/I3Module/i3module.py" $@
#nvcc --version
#PY_ENV=/home/tglauch/virtualenvs/tf_env3/
#PY_ENV=/home/tglauch/venv_new/
#META_PROJECT=/cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/RHEL_7_x86_64/metaprojects/icerec/V05-02-04/env-shell.sh
#source $META_PROJECT python "$SDIR/I3Module/i3module.py" $@
#source /home/tglauch/i3/combo/build/env-shell.sh python "$SDIR/I3Module/i3module.py" $@
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
#SDIR="$(dirname "$DIR")"
#export HDF5_USE_FILE_LOCKING='FALSE'
#PY_ENV=/home/tglauch/tf_env_gpu/
#IC_ENV=`/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`
#META_PROJECT=/home/tglauch/i3/combo/build/env-shell.shexport KERAS_BACKEND="tensorflow"eval $IC_ENV
