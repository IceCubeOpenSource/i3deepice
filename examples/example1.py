from icecube import dataio, icetray
from I3Tray import *
import sys
import numpy as np
import os
sys.path.append('../')
from i3deepice.i3module import DeepLearningModule, print_info
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--pulsemap", type=str, default="InIceDSTPulses")
    parser.add_argument(
        "--batch_size", type=int, default=64)
    parser.add_argument(
        "--n_cores", type=int, default=1)
    parser.add_argument(
        "--remove_daq", action='store_true', default=False)
    parser.add_argument(
        "--model", type=str, default='classification')
    parser.add_argument(
        "--outfile", type=str, default="./myhdf.i3")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()
    files = []
    for j in np.atleast_1d(args.files):
        if os.path.isdir(j):
            files.extend([os.path.join(j,i) for i in os.listdir(j) if '.i3' in i])
        else:
            files.append(j)
    files = files
    save_as_base = 'TUM_dnn_classifiation_test'
    save_as = [save_as_base + '_sat', save_as_base + '_sat_bright', save_as_base + '_bright']
    tray = I3Tray()
    tray.AddModule('I3Reader','reader',
                   FilenameList = files)
    tray.AddModule(DeepLearningModule, "DeepLearningMod",
                   pulsemap=args.pulsemap,
                   batch_size=args.batch_size,
                   saturation_windows=['SaturationWindows', 'SaturationWindows', 'None'],
                   bright_doms=['None', 'BrightDOMs', 'BrightDOMs'],
                   cpu_cores=1,
                   gpu_cores=1,
                   model=args.model,
                   save_as = save_as) #save_as_base
    tray.AddModule(print_info, 'printer',
                   save_as = save_as_base,
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule("I3Writer", 'writer',
                   Filename=args.outfile)
    tray.Execute()
    tray.Finish()
