from icecube import dataio, icetray
from I3Tray import *
import sys
import numpy as np
import os
sys.path.append(os.environ['DNN_BASE'])
from i3module import DeepLearningClassifier
import argparse

def print_info(phy_frame):
    print('Run_ID {} Event_ID {}'.format(phy_frame['I3EventHeader'].run_id,
                                         phy_frame['I3EventHeader'].event_id))
    print(phy_frame["Deep_Learning_Classification"])
    return


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
        "--keep_daq", action='store_true', default=True)
    parser.add_argument(
        "--model", type=str, default='classification')
    parser.add_argument(
        "--outfile", type=str, default="/home/tglauch/dnn_out_test.i3.bz2")    
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
    files = sorted(files)
    tray = I3Tray()
    tray.AddModule('I3Reader','reader',
                   FilenameList = files)
    tray.AddModule(DeepLearningClassifier, "DeepLearningClassifier",
                   pulsemap=args.pulsemap,
                   batch_size=args.batch_size,
                   n_cores=args.n_cores,
                   keep_daq=args.keep_daq,
                   model=args.model)
    tray.AddModule(print_info, 'printer',
                   Streams=[icetray.I3Frame.Physics])
    tray.AddModule("I3Writer", 'writer',
                   Filename=args.outfile)
    tray.Execute()
    tray.Finish()
