# coding: utf-8

import os
import sys
from icecube import icetray
from I3Tray import I3Tray
from icecube.dataclasses import I3MapStringDouble
from icecube import dataclasses, dataio
import argparse
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
#import keras
#import importlib

# Add more imports


class DeepLearningClassifier(icetray.I3ConditionalModule):
    """IceTray compatible class of the  Deep Learning Classifier
    """

    def __init__(self,context):
        """Initialize the Class
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("pulsemap","Define the name of the pulsemap",
                          "InIceDSTPulses")
        self.AddParameter("save_as", "Define the Output key",
                          "Deep_Learning_Classification")
        self.AddParameter("batch_size", "Size of the batches", 40)
        self.AddParameter("cpu_cores", "number of cores to be used", 1)
        self.AddParameter("gpu_cores", "number of gpu to be used", 1)
        self.AddParameter("remove_daq", "whether or not to remove Q-Frames", False)
        self.AddParameter("model", "which model to use", 'classification')

    def Configure(self):
        """Read the network architecture and input, output information from config files
        """
    
        # Initialize variables
        print('Initializing')
        self.__pulsemap = self.GetParameter("pulsemap")
        self.__save_as =  self.GetParameter("save_as")
        self.__batch_size =  self.GetParameter("batch_size")
        self.__cpu_cores =  self.GetParameter("cpu_cores")
        self.__gpu_cores =  self.GetParameter("gpu_cores")
        self.__remove_daq = self.GetParameter("remove_daq")
        self.__frame_buffer = []
        self.__buffer_length = 0
        self.__num_pframes = 0
        # Read additional information from a config file?!


        # Load a model and weights here

        # Example:
        #func_model_def = importlib.import_module('i3deepice.models.{}.model'.format(self.GetParameter("model")))
        #self.__model = func_model_def.model(self.__inp_shapes, self.__out_shapes)
        #self.__model.load_weights(os.path.join(dirname, 'models/{}/weights.npy'.format(self.GetParameter("model"))))

        config = tf.ConfigProto(intra_op_parallelism_threads=self.__cpu_cores,
                                inter_op_parallelism_threads=self.__cpu_cores,
                                device_count = {'GPU': self.__gpu_cores , 'CPU': self.__cpu_cores},
                                log_device_placement=False)
        sess = tf.Session(config=config)
        set_session(sess)
        return


    def BatchProcessBuffer(self, frames):
        """Batch Process a list of frames. This includes pre-processing, prediction and storage of the results
        """

        # Data Processing
        if self.__num_pframes == 0:
            return
        for frame in frames:
            if frame.Stop != icetray.I3Frame.Physics:
                continue
            # Implement the single event processing here

        # Do the Prediction
        predictions = self.__model.predict() # or something similar

        # Store the Results to the Frame
        i = 0
        for frame in frames:
            if frame.Stop != icetray.I3Frame.Physics:
                continue
            # output = some_function_on(predictions[i])
            frame.Put(self.__save_as, output)
            i += 1
        return


    def Physics(self, frame):
        """ Buffer physics frames until batch size is reached, then start processing
        """
        self.__frame_buffer.append(frame)
        self.__buffer_length += 1
        self.__num_pframes += 1
        if self.__buffer_length == self.__batch_size:
            self.BatchProcessBuffer(self.__frame_buffer)
            for frame in self.__frame_buffer:
                self.PushFrame(frame)
            self.__frame_buffer[:] = []
            self.__buffer_length = 0
            self.__num_pframes = 0
        return


    def DAQ(self,frame):
        """ Handel Q-Frames. Append to buffer if they should be kept
        """
        if not self.__remove_daq:
            self.__frame_buffer.append(frame)
        return

    def Finish(self):
        """ Process the remaining (incomplete) batch of frames
        """
        self.BatchProcessBuffer(self.__frame_buffer)
        for frame in self.__frame_buffer:
            self.PushFrame(frame)
        self.__frame_buffer[:] = []
        return

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--plot", action="store_true", default=False)
    parser.add_argument(
        "--pulsemap", type=str, default="InIceDSTPulses")
    parser.add_argument(
        "--batch_size", type=int, default=40)
    parser.add_argument(
        "--cpu_cores", type=int, default=1)
    parser.add_argument(
        "--gpu_cores", type=int, default=1)
    parser.add_argument(
        "--remove_daq", action='store_true', default=False)
    parser.add_argument(
        "--model", type=str, default='classification')
    parser.add_argument(
        "--outfile", type=str, default="~/myhdf.i3.bz2")
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
                   cpu_cores=args.cpu_cores,
                   gpu_cores=args.gpu_cores,
                   remove_daq=args.remove_daq,
                   model=args.model)
    tray.AddModule("I3Writer", 'writer',
                   Filename=args.outfile)
    tray.Execute()
    tray.Finish()
