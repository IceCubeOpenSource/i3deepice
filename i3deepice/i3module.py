# coding: utf-8

import sys
import os
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, 'lib/'))
sys.path.insert(0, os.path.join(dirname, 'models/'))
from model_parser import parse_functional_model
from helpers import *
import numpy as np
from icecube import icetray
from I3Tray import I3Tray
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser  # ver. < 3.0
from collections import OrderedDict
import time
from icecube.dataclasses import I3MapStringDouble
from icecube import dataclasses, dataio
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from mc_truth import classify_wrapper
import importlib
print('Using keras version {} from {}'.format(keras.__version__,
                                              keras.__path__))


class DeepLearningModule(icetray.I3ConditionalModule):
    """IceTray compatible class of the  Deep Learning Classifier
    """

    def __init__(self, context):
        """Initialize the Class
        """
        icetray.I3ConditionalModule.__init__(self, context)
        self.AddParameter("pulsemap", "Define the name of the pulsemap",
                          "InIceDSTPulses")
        self.AddParameter("save_as", "Define the Output key",
                          "TUM_dnn_classification")
        self.AddParameter("batch_size", "Size of the batches", 40)
        self.AddParameter("cpu_cores", "number of cores to be used", 1)
        self.AddParameter("gpu_cores", "number of gpu to be used", 1)
        self.AddParameter("remove_daq", "whether or not to remove Q-Frames",
                          False)
        self.AddParameter("calib_errata", "Key for Calibration Errata", 'None')
        self.AddParameter("bad_dom_list", "Key for Bad Doms", 'None')
        self.AddParameter("saturation_windows", "Key for Saturation Windows?",
                          'None')
        self.AddParameter("bright_doms", "Key for Bright DOMs", 'None')
        self.AddParameter("model", "which model to use", 'classification')
        self.AddParameter("benchmark", "store benchmark results?", False)
        self.AddParameter("add_truth", "Want to add the true classification?", True)

    def Configure(self):
        """Read the network architecture, input & output information from config files
        """

        print('Initialize the Deep Learning classifier..\
               this may take a few seconds')
        self.__runinfo = np.load(os.path.join(dirname,'models/{}/run_info.npy'.format(self.GetParameter("model"))),
                                 allow_pickle=True)[()]
        self.__grid = np.load(os.path.join(dirname, 'lib/grid.npy'),
                              allow_pickle=True)[()]
        self.__inp_shapes = self.__runinfo['inp_shapes']
        self.__out_shapes = self.__runinfo['out_shapes']
        self.__inp_trans = self.__runinfo['inp_trans']
        self.__out_trans = self.__runinfo['out_trans']
        self.__pulsemap = list(self.GetParameter("pulsemap"))
        self.__save_as = list(self.GetParameter("save_as"))
        self.__batch_size = self.GetParameter("batch_size")
        self.__cpu_cores = self.GetParameter("cpu_cores")
        self.__gpu_cores = self.GetParameter("gpu_cores")
        self.__remove_daq = self.GetParameter("remove_daq")
        self.__benchmark = self.GetParameter("benchmark")
        self.__calib_err_key = list(self.GetParameter("calib_errata"))
        self.__bad_dom_key = list(self.GetParameter("bad_dom_list"))
        self.__sat_window_key = list(self.GetParameter("saturation_windows"))
        self.__bright_doms_key = list(self.GetParameter("bright_doms"))
        self.__add_truth = self.GetParameter("add_truth")
        self.__frame_buffer = []
        self.__buffer_length = 0
        self.__num_pframes = 0
        pulse_info_vars = [self.__pulsemap, self.__calib_err_key, self.__bad_dom_key,
                           self.__sat_window_key, self.__bright_doms_key]
        pulse_info_lens = [len(i) for i in pulse_info_var]
        max_len = np.max(pulse_info_lens)
        for  pulse_info_var in pulse_info_vars:
            if len(pulse_info_var) == 1:
                pulse_info_var = pulse_info_var.extend([pulse_info_var[0] * (max_len - 1)])
            elif len(pulse_info_var) != max_len:
                raise ValueError('Array length do not match. Either give a full list or a single string, got {}'.format(pulse_info_var))
        if (len(self.__save_as) == 1) & (max_len > 1):
            s0 = self.__save_as[0]
            self.__save_as = [s0 + '_{}'.format(i) for i in range(max_len)]
        else:
            if len(self.__save_as) != max_len:
                raise ValueError('Save_as array length does not match. Either give full list or string')

        dataset_configparser = ConfigParser()
        dataset_configparser.read(os.path.join(dirname,'models/{}/config.cfg'.format(self.GetParameter("model"))))
        inp_defs = dict()
        for key in dataset_configparser.options('Input_Times'):
            inp_defs[key] = dataset_configparser.get('Input_Times', key)
        for key in dataset_configparser.options('Input_Charges'):
            inp_defs[key] = dataset_configparser.get('Input_Charges', key)
        self.__inputs = []
        for key in self.__inp_shapes.keys():
            binput = []
            branch = self.__inp_shapes[key]
            for bkey in branch.keys():
                if bkey == 'general':
                    continue
                elif 'charge_quantile' in bkey:
                    feature = 'pulses_quantiles(charges, times, {})'.format(float('0.' + bkey.split('_')[3]))
                else:
                    feature = inp_defs[bkey.replace('IC_', '')]
                trans = self.__inp_trans[key][bkey]
                binput.append((feature, trans))
            self.__inputs.append(binput)


        print("Pulsemap {},  Store results under {}".format(self.__pulsemap, self.__save_as))
        if __name__ == "__main__":
            func_model_def = importlib.import_module('models.{}.model'.format(self.GetParameter("model")))
        else:
            func_model_def = importlib.import_module('i3deepice.models.{}.model'.format(self.GetParameter("model")))
        self.__output_names = func_model_def.output_names
        self.__model = func_model_def.model(self.__inp_shapes,
                                            self.__out_shapes)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=self.__cpu_cores,
                                          inter_op_parallelism_threads=self.__cpu_cores,
                                          device_count={'GPU': self.__gpu_cores ,
                                                        'CPU': self.__cpu_cores},
                                          log_device_placement=False)
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        self.__model.load_weights(os.path.join(dirname, 'models/{}/weights.npy'.format(self.GetParameter("model"))))

    def get_cleaned_pulses(self, frame, pulse_key, bright_dom_key='None'
                           bad_dom_key = 'None', calib_err_key = 'None',
                           sat_window_key = 'None'):
        if isinstance(frame[pulse_key],
                      dataclasses.I3RecoPulseSeriesMapMask):
            pulses = frame[pulse_key].apply(frame)
        else:
            pulses = frame[pulse_key]
        if bright_dom_key in frame.keys():
            for bright_dom in frame[bright_dom_key]:
                if bright_dom in pulses.keys():
                    pulses.pop(bright_dom)
        if bad_dom_key in frame.keys():
            for bad_dom in frame[bad_dom_key]:
                if bad_dom in pulses.keys():
                    pulses.pop(bad_dom)
        if calib_err_key in frame.keys():
            for errata in frame[calib_err_key]:
                if errata.key() not in pulses.keys():
                    continue
                single_dom_pulses = pulses[errata.key()]
                for it_errata in errata.data():
                    for spulse in single_dom_pulses:
                        if (spulse.time > it_errata.start) &\
                           (spulse.time < it_errata.stop):
                            spulse.charge = 0
        if sat_window_key in frame.keys():
            for errata in frame[sat_window_key]:
                if errata.key() not in pulses.keys():
                    continue
                single_dom_pulses = pulses[errata.key()]
                for it_errata in errata.data():
                    for spulse in single_dom_pulses:
                        if (spulse.time > it_errata.start) &\
                           (spulse.time < it_errata.stop):
                            spulse.charge = 0
        return pulses

    def BatchProcessBuffer(self, frames):
        """Batch Process a list of frames.
        This includes pre-processing, prediction and storage of the results
        """
        for counter, pulse_key in enumerate(self.__pulsemap):
            f_slices = []
            timer_t0 = time.time()
            benchmark_times = [timer_t0]
            if self.__num_pframes == 0:
                return
            for frame in frames:
                if frame.Stop != icetray.I3Frame.Physics:
                    continue
                if pulse_key not in frame.keys():
                    print('No Pulsemap called {}..continue without prediction'.format(pulse_key))
                    continue
                f_slice = []
                pulses = self.get_cleaned_pulses(
                                frame, pulse_key, 
                                bright_dom_key = self.__bright_doms_key[counter],
                                bad_dom_key = self.__bad_dom_key[counter],
                                calib_err_key = self.__calib_err_key[counter],
                                sat_window_key = self.__sat_window_key[counter])
                t0 = get_t0(pulses)
                for key in self.__inp_shapes.keys():
                    f_slice.append(np.zeros(self.__inp_shapes[key]['general']))
                for omkey in pulses.keys():
                    dom = (omkey.string, omkey.om)
                    if dom not in self.__grid.keys():
                        continue
                    gpos = self.__grid[dom]
                    charges = np.array([p.charge for p in pulses[omkey][:]
                                        if p.charge > 0])
                    times = np.array([p.time for p in pulses[omkey][:]
                                      if p.charge > 0]) - t0
                    widths = np.array([p.width for p in pulses[omkey][:]
                                       if p.charge > 0])
                    if len(charges) == 0: # might happen through pulse cleaning
                        continue
                    for branch_c, inp_branch in enumerate(self.__inputs):
                        for inp_c, inp in enumerate(inp_branch):
                            f_slice[branch_c][gpos[0]][gpos[1]][gpos[2]][inp_c] =\
                                inp[1](eval(inp[0]))
                f_slices.append(f_slice)
                benchmark_times.append(time.time())
            predictions = self.__model.predict(np.array(np.squeeze(f_slices,
                                                                   axis=1),
                                                        ndmin=5),
                                               batch_size=self.__batch_size,
                                               verbose=0, steps=None)
            prediction_time = (time.time() - benchmark_times[-1])/len(f_slices)
            benchmark_times = np.diff(benchmark_times)
            i = 0
            for frame in frames:
                if frame.Stop != icetray.I3Frame.Physics:
                    continue
                if pulse_key not in frame.keys():
                    continue
                output = I3MapStringDouble()
                prediction = np.concatenate(np.atleast_2d(predictions[i]))
                for j in range(len(prediction)):
                    output[self.__output_names[j]] = float(prediction[j])
                frame.Put(self.__save_as[counter], output)
                if self.__benchmark & (counter==0):
                    output_bm = I3MapStringDouble()
                    output_bm['processing'] = benchmark_times[i]
                    output_bm['avg_prediction'] = prediction_time
                    output_bm['batch_size'] = len(f_slices)
                    frame.Put(self.__save_as[counter] + '_Benchmark', output_bm)
                if self.__add_truth and not ('classification_truth' in frame.keys()):
                    try:
                        classify_wrapper(frame)
                    except Exception as e:
                        print('Failed while calculating the truth...')
                        print(e)
                i += 1
            tot_time = time.time() - timer_t0
            t_str = 'Total Time {:.2f}s, Processing Time: {:.2f}s/event, Prediction Time {:.3f}s/event \n'
            print(t_str.format(tot_time, np.median(benchmark_times), prediction_time))
        return

    def Process(self):
        frame = self.PopFrame()
        if frame.Stop == icetray.I3Frame.Physics:
            self.Physics(frame)
        else:
            self.DAQ(frame)
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

    def DAQ(self, frame):
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


def print_info(phy_frame, save_as):
    print('Run_ID {} Event_ID {}'.format(phy_frame['I3EventHeader'].run_id,
                                         phy_frame['I3EventHeader'].event_id))
    if 'classification_truth' in phy_frame.keys():
        print('Truth:\n{}'.format(phy_frame['classification_truth'].value))
    if isinstance(save_as, str):
        keys = [key for key in phy_frame.keys() if save_as in key]
    for key in keys:
        if key in phy_frame.keys():
            print('Prediction ({}) :\n{}'.format(key, phy_frame[key]))
        else:
            print('Key {} does not exist in frame'.format(key_all))
    print('\n')
    return


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", help="files to be processed",
        type=str, nargs="+", required=True)
    parser.add_argument(
        "--plot", action="store_true", default=False)
    parser.add_argument(
        "--pulsemap", type=str, default="InIceDSTPulses", nargs='+')
    parser.add_argument(
        "--batch_size", type=int, default=48)
    parser.add_argument(
        "--cpu_cores", type=int, default=1)
    parser.add_argument(
        "--gpu_cores", type=int, default=1)
    parser.add_argument(
        "--remove_daq", action='store_true', default=False)
    parser.add_argument(
        "--model", type=str, default='classification')
    parser.add_argument(
        "--save_as", type=str, default="TUM_dnn_classification")
    parser.add_argument(
        "--outfile", type=str, default='None')
    parser.add_argument(
        "--benchmark", action='store_true',
        default=False)
    parser.add_argument(
        "--no_truth", action='store_false', default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArguments()
    if args.plot:
        from plotting import make_plot
    files = []
    for j in np.atleast_1d(args.files):
        if os.path.isdir(j):
            files.extend([os.path.join(j, i)
                          for i in os.listdir(j) if '.i3' in i])
        else:
            files.append(j)
    print('Filelist: {}'.format(files))
    tray = I3Tray()
    tray.AddModule('I3Reader', 'reader',
                   FilenameList=files)
    tray.AddModule(DeepLearningModule, "DeepLearningMod",
                   pulsemap=args.pulsemap,
                   batch_size=args.batch_size,
                   cpu_cores=args.cpu_cores,
                   gpu_cores=args.gpu_cores,
                   remove_daq=args.remove_daq,
                   calib_errata='CalibrationErrata',
                   bad_dom_list='BadDomsList',
                   saturation_windows='SaturationWindows',
#                  bright_doms='BrightDOMs',
                   model=args.model,
                   add_truth=args.no_truth,
                   save_as = args.save_as,
                   benchmark=args.benchmark)
    tray.AddModule(print_info, 'printer',
                   save_as=args.save_as,
                   Streams=[icetray.I3Frame.Physics])
    if args.outfile != 'None':
        tray.AddModule("I3Writer", 'writer',
                       Filename=args.outfile)
    if args.plot:
        tray.AddModule(make_plot, 'plotter',
                       Streams=[icetray.I3Frame.Physics])
    tray.Execute(20)
    tray.Finish()
