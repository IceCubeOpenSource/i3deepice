#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.stats import norm
from six.moves import configparser

def IC_divide_1000(x, r_vals=None):
    return  x / 1000.

def IC_divide_10(x, r_vals=None):
    return  x / 10.

def IC_divide_10000(x, r_vals=None):
    return  x / 10000.

def IC_divide_100(x, r_vals=None):
    return  x / 100.

def identity(x, r_vals=None):
    return x

def IC_std_one(x, r_vals=None, axis=(1,2,3)):
    return  x / np.std(x, axis=axis)[:,np.newaxis,np.newaxis,np.newaxis]

def IC_centralize(x, r_vals=None, axis=(1,2,3)):
    return ((x - np.mean(x, axis=axis)[:,np.newaxis,np.newaxis,np.newaxis]))\
            / np.std(x, axis=axis)[:,np.newaxis,np.newaxis,np.newaxis]

def log10(x, r_vals=None):
    mask = (x >0.)
    x[mask] = np.log10(1.*x[mask])
    x[~mask] = 0.
    return x

def centralize(x, r_vals=None):
    if np.std(x) > 0.:
        return ((x - np.mean(x)) / np.std(x))
    else:
        return (x - np.mean(x))


def waveform_offset(x):
    return (x - 10000) / 400


def max(x):
    return np.amax(x)


def max_min_delta(x):
    return np.max(x) - np.min(x)


def shift_min_to_zero(x):
    return x - np.amin(x)


def sort_input(x):
    return np.sort(np.ndarray.flatten(x))


def plus_one_log10(x):
    tmp = x + 1.
    return np.log10(tmp)


def zenith_prep(x, r_vals=None):
    x = x / np.pi
    return x


def log_handle_zeros_flatten_top30(x):
    return np.sort(np.ndarray.flatten(np.log10(1. + x)))[-30:]


def log_handle_zeros(x):
    return np.where(x != 0, np.log10(x), 0)


def sort_input_and_top20(x):
    return np.sort(np.ndarray.flatten(x))[-20:]


def smeared_one_hot_encode_logbinned(E):
    width = 0.16
    bins = np.linspace(3, 7, 50)
    gauss = norm(loc=np.log10(E), scale=width)
    smeared_hot_output = gauss.pdf(bins)
    return smeared_hot_output / np.sum(smeared_hot_output)


#def one_hot_encode_logbinned(x):
#    bins = np.linspace(3, 7, 50)
#    bin_indcs = np.digitize(np.log10(x), bins)
#    one_hot_output = to_categorical(bin_indcs, len(bins))
#    return one_hot_output


def zenith_to_binary(x):
    """
    returns boolean values for the zenith (0 or 1; up or down, > or < pi/2) as np.array.
    """
    ret = np.copy(x)
    ret[ret < 1.5707963268] = 0.0
    ret[ret > 1] = 1.0
    return ret


def time_prepare(x):
    """
    This function normalizes the finite values of input data to the interval [0,1] and 
    replaces all infinity-values with replace_with (defaults to 1).
    """
    replace_with = 1.0
    ret = np.copy(x)
    time_np_arr_max = np.max(ret[ret != np.inf])
    time_np_arr_min = np.min(ret)
    ret = (ret - time_np_arr_min) / (time_np_arr_max - time_np_arr_min)
    ret[ret == np.inf] = replace_with
    return ret


def oneHotEncode_EventType_simple(x):
    """
    This function one hot encodes the input for the event 
    types cascade, tracks, doubel-bang
    """
    # define universe of possible input values
    onehot_encoded = []
    universe = [1, 2, 3]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    return onehot_encoded


def oneHotEncode_EventType_noDoubleBang_simple(x):
    """
    This function one hot encodes the input
    """
    # define universe of possible input values
    onehot_encoded = []
    universe = [1, 2, 3]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    if onehot_encoded == [0., 0., 1.]:
        onehot_encoded = [1.0, 0.0, 0.0]
    return onehot_encoded[:-1]


def log_of_sum(x):
    return np.log10(np.sum(x) + 0.0001)


def max_min_delta_log(x):
    return np.log10(np.max(x) - np.min(x))


def oneHotEncode_01(x, r_vals=None):
    """
    This function one hot encodes the input for a binary label 
    """
    # define universe of possible input values
    onehot_encoded = []
    universe = [0, 1]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    return onehot_encoded


def oneHotEncode_EventType_exact(x, r_vals=None):
    """
    This function one hot encodes the input for the event
    types cascade, tracks, doubel-bang
    """
    # define universe of possible input values
    onehot_encoded = []
    # universe has to defined depending on the problem,
    # in this implementation integers are neccesary
    universe = [0, 1, 2, 3, 4, 5, 6]
    for i in range(len(universe)):
        if x == universe[i]:
            value = 1.
        else:
            value = 0.
        onehot_encoded.append(value)
    return onehot_encoded


def oneHotEncode_3_evtypes(x, r_vals=None):
    """
    This function one hot encodes the input for the event
    types cascade, tracks, doubel-bang
    """

    # define universe of possible input values
    cascade = [1., 0., 0.]
    track = [0., 1., 0.]
    s_track = [0., 0., 1.]
    # map x to possible classes
    mapping = {0: cascade, 1: cascade, 2: track, 3: s_track, 4: track,
               5: cascade, 6: cascade, 7: cascade, 8: track, 9: cascade}
    return mapping[int(x)]


def oneHotEncode_db(x, r_vals=None):
    """
    This function one hot encode for event type  doubel-bang, no-double bang
    """
    # define universe of possible input values

    ndoublebang = [1., 0.]
    doublebang = [0., 1.]

    cut_tau = 10. # aus config auslesen, no hardcode
    if int(x) in [5, 6]:
        if r_vals[8] >= cut_tau:
            return doublebang
        else:
            return ndoublebang

    mapping = {0: ndoublebang, 1: ndoublebang, 2: ndoublebang, 3: ndoublebang,
               4: ndoublebang, 5: doublebang, 6: doublebang, 7: ndoublebang,
               8: ndoublebang, 9: ndoublebang}
    return mapping[int(x)]


def oneHotEncode_4_evtypes(x, r_vals=None):
    """
    This function one hot encodes the input for the event types 
    cascade, tracks, doubel-bang, starting tracks
    """
    cascade = [1., 0., 0., 0.]
    track = [0., 1., 0., 0.]
    doublebang = [0., 0., 1., 0.]
    s_track = [0., 0., 0., 1.]
    # map x to possible classes
    mapping = {0: cascade, 1: cascade, 2: track, 3: s_track, 4: track,
               5: doublebang, 6: doublebang, 7: cascade, 8: track, 9: cascade}
    return mapping[int(x)]

def oneHotEncode_signature(x, r_vals=None):
    """
    This function one hot encodes the input for the event types 
    cascade, tracks, doubel-bang, starting tracks
    """
    starting = [1., 0.]
    non_starting = [0., 1.]
    # map x to possible classes
    mapping = {-1: non_starting, 0: starting, 1: non_starting, 2: non_starting}
    ret = np.zeros((len(np.atleast_1d(x)), 2))
    for i in mapping.keys():
        ret[x == i] = mapping[i]
    return ret

def oneHotEncode_new(x, r_vals=None):
    """
    This function one hot encodes the input for the event types 
    non-starting cascade, starting  cascade, through-going track, 
    starting track, stopping track
    """
    ns_cascade = [1., 0., 0., 0., 0.]
    s_cascade = [0., 1., 0., 0., 0.]
    tg_track = [0., 0., 1., 0., 0.]
    sta_track = [0., 0., 0., 1., 0.]
    sto_track = [0., 0., 0., 0., 1.]
    # map x to possible classes
    mapping = {0: ns_cascade, 1: s_cascade, 2: tg_track, 3: sta_track, 4: sto_track,
               11: ns_cascade, 22: tg_track, 23: sto_track}
    ret = np.zeros((len(np.atleast_1d(x)), 5))
    for i in mapping.keys():
        ret[x == i] = mapping[i]
    return ret


def oneHotEncode_4_evtypes_tau_decay_length(x, r_vals):
    """
    This function one hot encodes the input for the event types 
    cascade, tracks, doubel-bang, starting tracks
    """
    cascade = [1., 0., 0., 0.]
    track = [0., 1., 0., 0.]
    doublebang = [0., 0., 1., 0.]
    s_track = [0., 0., 0., 1.]
    
    cut = 5. # aus config auslesen, no hardcode
    # map x to possible classes
    if int(x) in [5, 6]:
        if r_vals[8] >= cut:
            return doublebang
        else:
            return cascade
    else:
        mapping = {0: cascade, 1: cascade, 2: track, 3: s_track, 4: track,
                   5: doublebang, 6: doublebang, 7: cascade, 8: track, 9: cascade}
        return mapping[int(x)]


def oneHotEncode_Starting_padding0(x, r_vals):
    nope = [1., 0.]
    starting = [0., 1.]
    pos = dataclasses.I3Position(r_vals["vert_x"], r_vals["vert_y"], r_vals["vert_z"])
    dir = dataclasses.I3Direction(r_vals["dir_x"], r_vals["dir_y"], r_vals["dir_z"])
    gcdfile = "/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2013.56429_V0.i3.gz"
    padding = 0 # aus config auslesen, no hardcode
#    print "Used Padding:  {}m".format(padding)
#    print parser_dict
#    print pos
#    print dir
#    print "Nice"  
 
    surface = icecube.MuonGun.ExtrudedPolygon.from_file(gcdfile,
                                                        padding=padding)
    intersections = surface.intersection(pos, dir)
    if intersections.first <= 0 and intersections.second > 0:
        starting = starting  # starting event
    else:
        starting = nope  # through-going or stopping event
    return starting


def oneHotEncode_4_evtypes_tau_decay_length_strack_length(x, r_vals):
    """
    This function one hot encodes the input for the event types 
    cascade, tracks, doubel-bang, starting tracks
    """
    cascade = [1., 0., 0., 0.]
    track = [0., 1., 0., 0.]
    doublebang = [0., 0., 1., 0.]
    s_track = [0., 0., 0., 1.]

    cut_tau = 5. # aus config auslesen, no hardcode
    cut_track = 75.  # aus config auslesen, no hardcode
    # map x to possible classes
    if int(x) in [5, 6]:
        if r_vals[8] >= cut_tau:
            return doublebang
        else:
            return cascade
    elif int(x) in [3]:
        if r_vals[22] >= cut_track:
            return s_track
        else:
            return cascade 
    else:
        mapping = {0: cascade, 1: cascade, 2: track, 3: s_track, 4: track,
                   5: doublebang, 6: doublebang, 7: cascade, 8: track, 9: cascade}
        return mapping[int(x)]


def oneHotEncode_4_evtypes_tau_decay_length_strack_length_10(x, r_vals):
    """
    This function one hot encodes the input for the event types 
    cascade, tracks, doubel-bang, starting tracks
    """
    cascade = [1., 0., 0., 0.]
    track = [0., 1., 0., 0.]
    doublebang = [0., 0., 1., 0.]
    s_track = [0., 0., 0., 1.]

    cut_tau = 10. # aus config auslesen, no hardcode
    cut_track = 75.  # aus config auslesen, no hardcode
    # map x to possible classes
    if int(x) in [5, 6]:
        if r_vals[8] >= cut_tau:
            return doublebang
        else:
            return cascade
    elif int(x) in [3]:
        if r_vals[22] >= cut_track:
            return s_track
        else:
            return cascade
    else:
        mapping = {0: cascade, 1: cascade, 2: track, 3: s_track, 4: track,
                   5: doublebang, 6: doublebang, 7: cascade, 8: track, 9: cascade}
        return mapping[int(x)]


def oneHotEncode_3_evtypes_strack_length_75(x, r_vals):
    """
    This function one hot encodes the input for the event types 
    cascade, tracks, doubel-bang, starting tracks
    """
    cascade_db = [1., 0., 0.]
    track = [0., 1., 0.]
    s_track = [0., 0., 1.]

    cut_track = 75.  # aus config auslesen, no hardcode
    # map x to possible classes
    if int(x) in [3]:
        if r_vals[22] >= cut_track:
            return s_track
        else:
            return cascade_db
    else:
        mapping = {0: cascade_db, 1: cascade_db, 2: track, 3: s_track, 4: track,
                   5: cascade_db, 6: cascade_db, 7: cascade_db, 8: track, 9: cascade_db}
        return mapping[int(x)]

