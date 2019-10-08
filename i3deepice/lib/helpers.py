import numpy as np
import scipy.stats as st

def get_t0(frame, puls_key='InIceDSTPulses'):
    pulses = frame[puls_key]
    pul = pulses.apply(frame)
    time = []
    charge = []
    for i in pul:
        for j in i[1]:
            charge.append(j.charge)
            time.append(j.time)
    return median(time, weights=charge)


def median(arr, weights=None):
    if weights is not None:
        weights = 1. * np.array(weights)
    else:
        weights = np.ones(len(arr))
    rv = st.rv_discrete(values=(arr, weights / weights.sum()))
    return rv.median()

def charge_after_time(charges, times, t=100):
    mask = (times - np.min(times)) < t
    return np.sum(charges[mask])


def time_of_percentage(charges, times, percentage):
    charges = charges.tolist()
    cut = np.sum(charges) / (100. / percentage)
    sum = 0
    for i in charges:
        sum = sum + i
        if sum > cut:
            tim = times[charges.index(i)]
            break
    return tim

#based on the pulses
def pulses_quantiles(charges, times, quantile):
    tot_charge = np.sum(charges)
    cut = tot_charge*quantile
    progress = 0
    for i, charge in enumerate(charges):
        progress += charge
        if progress >= cut:
            return times[i]
