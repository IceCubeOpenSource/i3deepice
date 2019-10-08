import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import os 

def figsize(scale, ratio=(np.sqrt(5.0)-1.0)/2.0):
    fig_width_pt = 455.8843                         # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*ratio            # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size


def plot_prediction(prediction, figax=(None,None)):
    data = np.arange(len(prediction))
    bins = np.arange(0, data.max() + 1.5) - 0.5
    fig = plt.figure(figsize=figsize(0.9, (np.sqrt(5.0)-1.0)/2.0))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.barh(data, width=prediction, height=0.2,
    tick_label=['Skimming', 'Starting Cascade', 'Through-Going Track',
                'Starting Track', 'Stopping Track'], alpha=0.9)
    ax.set_xlabel('Prediction Score')
    return fig, ax

def make_plot(frame, key="Deep_Learning_Classification", ofolder='~', figpath=''):
    ofolder = os.path.expanduser(ofolder)
    if figpath =='':
        figpath = os.path.join(ofolder, '{}_{}.pdf'.format(frame['I3EventHeader'].run_id,
                                                 frame['I3EventHeader'].event_id))
    prediction = [frame[key]['Skimming'],
                  frame[key]['Cascade'],
                  frame[key]['Through_Going_Track'],
                  frame[key]['Starting_Track'],
                  frame[key]['Stopping_Track']]
    prediction = np.array(prediction)
    fig, ax = plot_prediction(prediction)
    fig.savefig(figpath, bbox_inches='tight')
    return

