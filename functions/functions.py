## import libraries
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

## zero amplitude template
def zero_template_subtraction(signal, zero_template):
    """
        Returns ECAP signal with zero template subtracted.

        signal : 1D numpy array
            ECAP signal.
        zero_template : 1D numpy array
            Stimulus-free measurement.
    """
    return (signal - zero_template)


## scaled template subtraction (no arctan)
def scaled_template_subtraction(no_ECAP, ECAP, no_ECAP_current, ECAP_current):
    """
        Returns scaled template subtraction of an ECAP signal.
    
        Scaling factor is the ratio of stimulation currents between an ECAP signal above and below neural threshold.

        no_ECAP : 1D numpy array
            ECAP signal below neural threshold.
        ECAP : 1D numpy array
            ECAP signal above neural threshold.
        no_ECAP_current : float
            Stimulation current of ECAP signal below neural threshold.
        ECAP_current : float
            Stimulation current of ECAP signal above neural threshold.
    """
    return (ECAP - ECAP_current/no_ECAP_current * no_ECAP)


## scaled template subtraction (with arctan)
def damped_scaled_template_subtraction(no_ECAP, ECAP, no_ECAP_current, ECAP_current):
    """
        Returns scaled template subtraction of an ECAP signal.
    
        Scaling factor is the ratio of stimulation currents between an ECAP signal above and below neural threshold.
        The scaling factor is dampened by taking arctan of it.

        no_ECAP : 1D numpy array
            ECAP signal below neural threshold.
        ECAP : 1D numpy array
            ECAP signal above neural threshold.
        no_ECAP_current : float
            Stimulation current of ECAP signal below neural threshold.
        ECAP_current : float
            Stimulation current of ECAP signal above neural threshold.
    """
    return (ECAP - np.arctan(ECAP_current/no_ECAP_current) * no_ECAP)


## alternating stimulation
def alternating_stimulation_sum(anodic, cathodic, weight=0.5):
    """
        Returns the sum of alternated stimulated signals.

        anodic : 1D numpy array
            Anodic stimulated signal.
        cathodic : 1D numpy array
            Cathodic stimulated signal
        weight : float (between 0. and 1.)
            Differently weigh anodic and cathodic signals.
    """
    return (weight * anodic + (1 - weight) * cathodic)


## extract anodic ECAP signal, its cathodic counterpart, currents and time values for an electrodepair
def extract_electrodepair(dataset, taskkey, electrodepair):
    """
        Extract anodic ECAP signal, its cathodic counterpart, the currents and time values for a electrodepair.
    
        dataset : pandas Dataframe
        taskkey : str
        electrodepair : tuple
    """
    ## get anodic and cathodic measurements from dataset
    taskkey_data = dataset.groupby("taskkey").get_group(taskkey)    # get data from specified taskkey
    # get anodic and cathodic measurements for specified electrodepair
    electrodepair_data = taskkey_data.groupby(["stimulatingelectrode", "recordingelectrode"]).get_group(electrodepair)
    # split up anodic and cathodic measurements
    anodic_data = electrodepair_data.groupby(["polarity"]).get_group("anodiccathodic")
    cathodic_data = electrodepair_data.groupby(["polarity"]).get_group("cathodicanodic")
    
    ## extract time values
    anodic_data = anodic_data.set_index(["name"])   # sort by measurement name
    cathodic_data = cathodic_data.set_index(["name"])
    
    time = anodic_data.loc["time"]["datapoints"]    # assume that time values are the same for anodic and cathodic
    time = np.asarray(time)
    # drop time measurements
    anodic_data = anodic_data.drop(labels="time")
    cathodic_data = cathodic_data.drop(labels="time")
    
    ## extract current values
    currents_list = []
    
    # loop over every measurement
    for row in anodic_data.index:   # assume that current values are the same for anodic and cathodic
        if row == "zerotemplate":   # skip zero current because its just 0cu
            pass
        else:                       # get current of measurement
            current = row.replace("cu", "")
            currents_list.append(float(current))

    currents = np.asarray(currents_list)
    
    ## extract measurements and current values
    # note: tried to read the whole column "datapoints" and convert it to a numpy array
    # -> did not work and reverted back to for loops
    anodic_list = []
    cathodic_list = []
    
    # get datapoints of every measurement
    for row in anodic_data["datapoints"]:
        anodic_list.append(row)
    for row in cathodic_data["datapoints"]:
        cathodic_list.append(row)
    
    anodic = np.asarray(anodic_list)
    cathodic = np.asarray(cathodic_list)
    
    return anodic[1:], anodic[0], cathodic[1:], cathodic[0], time, currents


## plot anodic and cathodic ECAP signal
def plot_electrodepair(anodic, cathodic, time, currents, electrodepair=None):
    plot_number = anodic.shape[0]   # get number of subplots
    
    # set aesthetics
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure(figsize=(8, 8))
    
    # set up plot
    gs = fig.add_gridspec(plot_number, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    fig.suptitle(f"Electrodepair {electrodepair}")
    
    # plot every measurement in subplot
    for i in range(plot_number):
        # plot anodic and cathodic signals
        axs[i].plot(time*10e3, anodic[i]*10e3, color="red", linestyle="-", linewidth=1.5, label="anodic-first")
        axs[i].plot(time*10e3, cathodic[i]*10e3, color="blue", linestyle="--", linewidth=1.5, label="cathodic-first")
        # add grid and currents
        axs[i].text(1.02, 0.45, str(currents[i])+"cu", transform=axs[i].transAxes)
        axs[i].grid(linewidth=0.3)

    # set aesthetics
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=12, edgecolor="none")
    fig.supxlabel("Time (ms)")
    fig.supylabel("ECAP amplitude (mV)")
    
    plt.show()