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


## extract anodic ECAP signal, current and time values and its cathodic counterpart ECAP signal
def extract_electrodepair(dataset, taskkey, electrodepair):
    ## get anodic and cathodic measurements in a dataframe
    # extract taskkey data
    taskkey_data = dataset.groupby("taskkey").get_group(taskkey)
    
    # extract anodic and cathodic measurements for chosen electrodepair
    electrodepair_data = taskkey_data.groupby(["stimulatingelectrode", "recordingelectrode"]).get_group(electrodepair)
    # split up anodic and cathodic measurements
    anodic_data = electrodepair_data.groupby(["polarity"]).get_group("anodiccathodic")
    cathodic_data = electrodepair_data.groupby(["polarity"]).get_group("cathodicanodic")
    
    ## extract time values
    anodic_data = anodic_data.set_index(["name"])
    cathodic_data = cathodic_data.set_index(["name"])
    # assume that time values are the same for both measurements
    time = anodic_data.loc["time"]["datapoints"]
    # drop time measurements
    anodic_data = anodic_data.drop(labels="time")
    cathodic_data = cathodic_data.drop(labels="time")
    
    ## extract current values
    currents_list = []
    
    for row in anodic_data.index:
        if row == "zerotemplate":
            currents_list.append(0.)
        else:
            current = row.replace("cu", "")
            currents_list.append(float(current))

    currents = np.asarray(currents_list)
    
    ## extract measurements and current values
    # note: tried it to read the whole datapoint column and convert it to a numpy array
    # it did not work and I reverted back to for loops
    anodic_list = []
    cathodic_list = []
    
    for row in anodic_data["datapoints"]:
        anodic_list.append(row)
    for row in cathodic_data["datapoints"]:
        cathodic_list.append(row)
    
    anodic = np.asarray(anodic_list)
    cathodic = np.asarray(cathodic_list)
    
    ## return all values
    return anodic, cathodic, time, currents


## plot anodic (and cathodic) ECAP signal
def plot_electrodepair(anodic, cathodic, time, currents, electrodepair=None):
    plot_number = anodic.shape[0]
    
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(plot_number, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    fig.suptitle(f"{electrodepair}")
    for i, anodic in enumerate(anodic):
        axs[i].plot(time, anodic, color="red", linestyle="-", linewidth=1.5, label="anodic-first")
        axs[i].plot(time, cathodic[i], color="blue", linestyle="--", linewidth=1.5, label="cathodic-first")
        # plt.grid(linewidth=0.3)
    plt.legend() # loc="lower center"
    plt.show()