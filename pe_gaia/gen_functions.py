import pycbc.waveform as pycbc_wf
import pycbc.psd as pycbc_psd
import pycbc.noise as pycbc_noise
import pycbc.types as pycbcty
import pycbc.detector as pycbc_det
import pycbc.catalog as pycbc_cat
import pycbc.filter as pycbc_fil
import pycbc.psd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import shutil as sh
import os as os
from glob import glob
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# hyperparameters definition 

sampling_frequency = 4096
low_freq_cutoff = 30

delta_t = 1.0 / sampling_frequency
time_length = 4
length = int(time_length / delta_t) 

delta_f = 1.0 / time_length 
flength = int(length / 2 + 1)

V1 = pycbc_det.Detector('V1')
L1 = pycbc_det.Detector('L1')
H1 = pycbc_det.Detector('H1')

Virgo_PSD = pycbc_psd.analytical.AdvVirgo(flength, delta_f, low_freq_cutoff=15) 
Ligo_PSD = pycbc.psd.analytical.aLIGOZeroDetHighPower(flength, delta_f, low_freq_cutoff=15)

def get_params(n):
    '''
    This function returns intrinsic and extrinsic parameters of n GW events
    -----------------------------------------------------------------------
    Arguments:
    n -- number of events
    '''
    ### Masses ###
    mass1 = np.random.uniform(5, 90, n)
    mass2 = np.random.uniform(5, 90, n)
    swap_idx = mass1 < mass2
    mass1[swap_idx], mass2[swap_idx] = mass2[swap_idx], mass1[swap_idx]

    ### Luminosity distance ####    
    distance = np.random.uniform(200, 5000, n)

    ### Inclination angles ###
    inclination = np.random.uniform(0, np.pi/2, n)

    ### Spin components along z ### 
    spin1 = np.random.uniform(-1, 1, n)
    spin2 = np.random.uniform(-1, 1, n)

    ### Sky location drawn randomly ###
    ra = np.random.uniform(0, 2 * np.pi, n)
    dec = np.random.uniform(0, np.pi, n)

    return mass1, mass2, distance, inclination, spin1, spin2, ra, dec

def chirp(m1,m2):
    """
    This function returns the chirp mass of a binary
    ---------------------------------------------------
    Arguments:
    m1 and m2 -- individual masses
    
    Return:
    M -- chirp mass
    """
    M = ((m1*m2)**(3/5))/((m1+m2)**(1/5))
    return M
    
def template(m1, m2, z1, z2, incl, dist):
    """
    Builds a template por a GW merger of both masses using SEOBNRv4_opt approximation method with a
    sample of 4096 data per second
    ---------------------------------------------------------------------------------------------------
    Arguments:
    m1, m2 -- masses of the two components
    z1, z2 -- spins of the two components
    incl -- angle between angular momentum L and line of sight (from 0 to pi)
    dist -- distance in Mpc to the emitting source
    cut_off -- low frecuency limit
    
    Return:
    hp, hc -- pycbc.timeseries.TimeSeries, for the plus/cross polarization GW (strain)
    """
    hp, hc = pycbc_wf.get_td_waveform(approximant = "SEOBNRv4_opt", mass1 = m1, mass2 = m2, 
                                      spin1z = z1, spin2z = z2, inclination = incl, distance = dist, 
                                      delta_t = delta_t, f_lower = low_freq_cutoff)

    return hp, hc

def L1_gw(hp, hc, ra, dec):
    """
    Project the GW emitted by a source onto the detector to obtain the wave measured by a particular
    detector depending on coordinates of the detector in the Earth and coordinates of the source in the sky
    ------------------------------------------------------------------------------------------------------
    Arguments:
    hp, hc -- pycbc.timeseries-type for the plus/cross polarization GW generated at the source
    ra, dec -- sky location of the source

    Return:
    gw -- pycbc.timeseries.TimeSeries, GW as measured by the detector
    """
    gw = L1.project_wave(hp, hc, ra, dec, 1, method = 'lal')
    
    return gw

def H1_gw(hp, hc, ra, dec):
    gw = H1.project_wave(hp, hc, ra, dec, 1, method = 'lal')
    
    return gw

def V1_gw(hp, hc, ra, dec):
    gw = V1.project_wave(hp, hc, ra, dec, 1, method = 'lal')
    
    return gw

def noise_gen(PSD, seed=None):
    """
    From a previous simulated PSD it generates gaussian noise
    ---------------------------------------------------------
    Arguments:
    PSD -- Power Spectral Density of the desired noise

    Return:
    noise -- noise generated from PSD
    """
    noise = pycbc_noise.gaussian.noise_from_psd(psd=PSD, seed=seed, length=length, delta_t=delta_t)

    return noise

def injection_by_hand(strain, signal, t_inj=None):
    """
    This function injects a signal inside a strain of noise
    ---------------------------------------------------------------------------
    Arguments:
    strain -- noise data
    signal -- simulated GW signal
    t_inj -- if not None is the time when to inject the signal in the 4s strain 
    
    Returns:
    data -- noise + signal strain   
    t -- time of the injection of the signal 
    """
    if len(strain) < len(signal):
        raise ValueError('Strain data length should be bigger than signal data')
    if strain.delta_t != signal.delta_t:
        raise ValueError('Strain and signal must contain same delta_t')
    
    len_signal = len(signal)
    len_strain = len(strain)
    dif = len_strain - len_signal
    
    # if t_inj is not None, we insert the signal in the specified position.
    if t_inj:
        loc = int(round(t_inj * sampling_frequency))
        t = t_inj
        
    # if t_inj is None, we will introduce the signal in a random place inside de strain.
    # this place should be such that the signal length can fit inside.
    # since when whitening data we need to crop part of the beginning and end, we make 
    # sure we dont inject just at the beginning or end of the strain. 
    else:
        loc = np.random.randint(0.5 * sampling_frequency, dif - 0.5 * sampling_frequency)
        t = loc / sampling_frequency

    h = np.copy(strain)
    h[loc:loc+len_signal] = strain[loc:loc+len_signal] + np.array(signal)
    
    data = pycbcty.timeseries.TimeSeries(h, strain.delta_t)
    
    return data, t

def get_SNR(hp, strain, PSD=None):
    """
    Computes the Signal-to-Noise Ratio for a given data expressed as sum of a noise and model
    -----------------------------------------------------------------------------------------
    Arguments:
    hp -- GW template
    strain -- gw + detector noise
    PSD -- known detector PSD

    Return:
    SNR -- Signal to Noise ratio
    """
    
    # used on real data where we can compute the PSD directly from the strain
    if PSD==None:
        strain = pycbc.filter.highpass(strain, 15.0)
        conditioned = strain_V1.crop(2,2)
        psd = conditioned.psd(4)
        psd = pycbc.psd.interpolate(psd, conditioned.delta_f)
        psd = pycbc.psd.inverse_spectrum_truncation(psd, int(4 * conditioned.sample_rate),
                                          low_frequency_cutoff=15)
        
        template = hp.copy()
        template.resize(len(conditioned))
        template = template.cyclic_time_shift(template.start_time)
        
        snr = pycbc.filter.matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=15)
        snr = snr.crop(4 + 4, 4)
    
    else:
        template = hp.copy()
        template.resize(len(strain))
        template = template.cyclic_time_shift(template.start_time)

        snr = pycbc.filter.matched_filter(template, strain, psd=PSD, low_frequency_cutoff=15)

    peak = abs(snr).numpy().argmax()
    snr_value = abs(snr[peak])

    return snr_value

def process_data(data):
    '''
    This function will process the data, whitening, bandpassing and normalising it.
    '''

    length = len(data)
    delta_t = data.delta_t
    segment_duration = data.get_duration()
    sample_rate = data.get_sample_rate()
    max_filter_duration = segment_duration / 4
    
    frec_low_cutoff = 30
    frec_high_cutoff = 350

    # whiten the data
    data = data.whiten(segment_duration, max_filter_duration, remove_corrupted=True, 
                       low_frequency_cutoff=frec_low_cutoff, return_psd=False)

    # bandpass
    data = data.lowpass_fir(frec_high_cutoff, 8, beta=5.0, remove_corrupted=True) # bandpassing: supress data for frec>350
    data = data.highpass_fir(frec_low_cutoff, 8, beta=5.0, remove_corrupted=True) # bandpassing: supress data for frec<30

    # normalize the data
    data_array = np.array(data).reshape(-1, 1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data_array).flatten()
    data = pycbcty.timeseries.TimeSeries(data, delta_t)

    # append zeros to beginning and end of the data to keep the input shape unchanged after cropping corrupted segments
    length_cr = len(data)
    dif = length-length_cr
    data.prepend_zeros(int(dif / 2)) # append zeros at beginning
    data.append_zeros(int(dif / 2)) # append zeros at end

    # return an array
    data = np.squeeze(np.array(data)) 
    
    return data

def plot_distribution(ax, values, label, x_range, ideal_pdf=None, x_ticks=None, x_labels=None):
    '''
    This function plots the parameters distributions
    ------------------------------------------------
    Arguments:
    ax -- axes element 
    values -- array containing the parameter values
    label -- x axis label 
    x_range -- parameter range 
    ideal_pdf -- if not None, a function that defines the 'ideal' pdf the parameter should follow
    x_ticks -- x axis ticks
    x_labels -- x axis ticks labels
    '''
    
    # create x values for true pdf
    x_min, x_max = x_range
    x_plot = np.linspace(x_min, x_max, 500)
    
    # plot histogram 
    ax.hist(values, bins=20, label='Sampled', alpha=0.2, 
            density=True, color='green', edgecolor='gray', linewidth=1.2)
    
    if ideal_pdf is None:
        pdf = 1 / (x_max - x_min)
        pdf_values = np.full_like(x_plot, pdf)
    else:
        pdf_values = ideal_pdf(x_plot)
        
    ax.plot(x_plot, pdf_values, label='Ideal', color='r', linestyle='-', linewidth=2)
    
    ax.set_xlabel(label, fontsize=18)
    ax.legend(prop={'size':18})
    
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        
    if x_labels is not None:
        ax.set_xticklabels(x_labels)
    
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    
    return Nones