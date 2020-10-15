#sine wave generation function

import numpy as np


def generate_sine_wave_no(N, A, f0, fs, phi):

    '''
    generate sine wave with a given sample number
    N(int): no. of samples 
    A(float): amplitude 
    f0(float): frequency in Hz
    fs(flaot): sampling frequency in Hz
    phi(float): phase
    '''

    T = 1/fs
    n = np.arange(N)
    s = A*np.sin(2*np.pi*f0*n*T+phi)
    
    return s

def gaussian(N,FWHM,fd):
    '''
    
    :param N: 
    :param f0: 
    
    :return: 
    '''

    temp = np.linspace(0,N,N)
    sigma = FWHM/2.354826

    g = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(temp - fd) ** 2 / (2 * sigma) ** 2)
    g = g/g.max()
    
    return g



def numpy_SNR(reference_signal, target_signal):
    
    signal = np.sum(reference_signal ** 2)
    noise = np.sum((target_signal - reference_signal) ** 2)
    snr = 10 * np.log10(signal / noise)
    return snr