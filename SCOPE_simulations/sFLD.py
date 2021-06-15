import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral.io.aviris as aviris
from spectral import *

def get_simualted_spectral_df(file_pathname):
    """ Takes output spectral file from SCOPE model and transforms it to a df
        with wavelength values as column headers

    Parameters
    ----------
    file_pathname : string (pathname)
        pathname to file containing the simulated spectral data from SCOPE
        
    Outputs
    ----------
    df : pandas dataframe
        dataframe of spectral values with wavelengths (nm) at columns
    """
    df = pd.read_csv(file_pathname, skiprows = 2) # skip the rows containing file data
    df.columns = np.arange(400, 2562) # name the columns after wavelengths
    return(df)


def average_simulations(df):
    """
    Calculates the average of the simulations contained within the SCOPE spectral dfs
    
    input
    -----
    df: pandas df
    pandas dataframe containing spectra, columns named from wavelengths 400 - 2562 (nm)
    
    output
    -----
    means: np.array
    array containing the average value of the simulations at each wavelength from 400 - 2562
    """
    means = [] # initiate empty array for average values
    for col_name in range(400, 2562): # loop over each wavelength to calculate average values
        means.append(df[col_name].mean())
    return(np.asarray(means))


def plot_o2a_band(e_spectra_df, l_spectra_df, wavelengths):
    """ Plots the O2A band of the average E and L spectra

    Parameters
    ----------
    e_spectra_df : pandas df
        pandas dataframe containing E spectra, columns named from wavelengths matching wavelengths array
    l_spectra_df : pandas df
        pandas dataframe containing E spectra, columns named from wavelengths matching wavelengths array
    wavelengths : np.array
        np array containing wavelengths that spectra are generated over
    """
    
    # plot the average spectral values as a function of the wavelengths
    plt.plot(wavelengths, average_simulations(e_spectra_df), label = 'E / pi')
    plt.plot(wavelengths, average_simulations(l_spectra_df), label = 'L')
    plt.xlim(750, 780) # plot the O2A band
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(' Radiance (W m-2 um-1 sr-1)')
    plt.legend()
    plt.show()
    return()

def resample_spectra(fwhm, spectra, wavelengths = np.arange(400, 2562)):
    """Applies a Gaussian convolution to a spectra to resample to a desired FWHM

    Parameters
    ----------
    fwhm : float
        target FWHM for new spectra
    wavelengths : np array, default = np.arange(400, 2562)
        array of original wavelengths
    spectra : np array
        spectra to be re-sampled
        
    Outputs
    --------
    resampled_spectra: np array
        orignal spectra resampled at new FWHM
    bands2: np array
        wavelengths of newly resampled spectra
    """
    bands2 = np.arange(wavelengths[0], wavelengths[-1], fwhm) # define new wavelegths at target FWHM
    resample = BandResampler(wavelengths, bands2) # intiaite the resample function
    resampled_spectra = resample(spectra) # resample the spectra
    return(resampled_spectra, bands2)

def find_nearest(array, value):
    '''
    Returns the index of the item in the array closest to a given value
    
    Parameters
    -----------
    array: array
        array containing values to search over
    value: float
        target value
    
    Outputs
    --------
    idx: integer
        index of the array item closet to the input value
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return(idx)

def sFLD(e_spectra, l_spectra, wavelengths, e_directional = True, O2A_band = True):
    """Applies the sFLD algorithm for SIF retrieval at either the O2A or O2B absorption band

    Parameters
    ----------
    e_spectra : np.array
        np array containing the downwelling irradiance values at the wavelengths provided
    l_spectra : np.array
        np array containing the upwelling radiance values at the wavelengths provided
    wavelengths : np.array
        np array containing the wavelengths at which the E and L spectra are sampled at
    e_directional : bool, optional
        Determines whether the E value is directional (i.e. if directional units are W m-2 um-1 sr-1), by default True
    O2A_band : bool, optional
        Determines if the target retrieval band is the O2A absorption band, by default True
    """
    
    if e_directional == False:
        e_directional = e_directional / np.pi
    
    # O2A 760
    # O2B 687
    
    # look for index of value nearest 750, 775, 670 and 700
    e_o2a_left_index = find_nearest(wavelengths, 750)
    e_o2a_right_index = find_nearest(wavelengths, 775)
    l_o2a_left_index = find_nearest(wavelengths, 750)
    l_o2a_right_index = find_nearest(wavelengths, 775)
    
    e_o2b_left_index = find_nearest(wavelengths, 670)
    e_o2b_right_index = find_nearest(wavelengths, 700)
    l_o2b_left_index = find_nearest(wavelengths, 670)
    l_o2b_right_index = find_nearest(wavelengths, 700)
    
    
    
    
    # define spectral regions (index begins at 400 nm)
    if O2A_band == True:
        e_spectra = e_spectra[e_o2a_left_index:e_o2a_right_index]
        l_spectra = l_spectra[l_o2a_left_index:l_o2a_right_index]
    if O2A_band == False:
        e_spectra = e_spectra[e_o2b_left_index:e_o2b_right_index]
        l_spectra = l_spectra[l_o2b_left_index:l_o2b_right_index]
    
    # look for minima in spectral region
    
    e_argmin = np.argmin(e_spectra)
    l_argmin = np.argmin(l_spectra)
    
    
    # get this value for both
    
    e_in = e_spectra[e_argmin]
    l_in = l_spectra[l_argmin]
    
    
    # look to left of this region for shoulder maxima
    
    e_left_shoulder = e_spectra[:e_argmin]
    l_left_shoulder = l_spectra[:l_argmin]
    

    # get max from left shoulder
    e_out = e_left_shoulder[np.argmax(e_left_shoulder)]
    l_out = l_left_shoulder[np.argmax(l_left_shoulder)]
    
    # combine in equation
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in)
    
    return(fluorescence)

def get_fluorescence(e_pathname, l_pathname, fwhm = 3.5, plot = True):
    """ Takes the output downwelling irradiance and upwelling irradiance functions from SCOPE
    and applies the sFLD SIF retrieval algorithm.

    Parameters
    ----------
    e_pathname : string (file pathname)
        pathname of file containing the E spectral data (directional, Lout)
    l_pathname : string (file pathname)
        pathname of file containing the L spectral data (directional, Lout + F)
    fwhm : float, optional
        target FWHM of resampling, by default 3.5
    plot : bool, optional
        plot the O2A band of the average spectral values, by default True
    """
    
    # convert csv to dataframes
    e_spectra = get_simualted_spectral_df(e_pathname)
    l_spectra = get_simualted_spectral_df(l_pathname)
    
    # get average values
    e_average = average_simulations(e_spectra)
    l_average = average_simulations(l_spectra)
    
    # plot average values at O2A absorption band
    
    if plot == True:
        plot_o2a_band(e_spectra, l_spectra, np.arange(400, 2562))
    
    # resample the wavelengths at desired FWHM
    e_resampled, re_wave = resample_spectra(fwhm, e_average)
    l_resampled = resample_spectra(fwhm, l_average)[0]
    
    # apply sFLD method
    
    fluorescence = sFLD(e_resampled, l_resampled, re_wave)
    
    return(fluorescence)
    
    
    




# test
"""
# get spectra files
e_pathname = "/Users/jameswallace/Desktop/Project/data/verification_run_2021-06-14-1239/Lo_spectrum.csv"
l_pathname = "/Users/jameswallace/Desktop/Project/data/verification_run_2021-06-14-1239/Lo_spectrum_inclF.csv"
e_spectra = get_simualted_spectral_df(e_pathname)
l_spectra = get_simualted_spectral_df(l_pathname)
print(e_spectra.head())
print(l_spectra.head())

# get averages
e_average = average_simulations(e_spectra)
l_average = average_simulations(l_spectra)

print(e_average)
print(l_average)

# plot average values
plot_o2a_band(e_spectra, l_spectra, np.arange(400, 2562))

# resample
e_resampled, re_wave = resample_spectra(3.5, e_average)
l_resampled = resample_spectra(3.5, l_average)[0]
print(l_resampled)

# sFLD test

fluorescence = sFLD(e_resampled, l_resampled, re_wave)
print(fluorescence)


print(get_fluorescence(e_pathname, l_pathname))

"""