import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral.io.aviris as aviris
from spectral import *

def get_simulated_spectral_df(file_pathname):
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


def plot_o2a_band(e_spectra, l_spectra, wavelengths = np.arange(400, 2562)):
    """ Plots the O2A band of the average E and L spectra

    Parameters
    ----------
    e_spectra_df : pandas df
        pandas dataframe containing E spectra, columns named from wavelengths matching wavelengths array
    l_spectra_df : pandas df
        pandas dataframe containing E spectra, columns named from wavelengths matching wavelengths array
    wavelengths : np.array, optional
        np array containing wavelengths that spectra are generated over, default np.arange(400, 2562) 1 nm FWHM
    """
    
    e_spectra = e_spectra / np.pi # convert to same units
    
    # plot the average spectral values as a function of the wavelengths
    plt.plot(wavelengths, e_spectra, label = 'E / pi')
    plt.plot(wavelengths, l_spectra, label = 'L')
    plt.xlim(750, 780) # plot the O2A band
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(' Radiance (W m-2 um-1 sr-1)')
    plt.title('O2A Absorption Band')
    plt.legend()
    plt.show()
    return()

def resample_spectra(fwhm, spectra, wavelengths = np.arange(400, 2562)):
    """Applies a Gaussian convolution to a spectra to resample to a desired FWHM

    Parameters
    ----------
    fwhm : float
        target FWHM for new spectra
    wavelengths : np array, default = np.arange(400, 2562) 1 nm FWHM
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

def sFLD(e_spectra, l_spectra, wavelengths, plot, O2A_band = True):
    """Applies the sFLD algorithm for SIF retrieval at either the O2A or O2B absorption band

    Parameters
    ----------
    e_spectra : np.array
        np array containing the downwelling irradiance values at the wavelengths provided
    l_spectra : np.array
        np array containing the upwelling radiance values at the wavelengths provided
    wavelengths : np.array
        np array containing the wavelengths at which the E and L spectra are sampled at
    plot: bool
        Determines whether plot of spectra and points choosen for sFLD will be shown
    O2A_band : bool, optional
        Determines if the target retrieval band is the O2A absorption band, by default True
        
    Outputs
    --------
    fluorescence: float
        Fluorescence at the O2A absorption band retrieved using the sFLD method
    """
    
    e_spectra = e_spectra / np.pi
    
    
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
        e_spectra_range = e_spectra[e_o2a_left_index:e_o2a_right_index]
        l_spectra_range = l_spectra[l_o2a_left_index:l_o2a_right_index]
    if O2A_band == False:
        e_spectra_range = e_spectra[e_o2b_left_index:e_o2b_right_index]
        l_spectra_range = l_spectra[l_o2b_left_index:l_o2b_right_index]
    
    # look for minima in spectral region
    
    e_argmin = np.argmin(e_spectra_range)
    l_argmin = np.argmin(l_spectra_range)
    
    
    # get this value for both
    
    e_in = e_spectra_range[e_argmin]
    l_in = l_spectra_range[l_argmin]
    
    
    # look to left of this region for shoulder maxima
    
    e_left_shoulder = e_spectra_range[:e_argmin]
    l_left_shoulder = l_spectra_range[:l_argmin]
    
    # look for maxima at left shoulder
    
    e_argmax = np.argmax(e_left_shoulder)
    l_argmax = np.argmax(l_left_shoulder)
    
    # get max from left shoulder
    e_out = e_left_shoulder[e_argmax]
    l_out = l_left_shoulder[l_argmax]
    
    # combine in equation
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in)
    
    
    if plot == True:
        # plot points selected
        
        #plt.scatter(e_argmin + 750, e_in)
        plt.plot(wavelengths[e_o2a_left_index:e_o2a_right_index] ,e_spectra[e_o2a_left_index:e_o2a_right_index], color = 'orange')
        plt.plot(wavelengths[l_o2a_left_index:l_o2a_right_index] ,l_spectra[l_o2a_left_index:l_o2a_right_index], color = 'b')
        plt.scatter(wavelengths[e_argmin + e_o2a_left_index], e_in, color = 'orange', label = 'E_in')
        plt.scatter(wavelengths[e_argmax + e_o2a_left_index], e_out, color = 'red', label = 'E_out')
        plt.scatter(wavelengths[l_argmin + l_o2a_left_index], l_in, color = 'b', label = 'L_in')
        plt.scatter(wavelengths[l_argmax + l_o2a_left_index], l_out, color = 'green', label = 'L_out')
        plt.title('sFLD SIF Retrieval Method')
        plt.xlabel('Wavelengths (nm')
        plt.ylabel('Radiance (W m-2 um-1 sr-1)')
        plt.legend()
        plt.show()
    
    return(fluorescence)

def get_fluorescence(e_pathname, l_pathname, plot, fwhm = 3.5):
    """ Takes the output downwelling irradiance and upwelling irradiance functions from SCOPE
    and applies the sFLD SIF retrieval algorithm.
    From SCOPE model the E spectra is given in the output file Eout_spectrum.csv
    From SCOPE model the L spectra is given in the output file Lo_spectrum_inclF.csv

    Parameters
    ----------
    e_pathname : string (file pathname)
        pathname of file containing the E spectral data (directional, Lout)
    l_pathname : string (file pathname)
        pathname of file containing the L spectral data (directional, Lout + F)
    fwhm : float, optional
        target FWHM of resampling, by default 3.5
    plot : bool
        plot the O2A band of the average spectral values
    """
    
    # convert csv to dataframes
    e_spectra = get_simulated_spectral_df(e_pathname)
    l_spectra = get_simulated_spectral_df(l_pathname)
    
    # get average values
    e_average = average_simulations(e_spectra)
    l_average = average_simulations(l_spectra)
    
    # plot average values at O2A absorption band
    
    if plot == True:
        plot_o2a_band(e_average, l_average, np.arange(400, 2562))
    
    # resample the wavelengths at desired FWHM
    e_resampled, re_wave = resample_spectra(fwhm, e_average)
    l_resampled = resample_spectra(fwhm, l_average)[0]
    
    # apply sFLD method
    
    fluorescence = sFLD(e_resampled, l_resampled, re_wave, plot)
    
    return(fluorescence)
    
    
    


def alt_sFLD(e_spectra, l_spectra, wavelengths, plot, O2A_band = True):
    """Applies the sFLD algorithm for SIF retrieval at either the O2A or O2B absorption band
        Forces E_out to have the same wavelength values as L_out

    Parameters
    ----------
    e_spectra : np.array
        np array containing the downwelling irradiance values at the wavelengths provided
    l_spectra : np.array
        np array containing the upwelling radiance values at the wavelengths provided
    wavelengths : np.array
        np array containing the wavelengths at which the E and L spectra are sampled at
    plot: bool
        Determines whether plot of spectra and points choosen for sFLD will be shown
    O2A_band : bool, optional
        Determines if the target retrieval band is the O2A absorption band, by default True
        
    Outputs
    --------
    fluorescence: float
        Fluorescence at the O2A absorption band retrieved using the sFLD method
    """
    
    e_spectra = e_spectra / np.pi
    
    
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
        e_spectra_range = e_spectra[e_o2a_left_index:e_o2a_right_index]
        l_spectra_range = l_spectra[l_o2a_left_index:l_o2a_right_index]
    if O2A_band == False:
        e_spectra_range = e_spectra[e_o2b_left_index:e_o2b_right_index]
        l_spectra_range = l_spectra[l_o2b_left_index:l_o2b_right_index]
    
    # look for minima in spectral region
    
    e_argmin = np.argmin(e_spectra_range)
    l_argmin = np.argmin(l_spectra_range)
    
    
    # get this value for both
    
    e_in = e_spectra_range[e_argmin]
    l_in = l_spectra_range[l_argmin]
    
    
    # look to left of this region for shoulder maxima
    
    e_left_shoulder = e_spectra_range[:e_argmin]
    l_left_shoulder = l_spectra_range[:l_argmin]
    
    # look for maxima at left shoulder
    
    e_argmax = np.argmax(e_left_shoulder)
    l_argmax = np.argmax(l_left_shoulder)
    
    e_argmax = l_argmax
    
    # get max from left shoulder
    e_out = e_left_shoulder[e_argmax]
    l_out = l_left_shoulder[l_argmax]
    
    # combine in equation
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in)
    
    
    if plot == True:
        # plot points selected
        
        #plt.scatter(e_argmin + 750, e_in)
        plt.plot(wavelengths[e_o2a_left_index:e_o2a_right_index] ,e_spectra[e_o2a_left_index:e_o2a_right_index], color = 'orange')
        plt.plot(wavelengths[l_o2a_left_index:l_o2a_right_index] ,l_spectra[l_o2a_left_index:l_o2a_right_index], color = 'b')
        plt.scatter(wavelengths[e_argmin + e_o2a_left_index], e_in, color = 'orange', label = 'E_in')
        plt.scatter(wavelengths[e_argmax + e_o2a_left_index], e_out, color = 'red', label = 'E_out')
        plt.scatter(wavelengths[l_argmin + l_o2a_left_index], l_in, color = 'b', label = 'L_in')
        plt.scatter(wavelengths[l_argmax + l_o2a_left_index], l_out, color = 'green', label = 'L_out')
        plt.title('sFLD SIF Retrieval Method')
        plt.xlabel('Wavelengths (nm')
        plt.ylabel('Radiance (W m-2 um-1 sr-1)')
        plt.legend()
        plt.show()
    
    return(fluorescence)

def stats_on_spectra(wl, wl_start, wl_end, spectra, fun):
    '''
    Finds the value and index of the max or min within a given spectral range
    
    
    input
    ------
    wl: np arrays
    array of wavelengths
    
    wl_start: int
    start value of range for wavelengths
    
    wl_end: int
    end of value range for wavelengths
    
    spectra: np array
    spectra to conduct stats on
    
    fun: str 'min' or 'max'
    Locate the min or max of the region
    
    output
    -------
    value_index: int
    index of the wavelength and spectra matching the target value
    
    value: float
    spectra value at absorption feature
    
    '''
    # get index of wavelength array value at the start of spectral range
    # get index at wavelength array value at end of the spectral range
    # apply the function to the input spectra sliced across the range
    # return the value and the index
    
    index_start = find_nearest(wl, wl_start) - 1
    index_end = find_nearest(wl, wl_end) + 1
    
    
    if fun == 'max':
        value_index = np.argmax(spectra[index_start:index_end]) + index_start
    if fun == 'min':
        value_index = np.argmin(spectra[index_start:index_end]) + index_start
    
    value = spectra[value_index]
    
    
    return(value_index, value)
    
    
def new_sFLD(e_spectra, l_spectra, wavelengths, plot=True):
    """ final version of the sFLD algorithm

    Parameters
    ----------
    e_spectra : np array
        spectral array containing the incident solar radiance (directional)
    l_spectra : np array
        spectral array of the upwelling solar radiance
    wavelengths : np array
        array of the wavelength values
    plot : bool, optional
        produce plot of values, by default True
    """
    
    buffer_in = 5
    buffer_out = 7
    
    wl_in = 760
    
    e_in_index, e_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, e_spectra, 'min')
    l_in_index, l_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, l_spectra, 'min')
    e_out_index, e_out = stats_on_spectra(wavelengths, wl_in - buffer_out, wl_in + buffer_out, e_spectra, 'max')
    l_out_index, l_out = stats_on_spectra(wavelengths, wl_in - buffer_out, wl_in + buffer_out, l_spectra, 'max')
    
    if plot == True:
        plt.plot(wavelengths, e_spectra)
        plt.plot(wavelengths, l_spectra)
        
        plt.scatter(wavelengths[e_in_index], e_in, label = 'e_in')
        plt.scatter(wavelengths[l_in_index], l_in, label = 'l_in')
        plt.scatter(wavelengths[e_out_index], e_out, label = 'e_out')
        plt.scatter(wavelengths[l_out_index], l_out, label = 'l_out')
        plt.legend()
        plt.xlim(750, 775)
    
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in)
    
    return(fluorescence)   
    


# test


# get spectra files
e_pathname = "/Users/jameswallace/Desktop/Project/data/verification_run_2021-06-14-1239/Esun.csv"
l_pathname = "/Users/jameswallace/Desktop/Project/data/verification_run_2021-06-14-1239/Lo_spectrum_inclF.csv"
print(get_fluorescence(e_pathname, l_pathname, plot=False))


'''
e_spectra = get_simulated_spectral_df(e_pathname)
l_spectra = get_simulated_spectral_df(l_pathname)

#print(e_spectra.head())
#print(l_spectra.head())

# get averages
e_average = average_simulations(e_spectra)
l_average = average_simulations(l_spectra)

#print(e_average)
#print(l_average)


# plot average values
plot_o2a_band(e_average, l_average)


# resample
e_resampled, re_wave = resample_spectra(3.5, e_average)
l_resampled = resample_spectra(3.5, l_average)[0]
#print(len(l_resampled))

plt.plot(re_wave, e_resampled / np.pi, label = 'E / pi')
plt.plot(re_wave, l_resampled, label = 'L')
plt.title('Resampled Spectral Data')
plt.xlim(750, 775)
plt.legend()
plt.show()

# sFLD test
sFLD(e_resampled, l_resampled, re_wave)
fluorescence = sFLD(e_resampled, l_resampled, re_wave)
print(fluorescence)

'''