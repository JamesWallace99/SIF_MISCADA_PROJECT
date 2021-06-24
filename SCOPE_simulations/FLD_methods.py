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
    
    index_start = find_nearest(wl, wl_start)
    index_end = find_nearest(wl, wl_end) + 1
    
    if fun == 'max':
        value_index = np.argmax(spectra[index_start:index_end]) + index_start
    if fun == 'min':
        value_index = np.argmin(spectra[index_start:index_end]) + index_start
    
    value = spectra[value_index]
    
    
    return(value_index, value)
    
    
def sFLD(e_spectra, l_spectra, wavelengths, buffer_out = 6, plot=True):
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
    
    wl_in = 760
    
    e_in_index, e_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, e_spectra, 'min')
    l_in_index, l_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, l_spectra, 'min')
    e_out_index, e_out = stats_on_spectra(wavelengths, wl_in - buffer_out, wl_in + buffer_out, e_spectra, 'max')
    l_out_index, l_out = stats_on_spectra(wavelengths, wl_in - buffer_out, wl_in + buffer_out, l_spectra, 'max')
    
    if plot == True:
        
        plt.plot(wavelengths, e_spectra, color = 'orange')
        plt.plot(wavelengths, l_spectra, color = 'blue')
        plt.scatter(wavelengths[e_in_index], e_in, label = 'e_in')
        plt.scatter(wavelengths[l_in_index], l_in, label = 'l_in')
        plt.scatter(wavelengths[e_out_index], e_out, label = 'e_out')
        plt.scatter(wavelengths[l_out_index], l_out, label = 'l_out')
        #plt.legend()
        plt.xlim(750, 775)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Radiance (mW m−2 sr−1 nm−1)')
        plt.show()
    
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in)
    
    
    return(fluorescence)

def three_FLD(e_spectra, l_spectra, wavelengths, buffer_out = 12, plot=True):
    """Applies the 3FLD method for SIF extraction

    Parameters
    ----------
    e_spectra : np.array
        array containing the directional downwellling irradiance
    l_spectra : np.array
        array containing the directional downwellling irradiance
    wavelengths : np.array
        array containing the wavelengths at which the E and L spectra are sampled over
    plot : bool, optional
        generate plot of the O2A absorption band showing selected points, by default True
    """
    
    buffer_in = 1
    #buffer_out = 8
    
    wl_in = 760
    
    # get absorption well position
    e_in_index, e_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, e_spectra, 'min')
    l_in_index, l_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, l_spectra, 'min')
    
    # get absorption shoulders
    e_left_index, e_left = stats_on_spectra(wavelengths, wl_in - buffer_out, wl_in, e_spectra, 'max')
    l_left_index, l_left = stats_on_spectra(wavelengths, wl_in - buffer_out, wl_in, l_spectra, 'max')
    
    e_right_index, e_right = stats_on_spectra(wavelengths, wl_in, wl_in + buffer_out, e_spectra, 'max')
    l_right_index, l_right = stats_on_spectra(wavelengths, wl_in, wl_in + buffer_out, l_spectra, 'max')
    
    # interpolate between shoulders
    
    e_wavelengths_inter = wavelengths[e_left_index:e_right_index + 1]
    l_wavelengths_inter = wavelengths[l_left_index:l_right_index + 1]
    
    # get equation of straight line between two shoulders
    
    e_xp = [e_wavelengths_inter[0], e_wavelengths_inter[-1]] # get x values
    l_xp = [l_wavelengths_inter[0], l_wavelengths_inter[-1]]
    
    e_fp = [e_left, e_right] # get y values
    l_fp = [l_left, l_right]
    
    e_coefficients = np.polyfit(e_xp, e_fp, 1) # polyfit for equation
    l_coefficients = np.polyfit(l_xp, l_fp, 1)
    
    # apply to wavelengths in between shoulders
    
    e_interpolated = e_wavelengths_inter*e_coefficients[0] + e_coefficients[1]
    l_interpolated = l_wavelengths_inter*l_coefficients[0] + l_coefficients[1]
    
    # find interpolated value inside of absorption band
    
    e_out = e_interpolated[e_in_index - e_left_index]
    l_out = l_interpolated[l_in_index - l_left_index]
    
    
    if plot == True:
        
        # plot spectra
        plt.plot(wavelengths, e_spectra, color = 'orange')
        plt.plot(wavelengths, l_spectra, color = 'blue')
        
        # plot selected points
        plt.scatter(wavelengths[e_in_index], e_in, label = 'e_in')
        plt.scatter(wavelengths[l_in_index], l_in, label = 'l_in')
        plt.scatter(wavelengths[e_left_index], e_left, label = 'e_left')
        plt.scatter(wavelengths[l_left_index], l_left, label = 'l_left')
        plt.scatter(wavelengths[e_right_index], e_right, label = 'e_right')
        plt.scatter(wavelengths[l_right_index], l_right, label = 'l_right')
        
        # plot interpolation
        plt.plot(e_wavelengths_inter, e_interpolated)
        plt.plot(l_wavelengths_inter, l_interpolated)
        plt.scatter(wavelengths[e_in_index], e_out, label = 'e_out')
        plt.scatter(wavelengths[l_in_index], l_out, label = 'l_out')
        
        #plt.legend()
        plt.xlim(750, 775)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Radiance (mW m−2 sr−1 nm−1)')
        plt.show()
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in)
    

    
    return(fluorescence)

"""
# test sequence

# get csv pathnames
e_pathname = "/Users/jameswallace/Desktop/Project/data/verification_run_2021-06-14-1239/Esun.csv"
l_pathname = "/Users/jameswallace/Desktop/Project/data/verification_run_2021-06-14-1239/Lo_spectrum_inclF.csv"

# place them into dataframes
e_df = get_simulated_spectral_df(e_pathname)
l_df = get_simulated_spectral_df(l_pathname)

# test FLD methods
sFLD(np.asarray(e_df.iloc[0]) /np.pi, np.asarray(l_df.iloc[0]), np.arange(400, 2562), plot = True)
three_FLD(np.asarray(e_df.iloc[0]) /np.pi, np.asarray(l_df.iloc[0]), np.arange(400, 2562), plot = True)

# resample the dataframes at 3.5 nm FWHM
e_resampled, re_wave = resample_spectra(fwhm = 3.5, spectra = e_df.iloc[0])
l_resampled = resample_spectra(fwhm = 3.5, spectra = l_df.iloc[0])[0]

# test FLD methods on resampled spectras
print(sFLD(e_resampled / np.pi, l_resampled, re_wave))
print(three_FLD(e_resampled / np.pi, l_resampled, re_wave))
"""