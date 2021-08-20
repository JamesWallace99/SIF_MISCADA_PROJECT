'''
Contains functions to perform SIF extraction using the Fraunhofer Line Depth Methods sFLD, 3FLD and iFLD
Author: James Wallace
'''

# import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spectral.io.aviris as aviris
from spectral import *
from scipy import interpolate

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
    
    Outputs
    ---------
    None
    Shows the plot of the two spectra at the O2A absorption band region
    """
    
    e_spectra = e_spectra / np.pi # convert to same units
    
    # plot the average spectral values as a function of the wavelengths
    plt.plot(wavelengths, e_spectra, label = 'E / pi')
    plt.plot(wavelengths, l_spectra, label = 'L')
    plt.xlim(750, 780) # plot the O2A band
    # label the axis and include a title
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(' Radiance (W m-2 um-1 sr-1)')
    plt.title('O2A Absorption Band')
    plt.legend()
    plt.show() # show the plot
    return()

def resample_spectra(fwhm, spectra, wavelengths = np.arange(400, 2562)):
    """Applies a Gaussian convolution to a spectra to resample to a desired FWHM.

    Parameters
    ----------
    fwhm : float
        target FWHM for new spectra
    wavelengths : np array, default = np.arange(400, 2562) (i.e. 1 nm FWHM)
        np.array containing the original wavelengths the spectra was sampled at
    spectra : np.array
        spectra to be re-sampled
        
    Outputs
    --------
    resampled_spectra: np array
        original spectra resampled at new FWHM
    bands2: np array
        wavelengths of newly resampled spectra
    """
    bands2 = np.arange(wavelengths[0], wavelengths[-1], fwhm) # define new wavelegths at target FWHM
    resample = BandResampler(wavelengths, bands2) # intiaite the resample function
    resampled_spectra = resample(spectra) # resample the spectra
    return(resampled_spectra, bands2)


def add_noise(snr, spectra):
    """Add random Gaussian distributed noise with a mean value of zero 
    and a standard deviation equal to the ratio between signal level and SNR.

    Parameters
    ----------
    snr : int
        Signal to Noise Ratio (SNR), peak value at full resolution of the sensor
    spectra : np.array
        np.array containing the spectra to be resampled with the additional noise
    """
    spectra_noise = np.copy(spectra)
    for i in range(len(spectra)):
        noise = np.random.normal(0, (spectra[i]) / snr)
        spectra_noise[i] += noise
    return(spectra_noise)




def find_nearest(array, value):
    '''
    Returns the index of the item in the array closest to a given value
    
    Parameters
    -----------
    array: array
        array containing values to search over
    value: float
        value to search for within array
    
    Outputs
    --------
    idx: integer
        index of the array for the array value closet to the search value
    '''
    array = np.asarray(array) # treat the array as a np.array
    idx = (np.abs(array - value)).argmin() # get the index of the minimum value of array - search value
    return(idx)

def stats_on_spectra(wl, wl_start, wl_end, spectra, fun):
    '''
    Finds a defined statistic on a spectra for a given wavelength range.
    
    Parameters
    -----------
    wl: np.array
        array of wavelengths at which the spectra was sampled over
    wl_start: int
        wavelength value at the start of the desired range
    wl_end: int
        wavelength value at the end of the desired range
    spectra: np array
        spectra to extract statistic from, sampled at the given wavelength range
    fun: str ('min' or 'mean')
        Statistic to extract from the spectra in the given wavelength range
        Calculate the minimum or mean of the selected spectral region
    Outputs
    --------
    value index: int
        index value of the array at which the statistic is located (or nearest to for fun = 'mean')
    value: float
        value of the statistic extracted from the spectral array within the given wavelength range
    '''
    
    index_start = find_nearest(wl, wl_start) # finds the index of the wavelength array containing the start value
    index_end = find_nearest(wl, wl_end) + 1 # finds index of wavelengths for end array
    
    if fun == 'mean': # if mean statisitic is selected
        value = np.mean(spectra[index_start:index_end]) # find the average of the spectra in the spectral range
        value_index = find_nearest(wl, value) + index_start # find the index of the nearest value to the mean (for plotting)
        
    if fun == 'min': # if min statistic is selected
        value_index = np.argmin(spectra[index_start:index_end]) + index_start # get the index of minimum value in spectral range
        value = spectra[value_index] # get the spectra value at this index
    
    
    return(value_index, value)
    
    
def sFLD(e_spectra, l_spectra, wavelengths, fwhm, band = 'A', plot=True):
    """ Applies the sFLD method at either the O2A or O2B absorption bands to extract the SIF.

    Parameters
    ----------
    e_spectra : np.array
        spectral array containing the incident solar radiance
    l_spectra : np.array
        spectral array of the upwelling solar radiance
    wavelengths : np.array
        array of the wavelength values at which both the 'e_spectra' and 'l_spectra' were sampled
    fwhm: float
        full width half maximum at which the wavelengths were sampled
    band: str ('A' or 'B')
        Specifies which absorption band the retrieval algorithm should use, by default 'A' for O2A absorption band
    plot : bool, optional
        produce plot at absorption band showing points selected, by default True
    
    Output
    -------
    fluorescence: float
        SIF retrieved at the given absorption band
    """
    buffer_in = 5 #  range to search within absorption feature
    buffer_out = 1 # range to search outside of the absorption feature
    
    if band == 'A':
        out_in = 0.7535*fwhm+2.8937 # define amount to skip to left shoulder from minimum
        wl_in = 760 # standard location of O2A absorption feature defines start of search location
    if band == 'B':
        out_in = 0.697*fwhm + 1.245 # define amount to skip to left shoulder from minimum
        wl_in = 687 # standard location of the O2B aboorption band defines start of search location
    
    # find the points in given ranges using the stats_on_spectra function
    # find the minimum inside of the band for E_in and L_in starting at the standard locations for each band
    e_in_index, e_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, e_spectra, 'min')
    l_in_index, l_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, l_spectra, 'min')
    # locate the left shoulder of the band using the average of values within a given range
    e_out_index, e_out = stats_on_spectra(wavelengths, wl_in - buffer_out - out_in, wl_in - out_in, e_spectra, 'mean')
    l_out_index, l_out = stats_on_spectra(wavelengths, wl_in - buffer_out - out_in, wl_in - out_in, l_spectra, 'mean')
    
    if plot == True: # plot spectra and points at absorption feature
        plt.plot(wavelengths, e_spectra, color = 'orange')
        plt.plot(wavelengths, l_spectra, color = 'blue')
        plt.scatter(wavelengths[e_in_index], e_in, label = 'e_in')
        plt.scatter(wavelengths[l_in_index], l_in, label = 'l_in')
        plt.scatter(wavelengths[e_out_index], e_out, label = 'e_out')
        plt.scatter(wavelengths[l_out_index], l_out, label = 'l_out')
        #plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Radiance (mW m−2 sr−1 nm−1)')
        
        # zoom to absorption band
        
        if band == 'A':
            plt.xlim(750, 775)
            plt.title('O$_2$A Absorption Band')
        
        if band == 'B':
            plt.xlim(680, 700)
            plt.title('O$_2$B Absorption Band')
        
        plt.show() # show plot
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in) # calculate fluorescence using sFLD method
    
    return(fluorescence)

def three_FLD(e_spectra, l_spectra, wavelengths, fwhm, band = 'A', plot=True):
    """ Applies the 3FLD method at either the O2A or O2B absorption bands to extract the SIF.

    Parameters
    ----------
    e_spectra : np.array
        spectral array containing the incident solar radiance
    l_spectra : np.array
        spectral array of the upwelling solar radiance
    wavelengths : np.array
        array of the wavelength values at which both the 'e_spectra' and 'l_spectra' were sampled
    fwhm: float
        full width half maximum at which the wavelengths were sampled
    band: str ('A' or 'B')
        Specifies which absorption band the retrieval algorithm should use, by default 'A' for O2A absorption band
    plot : bool, optional
        produce plot at absorption band showing points selected, by default True
    
    Output
    -------
    fluorescence: float
        SIF retrieved at the given absorption band
    """
    # set the size of the point selection search regions inside and outside of the absorption band
    buffer_in = 5
    buffer_out = 1
    
    # adjust the shoulder skipping and right shoulder search range depending on which band is selected
    if band == 'A':
        out_in_first = 0.7535*fwhm+2.8937 # define amount to skip to left shoulder from minimum
        wl_in = 760 # standard location of O2A absorption feature
        out_in_second = 11 # define amount to skip to right shoulder from minimum
    if band == 'B':
        out_in_first = 0.697*fwhm + 1.245 # define amount to skip to left shoulder from minimum
        wl_in = 687 # standard location of the O2B aboorption band
        out_in_second = 8 # define amount to skip to right shoulder from minimum
    
    # get absorption well minima position
    e_in_index, e_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, e_spectra, 'min')
    l_in_index, l_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, l_spectra, 'min')
    # get absorption left and right shoulders
    e_left_index, e_left = stats_on_spectra(wavelengths, wl_in - buffer_out - out_in_first, wl_in - out_in_first, e_spectra, 'mean')
    l_left_index, l_left = stats_on_spectra(wavelengths, wl_in - buffer_out - out_in_first, wl_in - out_in_first, l_spectra, 'mean')
    e_right_index, e_right = stats_on_spectra(wavelengths, wl_in + out_in_second, wl_in + buffer_out + out_in_second, e_spectra, 'mean')
    l_right_index, l_right = stats_on_spectra(wavelengths, wl_in + out_in_second, wl_in + buffer_out + out_in_second, l_spectra, 'mean')
    # interpolate between shoulders using a linear fit
    e_wavelengths_inter = wavelengths[e_left_index:e_right_index + 1]
    l_wavelengths_inter = wavelengths[l_left_index:l_right_index + 1]
    # get equation of straight line between two shoulders
    e_xp = [e_wavelengths_inter[0], e_wavelengths_inter[-1]] # get x values
    l_xp = [l_wavelengths_inter[0], l_wavelengths_inter[-1]]
    e_fp = [e_left, e_right] # get y values
    l_fp = [l_left, l_right]
    e_coefficients = np.polyfit(e_xp, e_fp, 1) # polyfit with 1 DoF for linear fit
    l_coefficients = np.polyfit(l_xp, l_fp, 1)
    # apply fit to wavelengths between shoulders
    e_interpolated = e_wavelengths_inter*e_coefficients[0] + e_coefficients[1]
    l_interpolated = l_wavelengths_inter*l_coefficients[0] + l_coefficients[1]
    # find lineary interpolated value matching the index of absorption well minima
    e_out = e_interpolated[e_in_index - e_left_index]
    l_out = l_interpolated[l_in_index - l_left_index]
    
    if plot == True: # generate plots showing absorption band and selected points
        
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
        if band == 'A':
            plt.xlim(750, 775)
            plt.title('O$_2$A Absorption Band')
        
        if band == 'B':
            plt.xlim(680, 700)
            plt.title('O$_2$B Absorption Band')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Radiance (mW m−2 sr−1 nm−1)')
        plt.show()
    
    fluorescence = (e_out*l_in - l_out*e_in) / (e_out - e_in) # calculate fluorescence using interpolated values
    
    return(fluorescence)

def cubic_spline(x_vals, y_vals, target_x):
    """Fits and performs a cubic spline fit over a given set of x-values.

    Parameters
    ----------
    x_vals : np.array
        np.array of known x values
    y_vals : np.array
        np.array of known y values
    target_x : np.array
        np.array of x values to fit the spline fit over
    Output
    --------
    interpolated f(x) values: np.array
        np.array of the cubic spline fit fitted to the target x values
    """
    tck = interpolate.splrep(x_vals, y_vals)
    return(interpolate.splev(target_x, tck))

def fit_NaN(spectrum, wavelengths, plot):
    """Fits a cubic spline fit over a radiance spectra with NaN values at the O2 absorption bands.
        Returns a smoothed spectra with the NaN values replaced with the cubic spline interpolated values.

    Parameters
    ----------
    spectrum : np.array
        radiance spectra containing NaN values at the O2 absorption bands (686 - 697 nm for O2B and 759–770 nm for O2A)
    wavelengths : np.array
        wavelengths at which the spectrum was sampled over
    plot: bool
        generate plots showing smoothed spectra
    Output
    ---------
    smoothed: np.array
        np.array containing the spectrum with the NaN values replaced with a cubic spline interpolation of the surrounding values 
    """
    
    # get the indicies of the empty absorption band areas in the spectrum
    
    nan_indices = np.argwhere(np.isnan(spectrum)) # get the indices of the nan values
    # slice the nan_indices array to get the NaN indices of the O2B band
    o2b_indices = nan_indices[:find_nearest(wavelengths[nan_indices], 697) + 1] 
    
    # define the size of the area to construct the spline fit around the unknown points
    search_size = 10
    
    # get the wavelength values around the O2B band
    x_vals_left = wavelengths[int(o2b_indices[0]) - search_size: int(o2b_indices[0])]
    x_vals_right = wavelengths[int(o2b_indices[-1]) + 1: int(o2b_indices[-1]) + search_size + 1]
    x_vals = np.append(x_vals_left, x_vals_right)
    
    # get the spectrum values around the O2B band
    y_vals_left = spectrum[int(o2b_indices[0]) - search_size: int(o2b_indices[0])]
    y_vals_right = spectrum[int(o2b_indices[-1]) + 1: int(o2b_indices[-1]) + search_size + 1]
    y_vals = np.append(y_vals_left, y_vals_right)
    
    if plot == True:
        # plot points selected outside of band
        plt.scatter(x_vals, y_vals, label = 'Known Points')
        plt.plot(wavelengths, spectrum, color = 'orange', label = 'Spectra')
        plt.xlim(675, 707)

    # now interpolate the values within the band using the known values in the search areas around
    inter_ys_b = cubic_spline(x_vals, y_vals, wavelengths[o2b_indices])
    
    if plot == True:
        # plot the constructed spline fit within the absorption band area
        plt.plot(wavelengths[o2b_indices], inter_ys_b, color = 'green', label = 'Spline Fit')
        plt.scatter(wavelengths[o2b_indices], inter_ys_b, color = 'green', label = 'Constructed Points')

        plt.xlabel('Wavelengths (nm)')
        plt.ylabel('Spectral Value')
        plt.title('O2B Band: iFLD Interpolation')

        plt.legend()
        plt.show()
    
    # do the same for the O2A absorption band
    
    # get the NaN indices of the O2A band
    o2a_indices = nan_indices[find_nearest(wavelengths[nan_indices], 697) + 1:]

    # now get the values around the null values that we will use for the spline fit
    x_vals_left = wavelengths[int(o2a_indices[0]) - search_size: int(o2a_indices[0])]
    x_vals_right = wavelengths[int(o2a_indices[-1]) + 1: int(o2a_indices[-1]) + search_size + 1]
    x_vals = np.append(x_vals_left, x_vals_right)
    y_vals_left = spectrum[int(o2a_indices[0]) - search_size: int(o2a_indices[0])]
    y_vals_right = spectrum[int(o2a_indices[-1]) + 1: int(o2a_indices[-1]) + search_size + 1]
    y_vals = np.append(y_vals_left, y_vals_right)

    if plot == True:
        # plot points selected outside of band
        plt.scatter(x_vals, y_vals, label = 'Known points')
        plt.plot(wavelengths, spectrum, color = 'orange', label = 'Known Spectra')
        #plt.ylim(0.3, 0.5)
        plt.xlim(750, 775)

    # now interpolate within the band
    inter_ys = cubic_spline(x_vals, y_vals, wavelengths[o2a_indices])
    if plot == True:
        # plot the interpolated values within the band
        plt.plot(wavelengths[o2a_indices], inter_ys, color = 'green', label = 'Spline Fit')
        plt.scatter(wavelengths[o2a_indices], inter_ys, color = 'green', label = 'Constructed Points')

        plt.xlabel('Wavelengths (nm)')
        plt.ylabel('Spectral Value')
        plt.title('O2A Band: iFLD Interpolation')

        plt.legend()
        plt.show()
    
    # create smoothed array containing the interpolated values within the band
    
    smoothed = spectrum
    
    smoothed[o2a_indices] = inter_ys
    
    smoothed[o2b_indices] = inter_ys_b

    
    return(smoothed)
    
def iFLD(e_spectra, l_spectra, wavelengths, fwhm, band = 'A', plot=True):
    """ Applies the iFLD method at either the O2A or O2B absorption bands to extract the SIF.
        As described by Alonso et al. (2008) DOI: 10.1109/LGRS.2008.2001180

    Parameters
    ----------
    e_spectra : np.array
        spectral array containing the incident solar radiance
    l_spectra : np.array
        spectral array containing the upwelling solar radiance
    wavelengths : np.array
        array of the wavelength values at which both the 'e_spectra' and 'l_spectra' were sampled
    fwhm: float
        full width half maximum at which the wavelengths were sampled
    band: str ('A' or 'B')
        Specifies which absorption band the retrieval algorithm should use, by default 'A' for O2A absorption band
    plot : bool, optional
        produce plot at absorption band showing points selected, by default True
    
    Output
    -------
    fluorescence: float
        SIF retrieved at the given absorption band
    """
    
    r_app = l_spectra / e_spectra # caluclate array of apparent reflectance values
    
    # replace the absorption band areas with NaN values to construct the smoothed arrays
    # identify the areas containing the O2A and O2B bands
    o2b_left_index = find_nearest(wavelengths, 686)
    o2b_right_index = find_nearest(wavelengths, 697)
    o2a_left_index = find_nearest(wavelengths, 759)
    o2a_right_index = find_nearest(wavelengths, 770)
    
    r_app_nan = np.copy(r_app) # make copies of the arrays
    e_spectra_nan = np.copy(e_spectra)
    
    # set values within identified O2 absorption regions equal to NaN
    r_app_nan[o2b_left_index:o2b_right_index] = np.nan
    r_app_nan[o2a_left_index:o2a_right_index] = np.nan
    e_spectra_nan[o2b_left_index:o2b_right_index] = np.nan
    e_spectra_nan[o2a_left_index:o2a_right_index] = np.nan
    
    # calculate the smoothed spectrums by applying a cubic spline fit within the NaN regions
    r_smoothed = fit_NaN(r_app_nan, wavelengths, plot=False)
    e_smoothed = fit_NaN(e_spectra_nan, wavelengths, plot=False)
    
    # now select the points inside and outside of the absorption band to calculate the fluorescence
    buffer_in = 5 #  range to look over within absorption feature
    buffer_out = 1 # range to look over outside of the absorption feature
    
    # change the size of the shoulder skip and band position depending on the band of interest
    if band == 'A':
        out_in = 0.7535*fwhm+2.8937 # define amount to skip to shoulder from minimum
        wl_in = 760 # standard location of O2A absorption feature
    if band == 'B':
        out_in = 0.697*fwhm + 1.245 # define amount to skip to shoulder from minimum
        wl_in = 687 # standard location of the O2B aboorption band
    
    # find the points in given ranges
    # find the minimum inside of the band defining e_in and l_in
    e_in_index, e_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, e_spectra, 'min')
    l_in_index, l_in = stats_on_spectra(wavelengths, wl_in - buffer_in, wl_in + buffer_in, l_spectra, 'min')
    # find the average value at the left shoulder defining e_out and l_out
    e_out_index, e_out = stats_on_spectra(wavelengths, wl_in - buffer_out - out_in, wl_in - out_in, e_spectra, 'mean')
    l_out_index, l_out = stats_on_spectra(wavelengths, wl_in - buffer_out - out_in, wl_in - out_in, l_spectra, 'mean')
    
    # now calculate the apparent correction coefficients for the iFLD method
    # alpha_R = R_out / r_smoothed_in
    # alpha_F = E_out * alpha_R / e_smoothed_in
    r_out = l_out / e_out
    e_smoothed_in = e_smoothed[e_in_index] # match the index of the band minimum with the smoothed spectra
    r_smoothed_in = r_smoothed[e_in_index]
    
    alpha_R = r_out / r_smoothed_in # calculate the apparent correction coefficients
    alpha_F = e_out * alpha_R / e_smoothed_in
    

    if plot == True: # plot spectra and points at absorption feature
        plt.plot(wavelengths, e_spectra, color = 'orange')
        plt.plot(wavelengths, l_spectra, color = 'blue')
        plt.plot(wavelengths, e_smoothed)
        plt.scatter(wavelengths[e_in_index], e_smoothed_in)
        plt.scatter(wavelengths[e_in_index], e_in, label = 'e_in')
        plt.scatter(wavelengths[l_in_index], l_in, label = 'l_in')
        plt.scatter(wavelengths[e_out_index], e_out, label = 'e_out')
        plt.scatter(wavelengths[l_out_index], l_out, label = 'l_out')
        #plt.legend()
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Radiance (mW m−2 sr−1 nm−1)')
        
        # zoom to absorption band
        if band == 'A':
            plt.xlim(750, 775)
            plt.title('O$_2$A Absorption Band: iFLD Fitting')
        
        if band == 'B':
            plt.xlim(680, 700)
            plt.title('O$_2$B Absorption Band: iFLD Fitting')
        
        plt.show() # show plot
        
    fluorescence = (alpha_R*e_out*l_in - e_in*l_out) / (alpha_R*e_out - alpha_F*e_in) # calculate fluorescence using apparent correction coefficients
    
    return(fluorescence)

# test sequence

"""
# get csv pathnames
e_pathname = '/Users/jameswallace/Desktop/SCOPE_crops/dense_midold_unstressed_bean/Esun.csv'
l_pathname = '/Users/jameswallace/Desktop/SCOPE_crops/dense_midold_unstressed_bean/Lo_spectrum_inclF.csv'

# place them into dataframes
e_df = get_simulated_spectral_df(e_pathname)
l_df = get_simulated_spectral_df(l_pathname)

# test FLD methods
sFLD(np.asarray(e_df.iloc[0]) /np.pi, np.asarray(l_df.iloc[0]), np.arange(400, 2562), fwhm = 1, plot = True)
three_FLD(np.asarray(e_df.iloc[0]) /np.pi, np.asarray(l_df.iloc[0]), np.arange(400, 2562), fwhm = 1, plot = True)

# resample the dataframes at 3.5 nm FWHM
e_resampled, re_wave = resample_spectra(fwhm = 3.5, spectra = e_df.iloc[0])
l_resampled = resample_spectra(fwhm = 3.5, spectra = l_df.iloc[0])[0]

# test FLD methods on resampled spectras
print(sFLD(e_resampled / np.pi, l_resampled, re_wave, fwhm = 3.5))
print(three_FLD(e_resampled / np.pi, l_resampled, re_wave, fwhm = 3.5))
"""