import rioxarray
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import math
import sys
import json
sys.path.insert(1, '/Users/jameswallace/Desktop/SIF_MISCADA_PROJECT/SCOPE_simulations')
import FLD_methods


'''
--------------------------------------------
Functions to convert TIF images to csv files
--------------------------------------------
'''

def tif_to_csv(input_pathname, output_pathname):
    """
    Converts a hyperspectral TIF image file with 450 spectral bands to a csv file.
    Requires Rioxarray and Pandas python packages

    Parameters
    ----------
    input_pathname : str (pathname)
        imput tif image pathname
    output_pathname : str (pathname)
        export csv pathname ( e.g. dir\save_location.csv)
    """
    
    # verify input is tif file
    if input_pathname[-4:] != '.tif':
        return('Error: Input file is not in tif format')
    
    # verfiy output is to csv
    if output_pathname[-4:] != '.csv':
        return('Error: Output pathname is not to csv')
    
    
    # import the TIFF image as a riox DataArray type with a mask + scaling applied
    xds = rioxarray.open_rasterio(input_pathname, mask_and_scale = True)
    xds.name = "data"
    df = xds.to_dataframe() # create the dataframe

    df = df.drop('spatial_ref', 1) # drop the empty spatial reference column

    df = df.reset_index() # reset the index so no longer grouped with a multi-index

    # create a newly formatted df
    spectral_one = df[df['band'] == 1]
    df2 = pd.DataFrame({'y':spectral_one['y'], 'x':spectral_one['x'], '1':spectral_one['data']})

    # iterate to get the data for each of the bands
    for i in range(2, 450):
        column_name = str(i) # get the column name
        spectral_temp = df[df['band'] == i] # get the correct band values from the main df
        spectral_temp.reset_index(drop=True, inplace = True) # reset the index so they match new df2
        df2[column_name] = spectral_temp['data'] # create a new column for the band values

    df2 = df2.dropna() # drop the empty rows from the new df

    df2.reset_index(drop=True, inplace = True) # reset the index of the new df

    df2.to_csv(output_pathname)
    
    return(print('Image Conversion Succesful!'))


def convert_hs_files(input_file_pathname):
    """Converts all tif images in a folder to seperate csv files

    Parameters
    ----------
    input_file_pathname : string (folder pathname)
        pathname for the folder containing the tif images
    """
    
    # check input is str type
    if type(input_file_pathname) != str:
        return('Error: input not string type')
    
    # get a list of pathnames for tif files
    
    path = input_file_pathname
    
    tif_files = []

    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".tif") and entry.is_file():
                tif_files.append(entry.path)
    
    print('Found these tif files:', tif_files)
    
    # generate list of ouput csv names
    
    output_names = [] # initiate empty array for output names (same length as tif_files)

    for image in tif_files:
        output_names.append(image[:-3] + 'csv')
    
    print('Saving the csv files to these locations:', output_names)
    # for each item in list run tif_to_csv script
    
    for i in range(len(tif_files)):
        tif_to_csv(tif_files[i], output_names[i])
    
    
    return(print('Conversion Succesful!'))

'''
--------------------------------------------
Functions to convert csv files to readable pandas DataFrame
--------------------------------------------
'''

def spectral_csv_to_df(csv_pathname):
    """Converts the output spectral dataframe from the tif_to_csv function to a readable pandas dataframe

    Parameters
    ----------
    csv_pathname : str (pathname)
        pathname of the output csv from the 'tif_to_csv' function
    
    Output
    --------
    spectral_df: pandas.DataFrame
        pandas DataFrame containing the spectral information.
        Unique index, first column is the y-cords in UTM, second column contains x-cords in UTM
        Proceeding columns contain the spectral value at the sampled wavelength
    
    """
    spectral_df = pd.read_csv(csv_pathname, index_col = 0) # read the csv to a df 
    return(spectral_df)


def py6s_csv_to_array(csv_pathname):
    """Converts the output irradiance spectra csv from Py6S to a np.array

    Parameters
    ----------
    csv_pathname : str (pathname)
        pathname of the csv file containing the irradiance spectra output from Py6S
    
    Output
    --------
    e_spectra: np.array
        np.array containing the irradiance values from the Py6S simulation
    """
    e_spectra_df = pd.read_csv(csv_pathname, index_col = 0) # convert csv to df
    e_spectra_df = e_spectra_df.rename({'0':'irradiance'}, axis = 1) # name first column 'irradiance'
    e_spectra_df['irradiance'] = e_spectra_df.fillna(method = 'backfill') # backfill the irradiance to remove NaN values
    e_spectra = np.asarray(e_spectra_df['irradiance']) # convert irradiance column to a np.array
    return(e_spectra)

def bandnumber_to_wavelength(band_number_conversion_pathname):
    """ Generates a pandas dataframe of the measured wavelengths to convert 
        the bandnumber in the spectral data to wavelength (nm)

    Parameters
    ----------
    band_number_conversion_pathname : str (csv pathname)
        csv file containing the spectral bandnumber conversion information
        1st column: 'ID' contains bandnumbers from 1:449
        2nd column: 'Wavelength' contains wavelengths in nm
        
    Output    
    ---------
    band_num_conv_df: pandas Dataframe
        pandas dataframe indexed by band number 1:449
        1st column: 'Wavelength' contains wavelengths in nm
    
    """
    
    # input tests
    
    # check input is str type
    if type(band_number_conversion_pathname) != str:
        return('Error: input not string type')
    
    # check csv
    
    # verfiy input is to csv
    if band_number_conversion_pathname[-4:] != '.csv':
        return('Error: Output pathname is not to csv')
    
    
    # import csv containing band numbers conversion
    band_num_conv_df = pd.read_csv(band_number_conversion_pathname, index_col = 'ID')
    
    
    return(band_num_conv_df)

'''
--------------------------------------------
Convert UTM Coordinates to Latitude and Longlitude
--------------------------------------------
'''

def utmToLatLng(easting, northing, zone = 60, northernHemisphere=True):
    """Converts UTM co-ordinate system to lattitude and longlitude

    Parameters
    ----------
    easting : float
        x-coordinate
    northing : float
        y-coordinate
    zone : int, optional
        coordinate zone (N to W), by default 60 (New Zealand)
    northernHemisphere : bool, optional
        are the coordinates in the northen hemisphere, by default True

    Returns
    -------
    [type]
        [description]
    """
    
    
    
    if not northernHemisphere:
        northing = 10000000 - northing

    a = 6378137
    e = 0.081819191
    e1sq = 0.006739497
    k0 = 0.9996

    arc = northing / k0
    mu = arc / (a * (1 - math.pow(e, 2) / 4.0 - 3 * math.pow(e, 4) / 64.0 - 5 * math.pow(e, 6) / 256.0))

    ei = (1 - math.pow((1 - e * e), (1 / 2.0))) / (1 + math.pow((1 - e * e), (1 / 2.0)))

    ca = 3 * ei / 2 - 27 * math.pow(ei, 3) / 32.0

    cb = 21 * math.pow(ei, 2) / 16 - 55 * math.pow(ei, 4) / 32
    cc = 151 * math.pow(ei, 3) / 96
    cd = 1097 * math.pow(ei, 4) / 512
    phi1 = mu + ca * math.sin(2 * mu) + cb * math.sin(4 * mu) + cc * math.sin(6 * mu) + cd * math.sin(8 * mu)

    n0 = a / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (1 / 2.0))

    r0 = a * (1 - e * e) / math.pow((1 - math.pow((e * math.sin(phi1)), 2)), (3 / 2.0))
    fact1 = n0 * math.tan(phi1) / r0

    _a1 = 500000 - easting
    dd0 = _a1 / (n0 * k0)
    fact2 = dd0 * dd0 / 2

    t0 = math.pow(math.tan(phi1), 2)
    Q0 = e1sq * math.pow(math.cos(phi1), 2)
    fact3 = (5 + 3 * t0 + 10 * Q0 - 4 * Q0 * Q0 - 9 * e1sq) * math.pow(dd0, 4) / 24

    fact4 = (61 + 90 * t0 + 298 * Q0 + 45 * t0 * t0 - 252 * e1sq - 3 * Q0 * Q0) * math.pow(dd0, 6) / 720

    lof1 = _a1 / (n0 * k0)
    lof2 = (1 + 2 * t0 + Q0) * math.pow(dd0, 3) / 6.0
    lof3 = (5 - 2 * Q0 + 28 * t0 - 3 * math.pow(Q0, 2) + 8 * e1sq + 24 * math.pow(t0, 2)) * math.pow(dd0, 5) / 120
    _a2 = (lof1 - lof2 + lof3) / math.cos(phi1)
    _a3 = _a2 * 180 / math.pi

    latitude = 180 * (phi1 - fact1 * (fact2 + fact3 + fact4)) / math.pi

    if not northernHemisphere:
        latitude = -latitude

    longitude = ((zone > 0) and (6 * zone - 183.0) or 3.0) - _a3

    return (latitude, longitude)


'''
--------------------------------------------
Final function to calculate fluorscence values over the image and generate heat map plot
--------------------------------------------
'''

def get_tif_fluorescence(tif_pathname, method, e_pathname, bandnumber_pathname = '/Users/jameswallace/Desktop/Project/band_number_conversion.csv', plot = True, band = 'A'):
    """Retrieves the fluorescence values from a TIF image using a defined FLD method.

    Parameters
    ----------
    tif_pathname : str (pathname)
        pathname of the target TIF image
    method : str ('simple', 'three' or 'improved')
        defines the FLD method to be used for SIF retrieval. 
        Either: 'simple' (sFLD), 'three' (3FLD) or 'improved' (iFLD)
    e_pathname : str (pathname)
        pathname of the csv output file from the Py6S irradiance simulation, by default '/Users/jameswallace/Desktop/SIF_MISCADA_PROJECT/py6s_generate_irradiance/17_06_2021_13:18_irradiance.csv'
    bandnumber_pathname : str, optional
        pathname of the csv file containing the measurement wavelengths, by default '/Users/jameswallace/Desktop/Project/band_number_conversion.csv'
    plot : bool, optional
        dictates wether heat map and histogram plot of retrieved SIF is generated, by default True
    band: str ('A' or 'B')
        determines which O2 absorption band to use for retrieval, default 'A' the O2A absorption band
        take either 'A' for the O2A band or 'B' for the O2B band.
        
    Outputs
    --------
    Saves a csv file containing the fluorescence values at each coordinate.
    Saves a  txt file containing the parameters of the simulation.
    Presents graphs showing a heatmap of the fluorescence values and histogram of the distribution.
    """
    # convert tif image to spectral csv file
    dateTimeObj = datetime.now() # get time stamp for output name
    timestampStr = dateTimeObj.strftime("%d_%m_%Y_%H:%M_")
    output_name = 'temp_spectral_' + timestampStr + '.csv'
    tif_to_csv(tif_pathname, output_name) 
    print('Converting CSV files')
    # convert csv files contianing spectra and wavelength conversions to dataframes
    r_app_df = spectral_csv_to_df(output_name) # convert the spectral csv to dataframe
    wavelengths_df = bandnumber_to_wavelength(bandnumber_pathname) # convert wavelengths dataframe
    
    # convert e_spectra and wavelengths to np.arrays
    e_spectra = py6s_csv_to_array(e_pathname) # convert irradiance csv to np.array
    wavelengths = np.asarray(wavelengths_df['Wavelength']) # get np.array of wavelength values
    
    # initiate df for fluorescence values and co-ordinates
    d = {'x': np.asarray(r_app_df['x']), 'y': np.asarray(r_app_df['y']), 'fluor': np.empty(len(r_app_df))} # get the data
    fluorescence_df = pd.DataFrame(data = d) # create the dataframe
    print('Calculating Fluorescence using ' + method + ' FLD method')
    # calculate the fluorescence
    if method == 'simple':
        # get the fluorscence using sFLD
        for i in range(len(r_app_df)):
            l_spectra = np.asarray(r_app_df.iloc[i][2:]) * e_spectra / np.pi # for each pixel get the upwelling radiance from the apparent reflectance
            fluorescence_df['fluor'][i] = FLD_methods.sFLD(e_spectra / np.pi, l_spectra, wavelengths, fwhm = 3.5, band = band, plot = False)
    if method == 'three':
        # get the fluorscence using 3FLD
        for i in range(len(r_app_df)):
            l_spectra = np.asarray(r_app_df.iloc[i][2:]) * e_spectra / np.pi # for each pixel get the upwelling radiance from the apparent reflectance
            fluorescence_df['fluor'][i] = FLD_methods.three_FLD(e_spectra / np.pi, l_spectra, wavelengths, fwhm = 3.5, band = band, plot = False)
    if method == 'improved':
        # get the fluorscence using iFLD
        for i in range(len(r_app_df)):
            l_spectra = np.asarray(r_app_df.iloc[i][2:]) * e_spectra / np.pi # for each pixel get the upwelling radiance from the apparent reflectance
            fluorescence_df['fluor'][i] = FLD_methods.iFLD(e_spectra / np.pi, l_spectra, wavelengths, fwhm = 3.5, band = band, plot = False)
    
    # generate heatmap plot of fluorescence intensity
    if plot == True:
        plt.scatter(fluorescence_df['x'], fluorescence_df['y'], c=fluorescence_df['fluor'], s = 0.5, marker = 'h')
        plt.colorbar()
        plt.title(tif_pathname)
        plt.xlabel('X co-ord (UTM60)')
        plt.ylabel('Y co-ord (UTM60)')
        plt.show()
        plt.hist(fluorescence_df['fluor'], bins = 50)
        plt.xlabel('SIF')
        plt.ylabel('Frequency')
        plt.title(tif_pathname)
        plt.show()
    os.remove(output_name)
    
    # now save the new fluorescence dataframe to a csv file
    
    options = {'tif_image': tif_pathname, 'method': method, 'e_spectra': e_pathname, 'Wavelengths': bandnumber_pathname}
    
    csv_output_name = 'fluorescence' + timestampStr + '.csv'
    options_name = 'fluorescence' + timestampStr + 'parameters.txt'
    
    fluorescence_df.to_csv(csv_output_name) # write fluorescence values to csv
    
    with open(options_name, 'w') as file:
        file.write(json.dumps(options)) # write input parameters to txt file
    
    return(print('Fluorescence values succesfully saved!'))

# test
get_tif_fluorescence('/Users/jameswallace/Desktop/Project/data/gold/s6_5240_E.tif', method = 'improved', e_pathname = '/Users/jameswallace/Desktop/SIF_MISCADA_PROJECT/SCOPE_simulations/final_irradiance_df.csv')