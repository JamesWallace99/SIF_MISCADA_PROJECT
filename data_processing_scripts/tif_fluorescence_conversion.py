import rioxarray
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

import sys
# insert at 1, 0 is the script path (or '' in REPL)
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
    
    print('Conversion Succesful!')
    
    return(band_num_conv_df)


'''
--------------------------------------------
Final function to calculate fluorscence values over the image and generate heat map plot
--------------------------------------------
'''

def get_tif_fluorescence(tif_pathname, method, e_pathname = '/Users/jameswallace/Desktop/SIF_MISCADA_PROJECT/py6s_generate_irradiance/17_06_2021_13:18_irradiance.csv', bandnumber_pathname = '/Users/jameswallace/Desktop/Project/band_number_conversion.csv', plot = True):
    # convert tif image to csv
    dateTimeObj = datetime.now() # get time stamp for output name
    timestampStr = dateTimeObj.strftime("%d_%m_%Y_%H:%M_")
    output_name = 'temp_spectral_' + timestampStr + '.csv'
    tif_to_csv(tif_pathname, output_name) 
    
    # csv file to dataframe
    r_app_df = spectral_csv_to_df(output_name) # convert the spectral csv to dataframe
    wavelengths_df = bandnumber_to_wavelength(bandnumber_pathname) # convert wavelengths dataframe
    
    # csv to np.arrays
    e_spectra = py6s_csv_to_array(e_pathname) # convert irradiance csv to np.array
    wavelengths = np.asarray(wavelengths_df['Wavelength']) # get np.array of wavelength values
    
    # initiate df for fluorescence values and co-ordinates
    d = {'x': np.asarray(r_app_df['x']), 'y': np.asarray(r_app_df['y']), 'fluor': np.empty(len(r_app_df))}
    fluorescence_df = pd.DataFrame(data = d) # create the dataframe
    
    for i in range(len(r_app_df)):
        l_spectra = np.asarray(r_app_df.iloc[i][2:]) * e_spectra / np.pi
        fluorescence_df['fluor'][i] = FLD_methods.sFLD(e_spectra / np.pi, l_spectra, wavelengths, fwhm = 3.5, band = 'A', plot = False)
    
    if plot == True:
        plt.scatter(fluorescence_df['x'], fluorescence_df['y'], c=fluorescence_df['fluor'], s = 0.5, marker = 'h')
        plt.colorbar()
        plt.show()
    
    return(fluorescence_df)

get_tif_fluorescence('/Users/jameswallace/Desktop/Project/data/gold/s7_5240_W.tif', method = 1)
    