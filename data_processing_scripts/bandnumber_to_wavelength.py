import pandas as pd
import numpy as np


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


# test by printing out head of dataframe
# print(bandnumber_to_wavelength('/Users/jameswallace/Desktop/Project/band_number_conversion.csv').head())