# Import all of the Py6S code
from Py6S import *
import numpy as np
import pandas as pd
import csv
from datetime import datetime


def generate_simulated_irradiance(retrieval_month, retrieval_day, retrieval_decimal_hour, retrieval_lat,
                                  retrieval_long, wavelengths_pathname,
                                  retrieval_alt = None, retrieval_wvc = None, retrieval_vis = None, output_params = True):
    """Uses the Py6S wrapper to generate a simulated irradiance spectrum given a set of paramters.
        works within the Py6S conda env, saves the irradiance spectra and model parameters to csvs in same folder

    Parameters
    ----------
    retrieval_month : int
        month of retrieval (i.e. 03)
    retrieval_day : int
        day of the month of retrieval (i.e. 09)
    retrieval_decimal_hour : float
        hour of retrieval (i.e 10:30 is 10.5)
    retrieval_lat : float
        latitude of retrieval
    retrieval_long : float
        longlitude of retrieval
    wavelengths_pathname : str (pathname)
        pathname of csv file containing wavelengths to be generated over
        Index in first column, wavelengths in nanometers on the second column
    retrieval_alt : float, optional
        altitude of the retrieval, takes Py6S value by default
    retrieval_wvc : float, optional
        water apour column of the retrieval, takes Py6S value by default
    retrieval_vis : float, optional
        visibility of retrieval, takes Py6S value by default
    """
    
    
    s = SixS() # Create a SixS object called s (used as the standard name by convention)
    
    
    # define parameters for the model
    s.geometry.month = retrieval_month
    s.geometry.day = retrieval_day
    s.geometry.gmt_decimal_hour = retrieval_decimal_hour
    s.geometry.latitude = retrieval_lat
    s.geometry.longitude = retrieval_long
    
    # match the simulated wavelengths with those used in the measurements
    exp_wavelengths_df = pd.read_csv(wavelengths_pathname, index_col = 'ID')
    exp_wavelengths = exp_wavelengths_df['Wavelength'].to_numpy() / 1000 # convert from nanometers to micrometers
    
    # run the simulations for different irradiances and sum for final result
    wavelengths, envi = SixSHelpers.Wavelengths.run_wavelengths(s, exp_wavelengths, output_name="environmental_irradiance")
    wavelengths, direct = SixSHelpers.Wavelengths.run_wavelengths(s, exp_wavelengths, output_name="direct_solar_irradiance")
    wavelengths, diffuse = SixSHelpers.Wavelengths.run_wavelengths(s, exp_wavelengths, output_name="diffuse_solar_irradiance")
    results = envi + direct + diffuse
    
    
    # generate plot of values
    irradiance_plot = SixSHelpers.Wavelengths.plot_wavelengths(wavelengths, results, "Solar Irradiance")
    
    # store the ouputs as csv files
    
    # get timestamp 
    dateTimeObj = datetime.now()

    timestampStr = dateTimeObj.strftime("%d_%m_%Y_%H:%M_")
    
    irradiance_spectrum = pd.DataFrame(results).to_csv(timestampStr + "irradiance.csv") # store the spectra
    
    # store the parameters in  a sperate file
    
    if output_params is True:
        s.run()
        dict = s.outputs.values
        w = csv.writer(open(timestampStr + 'simulation_params.csv', "w"))
        for key, val in dict.items():
            w.writerow([key, val])
    
    
    return(irradiance_spectrum, irradiance_plot)



# test

# generate_simulated_irradiance(2, 10, 10.5, -37.687798, 176.165131, '/Users/jameswallace/Desktop/Project/band_number_conversion.csv')
