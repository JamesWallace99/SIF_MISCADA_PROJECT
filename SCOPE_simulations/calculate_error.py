from FLD_methods import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def average_dataframe(df, step):
    """Averages all column values of a dataframe for every "step" rows

    Parameters
    ----------
    df : pandas df
        dataframe to be averaged 
    step : int
        number of rows that will be grouped and averaged in each step
    """
    df = df.groupby(np.arange(len(df))//step).mean()
    return(df)

def calc_average_percentage_error(e_spectra, l_spectra, f_spectra, wavelengths, method):
    """Calculates the average percentage error between a simulated dataset and FLD method implementation
        The E, L and F dataframes should have matching indexes
    

    Parameters
    ----------
    e_spectra : pd.Dataframe
        pd dataframe containing the downwelling radiance values
    l_spectra : pd.Dataframe
        pd dataframe containing the upwelling radiance values
    f_spectra : pd.Series
        pandas series containing the fluorescence values at F761
    wavelengths : np.array
        contains the wavelengths at which the E, L and F spectra were sampled over
    method : str
        "sFLD" or "3FLD"
    """
    percentage_errors = []
    for i in range(len(e_spectra)):
        if method == 'sFLD':
            error = f_spectra.iloc[i] - sFLD(e_spectra, l_spectra, re_wave, plot = False)
        if method == '3FLD':
            error = f_spectra.iloc[i] - three_FLD(e_spectra, l_spectra, re_wave, plot = False)
        percentage_error = 100 * abs(error) / f_spectra.iloc[i]
        percentage_errors.append(percentage_error)
    return(np.mean(percentage_errors))


