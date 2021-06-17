from sFLD import *

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
    df = e_spectra_df.groupby(np.arange(len(df))//step).mean()
    return(df)

""" 

After averaging df

plot of spectra on single graph E and F to show changes

for each row in spectra
    calculate SIF at O2A band by appyling sFLD
    add the SIF to an array or new df
    compare the SIF with the actual value of the SIF at the O2A band and calculate the differce
Find the RMSE
"""