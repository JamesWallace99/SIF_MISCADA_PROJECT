# Import all of the Py6S code
from Py6S import *
import numpy as np
# Create a SixS object called s (used as the standard name by convention)
s = SixS()


# Run the 6S simulation defined by this SixS object across the
# whole VNIR range

# get parameters from XML file and QGIS

s.geometry.month = 3
s.geometry.day = 16
s.geometry.gmt_decimal_hour = 15.5
s.geometry.latitude = -37.819705
s.geometry.longitude = 176.335646

# set sensor altitude

# -37.819705, 176.335646

# import the experimental wavelengths

import pandas as pd

exp_wavelengths_df = pd.read_csv(r'/Users/jameswallace/Desktop/Project/data/band_number_conversion.csv', index_col = 'ID')

exp_wavelengths = exp_wavelengths_df['Wavelength'].to_numpy() / 1000

# np.arange(0.3, 2.5, 0.001)

wavelengths, envi = SixSHelpers.Wavelengths.run_wavelengths(s, exp_wavelengths, output_name="environmental_irradiance")

wavelengths, direct = SixSHelpers.Wavelengths.run_wavelengths(s, exp_wavelengths, output_name="direct_solar_irradiance")

wavelengths, diffuse = SixSHelpers.Wavelengths.run_wavelengths(s, exp_wavelengths, output_name="diffuse_solar_irradiance")

results = envi + direct + diffuse



# Plot these results, with the y axis label set to "Pixel Radiance"
SixSHelpers.Wavelengths.plot_wavelengths(wavelengths, results, "Solar Irradiance")



# save outputs of combined irradainces to csv with wavelengths tested. Then compare with spectra from EnMap-Box


pd.DataFrame(results).to_csv("/Users/jameswallace/Desktop/Project/Py6S/solar_irradiance2.csv")
pd.DataFrame(wavelengths).to_csv("/Users/jameswallace/Desktop/Project/Py6S/wavelengths2.csv")