# SIF_MISCADA_PROJECT

This repositry contains code used for extracting sun-induced chlorophyll fluorescence (SIF) from hyperspectral imagery.

---------
# Index
---------
# data_processing_scripts
  - Contains files for converting TIFF hyperspectral images to SIF maps and values.

# SCOPE_simulations
  - Contains files for extracting SIF from SCOPE simulations, including implementations for the sFLD, 3FLD and iFLD methods.
  - The file FLD_methods is used within the data_processing_scripts to extract the SIF values
  - Contains the modelling excercise for the HySpex VNIR-1800 sensor.

# py6s_generate_irradiance
  - Contains scripts for generating irradiance spectra from input parameters using the Py6S wrapper of the atmospheric radiative transfer model 6S.

# SCOPE_crops
  - Contains the SCOPE simulated data for the 16 modelled canopies within the report.

# GPR
  - Contains the scripts used to investigate the Gaussian Processes Regression models for O2B SIF prediction.

# standalone_figures
  - Contains the scripts used to generate the standalone figures within the report.
