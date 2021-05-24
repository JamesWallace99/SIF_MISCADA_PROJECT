import rioxarray

# get TIFF image to convert
input_file_path = input("TIFF Image File Path:")

#input_file_path = "/Users/jameswallace/Desktop/Project/data/red/s22_6562_W.tif"

# import the TIFF image as a riox DataArray type with a mask + scaling applied
xds = rioxarray.open_rasterio(input_file_path, mask_and_scale = True)

# now convert to dataframe from exporting

import pandas as pd

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

# set the csv file location 
#export_file_path = '/Users/jameswallace/Desktop/Project/data/test_export.csv'

export_file_path = input("CSV Export Location Path:")

df2.to_csv(export_file_path)
    


