import rioxarray
import pandas as pd
import os

# includes two functions
# tif_to_csv converts a single tif image to a csv
# convert_hs_files converts all the tif images within a folder to seperate csvs


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


# run test
#tif_to_csv('/Users/jameswallace/Desktop/Project/data/gold/s7_5240_W.tif', '/Users/jameswallace/Desktop/Project/data/gold_s75240_W.csv')


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
    
# test conversion
# convert_hs_files('/Users/jameswallace/Desktop/Project/data/red')