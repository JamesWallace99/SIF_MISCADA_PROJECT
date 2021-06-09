import rioxarray
import matplotlib.pyplot as plt

def quick_plot(input_pathname, bandnumber = 116):
    """ Generate a quick plot of a tif image at a given band number (default = 116 (~770 nm))

    Parameters
    ----------
    input_pathname : str (tif img pathname)
        pathname of the tif image with 449 bands
    bandnumber : int, optional
        bandnumber to generate image at (1 <= bandnumber <= 449), by default 116 ~ 770 nm
    """
    
    # import the tif image as an xarray and apply a mask and scale to the image
    xds = rioxarray.open_rasterio(input_pathname, mask_and_scale = True)
    
    xds_band = xds.sel(band = bandnumber) # select defined bandnumber
    
    imgplot = xds_band.plot.imshow(figsize=(9,7), cmap='Greys', robust = True)
    plt.show()
    
    return(print('Visualisation Succesful!'))
    
    
# test run
# quick_plot("/Users/jameswallace/Desktop/Project/data/red/s22_6562_W.tif", bandnumber=420)