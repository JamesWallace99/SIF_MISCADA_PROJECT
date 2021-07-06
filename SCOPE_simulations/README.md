# Basic Usage

<h2>Loading Relevant Libraries</h2>


```python
from FLD_methods import *
```

<h2>Load Example Spectral Data From SCOPE</h2>


```python
# total incident irradiance is the sum of the direct and diffuse irradiance
e_spectra = get_simulated_spectral_df('Esun_example.csv') + get_simulated_spectral_df('Esky_example.csv')
l_spectra = get_simulated_spectral_df('Lo_spectrum_inclF_example.csv')
```

The 'get_simulated_spectral_df' function extracts the hyperspectral data from a csv file and places it in a pandas dataframe with columns named from 400 to 2562.


```python
e_spectra.head() # show the top of the pandas dataframe containing the E_spectra
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>400</th>
      <th>401</th>
      <th>402</th>
      <th>403</th>
      <th>404</th>
      <th>405</th>
      <th>406</th>
      <th>407</th>
      <th>408</th>
      <th>409</th>
      <th>...</th>
      <th>2552</th>
      <th>2553</th>
      <th>2554</th>
      <th>2555</th>
      <th>2556</th>
      <th>2557</th>
      <th>2558</th>
      <th>2559</th>
      <th>2560</th>
      <th>2561</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>762.41540</td>
      <td>713.86390</td>
      <td>771.24480</td>
      <td>720.10360</td>
      <td>782.96950</td>
      <td>697.56130</td>
      <td>686.79500</td>
      <td>710.25660</td>
      <td>727.46720</td>
      <td>802.01720</td>
      <td>...</td>
      <td>1.275431</td>
      <td>1.182825</td>
      <td>1.094863</td>
      <td>1.010975</td>
      <td>0.938320</td>
      <td>0.875757</td>
      <td>0.816150</td>
      <td>0.759049</td>
      <td>0.710475</td>
      <td>0.664150</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37.02911</td>
      <td>34.67117</td>
      <td>37.45822</td>
      <td>34.97453</td>
      <td>38.02803</td>
      <td>33.88001</td>
      <td>33.35728</td>
      <td>34.49697</td>
      <td>35.33306</td>
      <td>38.95416</td>
      <td>...</td>
      <td>1.275200</td>
      <td>1.182610</td>
      <td>1.094665</td>
      <td>1.010793</td>
      <td>0.938149</td>
      <td>0.875598</td>
      <td>0.816003</td>
      <td>0.758913</td>
      <td>0.710346</td>
      <td>0.664030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>373.94310</td>
      <td>350.12990</td>
      <td>378.27350</td>
      <td>353.19020</td>
      <td>384.02420</td>
      <td>342.13400</td>
      <td>336.85360</td>
      <td>348.36090</td>
      <td>356.80240</td>
      <td>393.36730</td>
      <td>...</td>
      <td>1.275032</td>
      <td>1.182454</td>
      <td>1.094521</td>
      <td>1.010660</td>
      <td>0.938025</td>
      <td>0.875482</td>
      <td>0.815896</td>
      <td>0.758814</td>
      <td>0.710252</td>
      <td>0.663943</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1043.83830</td>
      <td>977.36360</td>
      <td>1055.92340</td>
      <td>985.90430</td>
      <td>1071.97490</td>
      <td>955.04320</td>
      <td>940.30570</td>
      <td>972.43060</td>
      <td>995.99730</td>
      <td>1098.06980</td>
      <td>...</td>
      <td>1.275389</td>
      <td>1.182785</td>
      <td>1.094827</td>
      <td>1.010941</td>
      <td>0.938288</td>
      <td>0.875728</td>
      <td>0.816123</td>
      <td>0.759024</td>
      <td>0.710451</td>
      <td>0.664128</td>
    </tr>
    <tr>
      <th>4</th>
      <td>524.21480</td>
      <td>490.82480</td>
      <td>530.26830</td>
      <td>495.09750</td>
      <td>538.31060</td>
      <td>479.58540</td>
      <td>472.18230</td>
      <td>488.30830</td>
      <td>500.13920</td>
      <td>551.39160</td>
      <td>...</td>
      <td>1.275198</td>
      <td>1.182608</td>
      <td>1.094663</td>
      <td>1.010791</td>
      <td>0.938147</td>
      <td>0.875597</td>
      <td>0.816001</td>
      <td>0.758912</td>
      <td>0.710345</td>
      <td>0.664029</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 2162 columns</p>
</div>



<h2>Apply the FLD Methods to the First Row of the Spectras to Extract the Fluorescence</h2>


```python
# get the first rows of the spectra dataframes
# the methods take the spectras a np arrays so convert the object to this type

e_first_row = np.asarray(e_spectra.iloc[0])
l_first_row = np.asarray(l_spectra.iloc[0])
wavelengths = np.arange(400, 2562) # define the wavelengths at which the spectras were sampled
```


```python
# apply the sFLD method and show the plot

sFLD(e_first_row / np.pi, l_first_row, wavelengths, fwhm = 1, band = 'A', plot = True)
```


    
![png](example_FLD_method_SCOPE_workflow/output_9_0.png)
    





    2.1825743558511697



The plot shows the E spectra (Orange) and L spectra (Blue) at the O2A absorption band. The plotted points show the sFLD selection for the spectra values inside and outside of the absorption feature. The value shown is the fluorescence retrieved at the O2A absorption band.


```python
# apply the 3FLD method and show the plot
three_FLD(e_first_row / np.pi, l_first_row, wavelengths, fwhm =1, band = 'A', plot = True)
```


    
![png](example_FLD_method_SCOPE_workflow/output_11_0.png)
    





    1.1659635408570277



The plot shows the E spectra (Orange) and L spectra (Blue) at the O2A absorption band. The plotted points show the 3FLD selection for the spectra values inside and on the shoulders of the absorption feature. The straight line plotted shows the interpolation between the two shoulders of the absorption feature and the interpolated point plotted shows the spectra value selected "outside" of the absorption feature. The value shown is the fluorescence retrieved at the O2A absorption band.


```python
# apply the iFLD method and show the plot
iFLD(e_first_row / np.pi, l_first_row, wavelengths, fwhm =1, band = 'A', plot = True)
```


    
![png](example_FLD_method_SCOPE_workflow/output_13_0.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_13_1.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_13_2.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_13_3.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_13_4.png)
    





    1.1813404672023262



<h2>Resample the Data at Different FWHM for Sensor Simulation</h2>


```python
# generate the spectras and wavelengths resampled at 3.5 nm
e_resampled, resampled_wavelengths = resample_spectra(fwhm = 3.5, spectra = e_first_row)
l_resampled = resample_spectra(fwhm=3.5, spectra= l_first_row)[0]
```

The 'resample_wavelengths' functions recieves the target fwhm and the desired spectra as inputs. A Gaussian convolution is then applied to the spectra and the function returns the spectra at the desired fwhm as well as the new sampling wavelengths for the spectra.


```python
# now apply the FLD methods to the resampled data

sFLD(e_resampled / np.pi, l_resampled, resampled_wavelengths, fwhm = 3.5, band = 'A', plot = True)
```


    
![png](example_FLD_method_SCOPE_workflow/output_17_0.png)
    





    5.210083739356226




```python
three_FLD(e_resampled / np.pi, l_resampled, resampled_wavelengths, fwhm = 3.5, band = 'A', plot = True)
```


    
![png](example_FLD_method_SCOPE_workflow/output_18_0.png)
    





    1.2545734449127466




```python
iFLD(e_resampled / np.pi, l_resampled, resampled_wavelengths, fwhm = 3.5, band = 'A', plot = True)
```


    
![png](example_FLD_method_SCOPE_workflow/output_19_0.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_19_1.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_19_2.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_19_3.png)
    



    
![png](example_FLD_method_SCOPE_workflow/output_19_4.png)
    





    1.0077131014942926




```python

```
