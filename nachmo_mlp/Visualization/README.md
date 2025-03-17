# Vizualization for the paper

## Requirements

-  If you run the code the first time , it is necessary to install the `texlive` package. Log on to your conda envirinment and run `conda install conda-forge::texlive-core`. The run the command in the terminal  `module load texlive/live2021-gcc-11.2.0`.

- If you ran the code previously, just run the command in the terminal  `module load texlive/live2021-gcc-11.2.0`.

## Structure

The visualization program plots the results of neural network estimates. It has the following structure:
* **driver.py** # Reads true and estimated concentrations and distributes the plotting tasks among the corresponding functions, initiates the writing of the corresponding figures on a hard drive.
     -  **passmaker** # Creates a user-specified folder (if not provided, it uses the default name `Figs/`), where it puts all figures. Plotting subroutines also call the passmaker. For each such subroutine, it creates subfolders with meaningful names and places these figures there. I.e., for the histograms, the passmaker creates in the folder `Figs/` a subfolder `histograms/` and writes figures with histograms there.  
     -  **drawing_spagetti** # Draws two spaghetti plots (similar to spaghetti diagrams in weather forecasting). The first plot shows all estimated concentration time series of one species in a single graph, and the second plot shows corresponding reference concentrations.
     -  **drawing_spagetti_absolute_error** # Draws the mean of the absolute error for the concentration estimates (in ppb), i.e., `Err_abs= mean(abs(c_est - c_true))`. The grey-shaded region represents the standard deviation of the absolute error, and the pink-shaded region represents the smallest negative and the largest positive values of the error, i.e., `c_est - c_true`.
     -  **drawing_spagetti_relative_error** # Draws spaghetti plots representing *a kind of* the relative error which we define as `Err = (c_est - c_true) / max(c_true) * 100% `. Each graph corresponds to a single NN estimate obtained for the corresponding IC. Although our error measure underestimates the true relative error, it avoids the infinite values when `c_true-> 0`.   
     -  **drawing_histogram**: plots the histograms of the corresponding species at `step = step`.

Note that all functions used by the driver are located in the files having the same name. We adopt the convention that each file contains only one function, and its name is the same as the corresponding function's name.

## Key variables

* `epoch =` specifies the epoch for which you whant to visualize, type: integer 

* `species = ["OH", "H2O2","HO2"]` Specifies the species, type: list of strings. Note, all chemicals must be capitalized, and their sequence must be the same as in the NN inputs.  
* `path = ''` path to the data.
* `pass_to_figures = ` absolute or relative path to the place where you whant to see your figures.  
* `step =` time step which is visualized in the histogram 

### Other variables
* `skip = int(timesteps/400)` skips equal intervals in the data to fit the resolution of the screen and speed up the plotting process.  For instance, if the estimated concentration consists of 500K timesteps, there is no sense in plotting all of them on a single graph, especially, if your monitor has a resolution order of 1K. Moreover, saving such figures may require a lot of memory. The denominator on the right specifies the length of the. intervals that you would like to skip.        

## Ticks, axes, titles etc.

Specify the corresponding ticks and figure titles in each drawing_ function, otherwise, drawing subroutines will use default ticks and titles. Note that figures are saved in the folder, specified by the `pass_to_figures ` variable under the same names as their titles.   

* `xticks =`,  ticks in X-axis, type: list of strings.
* `xlabel =`, label in X-axis, type: string
* `ylabel =`,  label in Y-axis, type: string
* `title =`,  figure's title, type: string


