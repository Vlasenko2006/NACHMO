# Disclamer 

This is a private copy of Kinetic Pre-Processor (KPP), taken from the [original source](https://kpp.readthedocs.io/en/stable/getting_started/01_installation.html#download-kpp-from-github), version 3.0.2., for the NACHMO project. It is not intended for use for any other purposes or public distribution.  For information about the original source, license agreement, and latest releases, see the section in section The Kinetic Pre-Processor (KPP). 

# The driver.py


This version is an original KPP, controlled by an external program `driver.py`. The driver generates the initial conditions for the chemical model, sets the start and end of the calculations, and time intervals for writing the outputs, and runs the KPP processor writing the results to the `Data/` folder. It also identifies the list of species, saves the stoichiometry matrix and KPP estimates as numpy arrays saving them into `Data_npy/`. The data fro this folder you can use directly to train your NN model. The initialization of the KPP processor provided by the `driver.py` overrides the original initialization in the KPP by the `Initialize.f90` subroutine. 

Note that the driver may run the KPP processor in a sequential loop (no MPI), generating the new initial conditions for each iteration. The new initial conditions are generated as follows. The user sets the reference initial concentrations in the driver (see installation), and the driver multiplies them by a random number from 0 to 1. 

## Installation 

Assume that you want to run the KPP processor for some chemical mechanism (later just mechanism). The installation is the same as installation of the original KPP on Levante, described [here](https://codebase.helmholtz.cloud/m-dml/nachmo_coding/-/tree/main/kpp?ref_type=heads) with the difference that instead of cloning the code from the original source (the second step in the section **"2. Clone KPP-engine from the repository"**), clone it from this repository. Repeat all steps listed before generating the executable (i.e., before running the command `make -f Makeile_<mechanism>`). Suppose you want to run KPP for some chemical mechanism <mechanism> 

  * From the folder `drv/` copy all python files in the folder, where you intend to run the KPP (let it be `mechanism/`).
  * Open `driver.py` and `mechanism_Initialize.f90` in your beloved editor, i.e., emacs, gedit. 
  * In the `driver.py` find the comment `#    INITIALIZE YOUR RUNTIME VARIABLES HERE!`
  * Find the variables  `TSTART `, `TEND`, `TSTEPS`, and `TEMP`   corresponding to the starting, and ending of the computational time, discrete output steps, and temperature respectively. Give them the desired values.
  * In the `driver.py` find the comment `~~~ Define scale factor for units` and set `CFACTOR` below it.
  * In the `mechanism_Initialize.f90`, find the comment `! ~~~ Set initial species concentrations` and copy the code below containing `C(some integer)` and `RCONST(some integer)`, and paste it under the same comment in the `driver.py`.  These are the reference concentrations and concentration rate constants. If you want to change their values in the future, this is the right place to do that. Change the curly brackets in the indexing into rectangular ones. Note that integers in `C(some integer)` correspond to a certain species. So, if you know the order of these species in the KPP solver, you may directly code the initial concentration values without copying and pasting them from `mechanism_Initialize.f90`. 
*  Run the command in your terminal `make -f Makeile_<mechanism>` which will compile the KPP processor. You need to compile the Fortran code only once. After that, you can control and run the KPP using the driver.  
*  In `driver.py` find and set the proper values to the variables: 
   - `passf = "Data"`  # Folder where the data will be stored. 
   - `common_file_name_of_your_results = "mechanism"`  # Folder where the data will be stored. common_file_name_of_your_results = "dynho"
   - `KPP_run_file = "mechanism.exe"` name of the executable
   - `times = 1` Number of independent experiments you want to run in the loop.

## How to run

Due to the conflicts between gcc in Python and Fortran, we need to unload the gcc associated with Fortran. In the terminal, type `module purge` and then `module load python3`. Run the command `python driver.py` and enjoy.

# The Kinetic Pre-Processor (KPP)

NOTE ! The original KPP solver is [here](https://kpp.readthedocs.io/en/stable/getting_started/01_installation.html#download-kpp-from-github), we took version 3.0.2.


[![Latest
Release](https://img.shields.io/github/v/release/KineticPreProcessor/KPP?label=Latest%20Release)](https://kpp.readthedocs.io) [![License: GPL
v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://github.com/KineticPreProcessor/KPP/blob/main/LICENSE.txt) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7308373.svg)](https://doi.org/10.5281/zenodo.7308373) [![C-I
tests](https://img.shields.io/azure-devops/build/KineticPreProcessor/KPP/1/main?label=C-I%20Tests)](https://dev.azure.com/KineticPreProcessor/KPP/_build) [![ReadTheDocs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://kpp.readthedocs.io/en/latest) [![ReadTheDocs](https://img.shields.io/readthedocs/kpp?label=ReadTheDocs)](https://kpp.readthedocs.io)

This is the repository for the **The Kinetic PreProcessor (KPP)** source code.

## License
KPP is distributed under [GPLv3, the general public license](https://github.com/KineticPreProcessor/KPP/blob/main/LICENSE.txt).

## Documentation
Please see our comprehensive KPP User's Guide ([html](https://kpp.readthedocs.io), [pdf](https://kpp.readthedocs.io/_/downloads/en/latest/pdf/)) for installation and usage instructions, as well as references.

