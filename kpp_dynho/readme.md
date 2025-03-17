# Disclamer

This software is provided "AS IS" without warranties of any kind (MIT License?)


# In a nutshell


This software represents the KPP solver for dynamic OH and Verwer mechanisms. It consists of the original KPP engine for computing chemical processes in the atmosphere distributed by the official [developer](https://kpp.readthedocs.io/en/stable/index.html) and our files containing reactions of dynamic OH and Verwer mechanisms (see below for more).   

# Requirements

The installation process described below is subject to the Levante HPC, located at the Deutche Klima Rechnung Zentrum. We anticipate that it is similar to other HPCs. For the personal computers, like desktops or laptops this procedure might slightly differ. In any case, for installation you need 

1. Linux machine
2. gcc, including fortran compiler
3. spack (this is a smart installer of the scientific software)
4. bison, gcc distribution (a code parser that modifies and merges parts of your code in accordance to the instructions) 
5. Python(???)

During the installation you download fortran, c++ and python(?) files from the original git repository. Fortran files contain pieces of the code, representing KPP solver for an abstract chemical mechanism. To make is specific to your mechanism, you must type down in the special way the reactions you whant to compute (see example below), and then start the `bison` parser. This parser analyses your reactions converts them into the fortran code and mergers it with the downloaded pieces of the fortran code. C++ files contain the instructions for the bison parser how to merge fiels. At the end you get the KPP solver specific to your reactions.        



# Installation of the KPP solver for OH and Verwer mechanisms

The installation process described below is subject to the Levante HPC, located at the Deutche Klima Rechnung Zentrum. The KPP engine 

## 1. Preliminary steps

**!!!! Important !!!!** All steps, described here, are valid for the date 04.12.2023

Upload and install the bison parser by compleeting the following sequence in your terminal:
- log in to **levante.dkrz.de** and goto home directory
- `bash`
- `module purge`
- `spack unload`
- `module load gcc`
- `spack load bison@3.8.2%gcc`
- `export KPP_FLEX_LIB_DIR=/usr/lib64`

If you are installing KPP on a personal computer, you may not need the `module` system. Then, install 

 To find bison package type a command `spack find -l bison`, which displays you all bison packages on Levante. Take the gcc distribution and type `spack load <current gcc version>`, i.e.,  `spack load bison@3.8.2%gcc`. You have to do this once.


## 2. Clone KPP-engine from the repository:
- `module load git`
- `git clone https://github.com/KineticPreProcessor/KPP.git`

KPP-engine contains all necessary subroutines to compute chemical reactions, like various versions of Rosenbrock solvers etc. These files are not specified to compute reactions yet. To enable reactions computaion you have to provide the reactions themselves, the corresponding chemical rates and the initial condiions. Using this information, the KPP-wrapper (described in the next section) will adjust your engine to compute the required reactions.

## 3. Install the KPP-wrapper:
- `cd KPP/`
- Open `$HOME/.bashrc` with your beloved text editor, and insert the following lines there:

        export KPP_HOME=$HOME/KPP
        export PATH=$PATH:$KPP_HOME/bin 
- type `source    $HOME/.bashrc`
- cd `$KPP_HOME/src`
- type `make`
- type `which kpp`. You should get something like `~/KPP/bin/kpp`

KPP-wrapper builds KPP-engine from fortran files, by merging them, inserting the required ICs, reaction rates and so on. In the output of KPP_wrapper you have the fortran code, computing reactions, specified by you. 

## How to run Dynamic OH and Verwer Systems
Cloning nachmo repository you get necessary Fortran files and the corresponding Makefile.  Both chemical mechanisms were tested compiled and run on 24.02.2025 on Levnte HPC under `intel-oneapi-compilers/2023.2.1-gcc-11.2.0` compiler. However, instant Fortran and bison updates are full of surprises....no guarantee that chemical mechanisms will work with other Fortran versions. 

Clone the Nachmo repo. Run in your terminal

```
chmod -R 777 .
```
and then 
```
module load intel-oneapi-compilers/2023.2.1-gcc-11.2.0
```
then goto `kpp_dynho` folder and run:
```
make -f Makefile_dynho
```
After compilation finishes, run `./dynho.exe`.


To run Verwer mechanism goto folder `kpp_verwer` and run in the terminal
```
make -f Makefile_verwer
```
After compilation finishes, run `./verwer.exe`. 



## Prepare and run your own model(at your own risk).

An example of how to prepare and run your own model for the NOx-O3 cycle is given [here](https://kpp.readthedocs.io/en/stable/getting_started/02_running_kpp_sample_mech.html#). Below er show how to build it on the example of Dynamic OH.  


 We focus on the chemical reactions relevant to our study. Omitting the details of preparing `.eqn` and the rest of the necessary files, we provide the path to the source. From this source, you can download them. To run our experimental setup, do the following:

- Create a `dynho` directoy in your $`HOME/KPP/` and `cd` to `dynho`
- Copy `../models/atoms.kpp` to `dynho` folder. This file containst the periodic table in the kpp format.  
- Upload files from ....(Issue to Ali for providing storage for an eperimental setup files )
- Create an empty file`dynho.kpp` and paste there the code: 
```
#MODEL      dynho
#INTEGRATOR rosenbrock 
#LANGUAGE   Fortran90
#DRIVER     general
#JACOBIAN   SPARSE_LU_ROW
#HESSIAN    on
#STOICMAT   on
#DOUBLE     on
```
-  Create a file `dynho.eqn`. Open an empty file and paste:

```
#EQUATIONS
{1} H2O2 + hv = 2 OH                :  3.0E-5 ;
{2} H2O2 + OH = HO2 + H2O           :  1.7E-12 ;
{3} HO2 + HO2 = H2O2 + O2           :  3.1E-12  ;
{4} OH + HO2 = H2O + O2 		:  1.1E-10 ;
```
- Save the file under the name `dynho.eqn`. This file contains reactions and reaction rates for KPP processor

- Create a file `dynho.def` specifying ICs, output time step an the start-end time of simulation,type of solvers and fortran compiler. A sample of such file for dynamic oh given below, copy the sample and save it as `dynho.def`. If you want to change ICs, find a string, containing `{Variable species}` and chacnge the initial concentrations for the corresponding species. Note that units are in ppb. To change the computational period, output timestepping, find the line `#INLINE F90_INIT`  and change the varriables `TSTART`, `TEND` , `DT` and `TEMP` corresponding to the starting time, end of computations, output time steppin and temperature respecively. 



 Open an empty file and paste:

```
#include dynho.spc
#include dynho.eqn

{#LANGUAGE Fortran90}
{#INTEGRATOR ros3_old}
#INTEGRATOR rosenbrock
#DRIVER general}

{#LOOKATALL}
#LOOKAT H2O2; HO2; OH;

#MONITOR H2O2; HO2; OH;



#INITVALUES
  CFACTOR = 2.46E+10; {ppb-to-mcm}
  ALL_SPEC= 1.0E-10;
{Variable species}
  HO2        = 2.0e-1 ; {ppb units}
  H2O2       = 2.0e+3 ;
  OH         = 2.0e-3 ;
  H2O        = 6.0e+3 ;
  O2         = 2.1e+4 ;

#INLINE F90_GLOBAL
	REAL(dp) :: RAD
#ENDINLINE



#ENDINLINE
```

- Save the file under the name `dynho.def`.
- Create `dynho.spc` file by pasting in the empty file  the following text:

```
#include atoms.kpp

#DEFVAR
    OH          = O + H ;       {hydroxyl radical}
    HO2         = H + 2O ;      {perhydroxyl radical}
    H2O2        = 2H + 2O ;     {hydrogen peroxide}

#DEFFIX
    H2O         = 2H + O ;      {water}
    O2          = 2O ;          {molecular oxygen}
```

- Save it by giving the corresponding name. Fortran does not know anything about species, it knows only individual atoms from the file's atoms. kpp`. This file specifies reacting species defining their content of separate atoms.

- goto `KPP/drv/general.f90` , between the comment `!~~~> Call the integrator` ,see line 53, and `CALL INTEGRATE(` add the code `ICNTRL(3)=2` . This is the switch for Ros3 solver. If you need another solver, goto `KPP/int/rosenbrock.f90` and see the fortran comments in the head of the file for the list of available solvers. 

- goto `KPP/dynho/` and run the command `kpp dynho.kpp`

Not mandatory: to check that the right solver is calles, open `dynho_Integrate.f90`, find the subroutine `SUBROUTINE Rosenbrock` and put right after variable declaration a print satetment `print *, "ICNTRL(3) = ", ICNTRL(3)`.Ater compiling (see below), run the executable and find the print satements `ICNTRL(3) = 2`. If you do not see it, something is wrong.  **!!!Important!!!** Remove this print statement from the code, if you intend to create the data. Print statements slow down your simulations!  

run the command `make -f Makefile` that creates **dynho.exe** executable. Run the command `./dynho.exe` and enjoy.  

