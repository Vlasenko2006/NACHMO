# Novel Atmospheric CHemistry MOdel (NACHMO)

## Introduction
Atmospheric chemistry significantly impacts weather and climate variability. Modeling these processes requires solving stiff chemical kinetic ordinary differential equations (ODEs), which can consume up to 80% of computational time. Machine learning and neural networks offer a way to reduce these computational costs. Our Neural Network NACHMO combines this with novel error-removal techniques using singular value decomposition, quadratic programming, and gating layers. These methods extend the period of low-error estimates and are applicable to any trained NN or stiff ODE emulator.

## Example of NACHMO etimates:
Compare real(red) and NACHMO-estimated (other colours) concentrations under varions NN-modifications.


![Sample Output](https://github.com/Vlasenko2006/NACHMO/blob/main/Example_of_concentration_setimates.png)

## Repository content

 - KPP_python - python wrapper for Fortran-based Implicit Runge-Kutta Solver (we need it to create traing and validation sets)
 - kpp_dynho - Fortran-based Implicit Runge-Kutta Solver for Dynamic OH mechanism
 - kpp_verwer - Fortran-based Implicit Runge-Kutta Solver for Verwer mechanism
 - nachmo_mlp - Neural network computing chemical concentrations
 - NACHMO-F nachmo_mlp translated int Fortran to bridge it with Climate models like ICON
