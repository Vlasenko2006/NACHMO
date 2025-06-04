# Novel Atmospheric CHemistry MOdel (NACHMO)

## Introduction
Atmospheric chemistry plays a crucial role in influencing weather patterns and climate variability. The prediction of temperature fluctuations is particularly challenging due to the complex, nonlinear feedbacks—both short- and long-term—among greenhouse gases such as CO₂, methane, and other atmospheric constituents. Traditionally, modeling these interactions has required solving intricate mathematical and chemical equations, which is only feasible through numerical simulations on high-performance computing clusters and typically demands significant computational time.

Machine learning and neural networks now provide a powerful alternative, enabling the modeling of these feedbacks with far less computational overhead. Our neural network, NACHMO, leverages this advantage by integrating novel error-removal techniques, including singular value decomposition, quadratic programming, and gating layers. NACHMO accepts atmospheric species concentrations as input and predicts their evolution in the atmosphere over extended time periods, offering an efficient and innovative approach to atmospheric modeling.


## Example of NACHMO etimates:
Compare real(red) and NACHMO-estimated (other colours) concentrations under varions NN-modifications.


![Sample Output](https://github.com/Vlasenko2006/NACHMO/blob/main/Example_of_concentration_setimates.png)

## Repository content

 - KPP_python - python wrapper for Fortran-based Implicit Runge-Kutta Solver (we need it to create traing and validation sets)
 - kpp_dynho - Fortran-based Implicit Runge-Kutta Solver for Dynamic OH mechanism
 - kpp_verwer - Fortran-based Implicit Runge-Kutta Solver for Verwer mechanism
 - nachmo_mlp - Neural network computing chemical concentrations
 - NACHMO-F nachmo_mlp translated int Fortran to bridge it with Climate models like ICON
