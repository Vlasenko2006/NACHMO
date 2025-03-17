MODULE verwer_Model

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!  Completely defines the model verwer
!    by using all the associated modules
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  USE verwer_Precision
  USE verwer_Parameters
  USE verwer_Global
  USE verwer_Function
  USE verwer_Integrator
  USE verwer_Rates
  USE verwer_Jacobian
  USE verwer_Hessian
  USE verwer_Stoichiom
  USE verwer_LinearAlgebra
  USE verwer_Monitor
  USE verwer_Util

END MODULE verwer_Model

