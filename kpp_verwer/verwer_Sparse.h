C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
C                                                                  
C Sparse Data Header File                                          
C                                                                  
C Generated by KPP-2.2.1_rs5 symbolic chemistry Kinetics PreProcessor
C       (http://www.cs.vt.edu/~asandu/Software/KPP)                
C KPP is distributed under GPL, the general public licence         
C       (http://www.gnu.org/copyleft/gpl.html)                     
C (C) 1995-1997, V. Damian & A. Sandu, CGRER, Univ. Iowa           
C (C) 1997-2005, A. Sandu, Michigan Tech, Virginia Tech            
C     With important contributions from:                           
C        M. Damian, Villanova University, USA                      
C        R. Sander, Max-Planck Institute for Chemistry, Mainz, Germany
C                                                                  
C File                 : verwer_Sparse.h                           
C Time                 : Mon Jul 31 15:37:30 2023                  
C Working directory    : /pfs/data5/home/kit/imk-tro/ii5664/kpp/verwer
C Equation file        : verwer.kpp                                
C Output root filename : verwer                                    
C                                                                  
C ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




C  ----------> Sparse Jacobian Data                                

C LU_IROW - Row indexes of the LU Jacobian of variables            
      INTEGER LU_IROW(95)
      COMMON /SDATA/ LU_IROW
C LU_ICOL - Column indexes of the LU Jacobian of variables         
      INTEGER LU_ICOL(95)
      COMMON /SDATA/ LU_ICOL
C LU_CROW - Compressed row indexes of the LU Jacobian of variables 
      INTEGER LU_CROW(21)
      COMMON /SDATA/ LU_CROW
C LU_DIAG - Diagonal indexes of the LU Jacobian of variables       
      INTEGER LU_DIAG(21)
      COMMON /SDATA/ LU_DIAG


C  ----------> Sparse Hessian Data                                 

C IHESS_I - Index i of Hessian element d^2 f_i/dv_j.dv_k           
      INTEGER IHESS_I(39)
      COMMON /HESSDATA/ IHESS_I
C IHESS_J - Index j of Hessian element d^2 f_i/dv_j.dv_k           
      INTEGER IHESS_J(39)
      COMMON /HESSDATA/ IHESS_J
C IHESS_K - Index k of Hessian element d^2 f_i/dv_j.dv_k           
      INTEGER IHESS_K(39)
      COMMON /HESSDATA/ IHESS_K


C  ----------> Sparse Stoichiometric Matrix                        

C STOICM - Stoichiometric Matrix in compressed column format       
      REAL*8 STOICM(75)
      COMMON /STOICM_VALUES/ STOICM
C IROW_STOICM - Row indices in STOICM                              
      INTEGER IROW_STOICM(75)
      COMMON /STOICM_DATA/ IROW_STOICM
C CCOL_STOICM - Beginning of columns in STOICM                     
      INTEGER CCOL_STOICM(26)
      COMMON /STOICM_DATA/ CCOL_STOICM
C ICOL_STOICM - Column indices in STOICM                           
      INTEGER ICOL_STOICM(75)
      COMMON /STOICM_DATA/ ICOL_STOICM


C  ----------> Sparse Data for Jacobian of Reactant Products       

C ICOL_JVRP - Column indices in JVRP                               
      INTEGER ICOL_JVRP(36)
      COMMON /JVRP/ ICOL_JVRP
C IROW_JVRP - Row indices in JVRP                                  
      INTEGER IROW_JVRP(36)
      COMMON /JVRP/ IROW_JVRP
C CROW_JVRP - Beginning of rows in JVRP                            
      INTEGER CROW_JVRP(26)
      COMMON /JVRP/ CROW_JVRP

