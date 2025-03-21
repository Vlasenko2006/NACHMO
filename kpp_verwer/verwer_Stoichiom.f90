! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! The Stoichiometric Chemical Model File
! 
! Generated by KPP-2.2.1_rs5 symbolic chemistry Kinetics PreProcessor
!       (http://www.cs.vt.edu/~asandu/Software/KPP)
! KPP is distributed under GPL, the general public licence
!       (http://www.gnu.org/copyleft/gpl.html)
! (C) 1995-1997, V. Damian & A. Sandu, CGRER, Univ. Iowa
! (C) 1997-2005, A. Sandu, Michigan Tech, Virginia Tech
!     With important contributions from:
!        M. Damian, Villanova University, USA
!        R. Sander, Max-Planck Institute for Chemistry, Mainz, Germany
! 
! File                 : verwer_Stoichiom.f90
! Time                 : Wed Aug  2 21:06:50 2023
! Working directory    : /pfs/data5/home/kit/imk-tro/ii5664/kpp/verwer
! Equation file        : verwer.kpp
! Output root filename : verwer
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MODULE verwer_Stoichiom

  USE verwer_Parameters
  USE verwer_StoichiomSP

  IMPLICIT NONE

CONTAINS


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! ReactantProd - Reactant Products in each equation
!   Arguments :
!      V         - Concentrations of variable species (local)
!      F         - Concentrations of fixed species (local)
!      ARP       - Reactant product in each equation
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE ReactantProd ( V, F, ARP )

! V - Concentrations of variable species (local)
  REAL(kind=dp) :: V(NVAR)
! F - Concentrations of fixed species (local)
  REAL(kind=dp) :: F(NFIX)
! ARP - Reactant product in each equation
  REAL(kind=dp) :: ARP(NREACT)


! Reactant Products in each equation are useful in the
!     stoichiometric formulation of mass action law
  ARP(1) = V(17)
  ARP(2) = V(14)*V(18)
  ARP(3) = V(16)*V(18)
  ARP(4) = V(13)
  ARP(5) = V(13)
  ARP(6) = V(13)*V(20)
  ARP(7) = V(8)
  ARP(8) = V(8)*V(20)
  ARP(9) = V(15)*V(18)
  ARP(10) = V(15)*V(17)
  ARP(11) = V(9)
  ARP(12) = V(18)*V(19)
  ARP(13) = V(10)*F(2)
  ARP(14) = V(17)*V(20)
  ARP(15) = V(7)*F(1)*F(2)
  ARP(16) = V(14)
  ARP(17) = V(14)
  ARP(18) = V(5)*F(3)
  ARP(19) = V(5)*F(1)
  ARP(20) = V(6)*V(20)
  ARP(21) = V(12)
  ARP(22) = V(12)
  ARP(23) = V(14)*V(17)
  ARP(24) = V(12)*V(17)
  ARP(25) = V(11)
      
END SUBROUTINE ReactantProd

! End of ReactantProd function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! JacReactantProd - Jacobian of Reactant Products vector
!   Arguments :
!      V         - Concentrations of variable species (local)
!      F         - Concentrations of fixed species (local)
!      JVRP      - d ARP(1:NREACT)/d VAR (1:NVAR)
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE JacReactantProd ( V, F, JVRP )

! V - Concentrations of variable species (local)
  REAL(kind=dp) :: V(NVAR)
! F - Concentrations of fixed species (local)
  REAL(kind=dp) :: F(NFIX)
! JVRP - d ARP(1:NREACT)/d VAR (1:NVAR)
  REAL(kind=dp) :: JVRP(NJVRP)


! Reactant Products in each equation are useful in the
!    stoichiometric formulation of mass action law
! Below we compute the Jacobian of the Reactant Products vector
!    w.r.t. variable species: d ARP(1:NREACT) / d Var(1:NVAR)

! JVRP(1) = dARP(1)/dV(17)
  JVRP(1) = 1
! JVRP(2) = dARP(2)/dV(14)
  JVRP(2) = V(18)
! JVRP(3) = dARP(2)/dV(18)
  JVRP(3) = V(14)
! JVRP(4) = dARP(3)/dV(16)
  JVRP(4) = V(18)
! JVRP(5) = dARP(3)/dV(18)
  JVRP(5) = V(16)
! JVRP(6) = dARP(4)/dV(13)
  JVRP(6) = 1
! JVRP(7) = dARP(5)/dV(13)
  JVRP(7) = 1
! JVRP(8) = dARP(6)/dV(13)
  JVRP(8) = V(20)
! JVRP(9) = dARP(6)/dV(20)
  JVRP(9) = V(13)
! JVRP(10) = dARP(7)/dV(8)
  JVRP(10) = 1
! JVRP(11) = dARP(8)/dV(8)
  JVRP(11) = V(20)
! JVRP(12) = dARP(8)/dV(20)
  JVRP(12) = V(8)
! JVRP(13) = dARP(9)/dV(15)
  JVRP(13) = V(18)
! JVRP(14) = dARP(9)/dV(18)
  JVRP(14) = V(15)
! JVRP(15) = dARP(10)/dV(15)
  JVRP(15) = V(17)
! JVRP(16) = dARP(10)/dV(17)
  JVRP(16) = V(15)
! JVRP(17) = dARP(11)/dV(9)
  JVRP(17) = 1
! JVRP(18) = dARP(12)/dV(18)
  JVRP(18) = V(19)
! JVRP(19) = dARP(12)/dV(19)
  JVRP(19) = V(18)
! JVRP(20) = dARP(13)/dV(10)
  JVRP(20) = F(2)
! JVRP(21) = dARP(14)/dV(17)
  JVRP(21) = V(20)
! JVRP(22) = dARP(14)/dV(20)
  JVRP(22) = V(17)
! JVRP(23) = dARP(15)/dV(7)
  JVRP(23) = F(1)*F(2)
! JVRP(24) = dARP(16)/dV(14)
  JVRP(24) = 1
! JVRP(25) = dARP(17)/dV(14)
  JVRP(25) = 1
! JVRP(26) = dARP(18)/dV(5)
  JVRP(26) = F(3)
! JVRP(27) = dARP(19)/dV(5)
  JVRP(27) = F(1)
! JVRP(28) = dARP(20)/dV(6)
  JVRP(28) = V(20)
! JVRP(29) = dARP(20)/dV(20)
  JVRP(29) = V(6)
! JVRP(30) = dARP(21)/dV(12)
  JVRP(30) = 1
! JVRP(31) = dARP(22)/dV(12)
  JVRP(31) = 1
! JVRP(32) = dARP(23)/dV(14)
  JVRP(32) = V(17)
! JVRP(33) = dARP(23)/dV(17)
  JVRP(33) = V(14)
! JVRP(34) = dARP(24)/dV(12)
  JVRP(34) = V(17)
! JVRP(35) = dARP(24)/dV(17)
  JVRP(35) = V(12)
! JVRP(36) = dARP(25)/dV(11)
  JVRP(36) = 1
      
END SUBROUTINE JacReactantProd

! End of JacReactantProd function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



! Begin Derivative w.r.t. Rate Coefficients

! ------------------------------------------------------------------------------
! Subroutine for the derivative of Fun with respect to rate coefficients
! -----------------------------------------------------------------------------

      SUBROUTINE  dFun_dRcoeff( V, F, NCOEFF, JCOEFF, DFDR )
       
      USE verwer_Parameters
      USE verwer_StoichiomSP
      IMPLICIT NONE 

! V - Concentrations of variable/radical/fixed species            
      REAL(kind=dp) V(NVAR), F(NFIX)
! NCOEFF - the number of rate coefficients with respect to which we differentiate
      INTEGER NCOEFF       
! JCOEFF - a vector of integers containing the indices of reactions (rate
!          coefficients) with respect to which we differentiate
      INTEGER JCOEFF(NCOEFF)       
! DFDR  - a matrix containg derivative values; specifically, 
!         column j contains d Fun(1:NVAR) / d RCT( JCOEFF(j) )
!         for each 1 <= j <= NCOEFF
!         This matrix is stored in a column-wise linearized format
      REAL(kind=dp) DFDR(NVAR*NCOEFF)

! Local vector with reactant products
      REAL(kind=dp) A_RPROD(NREACT)
      REAL(kind=dp) aj
      INTEGER i,j,k
      
! Compute the reactant products of all reactions     
      CALL ReactantProd ( V, F, A_RPROD )

! Compute the derivatives by multiplying column JCOEFF(j) of the stoichiometric matrix with A_RPROD       
      DO j=1,NCOEFF
!                  Initialize the j-th column of derivative matrix to zero       
         DO i=1,NVAR
           DFDR(i+NVAR*(j-1)) = 0.0_dp 
         END DO
!                  Column JCOEFF(j) in the stoichiometric matrix times the
!                  reactant product  of the JCOEFF(j)-th reaction      
!                  give the j-th column of the derivative matrix   
         aj = A_RPROD(JCOEFF(j))
         DO k=CCOL_STOICM(JCOEFF(j)),CCOL_STOICM(JCOEFF(j)+1)-1
           DFDR(IROW_STOICM(k)+NVAR*(j-1)) = STOICM(k)*aj
         END DO
      END DO
      stop
      END SUBROUTINE  dFun_dRcoeff

! End Derivative w.r.t. Rate Coefficients


! Begin Jacobian Derivative w.r.t. Rate Coefficients

! ------------------------------------------------------------------------------
! Subroutine for the derivative of Jac with respect to rate coefficients
! Times a user vector
! -----------------------------------------------------------------------------

      SUBROUTINE  dJac_dRcoeff( V, F, U, NCOEFF, JCOEFF, DJDR )
       
      USE verwer_Parameters
      USE verwer_StoichiomSP
      IMPLICIT NONE 

! V - Concentrations of variable/fixed species            
      REAL(kind=dp) V(NVAR), F(NFIX)
! U - User-supplied Vector           
      REAL(kind=dp) U(NVAR)
! NCOEFF - the number of rate coefficients with respect to which we differentiate
      INTEGER NCOEFF       
! JCOEFF - a vector of integers containing the indices of reactions (rate
!          coefficients) with respect to which we differentiate
      INTEGER JCOEFF(NCOEFF)       
! DFDR  - a matrix containg derivative values; specifically, 
!         column j contains d Jac(1:NVAR) / d RCT( JCOEFF(j) ) * U
!                     for each 1 <= j <= NCOEFF
!         This matrix is stored in a column-wise linearized format
      REAL(kind=dp) DJDR(NVAR*NCOEFF)

! Local vector for Jacobian of reactant products
      REAL(kind=dp) JV_RPROD(NJVRP)
      REAL(kind=dp) aj
      INTEGER i,j,k
      
! Compute the Jacobian of all reactant products   
      CALL JacReactantProd( V, F, JV_RPROD )

! Compute the derivatives by multiplying column JCOEFF(j) of the stoichiometric matrix with A_PROD       
      DO j=1,NCOEFF
!                  Initialize the j-th column of derivative matrix to zero       
         DO i=1,NVAR
           DJDR(i+NVAR*(j-1)) = 0.0_dp
         END DO
!                  Column JCOEFF(j) in the stoichiometric matrix times the
!                  ( Gradient of reactant product of the JCOEFF(j)-th reaction X user vector )    
!                  give the j-th column of the derivative matrix   
!
!          Row JCOEFF(j) of JV_RPROD times the user vector
         aj = 0.0_dp
         DO k=CROW_JVRP(JCOEFF(j)),CROW_JVRP(JCOEFF(j)+1)-1
             aj = aj + JV_RPROD(k)*U(ICOL_JVRP(k))
         END DO
!          Column JCOEFF(j) of Stoichiom. matrix times aj         
         DO k=CCOL_STOICM(JCOEFF(j)),CCOL_STOICM(JCOEFF(j)+1)-1
           DJDR(IROW_STOICM(k)+NVAR*(j-1)) = STOICM(k)*aj
         END DO
      END DO
      stop
      END SUBROUTINE  dJac_dRcoeff

! End Jacobian Derivative w.r.t. Rate Coefficients


END MODULE verwer_Stoichiom

