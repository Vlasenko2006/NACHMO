! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Hessian File
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
! File                 : verwer_Hessian.f90
! Time                 : Wed Aug  2 21:06:50 2023
! Working directory    : /pfs/data5/home/kit/imk-tro/ii5664/kpp/verwer
! Equation file        : verwer.kpp
! Output root filename : verwer
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MODULE verwer_Hessian

  USE verwer_Parameters
  USE verwer_HessianSP

  IMPLICIT NONE

CONTAINS


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Hessian - function for Hessian (Jac derivative w.r.t. variables)
!   Arguments :
!      V         - Concentrations of variable species (local)
!      F         - Concentrations of fixed species (local)
!      RCT       - Rate constants (local)
!      HESS      - Hessian of Var (i.e. the 3-tensor d Jac / d Var)
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE Hessian ( V, F, RCT, HESS )

! V - Concentrations of variable species (local)
  REAL(kind=dp) :: V(NVAR)
! F - Concentrations of fixed species (local)
  REAL(kind=dp) :: F(NFIX)
! RCT - Rate constants (local)
  REAL(kind=dp) :: RCT(NREACT)
! HESS - Hessian of Var (i.e. the 3-tensor d Jac / d Var)
  REAL(kind=dp) :: HESS(NHESS)

! --------------------------------------------------------
! Note: HESS is represented in coordinate sparse format:
!       HESS(m) = d^2 f_i / dv_j dv_k = d Jac_{i,j} / dv_k
!       where i = IHESS_I(m), j = IHESS_J(m), k = IHESS_K(m).
! --------------------------------------------------------
! Note: d^2 f_i / dv_j dv_k = d^2 f_i / dv_k dv_j,
!       therefore only the terms d^2 f_i / dv_j dv_k
!       with j <= k are computed and stored in HESS.
! --------------------------------------------------------

! Local variables
! D2A - Second derivatives of equation rates
  REAL(kind=dp) :: D2A(11)

! Computation of the second derivatives of equation rates
! D2A(1) = d^2 A(2) / dV(14)dV(18)
  D2A(1) = 1.8e-20
! D2A(2) = d^2 A(3) / dV(16)dV(18)
  D2A(2) = 8.1e-18
! D2A(3) = d^2 A(6) / dV(13)dV(20)
  D2A(3) = 1e-17
! D2A(4) = d^2 A(8) / dV(8)dV(20)
  D2A(4) = 1.6e-17
! D2A(5) = d^2 A(9) / dV(15)dV(18)
  D2A(5) = 1.1e-17
! D2A(6) = d^2 A(10) / dV(15)dV(17)
  D2A(6) = 6.1e-18
! D2A(7) = d^2 A(12) / dV(18)dV(19)
  D2A(7) = 8.1e-18
! D2A(8) = d^2 A(14) / dV(17)dV(20)
  D2A(8) = 1.1e-17
! D2A(9) = d^2 A(20) / dV(6)dV(20)
  D2A(9) = 8.4e-19
! D2A(10) = d^2 A(23) / dV(14)dV(17)
  D2A(10) = 3.1e-23
! D2A(11) = d^2 A(24) / dV(12)dV(17)
  D2A(11) = 1.2e-18

! Computation of the Jacobian derivative
! HESS(1) = d^2 Vdot(1)/{dV(13)dV(20)} = d^2 Vdot(1)/{dV(20)dV(13)}
  HESS(1) = D2A(3)
! HESS(2) = d^2 Vdot(2)/{dV(17)dV(20)} = d^2 Vdot(2)/{dV(20)dV(17)}
  HESS(2) = D2A(8)
! HESS(3) = d^2 Vdot(3)/{dV(6)dV(20)} = d^2 Vdot(3)/{dV(20)dV(6)}
  HESS(3) = D2A(9)
! HESS(4) = d^2 Vdot(4)/{dV(15)dV(18)} = d^2 Vdot(4)/{dV(18)dV(15)}
  HESS(4) = D2A(5)
! HESS(5) = d^2 Vdot(6)/{dV(6)dV(20)} = d^2 Vdot(6)/{dV(20)dV(6)}
  HESS(5) = -D2A(9)
! HESS(6) = d^2 Vdot(8)/{dV(8)dV(20)} = d^2 Vdot(8)/{dV(20)dV(8)}
  HESS(6) = -D2A(4)
! HESS(7) = d^2 Vdot(9)/{dV(15)dV(17)} = d^2 Vdot(9)/{dV(17)dV(15)}
  HESS(7) = D2A(6)
! HESS(8) = d^2 Vdot(10)/{dV(18)dV(19)} = d^2 Vdot(10)/{dV(19)dV(18)}
  HESS(8) = D2A(7)
! HESS(9) = d^2 Vdot(11)/{dV(12)dV(17)} = d^2 Vdot(11)/{dV(17)dV(12)}
  HESS(9) = D2A(11)
! HESS(10) = d^2 Vdot(12)/{dV(12)dV(17)} = d^2 Vdot(12)/{dV(17)dV(12)}
  HESS(10) = -D2A(11)
! HESS(11) = d^2 Vdot(12)/{dV(14)dV(17)} = d^2 Vdot(12)/{dV(17)dV(14)}
  HESS(11) = D2A(10)
! HESS(12) = d^2 Vdot(13)/{dV(13)dV(20)} = d^2 Vdot(13)/{dV(20)dV(13)}
  HESS(12) = -D2A(3)
! HESS(13) = d^2 Vdot(14)/{dV(14)dV(17)} = d^2 Vdot(14)/{dV(17)dV(14)}
  HESS(13) = -D2A(10)
! HESS(14) = d^2 Vdot(14)/{dV(14)dV(18)} = d^2 Vdot(14)/{dV(18)dV(14)}
  HESS(14) = -D2A(1)
! HESS(15) = d^2 Vdot(15)/{dV(8)dV(20)} = d^2 Vdot(15)/{dV(20)dV(8)}
  HESS(15) = D2A(4)
! HESS(16) = d^2 Vdot(15)/{dV(15)dV(17)} = d^2 Vdot(15)/{dV(17)dV(15)}
  HESS(16) = -D2A(6)
! HESS(17) = d^2 Vdot(15)/{dV(15)dV(18)} = d^2 Vdot(15)/{dV(18)dV(15)}
  HESS(17) = -D2A(5)
! HESS(18) = d^2 Vdot(16)/{dV(6)dV(20)} = d^2 Vdot(16)/{dV(20)dV(6)}
  HESS(18) = D2A(9)
! HESS(19) = d^2 Vdot(16)/{dV(13)dV(20)} = d^2 Vdot(16)/{dV(20)dV(13)}
  HESS(19) = D2A(3)
! HESS(20) = d^2 Vdot(16)/{dV(16)dV(18)} = d^2 Vdot(16)/{dV(18)dV(16)}
  HESS(20) = -D2A(2)
! HESS(21) = d^2 Vdot(17)/{dV(12)dV(17)} = d^2 Vdot(17)/{dV(17)dV(12)}
  HESS(21) = -D2A(11)
! HESS(22) = d^2 Vdot(17)/{dV(14)dV(17)} = d^2 Vdot(17)/{dV(17)dV(14)}
  HESS(22) = -D2A(10)
! HESS(23) = d^2 Vdot(17)/{dV(14)dV(18)} = d^2 Vdot(17)/{dV(18)dV(14)}
  HESS(23) = D2A(1)
! HESS(24) = d^2 Vdot(17)/{dV(15)dV(17)} = d^2 Vdot(17)/{dV(17)dV(15)}
  HESS(24) = -D2A(6)
! HESS(25) = d^2 Vdot(17)/{dV(15)dV(18)} = d^2 Vdot(17)/{dV(18)dV(15)}
  HESS(25) = D2A(5)
! HESS(26) = d^2 Vdot(17)/{dV(16)dV(18)} = d^2 Vdot(17)/{dV(18)dV(16)}
  HESS(26) = D2A(2)
! HESS(27) = d^2 Vdot(17)/{dV(17)dV(20)} = d^2 Vdot(17)/{dV(20)dV(17)}
  HESS(27) = -D2A(8)
! HESS(28) = d^2 Vdot(17)/{dV(18)dV(19)} = d^2 Vdot(17)/{dV(19)dV(18)}
  HESS(28) = D2A(7)
! HESS(29) = d^2 Vdot(18)/{dV(14)dV(18)} = d^2 Vdot(18)/{dV(18)dV(14)}
  HESS(29) = -D2A(1)
! HESS(30) = d^2 Vdot(18)/{dV(15)dV(18)} = d^2 Vdot(18)/{dV(18)dV(15)}
  HESS(30) = -D2A(5)
! HESS(31) = d^2 Vdot(18)/{dV(16)dV(18)} = d^2 Vdot(18)/{dV(18)dV(16)}
  HESS(31) = -D2A(2)
! HESS(32) = d^2 Vdot(18)/{dV(18)dV(19)} = d^2 Vdot(18)/{dV(19)dV(18)}
  HESS(32) = -D2A(7)
! HESS(33) = d^2 Vdot(19)/{dV(15)dV(18)} = d^2 Vdot(19)/{dV(18)dV(15)}
  HESS(33) = D2A(5)
! HESS(34) = d^2 Vdot(19)/{dV(18)dV(19)} = d^2 Vdot(19)/{dV(19)dV(18)}
  HESS(34) = -D2A(7)
! HESS(35) = d^2 Vdot(20)/{dV(6)dV(20)} = d^2 Vdot(20)/{dV(20)dV(6)}
  HESS(35) = -D2A(9)
! HESS(36) = d^2 Vdot(20)/{dV(8)dV(20)} = d^2 Vdot(20)/{dV(20)dV(8)}
  HESS(36) = -D2A(4)
! HESS(37) = d^2 Vdot(20)/{dV(13)dV(20)} = d^2 Vdot(20)/{dV(20)dV(13)}
  HESS(37) = -D2A(3)
! HESS(38) = d^2 Vdot(20)/{dV(16)dV(18)} = d^2 Vdot(20)/{dV(18)dV(16)}
  HESS(38) = D2A(2)
! HESS(39) = d^2 Vdot(20)/{dV(17)dV(20)} = d^2 Vdot(20)/{dV(20)dV(17)}
  HESS(39) = -D2A(8)
      
END SUBROUTINE Hessian

! End of Hessian function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! HessTR_Vec - Hessian transposed times user vectors
!   Arguments :
!      HESS      - Hessian of Var (i.e. the 3-tensor d Jac / d Var)
!      U1        - User vector
!      U2        - User vector
!      HTU       - Transposed Hessian times user vectors: (Hess x U2)^T * U1 = [d (Jac^T*U1)/d Var] * U2
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE HessTR_Vec ( HESS, U1, U2, HTU )

! HESS - Hessian of Var (i.e. the 3-tensor d Jac / d Var)
  REAL(kind=dp) :: HESS(NHESS)
! U1 - User vector
  REAL(kind=dp) :: U1(NVAR)
! U2 - User vector
  REAL(kind=dp) :: U2(NVAR)
! HTU - Transposed Hessian times user vectors: (Hess x U2)^T * U1 = [d (Jac^T*U1)/d Var] * U2
  REAL(kind=dp) :: HTU(NVAR)

! Compute the vector HTU =(Hess x U2)^T * U1 = d (Jac^T*U1)/d Var * U2
  HTU(1) = 0
  HTU(2) = 0
  HTU(3) = 0
  HTU(4) = 0
  HTU(5) = 0
  HTU(6) = HESS(3)*(U1(3)*U2(20))+HESS(5)*(U1(6)*U2(20))+HESS(18)*(U1(16)*U2(20))+HESS(35)*(U1(20)*U2(20))
  HTU(7) = 0
  HTU(8) = HESS(6)*(U1(8)*U2(20))+HESS(15)*(U1(15)*U2(20))+HESS(36)*(U1(20)*U2(20))
  HTU(9) = 0
  HTU(10) = 0
  HTU(11) = 0
  HTU(12) = HESS(9)*(U1(11)*U2(17))+HESS(10)*(U1(12)*U2(17))+HESS(21)*(U1(17)*U2(17))
  HTU(13) = HESS(1)*(U1(1)*U2(20))+HESS(12)*(U1(13)*U2(20))+HESS(19)*(U1(16)*U2(20))+HESS(37)*(U1(20)*U2(20))
  HTU(14) = HESS(11)*(U1(12)*U2(17))+HESS(13)*(U1(14)*U2(17))+HESS(14)*(U1(14)*U2(18))+HESS(22)*(U1(17)*U2(17))+HESS(23)&
              &*(U1(17)*U2(18))+HESS(29)*(U1(18)*U2(18))
  HTU(15) = HESS(4)*(U1(4)*U2(18))+HESS(7)*(U1(9)*U2(17))+HESS(16)*(U1(15)*U2(17))+HESS(17)*(U1(15)*U2(18))+HESS(24)&
              &*(U1(17)*U2(17))+HESS(25)*(U1(17)*U2(18))+HESS(30)*(U1(18)*U2(18))+HESS(33)*(U1(19)*U2(18))
  HTU(16) = HESS(20)*(U1(16)*U2(18))+HESS(26)*(U1(17)*U2(18))+HESS(31)*(U1(18)*U2(18))+HESS(38)*(U1(20)*U2(18))
  HTU(17) = HESS(2)*(U1(2)*U2(20))+HESS(7)*(U1(9)*U2(15))+HESS(9)*(U1(11)*U2(12))+HESS(10)*(U1(12)*U2(12))+HESS(11)&
              &*(U1(12)*U2(14))+HESS(13)*(U1(14)*U2(14))+HESS(16)*(U1(15)*U2(15))+HESS(21)*(U1(17)*U2(12))+HESS(22)*(U1(17)&
              &*U2(14))+HESS(24)*(U1(17)*U2(15))+HESS(27)*(U1(17)*U2(20))+HESS(39)*(U1(20)*U2(20))
  HTU(18) = HESS(4)*(U1(4)*U2(15))+HESS(8)*(U1(10)*U2(19))+HESS(14)*(U1(14)*U2(14))+HESS(17)*(U1(15)*U2(15))+HESS(20)&
              &*(U1(16)*U2(16))+HESS(23)*(U1(17)*U2(14))+HESS(25)*(U1(17)*U2(15))+HESS(26)*(U1(17)*U2(16))+HESS(28)*(U1(17)&
              &*U2(19))+HESS(29)*(U1(18)*U2(14))+HESS(30)*(U1(18)*U2(15))+HESS(31)*(U1(18)*U2(16))+HESS(32)*(U1(18)*U2(19))&
              &+HESS(33)*(U1(19)*U2(15))+HESS(34)*(U1(19)*U2(19))+HESS(38)*(U1(20)*U2(16))
  HTU(19) = HESS(8)*(U1(10)*U2(18))+HESS(28)*(U1(17)*U2(18))+HESS(32)*(U1(18)*U2(18))+HESS(34)*(U1(19)*U2(18))
  HTU(20) = HESS(1)*(U1(1)*U2(13))+HESS(2)*(U1(2)*U2(17))+HESS(3)*(U1(3)*U2(6))+HESS(5)*(U1(6)*U2(6))+HESS(6)*(U1(8)&
              &*U2(8))+HESS(12)*(U1(13)*U2(13))+HESS(15)*(U1(15)*U2(8))+HESS(18)*(U1(16)*U2(6))+HESS(19)*(U1(16)*U2(13))&
              &+HESS(27)*(U1(17)*U2(17))+HESS(35)*(U1(20)*U2(6))+HESS(36)*(U1(20)*U2(8))+HESS(37)*(U1(20)*U2(13))+HESS(39)&
              &*(U1(20)*U2(17))
      
END SUBROUTINE HessTR_Vec

! End of HessTR_Vec function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Hess_Vec - Hessian times user vectors
!   Arguments :
!      HESS      - Hessian of Var (i.e. the 3-tensor d Jac / d Var)
!      U1        - User vector
!      U2        - User vector
!      HU        - Hessian times user vectors: (Hess x U2) * U1 = [d (Jac*U1)/d Var] * U2
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE Hess_Vec ( HESS, U1, U2, HU )

! HESS - Hessian of Var (i.e. the 3-tensor d Jac / d Var)
  REAL(kind=dp) :: HESS(NHESS)
! U1 - User vector
  REAL(kind=dp) :: U1(NVAR)
! U2 - User vector
  REAL(kind=dp) :: U2(NVAR)
! HU - Hessian times user vectors: (Hess x U2) * U1 = [d (Jac*U1)/d Var] * U2
  REAL(kind=dp) :: HU(NVAR)

! Compute the vector HU =(Hess x U2) * U1 = d (Jac*U1)/d Var * U2
  HU(1) = HESS(1)*(U1(13)*U2(20))+HESS(1)*(U1(20)*U2(13))
  HU(2) = HESS(2)*(U1(17)*U2(20))+HESS(2)*(U1(20)*U2(17))
  HU(3) = HESS(3)*(U1(6)*U2(20))+HESS(3)*(U1(20)*U2(6))
  HU(4) = HESS(4)*(U1(15)*U2(18))+HESS(4)*(U1(18)*U2(15))
  HU(5) = 0
  HU(6) = HESS(5)*(U1(6)*U2(20))+HESS(5)*(U1(20)*U2(6))
  HU(7) = 0
  HU(8) = HESS(6)*(U1(8)*U2(20))+HESS(6)*(U1(20)*U2(8))
  HU(9) = HESS(7)*(U1(15)*U2(17))+HESS(7)*(U1(17)*U2(15))
  HU(10) = HESS(8)*(U1(18)*U2(19))+HESS(8)*(U1(19)*U2(18))
  HU(11) = HESS(9)*(U1(12)*U2(17))+HESS(9)*(U1(17)*U2(12))
  HU(12) = HESS(10)*(U1(12)*U2(17))+HESS(10)*(U1(17)*U2(12))+HESS(11)*(U1(14)*U2(17))+HESS(11)*(U1(17)*U2(14))
  HU(13) = HESS(12)*(U1(13)*U2(20))+HESS(12)*(U1(20)*U2(13))
  HU(14) = HESS(13)*(U1(14)*U2(17))+HESS(13)*(U1(17)*U2(14))+HESS(14)*(U1(14)*U2(18))+HESS(14)*(U1(18)*U2(14))
  HU(15) = HESS(15)*(U1(8)*U2(20))+HESS(15)*(U1(20)*U2(8))+HESS(16)*(U1(15)*U2(17))+HESS(16)*(U1(17)*U2(15))+HESS(17)&
             &*(U1(15)*U2(18))+HESS(17)*(U1(18)*U2(15))
  HU(16) = HESS(18)*(U1(6)*U2(20))+HESS(18)*(U1(20)*U2(6))+HESS(19)*(U1(13)*U2(20))+HESS(19)*(U1(20)*U2(13))+HESS(20)&
             &*(U1(16)*U2(18))+HESS(20)*(U1(18)*U2(16))
  HU(17) = HESS(21)*(U1(12)*U2(17))+HESS(21)*(U1(17)*U2(12))+HESS(22)*(U1(14)*U2(17))+HESS(22)*(U1(17)*U2(14))+HESS(23)&
             &*(U1(14)*U2(18))+HESS(23)*(U1(18)*U2(14))+HESS(24)*(U1(15)*U2(17))+HESS(24)*(U1(17)*U2(15))+HESS(25)*(U1(15)&
             &*U2(18))+HESS(25)*(U1(18)*U2(15))+HESS(26)*(U1(16)*U2(18))+HESS(26)*(U1(18)*U2(16))+HESS(27)*(U1(17)*U2(20))&
             &+HESS(27)*(U1(20)*U2(17))+HESS(28)*(U1(18)*U2(19))+HESS(28)*(U1(19)*U2(18))
  HU(18) = HESS(29)*(U1(14)*U2(18))+HESS(29)*(U1(18)*U2(14))+HESS(30)*(U1(15)*U2(18))+HESS(30)*(U1(18)*U2(15))+HESS(31)&
             &*(U1(16)*U2(18))+HESS(31)*(U1(18)*U2(16))+HESS(32)*(U1(18)*U2(19))+HESS(32)*(U1(19)*U2(18))
  HU(19) = HESS(33)*(U1(15)*U2(18))+HESS(33)*(U1(18)*U2(15))+HESS(34)*(U1(18)*U2(19))+HESS(34)*(U1(19)*U2(18))
  HU(20) = HESS(35)*(U1(6)*U2(20))+HESS(35)*(U1(20)*U2(6))+HESS(36)*(U1(8)*U2(20))+HESS(36)*(U1(20)*U2(8))+HESS(37)&
             &*(U1(13)*U2(20))+HESS(37)*(U1(20)*U2(13))+HESS(38)*(U1(16)*U2(18))+HESS(38)*(U1(18)*U2(16))+HESS(39)*(U1(17)&
             &*U2(20))+HESS(39)*(U1(20)*U2(17))
      
END SUBROUTINE Hess_Vec

! End of Hess_Vec function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



END MODULE verwer_Hessian

