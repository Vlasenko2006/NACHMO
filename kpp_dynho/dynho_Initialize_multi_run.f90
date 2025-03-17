! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Initialization File
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
! File                 : dynho_Initialize.f90
! Time                 : Wed Feb  1 13:13:53 2023
! Working directory    : /pfs/data5/home/kit/imk-tro/ii5664/kpp/simple_dynH
! Equation file        : dynho.kpp
! Output root filename : dynho
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MODULE dynho_Initialize

  USE dynho_Parameters, ONLY: dp, NVAR, NFIX
  IMPLICIT NONE

CONTAINS


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Initialize - function to initialize concentrations
!   Arguments :
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE Initialize ()


  USE dynho_Global

  INTEGER :: i
  REAL(kind=dp) :: x
  REAL(kind=dp) :: rand_oh, rand_ho2, rand_h2o2
!  CFACTOR = 2.550000e+10_dp ! temp 15 deg = 288.15 K at 1. atm
  CFACTOR = 2.460000e+10_dp  ! temp = 25 deg 298 K

  x = (1.0E-10)*CFACTOR
  DO i = 1, NVAR
    VAR(i) = x
  END DO

  x = (1.0E-10)*CFACTOR
  DO i = 1, NFIX
    FIX(i) = x
  END DO


 call random_number(rand_h2o2)
 call random_number(rand_ho2)
 call random_number(rand_oh)



  VAR(1) = 2.0e-3*CFACTOR * rand_oh
  VAR(2) = 2.0e-1*CFACTOR * rand_ho2
  VAR(3) = (2.0e+3)*CFACTOR * rand_h2o2 


!  VAR(1) = (0.0)*CFACTOR
!  VAR(2) = (0.0)*CFACTOR
!  VAR(3) = (1.0e+3)*CFACTOR
  FIX(1) = (6.0e+3)*CFACTOR
  FIX(2) = (2.1e+4)*CFACTOR
! constant rate coefficients
  RCONST(2) = 1.7e-12
  RCONST(3) = 3.1e-12
  RCONST(4) = 1.1e-10
! END constant rate coefficients

! INLINED initializations

        TSTART = 0
        TEND = TSTART + 60     !A.V. Start-End time in sec 
        DT =  1.D-2    ! output frequency in sec
        TEMP = 298.15 !288.15

! End INLINED initializations

      
END SUBROUTINE Initialize

! End of Initialize function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



END MODULE dynho_Initialize

