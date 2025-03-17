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
! File                 : verwer_Initialize.f90
! Time                 : Wed Aug  2 21:06:50 2023
! Working directory    : /pfs/data5/home/kit/imk-tro/ii5664/kpp/verwer
! Equation file        : verwer.kpp
! Output root filename : verwer
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MODULE verwer_Initialize

  USE verwer_Parameters, ONLY: dp, NVAR, NFIX
  IMPLICIT NONE

CONTAINS


! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Initialize - function to initialize concentrations
!   Arguments :
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUBROUTINE Initialize (  )


  USE verwer_Global

  INTEGER :: i
  REAL(kind=dp) :: x, factor, rn(6)

  CFACTOR = 2.460000e+10_dp

  x = (0)*CFACTOR
  DO i = 1, NVAR
    VAR(i) = x
  END DO

  x = (0)*CFACTOR
  DO i = 1, NFIX
    FIX(i) = x
  END DO


  call random_number(rn)

  !rn = rn*0.2

  VAR(1) = (300)*CFACTOR *(rn(1)+0)
  VAR(6) = (7)*CFACTOR *(rn(2)+0)
  VAR(8) = (10)*CFACTOR *(rn(3)+0)
  VAR(13) = (100)*CFACTOR *(rn(4)+0)
  VAR(14) = (40)*CFACTOR *(rn(5)+0)
  VAR(18) = (200)*CFACTOR *(rn(6)+0)
  FIX(1) = (1.0e+6)*CFACTOR
  FIX(2) = (2.1e+4)*CFACTOR
  FIX(3) = (6.0e+3)*CFACTOR
! constant rate coefficients
  RCONST(2) = 1.8e-20
  RCONST(3) = 8.1e-18
  RCONST(6) = 1e-17
  RCONST(8) = 1.6e-17
  RCONST(9) = 1.1e-17
  RCONST(10) = 6.1e-18
  RCONST(11) = 0.00037
  RCONST(12) = 8.1e-18
  RCONST(13) = 0.031
  RCONST(14) = 1.1e-17
  RCONST(15) = 80000
  RCONST(18) = 1.7e+06
  RCONST(19) = 7.4e+09
  RCONST(20) = 8.4e-19
  RCONST(23) = 3.1e-23
  RCONST(24) = 1.2e-18
  RCONST(25) = 0.052
! END constant rate coefficients

! INLINED initializations

        TSTART = 0
        TEND = TSTART + 3600*10 !84000 * 60 
        DT = 60.0  
        TEMP = 298.15

! End INLINED initializations

      
END SUBROUTINE Initialize

! End of Initialize function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



END MODULE verwer_Initialize

