! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! Main Program File
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
! File                 : dynho_Main.f90
! Time                 : Wed Feb  1 13:13:53 2023
! Working directory    : /pfs/data5/home/kit/imk-tro/ii5664/kpp/simple_dynH
! Equation file        : dynho.kpp
! Output root filename : dynho
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
! 
! MAIN - Main program - driver routine
!   Arguments :
! 
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PROGRAM dynho_Driver

  USE dynho_Model
  USE dynho_Initialize, ONLY: Initialize
  
    implicit none

      REAL(kind=dp) :: T, DVAL(NSPEC)
      REAL(kind=dp) :: RSTATE(20)
      INTEGER :: i
      INTEGER :: j
 
      

  
!~~~> Initialization 

      STEPMIN = 0.0d0
      STEPMAX = 0.0d0

      DO i=1,NVAR
        RTOL(i) = 1.0d-4
        ATOL(i) = 1.0d-3
      END DO
     
      CALL Initialize()
      CALL InitSaveData()

!~~~> Time loop
      T = TSTART
kron: DO WHILE (T < TEND)

        TIME = T
        CALL GetMass( C, DVAL )
        WRITE(6,991) (T-TSTART)/(TEND-TSTART)*100, T,       &
                   ( TRIM(SPC_NAMES(MONITOR(i))),           &
                     C(MONITOR(i))/CFACTOR, i=1,NMONITOR ), &
                   ( TRIM(SMASS(i)), DVAL(i)/CFACTOR, i=1,NMASS )
        CALL SaveData()
        CALL Update_SUN() 
        CALL Update_RCONST()

        CALL INTEGRATE( TIN = T, TOUT = T+DT, RSTATUS_U = RSTATE, &
        ICNTRL_U = (/ 0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 /) )
        T = RSTATE(1)

      END DO kron
!~~~> End Time loop
!~~~> End Time loop

      CALL GetMass( C, DVAL )
      WRITE(6,991) (T-TSTART)/(TEND-TSTART)*100, T,     &
               ( TRIM(SPC_NAMES(MONITOR(i))),           &
                 C(MONITOR(i))/CFACTOR, i=1,NMONITOR ), &
               ( TRIM(SMASS(i)), DVAL(i)/CFACTOR, i=1,NMASS )
      TIME = T
      CALL SaveData()
      CALL CloseSaveData()

991   FORMAT(F6.1,'%. T=',E9.3,2X,200(A,'=',E11.4,'; '))    !!!!!!


  

END PROGRAM dynho_Driver

! End of MAIN function
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


