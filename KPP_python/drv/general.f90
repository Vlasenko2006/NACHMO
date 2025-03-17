PROGRAM KPP_ROOT_Driver

!~~~> Global variables and routines
      USE KPP_ROOT_Model
      USE KPP_ROOT_Initialize, ONLY: Initialize

!~~~> Local variables
      KPP_REAL :: T, DVAL(NSPEC)
      KPP_REAL :: RSTATE(20)
      INTEGER :: i

!~~~> Control (in) arguments for the integration
!~~~> These are set to zero by default, which will invoke default behavior
      INTEGER :: ICNTRL(20)
      KPP_REAL :: RCNTRL(20)

!----> A.V. 
      INTEGER :: file_size, counter, no_tsteps
      KPP_REAL, allocatable :: TSTEPS(:)

      KPP_REAL, allocatable :: RCONST_TMP(:)
      INTEGER, allocatable :: RINDEX(:)
      KPP_REAL, allocatable :: C_TMP(:)
      INTEGER, allocatable :: CINDEX(:)
      KPP_REAL :: RUNTIME_PARAMETERS(3)


      counter = 1
      file_size = 0

      INQUIRE(FILE="tsteps.nml", SIZE=file_size)

      file_size = int(file_size/8) -1
      no_tsteps = file_size
      ALLOCATE(TSTEPS(file_size))
    

      INQUIRE(FILE="RCONST.nml", SIZE=file_size)

      file_size = int(file_size/8) -1
      ALLOCATE(RCONST_TMP(file_size))


      INQUIRE(FILE="RINDEX.nml", SIZE=file_size)
      file_size = int(file_size/4) -2
      ALLOCATE(RINDEX(file_size))


      INQUIRE(FILE="C.nml", SIZE=file_size)

      file_size = int(file_size/8) -1
      ALLOCATE(C_TMP(file_size))

      INQUIRE(FILE="CINDEX.nml", SIZE=file_size)
      file_size = int(file_size/4) -2
      ALLOCATE(CINDEX(file_size))

      C_TMP(:) = 0.
      CINDEX(:) = 0.
      RCONST_TMP(:) = 0.
      RINDEX(:) = 0.

      


      OPEN(1, file='tsteps.nml', form='unformatted')
      READ(1) TSTEPS
      CLOSE(1)


      OPEN(1, file='runtime_data.nml', form='unformatted')
      READ(1) RUNTIME_PARAMETERS
      CLOSE(1)

      
      OPEN(1, file='C.nml', form='unformatted')
      READ(1) C_TMP
      CLOSE(1)

      OPEN(1, file='CINDEX.nml', form='unformatted')
      READ(1) CINDEX
      CLOSE(1)
 

      OPEN(1, file='RCONST.nml', form='unformatted')
      READ(1) RCONST_TMP
      CLOSE(1)

      OPEN(1, file='RINDEX.nml', form='unformatted')
      READ(1) RINDEX
      CLOSE(1)



!----> A.V. 


      ICNTRL  = 0
      RCNTRL  = 0.d0

!~~~> Initialization
      STEPMIN = 0.0d0
      STEPMAX = 0.0d0

      DO i=1,NVAR
        RTOL(i) = 1.0d-4
        ATOL(i) = 1.0d-3
      END DO

!~~~> Set default species concentrations
      CALL Initialize()


!~~~> Open log file for write
      CALL InitSaveData()

!~~~> Time loop
!----> A.V. 
      TSTART = RUNTIME_PARAMETERS(1)
      TEND =   RUNTIME_PARAMETERS(2)
      TEMP =   RUNTIME_PARAMETERS(3)

      DO I = 1, size(CINDEX)
         IF(CINDEX(I).ne.0) C(CINDEX(I)) = C_TMP(CINDEX(I))
         PRINT *, "C_TMP = ", C_TMP(CINDEX(I)), "SIZE(CINDEX) = ", size(CINDEX, 1)
      ENDDO 



      DO I = 1, size(RINDEX)
         IF(RINDEX(I).ne.0) RCONST(CINDEX(I)) = RCONST_TMP(RINDEX(I))
         IF(RINDEX(I).ne.0) PRINT *, "RCONST_TMP = ", RCONST_TMP(RINDEX(I)), "RINDEX(I) = ",RINDEX(I)
      ENDDO 




      PRINT *, "RUNTIME_PARAMETERS = ", RUNTIME_PARAMETERS(:)

!----> A.V. 
      T = TSTART
kron: DO WHILE (T < TEND)

!----> A.V. 


        IF(counter .le. no_tsteps)DT = TSTEPS(counter+1) - TSTEPS(counter)
        counter = counter+1

        PRINT *, "TSTEPS(counter) = ", TSTEPS(counter), "counter = ", counter, "DT = ", DT ,"T = ", T, "TEND = ", TEND
   
        IF(DT.eq.0) THEN
          PRINT *, "WRONG DT"
          STOP
        ENDIF

!----> A.V. 

        TIME = T

!~~~> Write values of monitored species at each iteration
        CALL GetMass( C, DVAL )
        WRITE(6,991) (T-TSTART)/(TEND-TSTART)*100, T,       &
                   ( TRIM(SPC_NAMES(MONITOR(i))),           &
                     C(MONITOR(i))/CFACTOR, i=1,NMONITOR )
        CALL SaveData()

!~~~> Update sunlight intensity and reaction rates
!~~~> before calling the integrator.
        CALL Update_SUN()
        CALL Update_RCONST()

!~~~> Call the integrator
        ICNTRL(3)=2
        

        CALL INTEGRATE( TIN       = T,        &
                        TOUT      = T+DT,     &
                        ICNTRL_U  = ICNTRL,   &
                        RCNTRL_U  = RCNTRL,   &
                        RSTATUS_U = RSTATE   )
        T = RSTATE(1)

      END DO kron
!~~~> End Time loop


      PRINT *, "============  LOOP IS FINISHED =========== "

!~~~> Write final values of monitored species and close log file
      CALL GetMass( C, DVAL )
      WRITE(6,991) (T-TSTART)/(TEND-TSTART)*100, T,     &
               ( TRIM(SPC_NAMES(MONITOR(i))),           &
                 C(MONITOR(i))/CFACTOR, i=1,NMONITOR )
      TIME = T
      CALL SaveData()
      CALL CloseSaveData()

991   FORMAT(F6.1,'%. T=',E9.3,2X,200(A,'=',E11.4,'; '))

END PROGRAM KPP_ROOT_Driver
