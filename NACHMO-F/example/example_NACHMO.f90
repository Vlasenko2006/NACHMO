program NACHMO


      use mod_kinds, only: ik, rk
      use mod_network, only: network_type
      use, intrinsic :: ieee_arithmetic, only: IEEE_Value, IEEE_QUIET_NAN
      use, intrinsic :: iso_fortran_env, only: real32


      implicit none
  
      type(network_type) :: net
      type(network_type) :: net2, net3
 

      real(real32) :: nan


      INTEGER, PARAMETER :: ncells   = 50
      INTEGER, PARAMETER :: nlayers  = 10
      INTEGER, PARAMETER :: nspecies = 3
      INTEGER, PARAMETER :: nsteps   = 100
      INTEGER, PARAMETER :: nneurons   = 80
      INTEGER, PARAMETER :: npairs   = int(nlayers/2 -2)

      INTEGER, PARAMETER :: nspecies_net2  = nspecies
      INTEGER, PARAMETER :: out_features_net1 = 2*nspecies
      INTEGER, PARAMETER :: out_features_net2 = nspecies

      INTEGER(ik) :: c 
      INTEGER(ik) :: i
      INTEGER(ik) :: j 
      INTEGER(ik) :: k 

      INTEGER :: in_features_net1
      INTEGER :: ncross_products
 

      REAL(rk), ALLOCATABLE  :: bilinear_layer(:)
      REAL(rk), ALLOCATABLE  :: gates(:)
      REAL(rk), ALLOCATABLE  :: intermediate_output(:)
      REAL(rk), ALLOCATABLE  :: output(:)
      REAL(rk), ALLOCATABLE  :: rates(:)
      REAL(rk), ALLOCATABLE  :: ww_in(:,:)
      REAL(rk), ALLOCATABLE  :: x1(:)


      REAL(rk) :: estimate(ncells,nsteps,nspecies)
      REAL(rk) :: IC(ncells,nspecies)
      REAL(rk) :: max_c(nspecies)
      REAL(rk) :: start
      REAL(rk) :: tend
  

      REAL(rk) :: ww_perc(nneurons,nneurons,npairs)
      REAL(rk) :: ww_n1(nneurons,nneurons)
      REAL(rk) :: ww_n2(out_features_net1,out_features_net1)
      REAL(rk) :: ww_out(nneurons,out_features_net1)

      REAL(rk) :: b_in(nneurons)
      REAL(rk) :: b_perc(nneurons,npairs)
      REAL(rk) :: b_out(out_features_net1)


      CALL FRACT(nspecies, ncross_products)       
      in_features_net1 = nspecies + ncross_products

      ALLOCATE(gates(nspecies))
      ALLOCATE(intermediate_output(out_features_net1))
      ALLOCATE(bilinear_layer(in_features_net1))
      ALLOCATE(output(nspecies))
      ALLOCATE(rates(nspecies))
      ALLOCATE(x1(nspecies))

      ALLOCATE(ww_in(     in_features_net1 ,nneurons )  )


      net = network_type([in_features_net1,nneurons,nneurons, nneurons,nneurons,   nneurons,nneurons,   nneurons,nneurons,  out_features_net1,out_features_net1],'lin')
      net2 = network_type([nspecies,nspecies],'lin')


      b_in(:) = 0.
      b_perc(:,:) = 0.
      b_out(:) = 0.

      ww_in(:,:) = 0.
      ww_perc(:,:,:) = 0.
      ww_out(:,:) = 0.
      ww_n1(:,:) = 0.
      ww_n2(:,:) = 0.
 
      DO i = 2,nlayers + 1
         IF (mod(i,2).eq.0) CALL net % layers(i) % set_activation("lin")
         IF (mod(i,2).eq.1) CALL net % layers(i) % set_activation("prelu1")    
      ENDDO


      CALL net2 % layers(2) % set_activation('prelu2')

      net%layers(2) % w = ww_in
      net%layers(2) % b = b_in

      DO i = 3,nlayers
            net%layers(i) % w = ww_perc(:,:,1) ! Setting everything to the zeros in the first layer
            net%layers(i) % b = b_perc(:,1)
      ENDDO

      net%layers(nlayers+1) % w = ww_out
      net%layers(nlayers+1) % b = b_out


      net2%layers(2) % w = ww_n2
      net2%layers(2) % b = b_out

      DO i = 1, nneurons
         ww_n1(i,i) = 1.0
      END DO 

      DO i = 1, out_features_net1
         ww_n2(i,i) = 1.0
      END DO 


      CALL READ_WEIGTS(nlayers,nneurons,npairs,in_features_net1, out_features_net1, b_in,b_perc,b_out, ww_in,ww_perc,ww_out)

      net%layers(2) % w = ww_in
      net%layers(2) % b = b_in
      j = 0
      DO i = 3,nlayers - 1 
         IF (mod(i,2).eq.0)THEN
            j = j+1
            net%layers(i) % w = ww_perc(:,:,j)
            net%layers(i) % b = b_perc(:,j)
         ELSE
            net%layers(i) % w = ww_n1
            net%layers(i) % b = 0
         END IF
      END DO

      net%layers(nlayers) % w = ww_out
      net%layers(nlayers) % b = b_out
      net%layers(nlayers+1) % w = ww_n2

      net2%layers(2) % w = ww_n2


      print *," "
      print *," "
      print *," "


      open(unit=1, file = "/work/gg0302/g260141/p2f/OZNN-V1/build/IC/IC_OH.txt")
      read(unit=1,fmt=*) IC(:,1)
      close(1)

      open(unit=1, file = "/work/gg0302/g260141/p2f/OZNN-V1/build/IC/IC_HO2.txt")
      read(unit=1,fmt=*) IC(:,2)
      close(1)

      open(unit=1, file = "/work/gg0302/g260141/p2f/OZNN-V1/build/IC/IC_H2O2.txt")
      read(unit=1,fmt=*) IC(:,3)
      close(1)

      open(unit=1, file = "/work/gg0302/g260141/p2f/OZNN-V1/build/IC/max_c.txt")
      read(unit=1,fmt=*) max_c(:)
      close(1)

      DO i =1, size(max_c)
         IC(:,i) = IC(:,i)/max_c(i)
      ENDDO

      call cpu_time(start)

      DO i = 1, ncells
        x1 = IC(i,:)
        estimate(i,1,:) = IC(i,:)
        DO j = 1,nsteps - 1
        call OuterProductLayer(nspecies, x1,  ncross_products, bilinear_layer)
        intermediate_output = real(net% output(bilinear_layer))


        rates = intermediate_output(:nspecies)
        gates = intermediate_output(nspecies + 1:)

        output(:) = rates * (1 - exp(-gates * gates))
        output(:) = output(:) + real(net2% output(x1(:)))
      
        x1(:) = output

        estimate(i,j+1,:) = output(:)

      END DO

    END DO

    call cpu_time(tend)

    PRINT *, "Elapsed time = ", tend - start 

    DO  i = 1,ncells
        print *, estimate(1,i,:)
    ENDDO


      c = 0

      open(unit=1, file = "/work/gg0302/g260141/p2f/OZNN-V1/build/estimates/estimate.txt")
      DO i = 1, ncells
          DO j = 1, nsteps
             DO k = 1, size(max_c)
                  write(unit=1,fmt=*) estimate(i,j,k)
                  c = c + 1
             END DO
          END DO
      END DO
      close(1)

      print *, "c=", c, "estimate =", size(estimate)


      stop


end program NACHMO


SUBROUTINE OuterProductLayer(nspecies, x,  ncross_products, layer)

      use mod_kinds, only: ik, rk
      implicit none

      INTEGER, INTENT(IN) :: nspecies
      INTEGER, INTENT(IN) :: ncross_products
      REAL(rk), DIMENSION(nspecies), INTENT(IN) :: x
      REAL(rk), DIMENSION(nspecies + ncross_products), INTENT(out) :: layer
      INTEGER :: i, lstride, rstride


    lstride = nspecies +1 
    rstride = 2 * nspecies + 1 
    layer(1:nspecies) = x(:)

    DO i = 1, nspecies 
        layer(lstride:rstride) =  x(i) * x(i:)
        lstride = rstride 
        rstride = rstride + size(x(i:)) -1
    END DO


   RETURN

END SUBROUTINE






SUBROUTINE FRACT(nspecies, ncross_products)


      INTEGER, INTENT(IN) :: nspecies
      INTEGER, INTENT(OUT) :: ncross_products
      INTEGER :: i

      ncross_products = 0
    
    DO i = 1,nspecies
        ncross_products = ncross_products + i
    END DO

   RETURN 

END SUBROUTINE FRACT





SUBROUTINE READ_WEIGTS(nlayers,nneurons,npairs,in_features_net1, out_features_net1, b_in,b_perc,b_out, ww_in,ww_perc,ww_out)


      use mod_kinds, only: ik, rk

      use, intrinsic :: ieee_arithmetic, only: IEEE_Value, IEEE_QUIET_NAN
      use, intrinsic :: iso_fortran_env, only: real32
      real(real32) :: nan


      CHARACTER(len=1)  :: fld1

      INTEGER, INTENT(IN) :: in_features_net1
      INTEGER, INTENT(IN) :: out_features_net1
      INTEGER, INTENT(IN) :: nlayers

      REAL(rk), INTENT(INOUT) :: b_in(nneurons)
      REAL(rk), INTENT(INOUT) :: b_perc(nneurons, npairs) 
      REAL(rk), INTENT(INOUT) :: b_out(out_features_net1)

      REAL(RK), INTENT(INOUT)  :: ww_in(     in_features_net1 ,nneurons )  
      REAL(RK), INTENT(INOUT)  :: ww_perc(   nneurons,nneurons,npairs )  
      REAL(RK), INTENT(INOUT)  :: ww_out(    nneurons,out_features_net1 )  

      REAL(RK)               :: w_in(      in_features_net1 * nneurons )  
      REAL(RK)               :: w_perc(    nneurons*nneurons,npairs )  
      REAL(RK)               :: w_out (    nneurons*out_features_net1 )  


      nan = IEEE_VALUE(nan, IEEE_QUIET_NAN)
      w_in(:) = 0.
      w_perc(:,:) = 0.
      w_out(:) = 0.

      ww_in(:,:) = nan
      ww_perc(:,:,:) = nan
      ww_out(:,:) = nan

      b_in(:) = nan
      b_perc(:,:) = nan
      b_out(:) = nan

      DO i = 1, int(nlayers/2)

         WRITE(fld1,'(i1.1)') i  
         OPEN(unit=1, file = "/work/gg0302/g260141/p2f/weights/net.layers."//fld1//".weight.txt")
         OPEN(unit=2, file = "/work/gg0302/g260141/p2f/weights/net.layers."//fld1//".bias.txt")

         IF (i.eq.1) THEN 
            READ(unit=1,fmt=*) w_in(:)
            READ(unit=2,fmt=*) b_in(:)
            ww_in = TRANSPOSE(RESHAPE(w_in, (/nneurons,nneurons/)))
         END IF

         IF (i.ge.2.and.i.lt.int(nlayers/2)) THEN 
             READ(unit=1,fmt=*) w_perc(:,i-1)
             READ(unit=2,fmt=*) b_perc(:,i-1)
             ww_perc(:,:,i-1) = TRANSPOSE(RESHAPE(w_perc(:,i-1), (/nneurons,nneurons/)))
         END IF
         IF (i.eq.int(nlayers/2)) THEN
            READ(unit=1,fmt=*) w_out(:)
            READ(unit=2,fmt=*) b_out(:)
            ww_out(:,:) = TRANSPOSE(RESHAPE(w_out, (/out_features_net1,nneurons/)))
         END IF
         CLOSE(1)
         CLOSE(2)

      END DO 

     RETURN

END SUBROUTINE





