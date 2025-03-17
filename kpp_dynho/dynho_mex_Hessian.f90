
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!                  Matlab Gateway for the Function Hessian
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 USE dynho_Model

      INTEGER nlhs, nrhs
      INTEGER plhs(*), prhs(*)
      INTEGER mxGetPr, mxCreateFull, mxGetM, mxgetN
      INTEGER VPtr, FPtr, RPtr, HESSPtr
      REAL(kind=dp) V(3), F(2), RCT(4)
      REAL(kind=dp) HESS(7)

! Check for the right number of input arguments
      IF ( nrhs .ne. 3 ) THEN
         CALL mexErrMsgTxt('Hessian requires 3 input vectors: &
     &V(3), F(2), RCT(4)')
      END IF 
! Check for the right number of output arguments
      IF ( nlhs .ne. 1 ) THEN
         CALL mexErrMsgTxt('Hessian requires 1 output vector: &
     &HESS(7)')
      END IF 

      plhs(1) = mxCreateDoubleMatrix(7,1,0)

      VPtr = mxGetPr(prhs(1));
      CALL mxCopyPtrToReal8(VPtr,V,3)
      
      FPtr = mxGetPr(prhs(2));
      CALL mxCopyPtrToReal8(FPtr,F,2)
      
      RPtr = mxGetPr(prhs(3));
      CALL mxCopyPtrToReal8(RPtr,RCT,4)

      HESSPtr = mxGetPr(plhs(1))

      CALL Hessian( V, F, RCT, HESS )

      CALL mxCopyReal8ToPtr(HESS, HESSPtr, 7)

 END SUBROUTINE mexFunction
