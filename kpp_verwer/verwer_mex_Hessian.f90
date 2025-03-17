
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!                  Matlab Gateway for the Function Hessian
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 USE verwer_Model

      INTEGER nlhs, nrhs
      INTEGER plhs(*), prhs(*)
      INTEGER mxGetPr, mxCreateFull, mxGetM, mxgetN
      INTEGER VPtr, FPtr, RPtr, HESSPtr
      REAL(kind=dp) V(20), F(4), RCT(25)
      REAL(kind=dp) HESS(39)

! Check for the right number of input arguments
      IF ( nrhs .ne. 3 ) THEN
         CALL mexErrMsgTxt('Hessian requires 3 input vectors: &
     &V(20), F(4), RCT(25)')
      END IF 
! Check for the right number of output arguments
      IF ( nlhs .ne. 1 ) THEN
         CALL mexErrMsgTxt('Hessian requires 1 output vector: &
     &HESS(39)')
      END IF 

      plhs(1) = mxCreateDoubleMatrix(39,1,0)

      VPtr = mxGetPr(prhs(1));
      CALL mxCopyPtrToReal8(VPtr,V,20)
      
      FPtr = mxGetPr(prhs(2));
      CALL mxCopyPtrToReal8(FPtr,F,4)
      
      RPtr = mxGetPr(prhs(3));
      CALL mxCopyPtrToReal8(RPtr,RCT,25)

      HESSPtr = mxGetPr(plhs(1))

      CALL Hessian( V, F, RCT, HESS )

      CALL mxCopyReal8ToPtr(HESS, HESSPtr, 39)

 END SUBROUTINE mexFunction
