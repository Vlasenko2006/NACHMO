!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 SUBROUTINE mexFunction( nlhs, plhs, nrhs, prhs )
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!          Matlab Gateway for the Sparse Jacobian Function Jac_SP
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

 USE verwer_Model

      INTEGER nlhs, nrhs
      INTEGER plhs(*), prhs(*)
      INTEGER mxGetPr, mxCreateFull, mxGetM, mxgetN
      INTEGER VPtr, FPtr, RPtr, JVSPtr
      REAL(kind=dp) V(20), F(4), RCT(25)
      REAL(kind=dp) JVS(95)

! Check for the right number of input arguments
      IF ( nrhs .ne. 3 ) THEN
         CALL mexErrMsgTxt('Jac_SP requires 3 input vectors: &
     &V(20), F(4), RCT(25)')
      END IF 
! Check for the right number of output arguments
      IF ( nlhs .ne. 1 ) THEN
         CALL mexErrMsgTxt('Jac_SP requires 1 output vector: &
     &JVS(95)')
      END IF 

      plhs(1) = mxCreateDoubleMatrix(95,1,0)

      VPtr = mxGetPr(prhs(1))
      CALL mxCopyPtrToReal8(VPtr,V,20)
      
      FPtr = mxGetPr(prhs(2))
      CALL mxCopyPtrToReal8(FPtr,F,4)
      
      RPtr = mxGetPr(prhs(3))
      CALL mxCopyPtrToReal8(RPtr,RCT,25)

      JVSPtr = mxGetPr(plhs(1))

      CALL Jac_SP( V, F, RCT, JVS )

      CALL mxCopyReal8ToPtr(JVS, JVSPtr, 95)

 END SUBROUTINE mexFunction
