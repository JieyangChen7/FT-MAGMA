 /*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Stan Tomov
       @generated from zpotrf_gpu.cpp normal z -> d, Fri Jan 30 19:00:13 2015
*/
#include "common_magma.h"
#include<iostream>


//#include"dpotrfFT.h"
//#include"dtrsmFT.h"
//#include"dsyrkFT.h"
//#include"dgemmFT.h"
#include "FT.h"
#include "papi.h"


using namespace std;

#define PRECISION_d

// === Define what BLAS to use ============================================
//#if (defined(PRECISION_s) || defined(PRECISION_d))
    #define magma_dtrsm magmablas_dtrsm
//#endif
// === End defining what BLAS to use =======================================

/**
    Purpose
    -------
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.
    If the current stream is NULL, this version replaces it with a new
    stream to overlap computation with communication.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_dposv_comp
    ********************************************************************/

extern "C" magma_int_t
magma_dpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magma_int_t *info)
{
#define dA(i, j) (dA + (j)*ldda + (i))

    magma_int_t     j, jb, nb;
    const char* uplo_ = lapack_uplo_const( uplo );
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double *work;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    int upper = (uplo == MagmaUpper);

    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    nb = magma_get_dpotrf_nb(n);

    //** debug **//
     nb = 4;
        
        
    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    /* Define user stream if current stream is NULL */
    magma_queue_t stream[5];
    
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[2] );
    magma_queue_create( &stream[3] );
    magma_queue_create( &stream[4] );
    if (orig_stream == NULL) {
        magma_queue_create( &stream[1] );
        magmablasSetKernelStream(stream[1]);
    }
    else {
        stream[1] = orig_stream;
    }
    
    

    //variables for FT
    bool FT = true;
    bool DEBUG = true;
    bool VERIFY = true;
    ABFTEnv * abftEnv;
    abftEnv = new ABFTEnv();




	
	if (FT) {
		
        initializeABFTEnv(abftEnv, nb, dA, ldda, n, n, nb, nb, stream, DEBUG);
		
  //       cout << "checksums initiallization......";
		// initializeChecksum(abftEnv, dA, ldda, stream);
  //       cout << "done." << endl;

        // cout << "banchmarking:" << endl;
        // ChecksumRecalProfiler(abftEnv, dA, ldda, stream); 


        // benchmark(abftEnv, dA, ldda, stream);
		
	}

    
    if (0) {
    //if ((nb <= 1) || (nb >= n)) {
        /* Use unblocked code. */
        magma_dgetmatrix_async( n, n, dA, ldda, work, n, stream[1] );
        magma_queue_sync( stream[1] );
        lapackf77_dpotrf(uplo_, &n, work, &n, info);
        magma_dsetmatrix_async( n, n, work, n, dA, ldda, stream[1] );
    }
    else {
        /* Use blocked code. */
        if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j < n; j += nb) {
                
                /* Update and factorize the current diagonal block and test
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
                
                magma_dsyrk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dA(0, j), ldda,
                            d_one,     dA(j, j), ldda);

                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[0] );
                
                if ( (j+jb) < n) {
                    /* Compute the current block row. */
                    magma_dgemm(MagmaConjTrans, MagmaNoTrans,
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda,
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda);
                }
                
                magma_queue_sync( stream[0] );
                lapackf77_dpotrf(MagmaUpperStr, &jb, work, &jb, info);
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }

                if ( (j+jb) < n) {
                    magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, (n-j-jb),
                                 c_one, dA(j, j   ), ldda,
                                        dA(j, j+jb), ldda);
                }
            }
        }
        else {
        	
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                //  Update and factorize the current diagonal block and test
                //  for non-positive-definiteness. Computing MIN
                //jb = min(nb, (n-j));
            	int k = 5;
            	
            	if (j % k == 0) {
            		//VERIFY = true;
            	}
            	jb = nb;
                if (j > 0) {

					dsyrkFT(MagmaLower, MagmaNoTrans,
                            jb, j,
                            MAGMA_D_ONE * (-1),
                            dA(j, 0), ldda, 
                            MAGMA_D_ONE,
                            dA(j, j), ldda,
                            abftEnv,
							COL_CHK(j / abftEnv->chk_nb, 0), abftEnv->col_dchk_ld, 
							COL_CHK(j / abftEnv->chk_nb, j / abftEnv->chk_nb), abftEnv->col_dchk_ld,
							FT, DEBUG, VERIFY,
                            stream);
					
                }
                
                              
                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[0] );
                if (FT) {
                    magma_dgetmatrix_async( 2, jb,
                    						COL_CHK(j / abftEnv->chk_nb, j / abftEnv->chk_nb), abftEnv->col_dchk_ld,
                                            abftEnv->col_hchk, abftEnv->col_hchk_ld, stream[0] );
                }
                           
                if ( (j+jb) < n && j > 0) {	
     
                	dgemmFT( MagmaNoTrans, MagmaTrans,
                            (n-j-jb), jb, j, 
                            MAGMA_D_ONE * (-1),
                            dA(j+jb, 0), ldda,
                			dA(j,    0), ldda, 
                            MAGMA_D_ONE,
                            dA(j+jb, j), ldda, 
                            abftEnv,
                			COL_CHK(j / abftEnv->chk_nb + 1, 0), abftEnv->col_dchk_ld,
                            ROW_CHK(j / abftEnv->chk_nb + 1, 0), abftEnv->row_dchk_ld,
                			COL_CHK(j / abftEnv->chk_nb, 0), abftEnv->col_dchk_ld,
                            ROW_CHK(j / abftEnv->chk_nb, 0), abftEnv->row_dchk_ld,
                			COL_CHK(j / abftEnv->chk_nb + 1, j/abftEnv->chk_nb), abftEnv->col_dchk_ld,
                            ROW_CHK(j / abftEnv->chk_nb + 1, j/abftEnv->chk_nb), abftEnv->row_dchk_ld,
                			FT, DEBUG, VERIFY,
                            stream);
                	
                }
                

                magma_queue_sync( stream[0] );
                
                //magma_set_lapack_numthreads(64);
                dpotrfFT(work, nb, nb, info, abftEnv, FT, DEBUG, true);
                                
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[1] );
                if (FT) {
                	magma_dsetmatrix_async( 2, jb,
                                            abftEnv->col_hchk, abftEnv->col_hchk_ld, 
                                            COL_CHK(j / abftEnv->chk_nb, j / abftEnv->chk_nb), abftEnv->col_dchk_ld,
                                            stream[1] );
                }
                
//                if (*info != 0) {
//                    *info = *info + j;
//                    break;
//                }
//                
                if ( (j+jb) < n) {          	
                	dtrsmFT( MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
                            (n-j-jb), jb, MAGMA_D_ONE,
                            dA(j,    j), ldda,
                			dA(j+jb, j), ldda,
                            abftEnv,
                            COL_CHK(j/abftEnv->chk_nb, j/abftEnv->chk_nb), abftEnv->col_dchk_ld,
                            ROW_CHK(j/abftEnv->chk_nb, j/abftEnv->chk_nb), abftEnv->row_dchk_ld,
                			COL_CHK(j / abftEnv->chk_nb + 1, j/abftEnv->chk_nb), abftEnv->col_dchk_ld,
                            ROW_CHK(j / abftEnv->chk_nb + 1, j/abftEnv->chk_nb), abftEnv->row_dchk_ld,
                			FT, DEBUG, VERIFY, 
                            stream);
                }
                
            }
            magma_queue_sync( stream[0] );
            magma_queue_sync( stream[1] );
            magma_queue_sync( stream[2] );
            magma_queue_sync( stream[3] );
            magma_queue_sync( stream[4] );
//			if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
//				cout << "PAPI ERROR" << endl;
//				return -1;
//			}
//			if (FT) {
//					//cout << "FT enabled:" << endl;
//					FTtime = real_time;
//			} else {
//					//cout << "FT disabled:" << endl;
//					noFTtime = real_time;
//			}  
//			PAPI_shutdown();
//        
//        	float overhead = (FTtime - noFTtime) / noFTtime;
//        	cout << N <<"	no FT:" << noFTtime <<"		FT:"<< FTtime <<"		overhead:"<< overhead <<endl;
        	}
        }
  //  }

    magma_free_pinned( work );
    magma_queue_destroy( stream[0] );
    if (orig_stream == NULL) {
        magma_queue_destroy( stream[1] );
    }
    magmablasSetKernelStream( orig_stream );

    return *info;
} /* magma_dpotrf_gpu */
