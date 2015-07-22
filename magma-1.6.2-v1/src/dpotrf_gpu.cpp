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
#include"FT.h"
#include"papi.h"


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
    //    nb = 2;
        
        
    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    /* Define user stream if current stream is NULL */
    magma_queue_t stream[2];
    
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
    magma_queue_create( &stream[0] );
    if (orig_stream == NULL) {
        magma_queue_create( &stream[1] );
        magmablasSetKernelStream(stream[1]);
    }
    else {
        stream[1] = orig_stream;
    }
    
    
    //acommdation
    int B = nb;
    int N = n;
    //variables for FT
    bool FT = true;
    bool DEBUG = false;
	double * v1;
	double * v2;
	double * v1d;
	double * v2d;
	size_t v1d_pitch;
	size_t v2d_pitch;
	size_t v1d_ld;
	size_t v2d_ld;
	double * chk1;
	double * chk2;
	double * chk1d;
	double * chk2d;
	size_t chk1d_pitch;
	size_t chk2d_pitch;
	int chk1d_ld;
	int chk2d_ld;
	size_t checksum1_pitch;
	size_t checksum2_pitch;
	double * checksum1;
	double * checksum2;
	int checksum1_ld;
	int checksum2_ld;

	if (FT) {
		//cout<<"check sum initialization started"<<endl;
		//intialize checksum vector on CPU
//		v1 = new double[B];
//		v2 = new double[B];
		magma_dmalloc_pinned(&v1, B * sizeof(double));
		magma_dmalloc_pinned(&v2, B * sizeof(double));
		
		for (int i = 0; i < B; i++) {
			v1[i] = 1;
			v2[i] = i + 1;
		}
		
//		printMatrix_host(v1, B, 1);
//		printMatrix_host(v2, B, 1);
		//cout<<"checksum vector on CPU initialized"<<endl;

		//intialize checksum vector on GPU
//		cudaMallocPitch((void**) &v1d, &v1d_pitch, B * sizeof(double), 1);
//		cudaMemcpy2D(v1d, v1d_pitch, v1, B * sizeof(double), B * sizeof(double),
//				1, cudaMemcpyHostToDevice);
//		
//		cudaMallocPitch((void**) &v2d, &v2d_pitch, B * sizeof(double), 1);
//		cudaMemcpy2D(v2d, v2d_pitch, v2, B * sizeof(double), B * sizeof(double),
//				1, cudaMemcpyHostToDevice);
		
		v1d_pitch = magma_roundup(B * sizeof(double), 32);
		v2d_pitch = magma_roundup(B * sizeof(double), 32);
		v1d_ld = v1d_pitch / sizeof(double);
		v2d_ld = v2d_pitch / sizeof(double);
		
		magma_dmalloc(&v1d, v1d_pitch * sizeof(double));
		magma_dmalloc(&v2d, v2d_pitch * sizeof(double));
		magma_dsetmatrix(B, 1, v1, B, v1d, v1d_ld);
		magma_dsetmatrix(B, 1, v2, B, v2d, v2d_ld);
		
		
//		printMatrix_gpu(v1d, v1d_ld, B, 1);
//		printMatrix_gpu(v2d, v2d_ld, B, 1);
		//cout<<"checksum vector on gpu initialized"<<endl;

		//allocate space for recalculated checksum on CPU
//		chk1 = new double[B];
//		chk2 = new double[B];
		magma_dmalloc_pinned(&chk1, B * sizeof(double));
		magma_dmalloc_pinned(&chk2, B * sizeof(double));
		
		//cout<<"allocate space for recalculated checksum on CPU"<<endl;

		//allocate space for reclaculated checksum on GPU
//		cudaMallocPitch((void**) &chk1d, &chk1d_pitch, (N / B) * sizeof(double),
//				B);
//		cudaMallocPitch((void**) &chk2d, &chk2d_pitch, (N / B) * sizeof(double),
//				B);
		
		chk1d_pitch = magma_roundup((N / B) * sizeof(double) , 32);
		chk2d_pitch = magma_roundup((N / B) * sizeof(double) , 32);
		
		chk1d_ld = chk1d_pitch / sizeof(double);
		chk2d_ld = chk2d_pitch / sizeof(double);
		
		magma_dmalloc(&chk1d, chk1d_pitch * B);
		magma_dmalloc(&chk2d, chk2d_pitch * B);
		//cout<<"allocate space for recalculated checksum on GPU"<<endl;
 
		//initialize checksums
		checksum1_pitch = magma_roundup((N / B) * sizeof(double) , 32);
		checksum2_pitch = magma_roundup((N / B) * sizeof(double) , 32);
		
		checksum1_ld = checksum1_pitch / sizeof(double);
		checksum2_ld = checksum2_pitch / sizeof(double);
		
		magma_dmalloc(&checksum1, checksum1_pitch * N);
		magma_dmalloc(&checksum2, checksum2_pitch * N);
		
		initializeChecksum(dA, ldda, N, B, v1d, checksum1, checksum1_ld);
		initializeChecksum(dA, ldda, N, B, v2d, checksum2, checksum2_ld);
		//cout<<"checksums initialized"<<endl;
		 
//		printMatrix_gpu(dA, ldda, N, N);
//		printMatrix_gpu(checksum1, checksum1_ld, N / B, N);
//		printMatrix_gpu(checksum2, checksum2_ld, N / B, N);
		

	}
    
    
    
    if (0) {
//    if ((nb <= 1) || (nb >= n)) {
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
        	float real_time = 0.0;
			float proc_time = 0.0;
			long long flpins = 0.0;
			float mflops = 0.0;
        	//timing start***************
        	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
				cout << "PAPI ERROR" << endl;
				return -1;
			}
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                //  Update and factorize the current diagonal block and test
                //  for non-positive-definiteness. Computing MIN
                //jb = min(nb, (n-j));
            	
//            	printMatrix_gpu(checksum1, checksum1_ld, N / B, N);
//            	printMatrix_gpu(checksum2, checksum2_ld, N / B, N);
            	
            	jb = nb;
                if (j > 0) {
					dsyrkFT(jb, j, dA(j, 0), ldda, dA(j, j), ldda,
							checksum1 + j / jb, checksum1_ld, 
							checksum2 + j / jb, checksum2_ld, 
							checksum1 + (j / jb) + j * checksum1_ld, checksum1_ld,
							checksum2 + (j / jb) + j * checksum2_ld, checksum2_ld,
							v1d, v2d, chk1d, chk1d_ld, chk2d, chk2d_ld,
							FT, DEBUG);
                }
                
                
//                magma_dsyrk(MagmaLower, MagmaNoTrans, jb, j,
//                            d_neg_one, dA(j, 0), ldda,
//                            d_one,     dA(j, j), ldda);
                
                
                
                magma_queue_sync( stream[1] );
                magma_dgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[0] );
                if (FT) {
                	magma_dgetmatrix_async(1, jb,
                			               checksum1 + (j / B) + j * checksum1_ld, checksum1_ld,
                			               chk1, 1, stream[0]);
                	magma_dgetmatrix_async(1, jb,
                	                	   checksum2 + (j / B) + j * checksum2_ld, checksum2_ld,
                	                	   chk2, 1, stream[0]);
                }
                
                if ( (j+jb) < n && j > 0) {
                	dgemmFT((n-j-jb), jb, j, dA(j+jb, 0), ldda,
                			dA(j,    0), ldda, dA(j+jb, j), ldda, 
                			checksum1 + (j + jb) / jb, checksum1_ld, 
                			checksum2 + (j + jb) / jb, checksum2_ld,
                			checksum1 + j * checksum1_ld + (j + jb) / jb, checksum1_ld,
                			checksum2 + j * checksum2_ld + (j + jb) / jb, checksum2_ld,
                			v1d, v2d, chk1d, chk1d_ld, chk2d, chk2d_ld, FT, DEBUG);
//                    magma_dgemm( MagmaNoTrans, MagmaConjTrans,
//                                 (n-j-jb), jb, j,
//                                 c_neg_one, dA(j+jb, 0), ldda,
//                                            dA(j,    0), ldda,
//                                 c_one,     dA(j+jb, j), ldda);
                }

                magma_queue_sync( stream[0] );
                
                //lapackf77_dpotrf(MagmaLowerStr, &jb, work, &jb, info);
                dpotrfFT(work, B, B, info, chk1, 1, chk2, 1, v1, v2, FT, DEBUG);
                
                
                magma_dsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[1] );
                if (FT) {
					magma_dgetmatrix_async(1, jb,
							               chk1, 1,
							               checksum1 + (j / jb) + j * checksum1_ld, checksum1_ld, stream[0]);
					magma_dgetmatrix_async(1, jb,
							               chk2, 1,
										   checksum2 + (j / jb) + j * checksum2_ld, checksum2_ld, stream[0]);
				}
                
//                printMatrix_gpu(checksum1, checksum1_ld, N / B, N);
//                printMatrix_gpu(checksum2, checksum2_ld, N / B, N);
                
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                
                if ( (j+jb) < n) {
                	dtrsmFT((n-j-jb), jb, dA(j,    j), ldda,
                			dA(j+jb, j), ldda,
                			checksum1 + (j + jb) / jb + j * checksum1_ld, checksum1_ld,
                			checksum2 + (j + jb) / jb + j * checksum2_ld, checksum2_ld,
                			v1d, v2d, chk1d, chk1d_ld, chk2d, chk2d_ld, FT, DEBUG);
//                    magma_dtrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
//                                (n-j-jb), jb,
//                                c_one, dA(j,    j), ldda,
//                                       dA(j+jb, j), ldda);
                }
            }
            if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
				cout << "PAPI ERROR" << endl;
				return -1;
			}
            if (FT)
            		cout << "FT enabled" << endl;
			cout << "Size:" << N << "(" << B << ")---Real_time:"
					<< real_time << "---" << "Proc_time:"
					<< proc_time << "---" << "Total GFlops:" << endl;            
			PAPI_shutdown();
        }
    }

    magma_free_pinned( work );

    magma_queue_destroy( stream[0] );
    if (orig_stream == NULL) {
        magma_queue_destroy( stream[1] );
    }
    magmablasSetKernelStream( orig_stream );

    return *info;
} /* magma_dpotrf_gpu */
