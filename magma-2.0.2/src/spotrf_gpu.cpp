/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @author Stan Tomov
       @author Mark Gates
       
       @generated from src/zpotrf_gpu.cpp normal z -> s, Mon May  2 23:29:59 2016
*/
#include "magma_internal.h"
#include "abft_printer.h"
#include "abft_encoder.h"
#include "abft_kernels.h"

// === Define what BLAS to use ============================================
    #undef  magma_dtrsm
    #define magma_dtrsm magmablas_dtrsm
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
    dA      SINGLE PRECISION array on the GPU, dimension (LDDA,N)
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
magma_spotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda + dA_offset)
    #else
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #endif

    /* Define for ABFT */
    #define dA_colchk(i_, j_)   (dA_colchk   + ((i_)/nb)*2 + (j_)*ldda_colchk)
	#define dA_rowchk(i_, j_)   (dA_rowchk   + (i_)        + ((j_)/nb)*2*ldda_rowchk)
    #define dA_colchk_r(i_, j_) (dA_colchk_r + ((i_)/nb)*2 + (j_)*ldda_colchk_r)
	#define dA_rowchk_r(i_, j_) (dA_rowchk_r + (i_)        + ((j_)/nb)*2*ldda_rowchk_r)



    /* Constants */
    const float c_one     = MAGMA_D_ONE;
    const float c_neg_one = MAGMA_D_NEG_ONE;
    const float d_one     =  1.0;
    const float d_neg_one = -1.0;
    
    /* Local variables */
    const char* uplo_ = lapack_uplo_const( uplo );
    bool upper = (uplo == MagmaUpper);
    
    magma_int_t j, jb, nb;
    float *work;

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
    
    nb = magma_get_spotrf_nb( n );

    //printf("nb=%d\n", nb);
    nb = 128;
    if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );

    /* flags */
    bool FT = true;
    bool DEBUG = false;
    bool CHECK_BEFORE = true;
    bool CHECK_AFTER = true;
    
    /* matrix sizes to be checksumed */
    int cpu_row = nb;
    int cpu_col = nb;
    int gpu_row = n;
    int gpu_col = n;
    
    printf( "initialize checksum vector on CPU\n");
    float * chk_v;
    int ld_chk_v = nb;
    magma_smalloc_pinned(&chk_v, nb * 2 * sizeof(float));
    for (int i = 0; i < nb; ++i) {
        *(chk_v + i) = 1;
    }
    for (int i = 0; i < nb; ++i) {
        *(chk_v + ld_chk_v + i) = i + 1;
    }

    if (DEBUG) {
        printf("checksum vector on CPU:\n");
        printMatrix_host(chk_v, ld_chk_v, nb, 2, -1, -1);
    }

    printf( "initialize checksum vector on GPUs\n");
    float * dev_chk_v;
    size_t pitch_dev_chk_v = magma_roundup(nb * sizeof(float), 32);
    int ld_dev_chk_v;
    
    magma_smalloc(&dev_chk_v, pitch_dev_chk_v * 2);
    ld_dev_chk_v = pitch_dev_chk_v / sizeof(float);
    magma_sgetmatrix(nb, 2,
                     chk_v, ld_chk_v, 
                     dev_chk_v, ld_dev_chk_v,
                     queues[1]);
    if (DEBUG) {
        printMatrix_gpu(dev_chk_v, ld_dev_chk_v,
                        nb, 2, nb, nb, queues[1]);
    }


	printf( "allocate space for checksum on CPU......\n" );
    float * colchk;
    float * colchk_r;
    magma_smalloc_pinned(&colchk, (cpu_row / nb) * 2 * cpu_col * sizeof(float));
    int ld_colchk = (cpu_row / nb) * 2;
    magma_smalloc_pinned(&colchk_r, (cpu_row / nb) * 2 * cpu_col * sizeof(float));
    int ld_colchk_r = (cpu_row / nb) * 2;
    printf( "done.\n" );

    float * rowchk;
    float * rowchk_r;
    magma_smalloc_pinned(&rowchk, cpu_row * (cpu_col / nb) * 2 * sizeof(float));
    int ld_rowchk = cpu_row;
    magma_smalloc_pinned(&rowchk_r, cpu_row * (cpu_col / nb) * 2 * sizeof(float));
    int ld_rowchk_r = cpu_row;
    printf( "done.\n" );

    /* allocate space for col checksum on GPU */
    printf( "allocate space for checksums on GPUs......\n" );
    
    float * dA_colchk;
    size_t pitch_dA_colchk = magma_roundup((gpu_row / nb) * 2 * sizeof(float), 32);
    int ldda_colchk = pitch_dA_colchk / sizeof(float);
    magma_smalloc(&dA_colchk, pitch_dA_colchk * gpu_col);

    float * dA_colchk_r;
    size_t pitch_dA_colchk_r = magma_roundup((gpu_row / nb) * 2 * sizeof(float), 32);
    int ldda_colchk_r = pitch_dA_colchk_r / sizeof(float);
    magma_smalloc(&dA_colchk_r, pitch_dA_colchk_r * gpu_col);

    float * dA_rowchk;
    size_t pitch_dA_rowchk = magma_roundup(gpu_row * sizeof(float), 32);
    int ldda_rowchk = pitch_dA_rowchk / sizeof(float);
    magma_smalloc(&dA_rowchk, pitch_dA_rowchk * (gpu_col / nb) * 2);


    float * dA_rowchk_r;
    size_t pitch_dA_rowchk_r = magma_roundup(gpu_row * sizeof(float), 32);
    int ldda_rowchk_r = pitch_dA_rowchk_r / sizeof(float);
    magma_smalloc(&dA_rowchk_r, pitch_dA_rowchk_r * (gpu_col / nb) * 2);
       
    printf( "done.\n" );

   
    printf( "calculate initial checksum on GPUs......\n" );
  
    col_chk_enc(gpu_row, gpu_col, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_colchk, ldda_colchk, 
                queues[1]);

    row_chk_enc(gpu_row, gpu_col, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_rowchk, ldda_rowchk, 
                queues[1]);

    printf( "done.\n" );

    if (DEBUG) {

        printf( "input matrix A:\n" );
        printMatrix_gpu(dA, ldda, gpu_row, gpu_col, nb, nb, queues[1]);
        printf( "column chk:\n" );
        printMatrix_gpu(dA_colchk, ldda_colchk, 
                        (gpu_row / nb) * 2, gpu_col, 2, nb, queues[1]);
        printf( "row chk:\n" );
        printMatrix_gpu(dA_rowchk, ldda_rowchk,  
                        gpu_row, (gpu_col / nb) * 2, nb, 2, queues[1]);
    }




    if (nb <= 1 || nb >= n) {
        /* Use unblocked code. */
        magma_sgetmatrix( n, n, dA(0,0), ldda, work, n, queues[0] );
        lapackf77_spotrf( uplo_, &n, work, &n, info );
        magma_ssetmatrix( n, n, work, n, dA(0,0), ldda, queues[0] );
    }
    else {
        /* Use blocked code. */
        if (upper) {
            //=========================================================
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j < n; j += nb) {
                // apply all previous updates to diagonal block,
                // then transfer it to CPU
                jb = min( nb, n-j );
                magma_ssyrk( MagmaUpper, MagmaConjTrans, jb, j,
                             d_neg_one, dA(0, j), ldda,
                             d_one,     dA(j, j), ldda, queues[1] );
                
                magma_queue_sync( queues[1] );
                magma_sgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );
                
                // apply all previous updates to block row right of diagonal block
                if (j+jb < n) {
                    magma_sgemm( MagmaConjTrans, MagmaNoTrans,
                                 jb, n-j-jb, j,
                                 c_neg_one, dA(0, j   ), ldda,
                                            dA(0, j+jb), ldda,
                                 c_one,     dA(j, j+jb), ldda, queues[1] );
                }
                
                // simultaneous with above sgemm, transfer diagonal block,
                // factor it on CPU, and test for positive definiteness
                magma_queue_sync( queues[0] );
                lapackf77_spotrf( MagmaUpperStr, &jb, work, &jb, info );
                magma_ssetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                
                // apply diagonal block to block row right of diagonal block
                if (j+jb < n) {
                    magma_strsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                 jb, n-j-jb,
                                 c_one, dA(j, j),    ldda,
                                        dA(j, j+jb), ldda, queues[1] );
                }
            }
        }
        else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j < n; j += nb) {
                // apply all previous updates to diagonal block,
                // then transfer it to CPU
                jb = min( nb, n-j );
                // magma_ssyrk( MagmaLower, MagmaNoTrans, jb, j,
                //              d_neg_one, dA(j, 0), ldda,
                //              d_one,     dA(j, j), ldda, queues[1] );

                abft_ssyrk( MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dA(j, 0), ldda,
                            d_one,     dA(j, j), ldda,
		                 	nb,
		                    dA_colchk(j, 0),    ldda_colchk,
		                    dA_rowchk(j, 0),    ldda_rowchk,
		                    dA_colchk_r(j, 0),  ldda_colchk_r,
		                    dA_rowchk_r(j, 0),  ldda_rowchk_r,
		                    dA_colchk(j, j),    ldda_colchk,
		                    dA_rowchk(j, j),    ldda_rowchk,
		                    dA_colchk_r(j, j),  ldda_colchk_r,
		                    dA_rowchk_r(j, j),  ldda_rowchk_r,
		                    dev_chk_v,          ld_dev_chk_v, 
		                 	FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
		                 	queues[1], queues[1]);
                
                magma_queue_sync( queues[1] );
                magma_sgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, queues[0] );

              	magma_sgetmatrix_async( 2, jb,
				                        dA_colchk(j, j), ldda_colchk,
				                        colchk,     ld_colchk, queues[0] );

              	magma_sgetmatrix_async( jb, 2,
				                        dA_rowchk(j, j), ldda_rowchk,
				                        rowchk,     ld_rowchk, queues[0] );

                
                // apply all previous updates to block column below diagonal block
                if (j+jb < n) {
                    // magma_sgemm( MagmaNoTrans, MagmaConjTrans,
                    //              n-j-jb, jb, j,
                    //              c_neg_one, dA(j+jb, 0), ldda,
                    //                         dA(j,    0), ldda,
                    //              c_one,     dA(j+jb, j), ldda, queues[1] );

                    abft_sgemm( MagmaNoTrans, MagmaConjTrans,
                                n-j-jb, jb, j,
                                c_neg_one, dA(j+jb, 0), ldda,
                                           dA(j,    0), ldda,
                                c_one,     dA(j+jb, j), ldda,
					            nb,
					            dA_colchk(j+jb, 0),   ldda_colchk,
					            dA_rowchk(j+jb, 0),   ldda_rowchk,
					            dA_colchk_r(j+jb, 0), ldda_colchk_r,
					            dA_rowchk_r(j+jb, 0), ldda_rowchk_r,

					            dA_colchk(j,    0),   ldda_colchk,
					            dA_rowchk(j,    0),   ldda_rowchk,
					            dA_colchk_r(j,    0), ldda_colchk_r,
					            dA_rowchk_r(j,    0), ldda_rowchk_r,

					            dA_colchk(j+jb, j),   ldda_colchk,
					            dA_rowchk(j+jb, j),   ldda_rowchk,
					            dA_colchk_r(j+jb, j), ldda_colchk_r,
					            dA_rowchk_r(j+jb, j), ldda_rowchk_r,
					            dev_chk_v,          ld_dev_chk_v, 
			                 	FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
			                 	queues[1], queues[1]);
                }
                
                // simultaneous with above sgemm, transfer diagonal block,
                // factor it on CPU, and test for positive definiteness
                magma_queue_sync( queues[0] );
                lapackf77_spotrf( MagmaLowerStr, &jb, work, &jb, info );
                magma_ssetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, queues[1] );

               //  magma_dsetmatrix_async( 2, jb,
               //  						colchk,     ld_colchk,
				           //              dA_colchk(j, j), ldda_colchk, queues[0] );

              	// magma_dsetmatrix_async( jb, 2,
              	// 						rowchk,     ld_rowchk,
				           //              dA_rowchk(j, j), ldda_rowchk, queues[0] );


                if (*info != 0) {
                    *info = *info + j;
                    break;
                }
                
                // apply diagonal block to block column below diagonal
                if (j+jb < n) {
                    // magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                    //              n-j-jb, jb,
                    //              c_one, dA(j,    j), ldda,
                    //                     dA(j+jb, j), ldda, queues[1] );
                    abft_strsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                 n-j-jb, jb,
                                 c_one, dA(j,    j), ldda,
                                        dA(j+jb, j), ldda,
							    nb,
							    dA_colchk(j,    j),   ldda_colchk,
					            dA_rowchk(j,    j),   ldda_rowchk,
					            dA_colchk_r(j,    j), ldda_colchk_r,
					            dA_rowchk_r(j,    j), ldda_rowchk_r,
							    dA_colchk(j+jb, j),   ldda_colchk,
					            dA_rowchk(j+jb, j),   ldda_rowchk,
					            dA_colchk_r(j+jb, j), ldda_colchk_r,
					            dA_rowchk_r(j+jb, j), ldda_rowchk_r,
							    dev_chk_v,          ld_dev_chk_v, 
			                 	FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
			                 	queues[1], queues[1]);
                }
            }
        }
    }
    
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    magma_free_pinned( work );
    
    return *info;
} /* magma_dpotrf_gpu */
