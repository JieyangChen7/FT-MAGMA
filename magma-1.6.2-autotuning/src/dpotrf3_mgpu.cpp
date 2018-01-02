/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zpotrf3_mgpu.cpp normal z -> d, Fri Jan 30 19:00:14 2015

*/
#include "common_magma.h"
#include "trace.h"
#include "FT.h"
#include <iostream>

#define PRECISION_d

/* === Define what BLAS to use ============================================ */
#if defined(PRECISION_s) || defined(PRECISION_d)
#define DTRSM_WORK
//#define magma_dtrsm magmablas_dtrsm
#endif
/* === End defining what BLAS to use ======================================= */

/**
    Purpose
    -------
    DPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.
    Auxiliary subroutine for dpotrf2_ooc. It is multiple gpu interface to compute
    Cholesky of a "rectangular" matrix.

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
magma_dpotrf3_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDouble_ptr d_lA[],  magma_int_t ldda,
    magmaDouble_ptr d_lP[],  magma_int_t lddp,
    double *A,          magma_int_t lda, magma_int_t h,
    magma_queue_t queues[][3], magma_event_t events[][5],
    magma_int_t *info )
{
#define Alo(i, j)  (A +             ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i))
#define Aup(i, j)  (A + (nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i))

#define dlA(id, i, j)     (d_lA[(id)] + (j)*ldda + (i))
#define dlP(id, i, j, k)  (d_lP[(id)] + (k)*nb*lddp + (j)*lddp + (i))
#define dlPT(id, i, j, k) (d_lP[(id)] + (k)*nb*lddp + (j)*nb   + (i))

    magma_int_t     j, jb, nb0, nb2, d, dd, id, j_local, j_local2, buf;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    int upper = (uplo == MagmaUpper);
    double *dlpanel;
    magma_int_t n_local[MagmaMaxGPUs], ldpanel;
    const magma_int_t stream1 = 0, stream2 = 1, stream3 = 2;
    
    *info = 0;
    if (! upper && uplo != MagmaLower) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (!upper && ngpu*ldda < max(1,n)) {
        *info = -4;
    } else if (upper && ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );
    
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
    /* used by dtrsm_work */
    double c_zero    = MAGMA_D_ZERO;
    int trsm_nb = 128;
    int trsm_n = trsm_nb*((nb+trsm_nb-1)/trsm_nb);
    double *d_dinvA[MagmaMaxGPUs];
    double *d_x[MagmaMaxGPUs];
    #define dinvA(d,j) &(d_dinvA[(d)][(j)*trsm_nb*trsm_n])
    #define dx(d,j) &(d_x[(d)][(j)*nb*m])
    /*
     * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE
     */
    // TODO free memory on failure.
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        if ( (MAGMA_SUCCESS != magma_dmalloc( &d_dinvA[d], 2*trsm_nb*trsm_n )) ||
             (MAGMA_SUCCESS != magma_dmalloc( &d_x[d],     2*nb*(upper ? n : m) )) ) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
    }
    magma_setdevice(0);
#endif

    /* initialization */
    for( d=0; d < ngpu; d++ ) {
        /* local-n and local-ld */
        if (upper) {
            n_local[d] = (n/(nb*ngpu))*nb;
            if (d < (n/nb)%ngpu)
                n_local[d] += nb;
            else if (d == (n/nb)%ngpu)
                n_local[d] += n%nb;
        } else {
            n_local[d] = (m/(nb*ngpu))*nb;
            if (d < (m/nb)%ngpu)
                n_local[d] += nb;
            else if (d == (m/nb)%ngpu)
                n_local[d] += m%nb;
        }
    }
    
    /* flags */
    bool FT = false;
    bool DEBUG = false;

    /* matrix sizes to be checksumed */
    int cpu_row = nb;
    int cpu_col = nb;
    int * gpu_row = new int[ngpu];
    int * gpu_col = new int[ngpu];
    for (int d = 0; d < ngpu; d++) {
        gpu_row[d] = n_local[d];
        gpu_col[d] = n;
    }

    /* initialize checksum vector on CPU */
    double * chk_v;
    int ld_chk_v = nb;
    magma_dmalloc_pinned(&chk_v, nb * 2 * sizeof(double));
    for (int i = 0; i < nb; ++i) {
        *(chk_v + i) = 1;
    }
    for (int i = 0; i < nb; ++i) {
        *(chk_v + ld_chk_v + i) = i + 1;
    }

    if (DEBUG) {
        cout << "checksum vector on CPU:" << endl;
        printMatrix_host(chk_v, ld_chk_v, nb, 2, -1, -1);
    }

    /* initialize checksum vector on GPUs */
    double ** dchk_v = new double * [ngpu];
    size_t pitch_dchk_v = magma_roundup(nb * sizeof(double), 32);
    int * ld_dchk_v = new int[ngpu];
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        magma_dmalloc(&dchk_v[d], pitch_dchk_v * 2);
        ld_dchk_v[d] = pitch_dchk_v / sizeof(double);
        magma_dsetmatrix(nb, 2,
                         chk_v, ld_chk_v, 
                         dchk_v[d], ld_dchk_v[d]);
    }

    /* allocate space for update column checksum on CPU */
    cout << "allocate space for column checksum on CPU......";
    double * colchk;
    int ld_colchk;
    magma_dmalloc_pinned(&colchk, (cpu_row / nb) * 2 * cpu_col * sizeof(double));
    ld_colchk = (cpu_row / nb) * 2;
    cout << "done." << endl;

    /* allocate space for update column checksum on CPU */
    cout << "allocate space for row checksum on CPU......";
    double * rowchk;
    int ld_rowchk;
    magma_dmalloc_pinned(&rowchk, cpu_row * (cpu_col / nb) * 2 * sizeof(double));
    ld_rowchk = cpu_row;
    cout << "done." << endl;

    
    /* allocate space for update checksum on GPU */
    cout << "allocate space for column checksums on GPUs......";
    double ** dcolchk = new double * [ngpu];
    int * ld_dcolchk = new int[ngpu];
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        size_t pitch_dcolchk = magma_roundup((gpu_row[d] / nb) * 2 * sizeof(double), 32);
        ld_dcolchk[d] = pitch_dcolchk / sizeof(double);
        magma_dmalloc(&(abftEnv->col_dchk), pitch_dcolchk * gpu_col[d]);
    }
    cout << "done." << endl;


    /* allocate space for update checksum on GPU */
    cout << "allocate space for row checksums on GPUs......";
    double ** drowchk = new double * [ngpu];
    int * ld_drowchk = new int[ngpu];
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        size_t pitch_drowchk = magma_roundup(gpu_rowp[d] * sizeof(double), 32);
        ld_drowchk[d] = pitch_drowchk / sizeof(double);
        magma_dmalloc(&drowchk[d], pitch_drowchk * (gpu_col[d] / nb) * 2);
    }
    cout << "done." << endl;

    /* calculate initial column checksum on GPUs */
    cout << "calculate initial column checksum on GPUs......";
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        col_chkenc(d_lA[d], ldda, gpu_row[d], gpu_col[d], nb, dcolchk[d], ld_dcolchk[d], queues[d][stream1]);
        row_chkenc(d_lA[d], ldda, gpu_row[d], gpu_col[d], nb, drowchk[d], ld_drowchk[d], queues[d][stream1]);
    }
    cout << "done." << endl;
    

    

    /* == initialize the trace */
    trace_init( 1, ngpu, 3, (CUstream_st**)queues );

    if (upper) {
        /* ---------------------------------------------- */
        /* Upper-triangular case                          */
        /* > Compute the Cholesky factorization A = U'*U. */
        /* ---------------------------------------------- */
        for (j=0; j < m; j += nb) {
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%ngpu;
            buf = (j/nb)%ngpu; // right now, we have ngpu buffers, so id and buf are the same..
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*ngpu);
            jb = min(nb, (m-j));
 
            /* Update the current diagonal block on stream1 */
            magma_setdevice(id);
            if ( j > 0 ) {
                magmablasSetKernelStream( queues[id][stream1] );
                trace_gpu_start( id, stream1, "syrk", "syrk" );
                magma_dsyrk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dlA(id, 0, nb*j_local), ldda,
                            d_one,     dlA(id, j, nb*j_local), ldda);
                trace_gpu_end( id, stream1 );
            }
            
            /* send the diagonal to cpu on stream1 */
            trace_gpu_start( id, stream1, "comm", "D to CPU" );
            magma_dgetmatrix_async( jb, jb,
                                    dlA(id, j, nb*j_local), ldda,
                                    Aup(j,j),               lda,
                                    queues[id][stream1] );
            trace_gpu_end( id, stream1 );

            /* update off-diagonal blocks in the panel */
            if ( j > 0 ) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2 --;
                    nb0 = nb*j_local2; // number of local columns in the panel, while jb is panel-size (number of rows)
            
                    if ( n_local[d] > nb0 ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream( queues[d][stream2] );
                        if ( d == id ) {
                            dlpanel = dlA(d,0,nb*j_local);
                            ldpanel = ldda;
                            // the GPU owns the row from start, and no need of synch.
                            //magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // wait for look-ahead trsm to finish
                        } else {
                            dlpanel = dlP(d,nb,0,buf);
                            ldpanel = lddp;
                            magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                        }
                        trace_gpu_start( d, stream2, "gemm", "gemm" );
                        magma_dgemm(MagmaConjTrans, MagmaNoTrans,
                                    jb, n_local[d]-nb0, j,
                                    c_neg_one, dlpanel,        ldpanel,
                                               dlA(d, 0, nb0), ldda,
                                    c_one,     dlA(d, j, nb0), ldda);
                        trace_gpu_end( d, stream2 );
                        magma_event_record( events[d][2], queues[d][stream2] );
                    }
                    d = (d+1)%ngpu;
                }
            }

            /* wait for panel and factorize it on cpu */
            magma_setdevice(id);
            magma_queue_sync( queues[id][stream1] );
            trace_cpu_start( 0, "getrf", "getrf" );
            lapackf77_dpotrf(MagmaUpperStr, &jb, Aup(j,j), &lda, info);
            trace_cpu_end( 0 );
            if (*info != 0) {
                *info = *info + j;
                break;
            }
            
            /* send the diagonal to gpus on stream1 */
            if ( (j+jb) < n) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    if ( d == id ) {
                        dlpanel = dlA(d, j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d,0,0,buf);
                        ldpanel = lddp;
                    }
                    magma_setdevice(d);
                    trace_gpu_start( d, stream1, "comm", "comm" );
                    magma_dsetmatrix_async( jb, jb,
                                            Aup(j,j), lda,
                                            dlpanel,  ldpanel,
                                            queues[d][stream1] );
                    trace_gpu_end( d, stream1 );
                    magma_event_record( events[d][1], queues[d][stream1] );
                    d = (d+1)%ngpu;
                }
            } else {
                magma_setdevice(id);
                trace_gpu_start( id, stream1, "comm", "comm" );
                magma_dsetmatrix_async( jb, jb,
                                        Aup(j,j),               lda,
                                        dlA(id, j, nb*j_local), ldda,
                                        queues[id][stream1] );
                trace_gpu_end( id, stream1 );
            }
            
            /* panel-factorize the off-diagonal */
            if ( (j+jb) < n) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    if ( d == id ) {
                        dlpanel = dlA(d,j,nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d,0,0,buf);
                        ldpanel = lddp;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    
                    magma_setdevice(d);
                    if ( j+jb < m && d == (j/nb+1)%ngpu ) {
                        /* owns the next column, look-ahead next block on stream1 */
                        nb0 = min(nb, nb2);
                        magmablasSetKernelStream( queues[d][stream1] );
                        magma_queue_wait_event( queues[d][stream1], events[d][2] ); // wait for gemm update
                        trace_gpu_start( d, stream1, "trsm", "trsm" );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                        magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                        magmablas_dlaset( MagmaFull, nb0,     jb,     c_zero, c_zero, dx(d,0), nb0 );
                        magmablas_dtrsm_work( MagmaLeft, MagmaUpper,
                                              MagmaConjTrans, MagmaNonUnit,
                                              jb, nb0, c_one,
                                              dlpanel, ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                              1, dinvA(d,0), dx(d,0) );
#else
                        magma_dtrsm( MagmaLeft, MagmaUpper,
                                     MagmaConjTrans, MagmaNonUnit,
                                     jb, nb0, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, j, nb*j_local2), ldda);
#endif
                        magma_event_record( events[d][4], queues[d][stream1] );
                        trace_gpu_end( d, stream1 );
                    } else if ( nb2 > 0 ) {
                        /* update all the blocks on stream2 */
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for cholesky factor
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
                        magmablasSetKernelStream( queues[d][stream2] );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                        magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                        magmablas_dlaset( MagmaFull, nb2,     jb,     c_zero, c_zero, dx(d,0), nb2 );
                        magmablas_dtrsm_work( MagmaLeft, MagmaUpper,
                                              MagmaConjTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel, ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                              1, dinvA(d,0), dx(d,0) );
#else
                        magma_dtrsm( MagmaLeft, MagmaUpper,
                                     MagmaConjTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, j, nb*j_local2), ldda);
#endif
                        trace_gpu_end( d, stream2 );
                    }
                    d = (d+1)%ngpu;
                } /* end of for */

                /* ========================================================== */
                if ( j+jb < m ) {
                    d = (j/nb+1)%ngpu;
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    nb0 = min(nb, n_local[d]-nb*j_local2 );
                
                    /* even on 1 gpu, off-diagonals are copied to cpu (synchronize at the end).      *
                     * so we have the Cholesky factor, but only diagonal submatrix of the big panel, *
                     * on cpu at the end.                                                            */
                    int d2, buf2;
                    magma_setdevice(d);
                    /* lookahead done */
                    magma_queue_wait_event( queues[d][stream3], events[d][4] );
                
                    trace_gpu_start( d, stream3, "comm", "row to CPU" );
                    magma_dgetmatrix_async( (j+jb), nb0,
                                            dlA(d, 0, nb*j_local2), ldda,
                                            Aup(0,j+jb),            lda,
                                            queues[d][stream3] );
                    trace_gpu_end( d, stream3 );
                    magma_event_record( events[d][3], queues[d][stream3] );
                    /* needed on pluto */
                    //magma_queue_sync( queues[d][stream3] );
                
                    /* broadcast rows to gpus on stream2 */
                    buf2 = ((j+jb)/nb)%ngpu;
                    for( d2=0; d2 < ngpu; d2++ ) {
                        if ( d2 != d ) {
                            magma_setdevice(d2);
                            trace_gpu_start( d2, stream3, "comm", "row to GPUs" );
                            magma_queue_wait_event( queues[d2][stream3], events[d][3] ); // rows arrived at cpu on stream3
                            magma_dsetmatrix_async( j+jb, nb0,
                                                    Aup(0,j+jb),       lda,
                                                    dlP(d2,nb,0,buf2), lddp,
                                                    queues[d2][stream3] );
                            trace_gpu_end( d2, stream3 );
                            magma_event_record( events[d2][0], queues[d2][stream3] );
                        }
                    }

                    /* =========================== */
                    /* update the remaining blocks */
                    nb2 = n_local[d]-(nb*j_local2 + nb0);
                    if ( nb2 > 0 ) {
                        if ( d == id ) {
                            dlpanel = dlA(d, j, nb*j_local);
                            ldpanel = ldda;
                        } else {
                            dlpanel = dlP(d,0,0,buf);
                            ldpanel = lddp;
                        }
                        magma_setdevice(d);
                        magmablasSetKernelStream( queues[d][stream2] );
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                        int flag = 0;
                        if (flag == 0) {
                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // lookahead -> diagonal inversion
                        } else {
                            magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,flag), trsm_nb );
                            magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                        }
                        magmablas_dlaset( MagmaFull, nb2, jb, c_zero, c_zero, dx(d,1), nb2 );
                        magmablas_dtrsm_work( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel, ldpanel,
                                              dlA(d, j, nb*j_local2+nb0), ldda,
                                              flag, dinvA(d,flag), dx(d,1) );
#else
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for cholesky factor
                        magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel, ldpanel,
                                     dlA(d, j, nb*j_local2+nb0), ldda);
#endif
                        trace_gpu_end( d, stream2 );
                    }
                }
            } /* end of dtrsm */
        } /* end of for j=1, .., n */
    } else {
        /* ---------------------------------------------- */
        /* Lower-triangular case                          */
        /* > Compute the Cholesky factorization A = L*L'. */
        /* ---------------------------------------------- */
        for (j=0; j < n; j += nb) {
        
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%ngpu;
            buf = (j/nb)%ngpu;
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*ngpu);
            jb = min(nb, (n-j));

            /* Update the current diagonal block on stream1 */
            magma_setdevice(id);
            if ( j > 0 ) {
                magmablasSetKernelStream( queues[id][stream1] );
                magma_dsyrk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dlA(id, nb*j_local, 0), ldda,
                            d_one,     dlA(id, nb*j_local, j), ldda);
            }

            /* send the diagonal to cpu on stream1 */
            magma_dgetmatrix_async( jb, jb,
                                    dlA(id, nb*j_local, j), ldda,
                                    Alo(j,j),               lda,
                                    queues[id][stream1] );

            /* update off-diagonal blocks of the panel */
            if ( j > 0 ) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
            
                    if ( nb0 < n_local[d] ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream( queues[d][stream2] );
                        if ( d == id ) {
                            dlpanel = dlA(d, nb*j_local, 0);
                            ldpanel = ldda;
                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // wait for look-ahead trsm to finish
                        } else {
                            dlpanel = dlPT(d,0,nb,buf);
                            ldpanel = nb;
                            magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                        }
                        magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                                     n_local[d]-nb0, jb, j,
                                     c_neg_one, dlA(d, nb0, 0), ldda,
                                                dlpanel,        ldpanel,
                                     c_one,     dlA(d, nb0, j), ldda);
                        magma_event_record( events[d][2], queues[d][stream2] );
                    }
                    d = (d+1)%ngpu;
                }
            }

            /* wait for the panel and factorized it on cpu */
            magma_setdevice(id);
            magma_queue_sync( queues[id][stream1] );
            lapackf77_dpotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus on stream1 */
            if ( (j+jb) < m) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    if ( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    magma_setdevice(d);
                    magma_dsetmatrix_async( jb, jb,
                                            Alo(j,j), lda,
                                            dlpanel,  ldpanel,
                                            queues[d][stream1] );
                    magma_event_record( events[d][1], queues[d][stream1] );
                    d = (d+1)%ngpu;
                }
            } else {
                magma_setdevice(id);
                magma_dsetmatrix_async( jb, jb,
                                        Alo(j,j),               lda,
                                        dlA(id, nb*j_local, j), ldda,
                                        queues[id][stream1] );
            }

            /* panel factorize the off-diagonal */
            if ( (j+jb) < m) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    if ( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2);
                    
                    magma_setdevice(d);
                    if ( j+nb < n && d == (j/nb+1)%ngpu ) { /* owns next column, look-ahead next block on stream1 */
                        if ( j > 0 ) magma_queue_wait_event( queues[d][stream1], events[d][2] ); // wait for gemm update
                        magmablasSetKernelStream( queues[d][stream1] );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                        magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                        magmablas_dlaset( MagmaFull, nb0,     jb,     c_zero, c_zero, dx(d,0), nb0 );
                        magmablas_dtrsm_work( MagmaRight, MagmaLower,
                                              MagmaConjTrans, MagmaNonUnit,
                                              nb0, jb, c_one,
                                              dlpanel, ldpanel,
                                              dlA(d, nb*j_local2, j), ldda,
                                              1, dinvA(d,0), dx(d,0) );
#else
                        magma_dtrsm( MagmaRight, MagmaLower,
                                     MagmaConjTrans, MagmaNonUnit,
                                     nb0, jb, c_one,
                                     dlpanel, ldpanel,
                                     dlA(d, nb*j_local2, j), ldda);
#endif
                        magma_event_record( events[d][4], queues[d][stream1] );
                    } else if ( nb2 > 0 ) { /* other gpus updating all the blocks on stream2 */
                        /* update the entire column */
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for the cholesky factor
                        magmablasSetKernelStream( queues[d][stream2] );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                        magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                        magmablas_dlaset( MagmaFull, nb2,     jb,     c_zero, c_zero, dx(d,0), nb2 );
                        magmablas_dtrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                              nb2, jb, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, nb*j_local2, j), ldda,
                                              1, dinvA(d,0), dx(d,0) );
#else
                        magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                     nb2, jb, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, nb*j_local2, j), ldda);
#endif
                    }
                    d = (d+1)%ngpu;
                } /* end for d */

                /* ========================================================== */
                if ( j+jb < n ) {
                    d = (j/nb+1)%ngpu;
                    /* next column */
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2--;
                    nb0 = min(nb, n_local[d]-nb*j_local2 );
                
                    /* even on 1 gpu, we copy off-diagonal to cpu (but don't synchronize).  */
                    /* so we have the Cholesky factor on cpu at the end.                    */
                    int d2, buf2;
//#define DPOTRF_DEVICE_TO_DEVICE
#ifdef DPOTRF_DEVICE_TO_DEVICE
                    // lookahead done
                
                    /* broadcast the rows to gpus */
                    buf2 = ((j+jb)/nb)%ngpu;
                    for( d2=0; d2 < ngpu; d2++ ) {
                        magma_setdevice(d2);
                        magma_queue_wait_event( queues[d2][stream3], events[d][4] );
                        if ( d2 != d ) {
                            magma_dcopymatrix_async( nb0, j+jb,
                                                     dlPT(d2,0,nb,buf2), nb, // first nbxnb reserved for diagonal block
                                                     dlA(d, nb*j_local2, 0), ldda,
                                                     queues[d2][stream3] );
                            magma_event_record( events[d2][0], queues[d2][stream3] );
                        } else {
                            magma_dgetmatrix_async( nb0, j+jb,
                                                    dlA(d, nb*j_local2, 0), ldda,
                                                    Alo(j+jb,0),            lda,
                                                    queues[d][stream3] );
                        }
                    }
#else
                    // lookahead done
                    magma_setdevice(d);
                    magma_queue_wait_event( queues[d][stream3], events[d][4] );
                    magma_dgetmatrix_async( nb0, j+jb,
                                            dlA(d, nb*j_local2, 0), ldda,
                                            Alo(j+jb,0),            lda,
                                            queues[d][stream3] );
                    magma_event_record( events[d][3], queues[d][stream3] );
                    /* syn on rows on CPU, seem to be needed on Pluto */
                    //magma_queue_sync( queues[d][stream3] );
                
                    /* broadcast the rows to gpus */
                    buf2 = ((j+jb)/nb)%ngpu;
                    for( d2=0; d2 < ngpu; d2++ ) {
                        if ( d2 != d ) {
                            magma_setdevice(d2);
                            magma_queue_wait_event( queues[d2][stream3], events[d][3] ); // getmatrix done
                            magma_dsetmatrix_async( nb0, j+jb,
                                                    Alo(j+jb,0),        lda,
                                                    dlPT(d2,0,nb,buf2), nb, // first nbxnb reserved for diagonal block
                                                    queues[d2][stream3] );
                            magma_event_record( events[d2][0], queues[d2][stream3] );
                        }
                    }
#endif
                    /* =================================== */
                    /* updates remaining blocks on stream2 */
                    nb2 = n_local[d] - (j_local2*nb + nb0);
                    if ( nb2 > 0 ) {
                        if ( d == id ) {
                            dlpanel = dlA(d, nb*j_local, j);
                            ldpanel = ldda;
                        } else {
                            dlpanel = dlPT(d,0,0,buf);
                            ldpanel = nb;
                        }
                        magma_setdevice(d);
                        magmablasSetKernelStream( queues[d][stream2] );
                        /* update the remaining blocks in the column */
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                        int flag = 0;
                        if (flag == 0) {
                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // lookahead -> diagonal inversion
                        } else {
                            magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,flag), trsm_nb );
                            magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                        }
                        magmablas_dlaset( MagmaFull, nb2, jb, c_zero, c_zero, dx(d,1), nb2 );
                        magmablas_dtrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                              nb2, jb, c_one,
                                              dlpanel,                    ldpanel,
                                              dlA(d, nb*j_local2+nb0, j), ldda,
                                              flag, dinvA(d,flag), dx(d,1) );
#else
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                        magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                     nb2, jb, c_one,
                                     dlpanel,                    ldpanel,
                                     dlA(d, nb*j_local2+nb0, j), ldda);
#endif
                    }
                }
            }
        }
    } /* end of else not upper */

    /* == finalize the trace == */
    trace_finalize( "dpotrf.svg", "trace.css" );
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        for( j=0; j < 3; j++ ) {
            magma_queue_sync( queues[d][j] );
        }
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
        magma_free( d_dinvA[d] );
        magma_free( d_x[d] );
#endif
    }
    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );

    return *info;
} /* magma_dpotrf_mgpu */

#undef Alo
#undef Aup
#undef dlA
#undef dlP
#undef dlPT


#define A(i, j)  (A +(j)*lda  + (i))
#define dA(d, i, j) (dA[(d)]+(j)*ldda + (i))


// ----------------------------------------------------------------------
extern "C" magma_int_t
magma_dhtodpo(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    double    *A,    magma_int_t lda,
    magmaDouble_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info)
{
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    magma_int_t k;
    if (uplo == MagmaUpper) {
        magma_int_t j, jj, jb, mj;
        
        /* go through each column */
        for (j=off_j; j < n; j += nb) {
            jj = (j-off_j)/(nb*ngpu);
            k  = ((j-off_j)/nb)%ngpu;
            
            jb = min(nb, (n-j));
            if (j+jb < off_j+m)
                mj = (j-off_i)+jb;
            else
                mj = m;

            magma_setdevice(k);
            magma_dsetmatrix_async( mj, jb,
                                    A(off_i, j),     lda,
                                    dA(k, 0, jj*nb), ldda,
                                    queues[k][0] );
        }
    }
    else {
        magma_int_t i, ii, ib, ni;
        
        /* go through each row */
        for (i=off_i; i < m; i += nb) {
            ii = (i-off_i)/(nb*ngpu);
            k  = ((i-off_i)/nb)%ngpu;
            
            ib = min(nb, (m-i));
            if (i+ib < off_i+n)
                ni = (i-off_i)+ib;
            else
                ni = n;
            
            magma_setdevice(k);
            magma_dsetmatrix_async( ib, ni,
                                    A(i, off_j),     lda,
                                    dA(k, ii*nb, 0), ldda,
                                    queues[k][0] );
        }
    }
    for( k=0; k < ngpu; k++ ) {
        magma_setdevice(k);
        magma_queue_sync( queues[k][0] );
    }
    magma_setdevice( orig_dev );

    return *info;
}


// ----------------------------------------------------------------------
extern "C" magma_int_t
magma_ddtohpo(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    double    *A,    magma_int_t lda,
    magmaDouble_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info)
{
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    magma_int_t k;
    if (uplo == MagmaUpper) {
        magma_int_t j, jj, jb, mj;
        
        /* go through each column */
        for (j=off_j+NB; j < n; j += nb) {
            jj =  (j-off_j)/(nb*ngpu);
            k  = ((j-off_j)/nb)%ngpu;
            
            jb = min(nb, (n-j));
            if (j+jb < off_j+m)
                mj = (j-off_i)+jb;
            else
                mj = m;

            magma_setdevice(k);
            magma_dgetmatrix_async( mj, jb,
                                    dA(k, 0, jj*nb), ldda,
                                    A(off_i, j),     lda,
                                    queues[k][0] );
            magma_queue_sync( queues[k][0] );
        }
    } else {
        magma_int_t i, ii, ib, ni;
        
        /* go through each row */
        for (i=off_i+NB; i < m; i += nb) {
            ii = (i-off_i)/(nb*ngpu);
            k  = ((i-off_i)/nb)%ngpu;
            
            ib = min(nb, (m-i));
            if (i+ib < off_i+n)
                ni = (i-off_i)+ib;
            else
                ni = n;
            
            magma_setdevice(k);
            magma_dgetmatrix_async( ib, ni,
                                    dA(k, ii*nb, 0), ldda,
                                    A(i, off_j),     lda,
                                    queues[k][0] );
            magma_queue_sync( queues[k][0] );
        }
    }
    /*for( k=0; k < ngpu; k++ ) {
        magma_setdevice(k);
        magma_queue_sync( queues[k][0] );
    }*/
    magma_setdevice( orig_dev );

    return *info;
}

#undef A
#undef dA
