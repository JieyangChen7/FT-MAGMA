/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from src/zpotrf3_mgpu.cpp normal z -> d, Mon May  2 23:30:01 2016

*/
#include "magma_internal.h"
#include "trace.h"
#include "abft_printer.h"
#include "abft_encoder.h"
#include "abft_kernels.h"

#define PRECISION_d

/* === Define what BLAS to use ============================================ */
#if defined(PRECISION_s) || defined(PRECISION_d)
#define DTRSM_WORK
//#undef  magma_dtrsm
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
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    m       INTEGER
            The number of rows of the submatrix to be factorized.

    @param[in]
    n       INTEGER
            The number of columns of the submatrix to be factorized.

    @param[in]
    off_i   INTEGER
            The first row index of the submatrix to be factorized.

    @param[in]
    off_j   INTEGER
            The first column index of the submatrix to be factorized.

    @param[in]
    nb      INTEGER
            The block size used for the factorization and distribution.

    @param[in,out]
    d_lA    DOUBLE PRECISION array of pointers on the GPU, dimension (ngpu).
            On entry, the symmetric matrix dA distributed over GPU.
            (d_lAT[d] points to the local matrix on d-th GPU).
            If UPLO = MagmaLower or MagmaUpper, it respectively uses 
            a 1D block column or row cyclic format (with the block size 
            nb), and each local matrix is stored by column.
            If UPLO = MagmaUpper, the leading N-by-N upper triangular 
            part of dA contains the upper triangular part of the matrix dA, 
            and the strictly lower triangular part of dA is not referenced.  
            If UPLO = MagmaLower, the leading N-by-N lower triangular part 
            of dA contains the lower triangular part of the matrix dA, and 
            the strictly upper triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in,out]
    d_lP    DOUBLE PRECISION array of pointers on the GPU, dimension (ngpu).
            d_LAT[d] points to workspace of size h*lddp*nb on d-th GPU.

    @param[in]
    lddp    INTEGER
            The leading dimension of the array dP.  LDDA >= max(1,N).

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[in,out]
    A       DOUBLE PRECISION array on the CPU, dimension (LDA,H*NB)
            On exit, the panel is copied back to the CPU

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in]
    h       INTEGER
            It specifies the size of the CPU workspace, A.

    @param[in]
    queues  magma_queue_t
            queues is of dimension (ngpu,3) and contains the queues 
            used for the partial factorization.

    @param[in]
    events  magma_event_t
            events is of dimension(ngpu,5) and contains the events used 
            for the partial factorization.

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

#define Alo_colchk(i, j)    (colchk   + ((j)+off_j)*ld_colchk    + ((nb*(((i)/nb)%h)+off_i)/nb)*2)
#define Alo_colchk_r(i, j)  (colchk_r + ((j)+off_j)*ld_colchk_r  + ((nb*(((i)/nb)%h)+off_i)/nb)*2)
#define Alo_rowchk(i, j)    (rowchk   + (((j)+off_j)/nb)*2*ld_colchk    + (nb*(((i)/nb)%h)+off_i))
#define Alo_rowchk_r(i, j)  (rowchk_r + (((j)+off_j)/nb)*2*ld_colchk_r  + (nb*(((i)/nb)%h)+off_i))

#define dlA_colchk(id, i, j)       (d_lA_colchk[(id)]   + (j)*ldda_colchk[(id)]          + ((i)/nb)*2)
#define dlA_colchk_r(id, i, j)     (d_lA_colchk_r[(id)] + (j)*ldda_colchk_r[(id)]        + ((i)/nb)*2)
#define dlA_rowchk(id, i, j)       (d_lA_rowchk[(id)]   + ((j)/nb)*2*ldda_rowchk[(id)]   + (i))
#define dlA_rowchk_r(id, i, j)     (d_lA_rowchk_r[(id)] + ((j)/nb)*2*ldda_rowchk_r[(id)] + (i))

#define dlP_colchk(id, i, j, k)       (d_lP_colchk[(id)]   + (k)*nb*lddp_colchk[(id)]          + (j)*lddp_colchk        + ((i)/nb)*2)
#define dlP_colchk_r(id, i, j, k)     (d_lP_colchk_r[(id)] + (k)*nb*lddp_colchk_r[(id)]        + (j)*lddp_colchk_r      + ((i)/nb)*2)
#define dlP_rowchk(id, i, j, k)       (d_lP_rowchk[(id)]   + (k)*(nb/nb)*2*lddp_rowchk[(id)]   + (j/nb)*2*lddp_rowchk   + (i))
#define dlP_rowchk_r(id, i, j, k)     (d_lP_rowchk_r[(id)] + (k)*(nb/nb)*2*lddp_rowchk_r[(id)] + (j/nb)*2*lddp_rowchk_r + (i))

#define dlPT_colchk(id, i, j, k)       (d_lP_colchk[(id)]   + (k)*nb*lddp_colchk[(id)]          + (((j)*nb+(i))/nb)*2)
#define dlPT_colchk_r(id, i, j, k)     (d_lP_colchk_r[(id)] + (k)*nb*lddp_colchk_r[(id)]        + (((j)*nb+(i))/nb)*2)
#define dlPT_rowchk(id, i, j, k)       (d_lP_rowchk[(id)]   + (k)*(nb/nb)*2*lddp_rowchk[(id)]   + (j) * nb + (i))
#define dlPT_rowchk_r(id, i, j, k)     (d_lP_rowchk_r[(id)] + (k)*(nb/nb)*2*lddp_rowchk_r[(id)] + (j) * nb + (i))


    magma_int_t     j, jb, nb0, nb2, d, dd, id, j_local, j_local2, buf;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    bool upper = (uplo == MagmaUpper);
    double *dlpanel;

    double *dlpanel_colchk;
    double *dlpanel_rowchk;
    int ldpanel_colchk;
    int ldpanel_rowchk;

    double *dlpanel_colchk_r;
    double *dlpanel_rowchk_r;
    int ldpanel_colchk_r;
    int ldpanel_rowchk_r;

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
    
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
    /* used by dtrsm_work */
    double c_zero    = MAGMA_D_ZERO;
    magma_int_t trsm_nb = 128;
    magma_int_t trsm_n = magma_roundup( nb, trsm_nb );
    double *d_dinvA[MagmaMaxGPUs];
    double *dx[MagmaMaxGPUs];
    #define dinvA(d,j) &(d_dinvA[(d)][(j)*trsm_nb*trsm_n])
    #define dx(d,j) &(dx[(d)][(j)*nb*m])
    /*
     * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE
     */
    // TODO free memory on failure.
    magma_int_t dinvA_length = 2*trsm_nb*trsm_n;
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        if ( (MAGMA_SUCCESS != magma_dmalloc( &d_dinvA[d], dinvA_length )) ||
             (MAGMA_SUCCESS != magma_dmalloc( &dx[d],      2*nb*(upper ? n : m) )) ) {
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
    bool FT = true;
    bool DEBUG = true;
    bool CHECK_BEFORE = true;
    bool CHECK_AFTER = true;

    /* matrix sizes to be checksumed */
    int cpu_row = nb;
    int cpu_col = n;
    int * gpu_row = new int[ngpu];
    int * gpu_col = new int[ngpu];
    for (d = 0; d < ngpu; d++) {
        gpu_row[d] = n_local[d];
        gpu_col[d] = n;
    }

    printf( "initialize checksum vector on CPU\n");
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
        printf("checksum vector on CPU:\n");
        printMatrix_host(chk_v, ld_chk_v, nb, 2, -1, -1);
    }

    printf( "initialize checksum vector on GPUs\n");
    double ** dev_chk_v = new double * [ngpu];
    size_t pitch_dev_chk_v = magma_roundup(nb * sizeof(double), 32);
    int * ld_dev_chk_v = new int[ngpu];
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        magma_dmalloc(&dev_chk_v[d], pitch_dev_chk_v * 2);
        ld_dev_chk_v[d] = pitch_dev_chk_v / sizeof(double);
        magma_dgetmatrix(nb, 2,
                         chk_v, ld_chk_v, 
                         dev_chk_v[d], ld_dev_chk_v[d],
                         queues[d][stream1]);
        if (DEBUG) {
            printf("on GPU %d:\n", d);
            printMatrix_gpu(dev_chk_v[d], ld_dev_chk_v[d],
                            nb, 2, nb, nb, queues[d][stream1]);
        }
    }

    printf( "allocate space for column checksum on CPU......\n" );
    double * colchk;
    int ld_colchk;
    double * colchk_r;
    int ld_colchk_r;
    magma_dmalloc_pinned(&colchk, (cpu_row / nb) * 2 * cpu_col * sizeof(double));
    ld_colchk = (cpu_row / nb) * 2;
    magma_dmalloc_pinned(&colchk_r, (cpu_row / nb) * 2 * cpu_col * sizeof(double));
    ld_colchk_r = (cpu_row / nb) * 2;
    printf( "done.\n" );

    printf( "allocate space for row checksum on CPU......\n" );
    double * rowchk;
    int ld_rowchk;
    double * rowchk_r;
    int ld_rowchk_r;
    magma_dmalloc_pinned(&rowchk, cpu_row * (cpu_col / nb) * 2 * sizeof(double));
    ld_rowchk = cpu_row;
    magma_dmalloc_pinned(&rowchk_r, cpu_row * (cpu_col / nb) * 2 * sizeof(double));
    ld_rowchk_r = cpu_row;
    printf( "done.\n" );

    /* allocate space for col checksum on GPU */
    int panel_row;
    if (uplo == MagmaLower) {
        panel_row = nb;
    } else {
        panel_row = lddp;
    }

    printf( "allocate space for col column checksums on GPUs......\n" );
    double ** d_lA_colchk   = new double * [ngpu];
    int *     ldda_colchk   = new int      [ngpu];
    double ** d_lA_colchk_r = new double * [ngpu];
    int *     ldda_colchk_r = new int      [ngpu];

    double ** d_lP_colchk   = new double * [ngpu];
    int *     lddp_colchk   = new int      [ngpu];
    double ** d_lP_colchk_r = new double * [ngpu];
    int *     lddp_colchk_r = new int      [ngpu];
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        size_t pitch_d_lA_colchk = magma_roundup((gpu_row[d] / nb) * 2 * sizeof(double), 32);
        ldda_colchk[d] = pitch_d_lA_colchk / sizeof(double);
        magma_dmalloc(&d_lA_colchk[d], pitch_d_lA_colchk * gpu_col[d]);

        size_t pitch_d_lA_colchk_r = magma_roundup((gpu_row[d] / nb) * 2 * sizeof(double), 32);
        ldda_colchk_r[d] = pitch_d_lA_colchk_r / sizeof(double);
        magma_dmalloc(&d_lA_colchk_r[d], pitch_d_lA_colchk_r * gpu_col[d]);

        size_t pitch_d_lP_colchk = magma_roundup((panel_row / nb) * 2 * sizeof(double), 32);
        lddp_colchk[d] = pitch_d_lP_colchk / sizeof(double);
        magma_dmalloc(&d_lP_colchk[d], pitch_d_lP_colchk * ngpu * nb);

        size_t pitch_d_lP_colchk_r = magma_roundup((panel_row / nb) * 2 * sizeof(double), 32);
        lddp_colchk_r[d] = pitch_d_lP_colchk_r / sizeof(double);
        magma_dmalloc(&d_lP_colchk_r[d], pitch_d_lP_colchk_r * ngpu * nb);

    }
    printf( "done.\n" );

    printf( "allocate space for row checksums on GPUs......\n" );
    double ** d_lA_rowchk   = new double * [ngpu];
    int *     ldda_rowchk   = new int      [ngpu];
    double ** d_lA_rowchk_r = new double * [ngpu];
    int *     ldda_rowchk_r = new int      [ngpu];

    double ** d_lP_rowchk   = new double * [ngpu];
    int *     lddp_rowchk   = new int      [ngpu];
    double ** d_lP_rowchk_r = new double * [ngpu];
    int *     lddp_rowchk_r = new int      [ngpu];
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        size_t pitch_d_lA_rowchk = magma_roundup(gpu_row[d] * sizeof(double), 32);
        ldda_rowchk[d] = pitch_d_lA_rowchk / sizeof(double);
        magma_dmalloc(&d_lA_rowchk[d], pitch_d_lA_rowchk * (gpu_col[d] / nb) * 2);

        size_t pitch_d_lA_rowchk_r = magma_roundup(gpu_row[d] * sizeof(double), 32);
        ldda_rowchk_r[d] = pitch_d_lA_rowchk_r / sizeof(double);
        magma_dmalloc(&d_lA_rowchk_r[d], pitch_d_lA_rowchk_r * (gpu_col[d] / nb) * 2);

        size_t pitch_d_lP_rowchk = magma_roundup(panel_row * sizeof(double), 32);
        lddp_rowchk[d] = pitch_d_lP_rowchk / sizeof(double);
        magma_dmalloc(&d_lP_rowchk[d], pitch_d_lP_rowchk * ((ngpu * nb) / nb) * 2);

        size_t pitch_d_lP_rowchk_r = magma_roundup(panel_row * sizeof(double), 32);
        lddp_rowchk_r[d] = pitch_d_lP_rowchk_r / sizeof(double);
        magma_dmalloc(&d_lP_rowchk_r[d], pitch_d_lP_rowchk_r * ((ngpu * nb) / nb) * 2);
    }
    printf( "done.\n" );

    printf( "calculate initial checksum on GPUs......\n" );
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        col_chk_enc(gpu_row[d], gpu_col[d], nb, 
                    d_lA[d], ldda,  
                    dev_chk_v[d], ld_dev_chk_v[d], 
                    d_lA_colchk[d], ldda_colchk[d], 
                    queues[d][stream1]);

        row_chk_enc(gpu_row[d], gpu_col[d], nb, 
                    d_lA[d], ldda,  
                    dev_chk_v[d], ld_dev_chk_v[d], 
                    d_lA_rowchk[d], ldda_rowchk[d], 
                    queues[d][stream1]);
    }
    printf( "done.\n" );

    if (DEBUG) {

        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            printf( "on GPU %d:\n", d);
            printf( "input matrix A:\n" );
            printMatrix_gpu(d_lA[d], ldda, gpu_row[d], gpu_col[d], nb, nb, queues[d][stream1]);
            printf( "column chk:\n" );
            printMatrix_gpu(d_lA_colchk[d], ldda_colchk[d], 
                            (gpu_row[d] / nb) * 2, gpu_col[d], 2, nb, queues[d][stream1]);
            printf( "row chk:\n" );
            printMatrix_gpu(d_lA_rowchk[d], ldda_rowchk[d], 
                            gpu_row[d], (gpu_col[d] / nb) * 2, nb, 2, queues[d][stream1]);
        }
    }


    /* == initialize the trace */
    trace_init( 1, ngpu, 3, queues );

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
                trace_gpu_start( id, stream1, "syrk", "syrk" );
                magma_dsyrk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dlA(id, 0, nb*j_local), ldda,
                            d_one,     dlA(id, j, nb*j_local), ldda,
                            queues[id][stream1]);
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
                        if ( d == id ) {
                            dlpanel = dlA(d,0,nb*j_local);
                            ldpanel = ldda;
                            // the GPU owns the row from start, and no need of sync.
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
                                    c_one,     dlA(d, j, nb0), ldda,
                                    queues[d][stream2]);
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
                        magma_queue_wait_event( queues[d][stream1], events[d][2] ); // wait for gemm update
                        trace_gpu_start( d, stream1, "trsm", "trsm" );
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                            //magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_dlaset( MagmaFull, nb0,     jb,     c_zero, c_zero, dx(d,0), nb0 );
                            magmablas_dtrsm_work( MagmaLeft, MagmaUpper,
                                                  MagmaConjTrans, MagmaNonUnit,
                                                  jb, nb0, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, j, nb*j_local2), ldda,
                                                  dx(d,0), jb,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream1] );
                        #else
                            magma_dtrsm( MagmaLeft, MagmaUpper,
                                         MagmaConjTrans, MagmaNonUnit,
                                         jb, nb0, c_one,
                                         dlpanel,                ldpanel,
                                         dlA(d, j, nb*j_local2), ldda,
                                         queues[d][stream1] );
                        #endif
                        magma_event_record( events[d][4], queues[d][stream1] );
                        trace_gpu_end( d, stream1 );
                    } else if ( nb2 > 0 ) {
                        /* update all the blocks on stream2 */
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for cholesky factor
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                            //magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_dlaset( MagmaFull, nb2,     jb,     c_zero, c_zero, dx(d,0), nb2 );
                            magmablas_dtrsm_work( MagmaLeft, MagmaUpper,
                                                  MagmaConjTrans, MagmaNonUnit,
                                                  jb, nb2, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, j, nb*j_local2), ldda,
                                                  dx(d,0), jb,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream2] );
                        #else
                            magma_dtrsm( MagmaLeft, MagmaUpper,
                                         MagmaConjTrans, MagmaNonUnit,
                                         jb, nb2, c_one,
                                         dlpanel,                ldpanel,
                                         dlA(d, j, nb*j_local2), ldda,
                                         queues[d][stream2] );
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
                    magma_int_t d2, buf2;
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
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                            bool flag = 0;
                            if (flag == 0) {
                                magma_queue_wait_event( queues[d][stream2], events[d][4] ); // lookahead -> diagonal inversion
                            } else {
                                magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,flag), trsm_nb, queues[d][stream2] );
                                magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                            }
                            magmablas_dlaset( MagmaFull, nb2, jb, c_zero, c_zero, dx(d,1), nb2, queues[d][stream2] );
                            magmablas_dtrsm_work( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                                  jb, nb2, c_one,
                                                  dlpanel, ldpanel,
                                                  dlA(d, j, nb*j_local2+nb0), ldda,
                                                  dx(d,1), jb,
                                                  flag, dinvA(d,flag), dinvA_length,
                                                  queues[d][stream2] );
                        #else
                            magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for cholesky factor
                            magma_dtrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                         jb, nb2, c_one,
                                         dlpanel, ldpanel,
                                         dlA(d, j, nb*j_local2+nb0), ldda,
                                         queues[d][stream2] );
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
                // magma_dsyrk( MagmaLower, MagmaNoTrans, jb, j,
                //              d_neg_one, dlA(id, nb*j_local, 0), ldda,
                //              d_one,     dlA(id, nb*j_local, j), ldda,
                //              queues[id][stream1] );
                printf("syrk\n");
                abft_dsyrk(MagmaLower, MagmaNoTrans, jb, j,
                           d_neg_one, dlA(id, nb*j_local, 0), ldda,
                           d_one,     dlA(id, nb*j_local, j), ldda,
                           nb,
                           dlA_colchk(id, nb*j_local, 0),       ldda_colchk[id],
                           dlA_rowchk(id, nb*j_local, 0),       ldda_rowchk[id],
                           dlA_colchk_r(id, nb*j_local, 0),     ldda_colchk_r[id],
                           dlA_rowchk_r(id, nb*j_local, 0),     ldda_rowchk_r[id],
                           dlA_colchk(id, nb*j_local, j),       ldda_colchk[id],
                           dlA_rowchk(id, nb*j_local, j),       ldda_rowchk[id],
                           dlA_colchk_r(id, nb*j_local, j),     ldda_colchk_r[id],
                           dlA_rowchk_r(id, nb*j_local, j),     ldda_rowchk_r[id],
                           dev_chk_v[id],                       ld_dev_chk_v[id], 
                           FT,  DEBUG, CHECK_BEFORE, CHECK_AFTER,
                           queues[id][stream1], queues[id][stream1]);
            }

            /* send the diagonal to cpu on stream1 */
            magma_dgetmatrix_async( jb, jb,
                                    dlA(id, nb*j_local, j), ldda,
                                    Alo(j,j),               lda,
                                    queues[id][stream1] );
            if (FT) {
                /* send chk of diagonal to cpu on stream1 */
                magma_dgetmatrix_async( 2, jb,
                                        dlA_colchk(id, nb*j_local, j), ldda_colchk[id],
                                        Alo_colchk(j,j), ld_colchk,
                                        queues[id][stream1] );
            }

            /* update off-diagonal blocks of the panel */
            if ( j > 0 ) {
                d = (j/nb+1)%ngpu;
                for( dd=0; dd < ngpu; dd++ ) {
                    j_local2 = j_local+1;
                    if ( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
            
                    if ( nb0 < n_local[d] ) {
                        magma_setdevice(d);
                        if ( d == id ) {
                            dlpanel = dlA(d, nb*j_local, 0);
                            ldpanel = ldda;

                            dlpanel_colchk = dlA_colchk(d, nb*j_local, 0);
                            ldpanel_colchk = ldda_colchk[d];
                            dlpanel_rowchk = dlA_rowchk(d, nb*j_local, 0);
                            ldpanel_rowchk = ldda_rowchk[d];

                            dlpanel_colchk_r = dlA_colchk_r(d, nb*j_local, 0);
                            ldpanel_colchk_r = ldda_colchk_r[d];
                            dlpanel_rowchk_r = dlA_rowchk_r(d, nb*j_local, 0);
                            ldpanel_rowchk_r = ldda_rowchk_r[d];


                            magma_queue_wait_event( queues[d][stream2], events[d][4] ); // wait for look-ahead trsm to finish
                        } else {
                            dlpanel = dlPT(d,0,nb,buf);
                            ldpanel = nb;

                            dlpanel_colchk = dlPT_colchk(d, 0, nb, buf);
                            ldpanel_colchk = 2;
                            dlpanel_rowchk = dlPT_rowchk(d, 0, nb, buf);
                            ldpanel_rowchk = nb;

                            dlpanel_colchk_r= dlPT_colchk_r(d, 0, nb, buf);
                            ldpanel_colchk_r = 2;
                            dlpanel_rowchk_r = dlPT_rowchk_r(d, 0, nb, buf);
                            ldpanel_rowchk_r = nb;

                            magma_queue_wait_event( queues[d][stream2], events[d][0] ); // rows arrived at gpu
                        }
                        // magma_dgemm( MagmaNoTrans, MagmaConjTrans,
                        //              n_local[d]-nb0, jb, j,
                        //              c_neg_one, dlA(d, nb0, 0), ldda,
                        //                         dlpanel,        ldpanel,
                        //              c_one,     dlA(d, nb0, j), ldda,
                        //              queues[d][stream2] );
                        printf("dgemm\n");
                        abft_dgemm( MagmaNoTrans, MagmaConjTrans,
                                      n_local[d]-nb0, jb, j,
                                      c_neg_one, dlA(d, nb0, 0), ldda,
                                                 dlpanel,        ldpanel,
                                      c_one,     dlA(d, nb0, j), ldda,
                                      nb,
                                      dlA_colchk(d, nb0, 0),    ldda_colchk[d],
                                      dlA_rowchk(d, nb0, 0),    ldda_rowchk[d],
                                      dlA_colchk_r(d, nb0, 0),  ldda_colchk_r[d],
                                      dlA_rowchk_r(d, nb0, 0),  ldda_rowchk_r[d],
                                      dlpanel_colchk,   ldpanel_colchk,
                                      dlpanel_rowchk,   ldpanel_rowchk,
                                      dlpanel_colchk_r, ldpanel_colchk_r,
                                      dlpanel_rowchk_r, ldpanel_rowchk_r,
                                      dlA_colchk(d, nb0, j),    ldda_colchk[d],
                                      dlA_rowchk(d, nb0, j),    ldda_rowchk[d],
                                      dlA_colchk_r(d, nb0, j),  ldda_colchk_r[d],
                                      dlA_rowchk_r(d, nb0, j),  ldda_rowchk_r[d],
                                      dev_chk_v[d],     ld_dev_chk_v[d], 
                                      FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                                      queues[d][stream2], queues[d][stream2]);
                        magma_event_record( events[d][2], queues[d][stream2] );
                    }
                    d = (d+1)%ngpu;
                }
            }

            /* wait for the panel and factorized it on cpu */
            magma_setdevice(id);
            magma_queue_sync( queues[id][stream1] );
            //lapackf77_dpotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
            printf("j = %d, h = %d, size of A = %d, offset = %d\n", j, h, h*n*nb, ((j)+off_j)*lda  + (nb*(((j)/nb)%h)+off_i));
            abft_dpotf2(*MagmaLowerStr, jb, Alo(j,j), lda, info, 
                         nb, 
                         Alo_colchk(j,j),   ld_colchk, 
                         Alo_rowchk(j,j),   ld_rowchk, 
                         Alo_colchk_r(j,j),  ld_colchk_r, 
                         Alo_rowchk_r(j,j), ld_rowchk_r,
                         chk_v,  ld_chk_v, 
                         FT,  DEBUG, CHECK_BEFORE, CHECK_AFTER);

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

                        dlpanel_colchk = dlA_colchk(d, nb*j_local, j);
                        ldpanel_colchk = ldda_colchk[d];
                        dlpanel_rowchk = dlA_rowchk(d, nb*j_local, j);
                        ldpanel_rowchk = ldda_rowchk[d];

                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;

                        dlpanel_colchk = dlPT_colchk(d, 0, 0, buf);
                        ldpanel_colchk = 2;
                        dlpanel_rowchk = dlPT_rowchk(d, 0, 0, buf);
                        ldpanel_rowchk = nb;
                    }
                    magma_setdevice(d);
                    magma_dsetmatrix_async( jb, jb,
                                            Alo(j,j), lda,
                                            dlpanel,  ldpanel,
                                            queues[d][stream1] );

                    if (FT) {
                        magma_dsetmatrix_async( 2, jb,
                                                Alo_colchk(j,j), ld_colchk,
                                                dlpanel_colchk,  ldpanel_colchk,
                                                queues[d][stream1] );
                        magma_dsetmatrix_async( jb, 2,
                                                Alo_rowchk(j,j), ld_rowchk,
                                                dlpanel_rowchk,  ldpanel_rowchk,
                                                queues[d][stream1] );
                    }

                    magma_event_record( events[d][1], queues[d][stream1] );
                    d = (d+1)%ngpu;
                }
            } else {
                magma_setdevice(id);
                magma_dsetmatrix_async( jb, jb,
                                        Alo(j,j),               lda,
                                        dlA(id, nb*j_local, j), ldda,
                                        queues[id][stream1] );
                if (FT) {
                    magma_dsetmatrix_async( 2, jb,
                                            Alo_colchk(j,j), ld_colchk,
                                            dlA_colchk(d, nb*j_local, j),  ldda_colchk[d],
                                            queues[d][stream1] );
                    magma_dsetmatrix_async( jb, 2,
                                            Alo_rowchk(j,j), ld_rowchk,
                                            dlA_rowchk(d, nb*j_local, j),  ldda_rowchk[d],
                                            queues[d][stream1] );
                }
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

                        dlpanel_colchk = dlA_colchk(d, nb*j_local, j);
                        ldpanel_colchk = ldda_colchk[d];
                        dlpanel_rowchk = dlA_rowchk(d, nb*j_local, j);
                        ldpanel_rowchk = ldda_rowchk[d];

                        dlpanel_colchk_r = dlA_colchk_r(d, nb*j_local, j);
                        ldpanel_colchk_r = ldda_colchk_r[d];
                        dlpanel_rowchk_r = dlA_rowchk_r(d, nb*j_local, j);
                        ldpanel_rowchk_r = ldda_rowchk_r[d];

                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;

                        dlpanel_colchk = dlPT_colchk(d, 0, 0, buf);
                        ldpanel_colchk = 2;
                        dlpanel_rowchk = dlPT_rowchk(d, 0, 0, buf);
                        ldpanel_rowchk = nb;

                        dlpanel_colchk_r= dlPT_colchk_r(d, 0, 0, buf);
                        ldpanel_colchk_r = 2;
                        dlpanel_rowchk_r = dlPT_rowchk_r(d, 0, 0, buf);
                        ldpanel_rowchk_r = nb;

                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2);
                    
                    magma_setdevice(d);
                    if ( j+nb < n && d == (j/nb+1)%ngpu ) { /* owns next column, look-ahead next block on stream1 */
                        if ( j > 0 ) magma_queue_wait_event( queues[d][stream1], events[d][2] ); // wait for gemm update
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                            //magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_dlaset( MagmaFull, nb0,     jb,     c_zero, c_zero, dx(d,0), nb0 );
                            // magmablas_dtrsm_work( MagmaRight, MagmaLower,
                            //                       MagmaConjTrans, MagmaNonUnit,
                            //                       nb0, jb, c_one,
                            //                       dlpanel, ldpanel,
                            //                       dlA(d, nb*j_local2, j), ldda,
                            //                       dx(d,0), nb0,
                            //                       1, dinvA(d,0), dinvA_length,
                            //                       queues[d][stream1] ); 

                            printf("dtrsm_work\n");
                            abft_dtrsm_work(MagmaRight, MagmaLower,
                                            MagmaConjTrans, MagmaNonUnit,
                                            nb0, jb, c_one,
                                            dlpanel, ldpanel,
                                            dlA(d, nb*j_local2, j), ldda,
                                            dx(d,0), nb0,
                                            1, dinvA(d,0), dinvA_length,   
                                            nb,
                                            dlpanel_colchk,    ldpanel_colchk,
                                            dlpanel_rowchk,    ldpanel_rowchk,
                                            dlpanel_colchk_r,  ldpanel_colchk_r,
                                            dlpanel_rowchk_r,  ldpanel_rowchk_r,
                                            dlA_colchk(d, nb*j_local2, j),   ldda_colchk[d],
                                            dlA_rowchk(d, nb*j_local2, j),   ldda_rowchk[d],
                                            dlA_colchk_r(d, nb*j_local2, j), ldda_colchk_r[d],
                                            dlA_rowchk_r(d, nb*j_local2, j), ldda_rowchk_r[d],
                                            dev_chk_v[d],                    ld_dev_chk_v[d],
                                            FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                                            queues[d][stream1], queues[d][stream1]);
                            
                        #else
                            printf("dtrsm\n");
                            magma_dtrsm( MagmaRight, MagmaLower,
                                         MagmaConjTrans, MagmaNonUnit,
                                         nb0, jb, c_one,
                                         dlpanel, ldpanel,
                                         dlA(d, nb*j_local2, j), ldda,
                                         queues[d][stream1] );
                            
                        #endif
                        magma_event_record( events[d][4], queues[d][stream1] );
                    } else if ( nb2 > 0 ) { /* other gpus updating all the blocks on stream2 */
                        /* update the entire column */
                        magma_queue_wait_event( queues[d][stream2], events[d][1] ); // wait for the cholesky factor
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                            //magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,0), trsm_nb );
                            //magmablas_dlaset( MagmaFull, nb2,     jb,     c_zero, c_zero, dx(d,0), nb2 );
                            printf("dtrsm_work-other\n");
                            magmablas_dtrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                                  nb2, jb, c_one,
                                                  dlpanel,                ldpanel,
                                                  dlA(d, nb*j_local2, j), ldda,
                                                  dx(d,0), nb2,
                                                  1, dinvA(d,0), dinvA_length,
                                                  queues[d][stream2] );
                            
                        #else
                            printf("dtrsm-other\n");
                            magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                         nb2, jb, c_one,
                                         dlpanel,                ldpanel,
                                         dlA(d, nb*j_local2, j), ldda,
                                         queues[d][stream2] );
                            
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
                    magma_int_t d2, buf2;
//#define DPOTRF_DEVICE_TO_DEVICE
#ifdef DPOTRF_DEVICE_TO_DEVICE
                    // lookahead done
                    printf("gpu-2-gpu\n");
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
                    printf("gpu-2-cpu-2-gpu\n");
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

                            dlpanel_colchk = dlA_colchk(d, nb*j_local, j);
                            ldpanel_colchk = ldda_colchk[d];
                            dlpanel_rowchk = dlA_rowchk(d, nb*j_local, j);
                            ldpanel_rowchk = ldda_rowchk[d];

                            dlpanel_colchk_r = dlA_colchk_r(d, nb*j_local, j);
                            ldpanel_colchk_r = ldda_colchk_r[d];
                            dlpanel_rowchk_r = dlA_rowchk_r(d, nb*j_local, j);
                            ldpanel_rowchk_r = ldda_rowchk_r[d];

                        } else {
                            dlpanel = dlPT(d,0,0,buf);
                            ldpanel = nb;

                            dlpanel_colchk = dlPT_colchk(d, 0, 0, buf);
                            ldpanel_colchk = 2;
                            dlpanel_rowchk = dlPT_rowchk(d, 0, 0, buf);
                            ldpanel_rowchk = nb;

                            dlpanel_colchk_r= dlPT_colchk_r(d, 0, 0, buf);
                            ldpanel_colchk_r = 2;
                            dlpanel_rowchk_r = dlPT_rowchk_r(d, 0, 0, buf);
                            ldpanel_rowchk_r = nb;
                        
                        }
                        magma_setdevice(d);
                        /* update the remaining blocks in the column */
                        #if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(DTRSM_WORK)
                            bool flag = 0;
                            if (flag == 0) {
                                magma_queue_wait_event( queues[d][stream2], events[d][4] ); // lookahead -> diagonal inversion
                            } else {
                                magmablas_dlaset( MagmaFull, trsm_nb, trsm_n, c_zero, c_zero, dinvA(d,flag), trsm_nb, queues[d][stream2] );
                                magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                            }
                            magmablas_dlaset( MagmaFull, nb2, jb, c_zero, c_zero, dx(d,1), nb2, queues[d][stream2] );
                            // magmablas_dtrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                            //                       nb2, jb, c_one,
                            //                       dlpanel,                    ldpanel,
                            //                       dlA(d, nb*j_local2+nb0, j), ldda,
                            //                       dx(d,1), nb2,
                            //                       flag, dinvA(d,flag), dinvA_length,
                            //                       queues[d][stream2] );
                            printf("dtrsm_work-other2\n");
                            abft_dtrsm_work(MagmaRight, MagmaLower,
                                            MagmaConjTrans, MagmaNonUnit,
                                            nb2, jb, c_one,
                                            dlpanel, ldpanel,
                                            dlA(d, nb*j_local2+nb0, j), ldda,
                                            dx(d,1), nb2,
                                            flag, dinvA(d,flag), dinvA_length,
                                            nb,
                                            dlpanel_colchk,    ldpanel_colchk,
                                            dlpanel_rowchk,    ldpanel_rowchk,
                                            dlpanel_colchk_r,  ldpanel_colchk_r,
                                            dlpanel_rowchk_r,  ldpanel_rowchk_r,
                                            dlA_colchk(d, nb*j_local2+nb0, j),   ldda_colchk[d],
                                            dlA_rowchk(d, nb*j_local2+nb0, j),   ldda_rowchk[d],
                                            dlA_colchk_r(d, nb*j_local2+nb0, j), ldda_colchk_r[d],
                                            dlA_rowchk_r(d, nb*j_local2+nb0, j), ldda_rowchk_r[d],
                                            dev_chk_v[d],                    ld_dev_chk_v[d],
                                            FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
                                            queues[d][stream2], queues[d][stream2]);

                            
                        #else
                            magma_queue_wait_event( queues[d][stream2], events[d][1] ); // panel received
                            printf("dtrsm-other2\n");
                            magma_dtrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                         nb2, jb, c_one,
                                         dlpanel,                    ldpanel,
                                         dlA(d, nb*j_local2+nb0, j), ldda,
                                         queues[d][stream2] );
                            
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
        magma_free( dx[d] );
        #endif
    }
    magma_setdevice( orig_dev );

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
