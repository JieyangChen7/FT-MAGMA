/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zgeqrf_gpu.cpp normal z -> d, Fri Jan 30 19:00:15 2015
*/
#include "common_magma.h"
#include <iostream>
#include "FT.h"    

using namespace std;

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: 'a' is pointer to the current panel holding the
      Householder vectors for the QR factorization of the panel. This routine
      puts ones on the diagonal and zeros in the upper triangular part of 'a'.
      The upper triangular values are stored in work.
      
      Then, the inverse is calculated in place in work, so as a final result,
      work holds the inverse of the upper triangular diagonal block.
*/
void dsplit_diag_block(magma_int_t ib, double *a, magma_int_t lda, double *work)
{
    magma_int_t i, j, info;
    double *cola, *colw;
    double c_zero = MAGMA_D_ZERO;
    double c_one  = MAGMA_D_ONE;

    for (i=0; i < ib; i++) {
        cola = a    + i*lda;
        colw = work + i*ib;
        for (j=0; j < i; j++) {
            colw[j] = cola[j];
            cola[j] = c_zero;
        }
        colw[i] = cola[i];
        cola[i] = c_one;
    }
    lapackf77_dtrtri( MagmaUpperStr, MagmaNonUnitStr, &ib, work, &ib, &info);
}

/**
    Purpose
    -------
    DGEQRF computes a QR factorization of a real M-by-N matrix A:
    A = Q * R.
    
    This version stores the triangular dT matrices used in
    the block QR factorization so that they can be applied directly (i.e.,
    without being recomputed) later. As a result, the application
    of Q is much faster. Also, the upper triangular matrices for V have 0s
    in them. The corresponding parts of the upper triangular R are inverted
    and stored separately in dT.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      DOUBLE_PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    tau     DOUBLE_PRECISION array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    dT      (workspace) DOUBLE_PRECISION array on the GPU,
            dimension (2*MIN(M, N) + (N+31)/32*32 )*NB,
            where NB can be obtained through magma_get_dgeqrf_nb(M).
            It starts with MIN(M,N)*NB block that store the triangular T
            matrices, followed by the MIN(M,N)*NB block of the diagonal
            inverses for the R matrix. The rest of the array is used as workspace.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a real scalar, and v is a real vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_dgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_dgeqrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA,   magma_int_t ldda,
    double *tau,
    magmaDouble_ptr dT,
    magma_int_t *info )
{
    #define dA(a_1,a_2) (dA + (a_2)*(ldda) + (a_1))
    #define dT(a_1)     (dT + (a_1)*nb)
    #define d_ref(a_1)  (dT + (  minmn+(a_1))*nb)
    #define dd_ref(a_1) (dT + (2*minmn+(a_1))*nb)
    #define work(a_1)   (work + (a_1))
    #define hwork       (work + (nb)*(m))

    magma_int_t i, k, minmn, old_i, old_ib, rows, cols;
    magma_int_t ib, nb;
    magma_int_t ldwork, lddwork, lwork, lhwork;
    double *work, *ut;

    /* check arguments */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    k = minmn = min(m,n);
    if (k == 0)
        return *info;

    nb = magma_get_dgeqrf_nb(m);
    //nb = 4;
    nb = 128;
    lwork  = (m + n + nb)*nb;
    lhwork = lwork - m*nb;

    if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    
    ut = hwork+nb*(n);
    memset( ut, 0, nb*nb*sizeof(double));

    magma_queue_t stream[5];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_queue_create( &stream[2] );
    magma_queue_create( &stream[3] );
    magma_queue_create( &stream[4] );

    ldwork = m;
    lddwork= n;

    /* flags */
    bool FT = false;
    bool DEBUG = false;
    bool VERIFY = false;

    double * dT_col_chk;
    int dT_col_chk_ld;

    double * dT_row_chk;
    int dT_row_chk_ld;

    double * dwork_col_chk;
    int dwork_col_chk_ld;

    double * dwork_row_chk;
    int dwork_row_chk_ld;



    ABFTEnv * abftEnv = new ABFTEnv();
    initializeABFTEnv(abftEnv, nb, dA, ldda, m, n, m, nb, stream, 3, DEBUG);
    if (true) {
        

        /* allocate space for checksum of dT */
        cout << "allocate space for row checksum of dT......";
        size_t dT_row_chk_pitch = magma_roundup(nb * sizeof(double), 32);
        dT_row_chk_ld = dT_row_chk_pitch / sizeof(double);
        magma_dmalloc(&dT_row_chk, dT_row_chk_pitch * 2);
        cout << "done." << endl;

        //allocate space for checksum of dAP 
        cout << "allocate space for column checksum of dT......";
        size_t dT_col_chk_pitch = magma_roundup(2 * sizeof(double), 32);
        dT_col_chk_ld = dT_col_chk_pitch / sizeof(double);
        magma_dmalloc(&dT_col_chk, dT_col_chk_pitch * nb);
        cout << "done." << endl;


        /* allocate space for checksum of dT */
        cout << "allocate space for row checksum of dwork......";
        size_t dwork_row_chk_pitch = magma_roundup(n * sizeof(double), 32);
        dwork_row_chk_ld = dwork_row_chk_pitch / sizeof(double);
        magma_dmalloc(&dwork_row_chk, dT_row_chk_pitch * 2);
        cout << "done." << endl;

        //allocate space for checksum of dAP 
        cout << "allocate space for column checksum of dwork......";
        size_t dwork_col_chk_pitch = magma_roundup((n / abftEnv->chk_nb) * 2 * sizeof(double), 32);
        dwork_col_chk_ld = dwork_col_chk_pitch / sizeof(double);
        magma_dmalloc(&dwork_col_chk, dwork_col_chk_pitch * nb);
        cout << "done." << endl;

        // col_benchmark(abftEnv, dA, ldda);
        // row_benchmark(abftEnv, dA, ldda);

    }


    if ( (nb > 1) && (nb < k) ) {
        /* Use blocked code initially */
        cout << "nb=" << nb << endl;
        old_i = 0; old_ib = nb;
        for (i = 0; i < k-nb; i += nb) {
            //cout << "i=" << i << endl;
            ib = min(k-i, nb);
            rows = m -i;
            magma_dgetmatrix_async( rows, ib,
                                    dA(i,i),  ldda,
                                    work(i), ldwork, stream[1] );
            if (FT) {
                //transfer checksums to CPU
                cout << "ib=" << ib << endl;
                magma_dgetmatrix_async( rows, (ib / abftEnv->chk_nb) * 2,
                                        ROW_CHK(i, i),  abftEnv->row_dchk_ld,
                                        abftEnv->row_hchk, abftEnv->row_hchk_ld, stream[1] );
                magma_dgetmatrix_async( (rows /abftEnv->chk_nb) * 2, ib,
                                        COL_CHK(i, i),  abftEnv->col_dchk_ld,
                                        abftEnv->col_hchk, abftEnv->col_hchk_ld, stream[1] );
            }
            if (i > 0) {
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                cols = n-old_i-2*old_ib;
                // magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                //                   m-old_i, cols, old_ib,
                //                   dA(old_i, old_i         ), ldda, 
                //                   dT(old_i), nb,
                //                   dA(old_i, old_i+2*old_ib), ldda, 
                //                   dd_ref(0),    lddwork);

                if (FT) {
                    cudaMemset2D(dd_ref(0), lddwork * sizeof(double), 0, n * sizeof(double), nb);
                    cudaMemset2D(dwork_row_chk, dwork_row_chk_ld * sizeof(double), 0, n * sizeof(double), 2);
                    cudaMemset2D(dwork_col_chk, dwork_col_chk_ld * sizeof(double), 0, (n / abftEnv->chk_nb) * 2 * sizeof(double), nb);
                }
                VERIFY = updateCounter(abftEnv, old_i / nb, m / nb - 1, (old_i+2*old_ib) / nb, n / nb - 1, 1);
                dlarfbFT( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                           m-old_i, cols, old_ib,
                              dA(old_i, old_i         ), ldda, 
                              dT(old_i), nb,
                              dA(old_i, old_i+2*old_ib), ldda, 
                              dd_ref(0),    lddwork,
                              abftEnv,
                              COL_CHK(old_i / abftEnv->chk_nb, old_i /abftEnv->chk_nb), abftEnv->col_dchk_ld,
                              ROW_CHK(old_i / abftEnv->chk_nb, old_i /abftEnv->chk_nb), abftEnv->row_dchk_ld,
                              dT_col_chk, dT_col_chk_ld,
                              dT_row_chk, dT_row_chk_ld,
                              COL_CHK(old_i / abftEnv->chk_nb, (old_i+2*old_ib) /abftEnv->chk_nb), abftEnv->col_dchk_ld,
                              ROW_CHK(old_i / abftEnv->chk_nb, (old_i+2*old_ib) /abftEnv->chk_nb), abftEnv->row_dchk_ld,
                              dwork_col_chk, dwork_col_chk_ld,
                              dwork_row_chk, dwork_row_chk_ld,
                              FT, DEBUG, VERIFY, stream);
                
                /* store the diagonal */
                magma_dsetmatrix_async( old_ib, old_ib,
                                        ut,           old_ib,
                                        d_ref(old_i), old_ib, stream[0] );
            }

            magma_queue_sync( stream[1] );


            //lapackf77_dgeqrf(&rows, &ib, work(i), &ldwork, tau+i, hwork, &lhwork, info);
            dgeqrfFT(rows, ib, work(i), ldwork, tau+i, hwork, lhwork, info, abftEnv, false, false, false);
            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_dlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              work(i), &ldwork, tau+i, hwork, &ib);

            /* Put 0s in the upper triangular part of a panel (and 1s on the
               diagonal); copy the upper triangular in ut and invert it. */
            magma_queue_sync( stream[0] );
            dsplit_diag_block(ib, work(i), ldwork, ut);
            magma_dsetmatrix( rows, ib, work(i), ldwork, dA(i,i), ldda );
            if (FT) {

                //transfer checksums to GPU
                magma_dgetmatrix_async( rows, (ib / abftEnv->chk_nb) * 2,
                                        abftEnv->row_hchk, abftEnv->row_hchk_ld, 
                                        ROW_CHK(i, i),  abftEnv->row_dchk_ld, stream[1] );
                magma_dgetmatrix_async( (rows /abftEnv->chk_nb) * 2, ib,
                                        abftEnv->col_hchk, abftEnv->col_hchk_ld,
                                        COL_CHK(i, i),  abftEnv->col_dchk_ld, stream[1] );
            }

            if (i + ib < n) {
                /* Send the triangular factor T to the GPU */
                magma_dsetmatrix( ib, ib, hwork, ib, dT(i), nb );

                if (FT) {
                    /* calucate the row/col checksums for dT*/
                    // magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                    //     2, abftEnv->chk_nb, abftEnv->chk_nb,
                    //     MAGMA_D_ONE, 
                    //     abftEnv->hrz_vd, abftEnv->hrz_vd_ld,
                    //     dT(i), nb,
                    //     MAGMA_D_ZERO, 
                    //     dT_col_chk, dT_col_chk_ld);  

                    col_checksum_kernel_ccns4(ib, ib, abftEnv->chk_nb,
                                            dT(i), nb,
                                            abftEnv->vrt_vd, abftEnv->vrt_vd_ld,
                                            dT_col_chk, dT_col_chk_ld,
                                            abftEnv->stream);

                    row_checksum_kernel_cccs4(ib, ib, abftEnv->chk_nb,
                                            dT(i), nb,
                                            abftEnv->vrt_vd, abftEnv->vrt_vd_ld,
                                            dT_row_chk, dT_row_chk_ld,
                                            abftEnv->stream);

                    // magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                    //     abftEnv->chk_nb, 2, abftEnv->chk_nb,
                    //     MAGMA_D_ONE, 
                    //     dT(i), nb,
                    //     abftEnv->vrt_vd, abftEnv->vrt_vd_ld,
                    //     MAGMA_D_ZERO, 
                    //     dT_row_chk, dT_row_chk_ld);     


                    col_checksum_kernel_ccns4(m - i, ib, abftEnv->chk_nb,
                                            dA(i, i   ), ldda,
                                            abftEnv->vrt_vd, abftEnv->vrt_vd_ld,
                                            COL_CHK(i / abftEnv->chk_nb, i /abftEnv->chk_nb), abftEnv->col_dchk_ld,
                                            abftEnv->stream);

                    row_checksum_kernel_cccs4(m - i, ib, abftEnv->chk_nb,
                                            dA(i, i   ), ldda,
                                            abftEnv->vrt_vd, abftEnv->vrt_vd_ld,
                                            ROW_CHK(i / abftEnv->chk_nb, i /abftEnv->chk_nb), abftEnv->row_dchk_ld,
                                            abftEnv->stream);

                    // /* calucate the row/col checksums for dV*/
                    // for (int p = i; p < m; p += nb) {
                    //     magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                    //         2, abftEnv->chk_nb, abftEnv->chk_nb,
                    //         MAGMA_D_ONE, 
                    //         abftEnv->hrz_vd, abftEnv->hrz_vd_ld,
                    //         dA(p, i   ), ldda,
                    //         MAGMA_D_ZERO, 
                    //         COL_CHK(p / abftEnv->chk_nb, i /abftEnv->chk_nb), abftEnv->col_dchk_ld);  

                    //     magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                    //         abftEnv->chk_nb, 2, abftEnv->chk_nb,
                    //         MAGMA_D_ONE, 
                    //         dA(p, i   ), ldda,
                    //         abftEnv->vrt_vd, abftEnv->vrt_vd_ld,
                    //         MAGMA_D_ZERO, 
                    //         ROW_CHK(p / abftEnv->chk_nb, i /abftEnv->chk_nb), abftEnv->row_dchk_ld);     
                    // }
                }

                if (FT) {
                    cudaMemset2D(dd_ref(0), lddwork * sizeof(double), 0, n * sizeof(double), nb);
                    cudaMemset2D(dwork_row_chk, dwork_row_chk_ld * sizeof(double), 0, n * sizeof(double), 2);
                    cudaMemset2D(dwork_col_chk, dwork_col_chk_ld * sizeof(double), 0, (n / abftEnv->chk_nb) * 2 * sizeof(double), nb);
                }
                if (i+nb < k-nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left */
                    VERIFY = updateCounter(abftEnv, i / nb, m / nb - 1, (i+ib) / nb, (i+2*ib) / nb - 1, 1);
                    dlarfbFT( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                              rows, ib, ib,
                              dA(i, i   ), ldda, dT(i),  nb,
                              dA(i, i+ib), ldda, dd_ref(0), lddwork,
                              abftEnv,
                              COL_CHK(i / abftEnv->chk_nb, i /abftEnv->chk_nb), abftEnv->col_dchk_ld,
                              ROW_CHK(i / abftEnv->chk_nb, i /abftEnv->chk_nb), abftEnv->row_dchk_ld,
                              dT_col_chk, dT_col_chk_ld,
                              dT_row_chk, dT_row_chk_ld,
                              COL_CHK(i / abftEnv->chk_nb, i /abftEnv->chk_nb + 1), abftEnv->col_dchk_ld,
                              ROW_CHK(i / abftEnv->chk_nb, i /abftEnv->chk_nb + 1), abftEnv->row_dchk_ld,
                              dwork_col_chk, dwork_col_chk_ld,
                              dwork_row_chk, dwork_row_chk_ld,
                              FT, DEBUG, VERIFY, stream);
                }
                else {
                    cols = n-i-ib;
                    magma_dlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, cols, ib,
                                      dA(i, i   ), ldda, dT(i),  nb,
                                      dA(i, i+ib), ldda, dd_ref(0), lddwork);
                    /* Fix the diagonal block */
                    magma_dsetmatrix( ib, ib, ut, ib, d_ref(i), ib );
                }
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }

    /* Use unblocked code to factor the last or only block. */
    if (i < k) {
        ib   = n-i;
        rows = m-i;
        magma_dgetmatrix( rows, ib, dA(i, i), ldda, work, rows );
        lhwork = lwork - rows*ib;
        lapackf77_dgeqrf(&rows, &ib, work, &rows, tau+i, work+ib*rows, &lhwork, info);
        
        magma_dsetmatrix( rows, ib, work, rows, dA(i, i), ldda );
    }

    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_free_pinned( work );
    return *info;
} /* magma_dgeqrf_gpu */

#undef dA
#undef dT
#undef d_ref
#undef work
