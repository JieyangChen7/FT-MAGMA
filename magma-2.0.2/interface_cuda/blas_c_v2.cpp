/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
 
       @author Mark Gates
       @generated from interface_cuda/blas_z_v2.cpp normal z -> c, Mon May  2 23:29:59 2016
*/
#include "magma_internal.h"
#include "error.h"

#define COMPLEX

#ifdef HAVE_CUBLAS

// ========================================
// Level 1 BLAS

// --------------------
/** Returns index of element of vector x having max. absolute value;
    i.e., max (infinity) norm.
    
    @param[in]
    n       Number of elements in vector x. n >= 0.
            
    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            
    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.
            
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" magma_int_t
magma_icamax_q(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue )
{
    int result; /* not magma_int_t */
    cublasIcamax( queue->cublas_handle(), n, dx, incx, &result );
    return result;
}

// --------------------
/** Returns index of element of vector x having min. absolute value.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" magma_int_t
magma_icamin_q(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue )
{
    int result; /* not magma_int_t */
    cublasIcamin( queue->cublas_handle(), n, dx, incx, &result );
    return result;
}

// --------------------
/** Returns the sum of absolute values of vector x; i.e., one norm.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" float
magma_scasum_q(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue )
{
    float result;
    cublasScasum( queue->cublas_handle(), n, dx, incx, &result );
    return result;
}

// --------------------
/** Constant times a vector plus a vector; \f$ y = \alpha x + y \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_caxpy_q(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue )
{
    cublasCaxpy( queue->cublas_handle(), n, &alpha, dx, incx, dy, incy );
}

// --------------------
/** Copy vector x to vector y; \f$ y = x \f$.

    @param[in]
    n       Number of elements in vectors x and y. n >= 0.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_ccopy_q(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue )
{
    cublasCcopy( queue->cublas_handle(), n, dx, incx, dy, incy );
}

// --------------------
/** Returns dot product of vectors x and y; \f$ x^H y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C"
magmaFloatComplex magma_cdotc(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    magmaFloatComplex result;
    cublasCdotc( queue->cublas_handle(), n, dx, incx, dy, incy, &result );
    return result;
}

#ifdef COMPLEX
// --------------------
/** Returns dot product (unconjugated) of vectors x and y; \f$ x^T y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C"
magmaFloatComplex magma_cdotu(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    magmaFloatComplex result;
    cublasCdotu( queue->cublas_handle(), n, dx, incx, dy, incy, &result );
    return result;
}
#endif // COMPLEX

// --------------------
/** Returns 2-norm of vector x. Avoids unnecesary over/underflow.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" float
magma_scnrm2_q(
    magma_int_t n,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue )
{
    float result;
    cublasScnrm2( queue->cublas_handle(), n, dx, incx, &result );
    return result;
}

// --------------------
/** Apply Givens plane rotation, where cos (c) is real and sin (s) is complex.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       float. cosine.

    @param[in]
    s       COMPLEX. sine. c and s define a rotation
            [ c         s ]  where c*c + s*conj(s) = 1.
            [ -conj(s)  c ]

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_crot_q(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float c, magmaFloatComplex s,
    magma_queue_t queue )
{
    cublasCrot( queue->cublas_handle(), n, dx, incx, dy, incy, &c, &s );
}

#ifdef COMPLEX
// --------------------
/** Apply Givens plane rotation, where cos (c) and sin (s) are real.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).
            On output, overwritten with c*x + s*y.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).
            On output, overwritten with -conj(s)*x + c*y.

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    c       float. cosine.

    @param[in]
    s       float. sine. c and s define a rotation
            [  c  s ]  where c*c + s*s = 1.
            [ -s  c ]

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_csrot_q(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    float c, float s,
    magma_queue_t queue )
{
    cublasCsrot( queue->cublas_handle(), n, dx, incx, dy, incy, &c, &s );
}
#endif // COMPLEX

#ifdef REAL
// --------------------
/** Apply modified plane rotation.

    @ingroup magma_cblas1
*/
extern "C" void
magma_crotm_q(
    magma_int_t n,
    float *dx, magma_int_t incx,
    float *dy, magma_int_t incy,
    const float *param,
    magma_queue_t queue )
{
    cublasCrotm( queue->cublas_handle(), n, dx, incx, dy, incy, param );
}

// --------------------
/** Generate modified plane rotation.

    @ingroup magma_cblas1
*/
extern "C" void
magma_crotmg_q(
    float *d1, float       *d2,
    float *x1, const float *y1,
    float *param,
    magma_queue_t queue )
{
    cublasCrotmg( queue->cublas_handle(), d1, d2, x1, y1, param );
}
#endif // REAL

// --------------------
/** Scales a vector by a constant; \f$ x = \alpha x \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_cscal_q(
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magma_queue_t queue )
{
    cublasCscal( queue->cublas_handle(), n, &alpha, dx, incx );
}

#ifdef COMPLEX
// --------------------
/** Scales a vector by a real constant; \f$ x = \alpha x \f$.

    @param[in]
    n       Number of elements in vector x. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$ (real)

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx > 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_csscal_q(
    magma_int_t n,
    float alpha,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magma_queue_t queue )
{
    cublasCsscal( queue->cublas_handle(), n, &alpha, dx, incx );
}
#endif // COMPLEX

// --------------------
/** Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas1
*/
extern "C" void
magma_cswap_q(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    cublasCswap( queue->cublas_handle(), n, dx, incx, dy, incy );
}


// ========================================
// Level 2 BLAS

// --------------------
/** Perform matrix-vector product.
        \f$ y = \alpha A   x + \beta y \f$  (transA == MagmaNoTrans), or \n
        \f$ y = \alpha A^T x + \beta y \f$  (transA == MagmaTrans),   or \n
        \f$ y = \alpha A^H x + \beta y \f$  (transA == MagmaConjTrans).

    @param[in]
    transA  Operation to perform on A.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX array on GPU device.
            If transA == MagmaNoTrans, the n element vector x of dimension (1 + (n-1)*incx); \n
            otherwise,                 the m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX array on GPU device.
            If transA == MagmaNoTrans, the m element vector y of dimension (1 + (m-1)*incy); \n
            otherwise,                 the n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_cgemv_q(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue )
{
    cublasCgemv(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        m, n,
        &alpha, dA, ldda,
                dx, incx,
        &beta,  dy, incy );
}

// --------------------
/** Perform rank-1 update, \f$ A = \alpha x y^H + A \f$.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX array on GPU device.
            The m-by-n matrix A of dimension (ldda,n), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_cgerc_q(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue )
{
    cublasCgerc(
        queue->cublas_handle(),
        m, n,
        &alpha, dx, incx,
                dy, incy,
                dA, ldda );
}

#ifdef COMPLEX
// --------------------
/** Perform rank-1 update (unconjugated), \f$ A = \alpha x y^H + A \f$.

    @param[in]
    m       Number of rows of A. m >= 0.

    @param[in]
    n       Number of columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,m).
            The m-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_cgeru_q(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue )
{
    cublasCgeru(
        queue->cublas_handle(),
        m, n,
        &alpha, dx, incx,
                dy, incy,
                dA, ldda );
}
#endif // COMPLEX

// --------------------
/** Perform Hermitian matrix-vector product, \f$ y = \alpha A x + \beta y \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX array on GPU device.
            The m element vector x of dimension (1 + (m-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_chemv_q(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue )
{
    cublasChemv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        n,
        &alpha, dA, ldda,
                dx, incx,
        &beta,  dy, incy );
}

// --------------------
/** Perform Hermitian rank-1 update, \f$ A = \alpha x x^H + A \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced. 

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_cher_q(
    magma_uplo_t uplo,
    magma_int_t n,
    float alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue )
{
    cublasCher(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        n,
        &alpha, dx, incx,
                dA, ldda );
}

// --------------------
/** Perform Hermitian rank-2 update, \f$ A = \alpha x y^H + conj(\alpha) y x^H + A \f$.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced. 

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in,out]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_cher2_q(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dx, magma_int_t incx,
    magmaFloatComplex_const_ptr dy, magma_int_t incy,
    magmaFloatComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue )
{
    cublasCher2(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        n,
        &alpha, dx, incx,
                dy, incy,
                dA, ldda );
}

// --------------------
/** Perform triangular matrix-vector product.
        \f$ x = A   x \f$  (trans == MagmaNoTrans), or \n
        \f$ x = A^T x \f$  (trans == MagmaTrans),   or \n
        \f$ x = A^H x \f$  (trans == MagmaConjTrans).

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced. 

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_ctrmv_q(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx,
    magma_queue_t queue )
{
    cublasCtrmv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// --------------------
/** Solve triangular matrix-vector system (one right-hand side).
        \f$ A   x = b \f$  (trans == MagmaNoTrans), or \n
        \f$ A^T x = b \f$  (trans == MagmaTrans),   or \n
        \f$ A^H x = b \f$  (trans == MagmaConjTrans).

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced. 

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    n       Number of rows and columns of A. n >= 0.

    @param[in]
    dA      COMPLEX array of dimension (ldda,n), ldda >= max(1,n).
            The n-by-n matrix A, on GPU device.

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            On entry, the n element RHS vector b of dimension (1 + (n-1)*incx).
            On exit, overwritten with the solution vector x.

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas2
*/
extern "C" void
magma_ctrsv_q(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dx, magma_int_t incx,
    magma_queue_t queue )
{
    cublasCtrsv(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        n,
        dA, ldda,
        dx, incx );
}

// ========================================
// Level 3 BLAS

// --------------------
/** Perform matrix-matrix product, \f$ C = \alpha op(A) op(B) + \beta C \f$.

    @param[in]
    transA  Operation op(A) to perform on matrix A.

    @param[in]
    transB  Operation op(B) to perform on matrix B.

    @param[in]
    m       Number of rows of C and op(A). m >= 0.

    @param[in]
    n       Number of columns of C and op(B). n >= 0.

    @param[in]
    k       Number of columns of op(A) and rows of op(B). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If transA == MagmaNoTrans, the m-by-k matrix A of dimension (ldda,k), ldda >= max(1,m); \n
            otherwise,                 the k-by-m matrix A of dimension (ldda,m), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX array on GPU device.
            If transB == MagmaNoTrans, the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k); \n
            otherwise,                 the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_cgemm_q(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasCgemm(
        queue->cublas_handle(),
        cublas_trans_const( transA ),
        cublas_trans_const( transB ),
        m, n, k,
        &alpha, dA, ldda,
                dB, lddb,
        &beta,  dC, lddc );
}

// --------------------
/** Perform symmetric matrix-matrix product.
        \f$ C = \alpha A B + \beta C \f$ (side == MagmaLeft), or \n
        \f$ C = \alpha B A + \beta C \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is symmetric.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    m       Number of rows of C. m >= 0.

    @param[in]
    n       Number of columns of C. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If side == MagmaLeft, the m-by-m symmetric matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n symmetric matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_csymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasCsymm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        &alpha, dA, ldda,
                dB, lddb,
        &beta,  dC, lddc );
}

// --------------------
/** Perform symmetric rank-k update.
        \f$ C = \alpha A A^T + \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T A + \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A (for MagmaNoTrans) or rows of A (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_csyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasCsyrk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        &alpha, dA, ldda,
        &beta,  dC, lddc );
}

// --------------------
/** Perform symmetric rank-2k update.
        \f$ C = \alpha A B^T + \alpha B A^T \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T B + \alpha B^T A \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is symmetric.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A and B.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A and B (for MagmaNoTrans) or rows of A and B (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The n-by-n symmetric matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_csyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasCsyr2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        &alpha, dA, ldda,
                dB, lddb,
        &beta,  dC, lddc );
}

#ifdef COMPLEX
// --------------------
/** Perform Hermitian matrix-matrix product.
        \f$ C = \alpha A B + \beta C \f$ (side == MagmaLeft), or \n
        \f$ C = \alpha B A + \beta C \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is Hermitian.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether the upper or lower triangle of A is referenced.

    @param[in]
    m       Number of rows of C. m >= 0.

    @param[in]
    n       Number of columns of C. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If side == MagmaLeft, the m-by-m Hermitian matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n Hermitian matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The m-by-n matrix C of dimension (lddc,n), lddc >= max(1,m).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_chemm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasChemm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        m, n,
        &alpha, dA, ldda,
                dB, lddb,
        &beta,  dC, lddc );
}

// --------------------
/** Perform Hermitian rank-k update.
        \f$ C = \alpha A A^T + \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T A + \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A (for MagmaNoTrans) or rows of A (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The n-by-n Hermitian matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_cherk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    float alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasCherk(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        &alpha, dA, ldda,
        &beta,  dC, lddc );
}

// --------------------
/** Perform Hermitian rank-2k update.
        \f$ C = \alpha A B^T + \alpha B A^T \beta C \f$ (trans == MagmaNoTrans), or \n
        \f$ C = \alpha A^T B + \alpha B^T A \beta C \f$ (trans == MagmaTrans),      \n
        where \f$ C \f$ is Hermitian.

    @param[in]
    uplo    Whether the upper or lower triangle of C is referenced.

    @param[in]
    trans   Operation to perform on A and B.

    @param[in]
    n       Number of rows and columns of C. n >= 0.

    @param[in]
    k       Number of columns of A and B (for MagmaNoTrans) or rows of A and B (for MagmaTrans). k >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix A of dimension (ldda,k), ldda >= max(1,n); \n
            otherwise,                the k-by-n matrix A of dimension (ldda,n), ldda >= max(1,k).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX array on GPU device.
            If trans == MagmaNoTrans, the n-by-k matrix B of dimension (lddb,k), lddb >= max(1,n); \n
            otherwise,                the k-by-n matrix B of dimension (lddb,n), lddb >= max(1,k).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    beta    Scalar \f$ \beta \f$

    @param[in,out]
    dC      COMPLEX array on GPU device.
            The n-by-n Hermitian matrix C of dimension (lddc,n), lddc >= max(1,n).

    @param[in]
    lddc    Leading dimension of dC.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_cher2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    float beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    cublasCher2k(
        queue->cublas_handle(),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        n, k,
        &alpha, dA, ldda,
                dB, lddb,
        &beta,  dC, lddc );
}
#endif // COMPLEX

// --------------------
/** Perform triangular matrix-matrix product.
        \f$ B = \alpha op(A) B \f$ (side == MagmaLeft), or \n
        \f$ B = \alpha B op(A) \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is triangular.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether A is upper or lower triangular.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    m       Number of rows of B. m >= 0.

    @param[in]
    n       Number of columns of B. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If side == MagmaLeft, the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n); \n
            otherwise,            the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in]
    dB      COMPLEX array on GPU device.
            The m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_ctrmm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    cublasCtrmm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        &alpha, dA, ldda,
                dB, lddb,
                dB, lddb );  /* C same as B; less efficient */
}

// --------------------
/** Solve triangular matrix-matrix system (multiple right-hand sides).
        \f$ op(A) X = \alpha B \f$ (side == MagmaLeft), or \n
        \f$ X op(A) = \alpha B \f$ (side == MagmaRight),   \n
        where \f$ A \f$ is triangular.

    @param[in]
    side    Whether A is on the left or right.

    @param[in]
    uplo    Whether A is upper or lower triangular.

    @param[in]
    trans   Operation to perform on A.

    @param[in]
    diag    Whether the diagonal of A is assumed to be unit or non-unit.

    @param[in]
    m       Number of rows of B. m >= 0.

    @param[in]
    n       Number of columns of B. n >= 0.

    @param[in]
    alpha   Scalar \f$ \alpha \f$

    @param[in]
    dA      COMPLEX array on GPU device.
            If side == MagmaLeft, the m-by-m triangular matrix A of dimension (ldda,m), ldda >= max(1,m); \n
            otherwise,            the n-by-n triangular matrix A of dimension (ldda,n), ldda >= max(1,n).

    @param[in]
    ldda    Leading dimension of dA.

    @param[in,out]
    dB      COMPLEX array on GPU device.
            On entry, m-by-n matrix B of dimension (lddb,n), lddb >= max(1,m).
            On exit, overwritten with the solution matrix X.

    @param[in]
    lddb    Leading dimension of dB.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_cblas3
*/
extern "C" void
magma_ctrsm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue )
{
    cublasCtrsm(
        queue->cublas_handle(),
        cublas_side_const( side ),
        cublas_uplo_const( uplo ),
        cublas_trans_const( trans ),
        cublas_diag_const( diag ),
        m, n,
        &alpha, dA, ldda,
                dB, lddb );
}

#endif // HAVE_CUBLAS

#undef COMPLEX
