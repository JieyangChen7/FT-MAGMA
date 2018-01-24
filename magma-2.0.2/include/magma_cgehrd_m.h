/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016

       @generated from include/magma_zgehrd_m.h normal z -> c, Mon May  2 23:31:25 2016
       @author Mark Gates
*/

#ifndef MAGMA_CGEHRD_H
#define MAGMA_CGEHRD_H

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cgehrd_data
{
    magma_int_t ngpu;
    
    magma_int_t ldda;
    magma_int_t ldv;
    magma_int_t ldvd;
    
    magmaFloatComplex *A    [ MagmaMaxGPUs ];  // ldda*nlocal
    magmaFloatComplex *V    [ MagmaMaxGPUs ];  // ldv *nb, whole panel
    magmaFloatComplex *Vd   [ MagmaMaxGPUs ];  // ldvd*nb, block-cyclic
    magmaFloatComplex *Y    [ MagmaMaxGPUs ];  // ldda*nb
    magmaFloatComplex *W    [ MagmaMaxGPUs ];  // ldda*nb
    magmaFloatComplex *Ti   [ MagmaMaxGPUs ];  // nb*nb
    
    magma_queue_t queues[ MagmaMaxGPUs ];
};

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMA_CGEHRD_H
