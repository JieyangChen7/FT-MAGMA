/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zcaxpycp.cu mixed zc -> ds, Fri Jan 30 19:00:07 2015

*/
#include "FT.h"
#include "common_magma.h"

#define NB 64

// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
__global__ void
test(
    int m)
{
    
}



// ----------------------------------------------------------------------
// adds   x += r (including conversion to double)  --and--
// copies w = b
extern "C" void
test_abft() 
{
	
	
	test <<< 1, 1, 1>>> ( 1 );
}

