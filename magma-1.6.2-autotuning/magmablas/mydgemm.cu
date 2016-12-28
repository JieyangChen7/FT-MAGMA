/*
    Enhanced Online ABFT
    UC Riverside
    Jieyang Chen
*/
#include "FT.h"
#include "common_magma.h"
#include "magma.h"
#include <stdlib.h>

#define NB 512
// encoding checksum for A

__global__ void
chkenc_kernel(double * A, int lda, double * Chk , int ldchk)
{
//	printf("start kernel\n");
    //blockIdx.x: determin the column to process
	A = A + blockIdx.x * lda;

	__shared__ double cache1[NB];
	__shared__ double cache2[NB];
	
	//load one column to cache
	cache1[threadIdx.x] = A[threadIdx.x];
	cache2[threadIdx.x] = cache1[threadIdx.x] * (threadIdx.x + 1); //add weights

	__syncthreads();

	int i = blockDim.x / 2;

	while (i != 0) {
		if (threadIdx.x < i)
			cache1[threadIdx.x] += cache1[threadIdx.x + i];
		    cache2[threadIdx.x] += cache2[threadIdx.x + i];
		__syncthreads();
		i /= 2;
		if (threadIdx.x == 0) {
			printf("i=%d\n", i);
		}
	}
/*
	if (threadIdx.x == 0) {
		*(Chk + blockIdx.x * ldchk) = cache1[0];
		*(Chk + blockIdx.x * ldchk + 1) = cache2[0];

	}  

	*/ 
}


void chkenc(double * A, int lda, int m, int n, double * Chk , int ldchk, magma_queue_t stream) {
	chkenc_kernel<<<n, m>>>(A, lda, Chk, ldchk);
}



