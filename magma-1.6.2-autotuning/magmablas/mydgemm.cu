/*
    Enhanced Online ABFT
    UC Riverside
    Jieyang Chen
*/
#include "FT.h"
#include "common_magma.h"
#include "magma.h"
#include <stdlib.h>

#define NB 4
// encoding checksum for A
#define B 32
#define rB 32
#define cB 32
#define N 30720

__global__ void
chkenc_kernel(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
	A = A + blockIdx.x * lda;

	__shared__ double cache[NB];
	
	//load one column to cache
	cache[threadIdx.x] = A[threadIdx.x];

	__syncthreads();

	/* logrithm reduction */
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i)
			cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		*(Chk + blockIdx.x * ldchk + 1) = cache[0];
	}


	//load one column to cache
	cache[threadIdx.x] = A[threadIdx.x] * (threadIdx.x + 1);

	__syncthreads();

	i = blockDim.x / 2;

	while (i != 0) {
		if (threadIdx.x < i)
			cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		*(Chk + blockIdx.x * ldchk + 1) = cache[0];
	}

	
}


__global__ void
chkenc_kernel2(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    int idx = blockIdx.x * NB + threadIdx.x;

	A = A + idx * lda;

	double temp = 0;
	double temp2 = 0;
	for (int i = 0; i < NB; i++) {
		temp += A[i];
		temp2 += A[i] * (i+1);
	}
	*(Chk + idx * ldchk) = temp;
	*(Chk + idx * ldchk+1) = temp2;
	
}


__global__ void
chkenc_kernel3(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    int idx = blockIdx.x * B;

    double sum1 = 0;
    double sum2 = 0;

	A = A + idx * lda;

	__shared__ double cache[B][B];

	for (int i = 0; i < NB; i += B) {
		
		//load a block to cache
		for (int j = 0; j < B; j++) {
			cache[threadIdx.x][j] = *(A + j * lda + threadIdx.x);
			//if (blockIdx.x == 0 && threadIdx.x == 0) {
			//	printf("%f ", cache[threadIdx.x][j]);
			//}
		}

		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//		printf("\n");
	    //}
		__syncthreads();

		for (int j = 0; j < B; j++) {
			sum1 += cache[j][threadIdx.x];
			sum2 += cache[j][threadIdx.x] * (i + j + 1);
			//if (blockIdx.x == 0 && threadIdx.x == 0) {
			//	printf("%f ", cache[j][threadIdx.x]);
			//}
		}
		//if (blockIdx.x == 0 && threadIdx.x == 0) {
		//		printf("\n");
	    //}
		__syncthreads();
		A = A + B;
	}

	idx += threadIdx.x;

	*(Chk + idx * ldchk) = sum1;
	*(Chk + idx * ldchk+1) = sum2;
	
}




void chkenc(double * A, int lda, int m, int n, double * Chk , int ldchk, magma_queue_t stream) {
  /*  int numBlocks; // Occupancy in terms of active blocks 
    int blockSize = 32; 
	int device; 
	cudaDeviceProp prop; 
	int activeWarps; 
	int maxWarps; 
	cudaGetDevice(&device); 
	cudaGetDeviceProperties(&prop, device); cudaOccupancyMaxActiveBlocksPerMultiprocessor( &numBlocks, chkenc_kernel4, blockSize, 0); 
	activeWarps = numBlocks * blockSize / prop.warpSize; 
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize; 
	//printf("Occupancy: %f \n", (double)activeWarps / maxWarps * 100 );
	*/
	cudaFuncSetCacheConfig(chkenc_kernel, cudaFuncCachePreferShared);
	chkenc_kernel<<<n, NB, 0, stream>>>(A, lda, Chk, ldchk);

}



