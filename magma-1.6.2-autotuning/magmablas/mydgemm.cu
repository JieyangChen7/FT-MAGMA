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
#define B 32
#define rB 64
#define cB 8
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
		*(Chk + blockIdx.x * ldchk) = cache[0];
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
chkenc_kernel1_5(double * A, int lda, double * Chk , int ldchk)
{

	//blockIdx.x: determin the column to process
	A = A + blockIdx.x * lda;

	__shared__ double cache[NB];
	
	//load one column to cache
	cache[threadIdx.x] = A[threadIdx.x];

	__syncthreads();


	double sum = 0;
	if (threadIdx.x == 0) {

		for (int i = 0; i < NB; i++) {
			sum += cache[i];
		}
		*(Chk + blockIdx.x * ldchk) = sum;
	}

	__syncthreads();

	//load one column to cache
	cache[threadIdx.x] = A[threadIdx.x] * (threadIdx.x + 1);

	__syncthreads();


	sum = 0;
	if (threadIdx.x == 0) {

		for (int i = 0; i < NB; i++) {
			sum += cache[i];
		}
		*(Chk + blockIdx.x * ldchk + 1) = sum;
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
		}

		__syncthreads();

		for (int j = 0; j < B; j++) {
			sum1 += cache[j][threadIdx.x];
			sum2 += cache[j][threadIdx.x] * (i + j + 1);
			
		}
		
		__syncthreads();

		A = A + B;
	}

	idx += threadIdx.x;

	*(Chk + idx * ldchk) = cache[0][threadIdx.x];
	*(Chk + idx * ldchk+1) = cache[0][threadIdx.x];
	
}


__global__ void
chkenc_kernel3_5(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    int idx = blockIdx.x * cB;

    double sum1 = 0;
    double sum2 = 0;

	A = A + idx * lda;

	__shared__ double cache[rB][cB];

	for (int i = 0; i < NB; i += rB) {
		
		//load a block to cache
		cache[threadIdx.x][threadIdx.y] = *(A + threadIdx.y * lda + threadIdx.x);
		__syncthreads();
		int k = rB / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.x][threadIdx.y] += cache[threadIdx.x + k][threadIdx.y];
			}
			
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum1 += cache[0][threadIdx.y];
		}

		cache[threadIdx.x][threadIdx.y] = *(A + threadIdx.y * lda + threadIdx.x) * (i + threadIdx.x + 1);
		__syncthreads();
		k = rB / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.x][threadIdx.y] += cache[threadIdx.x + k][threadIdx.y];
			}
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum2 += cache[0][threadIdx.y];
		}
				
		A = A + rB;
	}

	idx += threadIdx.y;

	if (threadIdx.x == 0) {
		*(Chk + idx * ldchk) = sum1;
		*(Chk + idx * ldchk+1) = sum2;
	}
	
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
	//dim3 d(rB, cB, 1);
	chkenc_kernel3<<<n/B, B, 0, stream>>>(A, lda, Chk, ldchk);

}



