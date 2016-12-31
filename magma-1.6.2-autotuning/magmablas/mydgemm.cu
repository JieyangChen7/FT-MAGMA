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
#define rB 8
#define cB 64
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
    int idx = blockIdx.x * cB;

    double sum1 = 0;
    double sum2 = 0;

	A = A + idx * lda;

	__shared__ double cache[rB][cB];

	for (int i = 0; i < NB; i += rB) {
		
		//load a block to cache
		if (threadIdx.x < rB) {
			for (int j = 0; j < cB; j++) {
				cache[threadIdx.x][j] = *(A + j * lda + threadIdx.x);
				//cache[j][threadIdx.x] = *(A + j * lda + threadIdx.x);
			}
		}

		__syncthreads();

		for (int j = 0; j < rB; j++) {
			sum1 += cache[j][threadIdx.x];
			sum2 += cache[j][threadIdx.x] * (i + j + 1);
			
		}
		
		__syncthreads();

		A = A + B;
	}

	idx += threadIdx.x;

	*(Chk + idx * ldchk) = sum1;
	*(Chk + idx * ldchk+1) = sum2;
	
}

__global__ void
chkenc_kernel3_5(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    

    int rb = blockDim.x;
    int cb = blockDim.y; 

    int idx = blockIdx.x * cb;

    double sum1 = 0;
    double sum2 = 0;

	A = A + idx * lda;

	extern __shared__ double cache[]; //rB * cB

	for (int i = 0; i < NB; i += rb) {
		
		//load a block to cache
		cache[threadIdx.x + threadIdx.y * rb] = *(A + threadIdx.y * lda + threadIdx.x);
		__syncthreads();
		int k = rb / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.x + threadIdx.y * rb] += cache[threadIdx.x + k + threadIdx.y * rb];
			}
			
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum1 += cache[0 + threadIdx.y * rb];
		}

		cache[threadIdx.x + threadIdx.y * rb] = *(A + threadIdx.y * lda + threadIdx.x) * (i + threadIdx.x + 1);
		__syncthreads();
		k = rb / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.x + threadIdx.y * rb] += cache[threadIdx.x + k + threadIdx.y * rb];
			}
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum2 += cache[0 + threadIdx.y * rb];
		}
				
		A = A + rb;
	}

	idx += threadIdx.y;

	if (threadIdx.x == 0) {
		*(Chk + idx * ldchk) = sum1;
		*(Chk + idx * ldchk+1) = sum2;
	}
	
}



void chkenc(double * A, int lda, int m, int n, double * chk , int ldchk, magma_queue_t stream) {
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
	//int rb = 8;
	//int cb = 16;
	//dim3 d(rb, cb, 1);
	//chkenc_kernel3_5<<<N/cb, d, rb*cb*sizeof(double), stream>>>(A, lda, chk, ldchk);
	chkenc_kernel3<<<n/cB, cB, 0, stream>>>(A, lda, chk, ldchk);

}



