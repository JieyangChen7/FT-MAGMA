
#include<stdio.h>
#include<iostream>
#include"papi.h"
#define N 30720
#define NB 512
#define rB 32
#define cB 32

using namespace std;

// encoding checksum for A

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
		for (int j = 0; j < rB; j++) {
			cache[threadIdx.x][j] = *(A + j * lda + threadIdx.x);
		}
		__syncthreads();

		for (int j = 0; j < rB; j++) {
			sum1 += cache[j][threadIdx.x];
			sum2 += cache[j][threadIdx.x] * (i + j + 1);
		}
		__syncthreads();
		A = A + rB;
	}

	*(Chk + idx * ldchk) = sum1;
	*(Chk + idx * ldchk+1) = sum2;
	
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


	if (threadIdx.x == 0) {
		*(Chk + idx * ldchk) = sum1;
		*(Chk + idx * ldchk+1) = sum2;
	}
	
}


__global__ void
chkenc_kernel4(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    for(int k = 0; k < N; k += cB) {

	    double sum1 = 0;
	    double sum2 = 0;

		A = A + k * lda;

		__shared__ double cache[rB][cB];

		for (int i = 0; i < NB; i += rB) {
			
			//load a block to cache
			for (int j = 0; j < rB; j++) {
				cache[threadIdx.x][j] = *(A + j * lda + threadIdx.x);
			}
			__syncthreads();

			for (int j = 0; j < rB; j++) {
				sum1 += cache[j][threadIdx.x];
				sum2 += cache[j][threadIdx.x] * (i + j + 1);
			}
			__syncthreads();
			A = A + rB;
		}

		*(Chk + k * ldchk) = sum1;
		*(Chk + k * ldchk+1) = sum2;
	}
	
}


void chkenc(double * A, int lda, int m, int n, double * Chk , int ldchk, cudaStream_t stream) {
    int numBlocks; // Occupancy in terms of active blocks 
    int blockSize = cB; 
	int device; 
	cudaDeviceProp prop; 
	int activeWarps; 
	int maxWarps; 
	cudaGetDevice(&device); 
	cudaGetDeviceProperties(&prop, device); cudaOccupancyMaxActiveBlocksPerMultiprocessor( &numBlocks, chkenc_kernel4, blockSize, 0); 
	activeWarps = numBlocks * blockSize / prop.warpSize; 
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize; 
	printf("Occupancy: %f \n", (double)activeWarps / maxWarps * 100 );

	cudaFuncSetCacheConfig(chkenc_kernel, cudaFuncCachePreferShared);
	chkenc_kernel<<<N, NB, 0, stream>>>(A, lda, Chk, ldchk);

	//dim3 d(cB, rB, 1);
	//chkenc_kernel3_5<<<N/cB, d, 0, stream>>>(A, lda, Chk, ldchk);
	//chkenc_kernel4<<<1, cB, 0, stream>>>(A, lda, Chk, ldchk);
}

int main(){
	
	double * A = new double[NB * N];
	for (int i = 0; i < NB*N; i++) {
		A[i] = i;
	}
	double * dA;
	size_t dApitch;
	cudaMallocPitch(&dA, &dApitch, NB*sizeof(double), N);
	cudaMemcpy2D(dA, dApitch, A, NB, NB, N, cudaMemcpyHostToDevice);
	int ldda = dApitch/sizeof(double);

	double * chk;
	size_t chkpitch;
	cudaMallocPitch(&chk, &chkpitch, 2*sizeof(double), N);
	int ldchk = chkpitch/sizeof(double);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	float real_time = 0.0;
	float proc_time = 0.0;
	long long flpins = 0.0;
	float mflops = 0.0;
	long long flops = 2 * NB * N * 2;


	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}

	chkenc_kernel<<<N, NB, 0, stream>>>(dA, ldda, chk, ldchk);
	cudaStreamSynchronize(stream);
	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	cout << real_time << "\t" << (flops/real_time)/1e9 << "\t";

	PAPI_shutdown();


	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	//chkenc(dA, ldda, NB, N, chk , ldchk, stream);
	
	chkenc_kernel<<<N, NB, 0, stream>>>(dA, ldda, chk, ldchk);
	cudaStreamSynchronize(stream);
	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	
	cout << real_time << "\t" << (flops/real_time)/1e9 << "\t";



	real_time = 0.0;
	proc_time = 0.0;
	flpins = 0.0;
	mflops = 0.0;

	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	
	chkenc_kernel1_5<<<N, NB, 0, stream>>>(dA, ldda, chk, ldchk);
	cudaStreamSynchronize(stream);
	if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	cout << real_time << "\t" << (flops/real_time)/1e9;


	return 0;





}



