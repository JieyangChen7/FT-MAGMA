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
#define B 16
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

//N=32
__global__ void
chkenc_kernel3(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    int idx = blockIdx.x * B;

    double sum1 = 0;
    double sum2 = 0;

    double temp = 0;

	A = A + idx * lda;



	__shared__ double cache[B][B];

	for (int i = 0; i < NB; i += B) {
		
		//load a block to cache
		
			for (int j = 0; j < B; j++) {
				cache[threadIdx.x][j] = *(A + j * lda + threadIdx.x);
			}

		__syncthreads();

		for (int j = 0; j < B; j++) {
			temp = cache[j][threadIdx.x];
			sum1 += temp;
			sum2 += temp * (i + j + 1);
			
		}
		
		__syncthreads();

		A = A + B;
	}

	idx += threadIdx.x;

	*(Chk + idx * ldchk) = sum1;
	*(Chk + idx * ldchk+1) = sum2;
	
}


//N=16
__global__ void
chkenc_kernel3_P(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    int idx = blockIdx.x * B;

    double sum1 = 0;
    double sum2 = 0;

    double temp = 0;

	A = A + idx * lda;

	

	__shared__ double cache[B][B];


	double r0 = *(A + 0 * lda + threadIdx.x);
	double r1 = *(A + 1 * lda + threadIdx.x);
	double r2 = *(A + 2 * lda + threadIdx.x);
	double r3 = *(A + 3 * lda + threadIdx.x);
	double r4 = *(A + 4 * lda + threadIdx.x);
	double r5 = *(A + 5 * lda + threadIdx.x);
	double r6 = *(A + 6 * lda + threadIdx.x);
	double r7 = *(A + 7 * lda + threadIdx.x);
	
	double r8 = *(A + 8 * lda + threadIdx.x);
	double r9 = *(A + 9 * lda + threadIdx.x);
	double r10 = *(A + 10 * lda + threadIdx.x);
	double r11 = *(A + 11 * lda + threadIdx.x);
	double r12 = *(A + 12 * lda + threadIdx.x);
	double r13 = *(A + 13 * lda + threadIdx.x);
	double r14 = *(A + 14 * lda + threadIdx.x);
	double r15 = *(A + 15 * lda + threadIdx.x);
	/*
	double r16 = *(A + 16 * lda + threadIdx.x);
	double r17 = *(A + 17 * lda + threadIdx.x);
	double r18 = *(A + 18 * lda + threadIdx.x);
	double r19 = *(A + 19 * lda + threadIdx.x);
	double r20 = *(A + 20 * lda + threadIdx.x);
	double r21 = *(A + 21 * lda + threadIdx.x);
	double r22 = *(A + 22 * lda + threadIdx.x);
	double r23 = *(A + 23 * lda + threadIdx.x);
	double r24 = *(A + 24 * lda + threadIdx.x);
	double r25 = *(A + 25 * lda + threadIdx.x);
	double r26 = *(A + 26 * lda + threadIdx.x);
	double r27 = *(A + 27 * lda + threadIdx.x);
	double r28 = *(A + 28 * lda + threadIdx.x);
	double r29 = *(A + 29 * lda + threadIdx.x);
	double r30 = *(A + 30 * lda + threadIdx.x);
	double r31 = *(A + 31 * lda + threadIdx.x);
	*/



	for (int i = 0; i < NB; i += B) {

		//load current register->shared mem.
		cache[threadIdx.x][0] = r0;
		cache[threadIdx.x][1] = r1;
		cache[threadIdx.x][2] = r2;
		cache[threadIdx.x][3] = r3;
		cache[threadIdx.x][4] = r4;
		cache[threadIdx.x][5] = r5;
		cache[threadIdx.x][6] = r6;
		cache[threadIdx.x][7] = r7;
		
		cache[threadIdx.x][8] = r8;
		cache[threadIdx.x][9] = r9;
		cache[threadIdx.x][10] = r10;
		cache[threadIdx.x][11] = r11;
		cache[threadIdx.x][12] = r12;
		cache[threadIdx.x][13] = r13;
		cache[threadIdx.x][14] = r14;
		cache[threadIdx.x][15] = r15;
		/*
		cache[threadIdx.x][16] = r16;
		cache[threadIdx.x][17] = r17;
		cache[threadIdx.x][18] = r18;
		cache[threadIdx.x][19] = r19;
		cache[threadIdx.x][20] = r20;
		cache[threadIdx.x][21] = r21;
		cache[threadIdx.x][22] = r22;
		cache[threadIdx.x][23] = r23;
		cache[threadIdx.x][24] = r24;
		cache[threadIdx.x][25] = r25;
		cache[threadIdx.x][26] = r26;
		cache[threadIdx.x][27] = r27;
		cache[threadIdx.x][28] = r28;
		cache[threadIdx.x][29] = r29;
		cache[threadIdx.x][30] = r30;
		cache[threadIdx.x][31] = r31;
		*/

		__syncthreads();

		A = A + B;

		//load a next block to register
		
		 r0 = *(A + 0 * lda + threadIdx.x);
		 r1 = *(A + 1 * lda + threadIdx.x);
		 r2 = *(A + 2 * lda + threadIdx.x);
		 r3 = *(A + 3 * lda + threadIdx.x);
		 r4 = *(A + 4 * lda + threadIdx.x);
		 r5 = *(A + 5 * lda + threadIdx.x);
		 r6 = *(A + 6 * lda + threadIdx.x);
		 r7 = *(A + 7 * lda + threadIdx.x);
		 
		 r8 = *(A + 8 * lda + threadIdx.x);
		 r9 = *(A + 9 * lda + threadIdx.x);
		 r10 = *(A + 10 * lda + threadIdx.x);
		 r11 = *(A + 11 * lda + threadIdx.x);
		 r12 = *(A + 12 * lda + threadIdx.x);
		 r13 = *(A + 13 * lda + threadIdx.x);
		 r14 = *(A + 14 * lda + threadIdx.x);
		 r15 = *(A + 15 * lda + threadIdx.x);
		 /*
		 r16 = *(A + 16 * lda + threadIdx.x);
		 r17 = *(A + 17 * lda + threadIdx.x);
		 r18 = *(A + 18 * lda + threadIdx.x);
		 r19 = *(A + 19 * lda + threadIdx.x);
		 r20 = *(A + 20 * lda + threadIdx.x);
		 r21 = *(A + 21 * lda + threadIdx.x);
		 r22 = *(A + 22 * lda + threadIdx.x);
		 r23 = *(A + 23 * lda + threadIdx.x);
		 r24 = *(A + 24 * lda + threadIdx.x);
		 r25 = *(A + 25 * lda + threadIdx.x);
		 r26 = *(A + 26 * lda + threadIdx.x);
		 r27 = *(A + 27 * lda + threadIdx.x);
		 r28 = *(A + 28 * lda + threadIdx.x);
		 r29 = *(A + 29 * lda + threadIdx.x);
		 r30 = *(A + 30 * lda + threadIdx.x);
		 r31 = *(A + 31 * lda + threadIdx.x);
		 */


		for (int j = 0; j < B; j++) {
			temp = cache[j][threadIdx.x];
			sum1 += temp;
			sum2 += temp * (i + j + 1);
			
		}
		
		__syncthreads();

		
	}

	idx += threadIdx.x;

	*(Chk + idx * ldchk) = sum1;
	*(Chk + idx * ldchk+1) = sum2;
	
}


__global__ void
chkenc_kernel3_P_R(double * A, int lda, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    int idx = blockIdx.x * B;

    double sum1 = 0;
    double sum2 = 0;

    double temp = 0;

	A = A + idx;

	

	__shared__ double cache[B][B];


	double r0 = *(A + 0 * lda + threadIdx.x);
	double r1 = *(A + 1 * lda + threadIdx.x);
	double r2 = *(A + 2 * lda + threadIdx.x);
	double r3 = *(A + 3 * lda + threadIdx.x);
	double r4 = *(A + 4 * lda + threadIdx.x);
	double r5 = *(A + 5 * lda + threadIdx.x);
	double r6 = *(A + 6 * lda + threadIdx.x);
	double r7 = *(A + 7 * lda + threadIdx.x);
	
	double r8 = *(A + 8 * lda + threadIdx.x);
	double r9 = *(A + 9 * lda + threadIdx.x);
	double r10 = *(A + 10 * lda + threadIdx.x);
	double r11 = *(A + 11 * lda + threadIdx.x);
	double r12 = *(A + 12 * lda + threadIdx.x);
	double r13 = *(A + 13 * lda + threadIdx.x);
	double r14 = *(A + 14 * lda + threadIdx.x);
	double r15 = *(A + 15 * lda + threadIdx.x);
	/*
	double r16 = *(A + 16 * lda + threadIdx.x);
	double r17 = *(A + 17 * lda + threadIdx.x);
	double r18 = *(A + 18 * lda + threadIdx.x);
	double r19 = *(A + 19 * lda + threadIdx.x);
	double r20 = *(A + 20 * lda + threadIdx.x);
	double r21 = *(A + 21 * lda + threadIdx.x);
	double r22 = *(A + 22 * lda + threadIdx.x);
	double r23 = *(A + 23 * lda + threadIdx.x);
	double r24 = *(A + 24 * lda + threadIdx.x);
	double r25 = *(A + 25 * lda + threadIdx.x);
	double r26 = *(A + 26 * lda + threadIdx.x);
	double r27 = *(A + 27 * lda + threadIdx.x);
	double r28 = *(A + 28 * lda + threadIdx.x);
	double r29 = *(A + 29 * lda + threadIdx.x);
	double r30 = *(A + 30 * lda + threadIdx.x);
	double r31 = *(A + 31 * lda + threadIdx.x);
	*/



	for (int i = 0; i < NB; i += B) {

		//load current register->shared mem.
		cache[0][threadIdx.x] = r0;
		cache[1][threadIdx.x] = r1;
		cache[2][threadIdx.x] = r2;
		cache[3][threadIdx.x] = r3;
		cache[4][threadIdx.x] = r4;
		cache[5][threadIdx.x] = r5;
		cache[6][threadIdx.x] = r6;
		cache[7][threadIdx.x] = r7;
		
		cache[8][threadIdx.x] = r8;
		cache[9][threadIdx.x] = r9;
		cache[10][threadIdx.x] = r10;
		cache[11][threadIdx.x] = r11;
		cache[12][threadIdx.x] = r12;
		cache[13][threadIdx.x] = r13;
		cache[14][threadIdx.x] = r14;
		cache[15][threadIdx.x] = r15;
		/*
		cache[threadIdx.x][16] = r16;
		cache[threadIdx.x][17] = r17;
		cache[threadIdx.x][18] = r18;
		cache[threadIdx.x][19] = r19;
		cache[threadIdx.x][20] = r20;
		cache[threadIdx.x][21] = r21;
		cache[threadIdx.x][22] = r22;
		cache[threadIdx.x][23] = r23;
		cache[threadIdx.x][24] = r24;
		cache[threadIdx.x][25] = r25;
		cache[threadIdx.x][26] = r26;
		cache[threadIdx.x][27] = r27;
		cache[threadIdx.x][28] = r28;
		cache[threadIdx.x][29] = r29;
		cache[threadIdx.x][30] = r30;
		cache[threadIdx.x][31] = r31;
		*/

		__syncthreads();

		A = A + B * lda;

		//load a next block to register
		
		 r0 = *(A + 0 * lda + threadIdx.x);
		 r1 = *(A + 1 * lda + threadIdx.x);
		 r2 = *(A + 2 * lda + threadIdx.x);
		 r3 = *(A + 3 * lda + threadIdx.x);
		 r4 = *(A + 4 * lda + threadIdx.x);
		 r5 = *(A + 5 * lda + threadIdx.x);
		 r6 = *(A + 6 * lda + threadIdx.x);
		 r7 = *(A + 7 * lda + threadIdx.x);
		 
		 r8 = *(A + 8 * lda + threadIdx.x);
		 r9 = *(A + 9 * lda + threadIdx.x);
		 r10 = *(A + 10 * lda + threadIdx.x);
		 r11 = *(A + 11 * lda + threadIdx.x);
		 r12 = *(A + 12 * lda + threadIdx.x);
		 r13 = *(A + 13 * lda + threadIdx.x);
		 r14 = *(A + 14 * lda + threadIdx.x);
		 r15 = *(A + 15 * lda + threadIdx.x);
		 /*
		 r16 = *(A + 16 * lda + threadIdx.x);
		 r17 = *(A + 17 * lda + threadIdx.x);
		 r18 = *(A + 18 * lda + threadIdx.x);
		 r19 = *(A + 19 * lda + threadIdx.x);
		 r20 = *(A + 20 * lda + threadIdx.x);
		 r21 = *(A + 21 * lda + threadIdx.x);
		 r22 = *(A + 22 * lda + threadIdx.x);
		 r23 = *(A + 23 * lda + threadIdx.x);
		 r24 = *(A + 24 * lda + threadIdx.x);
		 r25 = *(A + 25 * lda + threadIdx.x);
		 r26 = *(A + 26 * lda + threadIdx.x);
		 r27 = *(A + 27 * lda + threadIdx.x);
		 r28 = *(A + 28 * lda + threadIdx.x);
		 r29 = *(A + 29 * lda + threadIdx.x);
		 r30 = *(A + 30 * lda + threadIdx.x);
		 r31 = *(A + 31 * lda + threadIdx.x);
		 */


		for (int j = 0; j < B; j++) {
			temp = cache[j][threadIdx.x];
			sum1 += temp;
			sum2 += temp * (i + j + 1);
			
		}
		
		__syncthreads();

		
	}

	idx += threadIdx.x;

	*(Chk + idx) = sum1;
	*(Chk + idx + ldchk) = sum2;
	
}


//N=16 Prefetch - full 
__global__ void
chkenc_kernel3_P_F(double * A, int lda, int nb, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    //int idx = blockIdx.x;


    double sum1 = 0;
    double sum2 = 0;

    double temp = 0;

	A = A + blockIdx.x * nb + blockIdx.y * nb * lda;

	

	__shared__ double cache[B][B];

	double r0 = 0;
	double r1 = 0;
	
	double r2 = 0;
	double r3 = 0;
	double r4 = 0;
	double r5 = 0;
	double r6 = 0;
	double r7 = 0;
	
	double r8 = 0;
	double r9 = 0;
	double r10 = 0;
	double r11 = 0;
	double r12 = 0;
	double r13 = 0;
	double r14 = 0;
	double r15 = 0;
	/*
	double r16 = 0;
	double r17 = 0;
	double r18 = 0;
	double r19 = 0;
	double r20 = 0;
	double r21 = 0;
	double r22 = 0;
	double r23 = 0;
	double r24 = 0;
	double r25 = 0;
	double r26 = 0;
	double r27 = 0;
	double r28 = 0;
	double r29 = 0;
	double r30 = 0;
	double r31 = 0;
	*/

	double * tA = A;
	for (int k = 0; k < nb; k += B) {
		
		r0 = *(A + 0 * lda + threadIdx.x);
		r1 = *(A + 1 * lda + threadIdx.x);
		
		r2 = *(A + 2 * lda + threadIdx.x);
		r3 = *(A + 3 * lda + threadIdx.x);
		r4 = *(A + 4 * lda + threadIdx.x);
		r5 = *(A + 5 * lda + threadIdx.x);
		r6 = *(A + 6 * lda + threadIdx.x);
		r7 = *(A + 7 * lda + threadIdx.x);
		
		r8 = *(A + 8 * lda + threadIdx.x);
		r9 = *(A + 9 * lda + threadIdx.x);
		r10 = *(A + 10 * lda + threadIdx.x);
		r11 = *(A + 11 * lda + threadIdx.x);
		r12 = *(A + 12 * lda + threadIdx.x);
		r13 = *(A + 13 * lda + threadIdx.x);
		r14 = *(A + 14 * lda + threadIdx.x);
		r15 = *(A + 15 * lda + threadIdx.x);
		/*
		r16 = *(A + 16 * lda + threadIdx.x);
		r17 = *(A + 17 * lda + threadIdx.x);
		r18 = *(A + 18 * lda + threadIdx.x);
		r19 = *(A + 19 * lda + threadIdx.x);
		r20 = *(A + 20 * lda + threadIdx.x);
		r21 = *(A + 21 * lda + threadIdx.x);
		r22 = *(A + 22 * lda + threadIdx.x);
		r23 = *(A + 23 * lda + threadIdx.x);
		r24 = *(A + 24 * lda + threadIdx.x);
		r25 = *(A + 25 * lda + threadIdx.x);
		r26 = *(A + 26 * lda + threadIdx.x);
		r27 = *(A + 27 * lda + threadIdx.x);
		r28 = *(A + 28 * lda + threadIdx.x);
		r29 = *(A + 29 * lda + threadIdx.x);
		r30 = *(A + 30 * lda + threadIdx.x);
		r31 = *(A + 31 * lda + threadIdx.x);
		*/

		sum1 = 0;
		sum2 = 0;
		temp = 0;


		for (int i = 0; i < nb; i += B) {

			//load current register->shared mem.
			cache[threadIdx.x][0] = r0;
			cache[threadIdx.x][1] = r1;
			
			cache[threadIdx.x][2] = r2;
			cache[threadIdx.x][3] = r3;
			cache[threadIdx.x][4] = r4;
			cache[threadIdx.x][5] = r5;
			cache[threadIdx.x][6] = r6;
			cache[threadIdx.x][7] = r7;
			
			cache[threadIdx.x][8] = r8;
			cache[threadIdx.x][9] = r9;
			cache[threadIdx.x][10] = r10;
			cache[threadIdx.x][11] = r11;
			cache[threadIdx.x][12] = r12;
			cache[threadIdx.x][13] = r13;
			cache[threadIdx.x][14] = r14;
			cache[threadIdx.x][15] = r15;
			/*
			cache[threadIdx.x][16] = r16;
			cache[threadIdx.x][17] = r17;
			cache[threadIdx.x][18] = r18;
			cache[threadIdx.x][19] = r19;
			cache[threadIdx.x][20] = r20;
			cache[threadIdx.x][21] = r21;
			cache[threadIdx.x][22] = r22;
			cache[threadIdx.x][23] = r23;
			cache[threadIdx.x][24] = r24;
			cache[threadIdx.x][25] = r25;
			cache[threadIdx.x][26] = r26;
			cache[threadIdx.x][27] = r27;
			cache[threadIdx.x][28] = r28;
			cache[threadIdx.x][29] = r29;
			cache[threadIdx.x][30] = r30;
			cache[threadIdx.x][31] = r31;
			*/

			__syncthreads();

			A = A + B;

			//load a next block to register
			
			 r0 = *(A + 0 * lda + threadIdx.x);
			 r1 = *(A + 1 * lda + threadIdx.x);
			 
			 r2 = *(A + 2 * lda + threadIdx.x);
			 r3 = *(A + 3 * lda + threadIdx.x);
			 r4 = *(A + 4 * lda + threadIdx.x);
			 r5 = *(A + 5 * lda + threadIdx.x);
			 r6 = *(A + 6 * lda + threadIdx.x);
			 r7 = *(A + 7 * lda + threadIdx.x);
			 
			 r8 = *(A + 8 * lda + threadIdx.x);
			 r9 = *(A + 9 * lda + threadIdx.x);
			 r10 = *(A + 10 * lda + threadIdx.x);
			 r11 = *(A + 11 * lda + threadIdx.x);
			 r12 = *(A + 12 * lda + threadIdx.x);
			 r13 = *(A + 13 * lda + threadIdx.x);
			 r14 = *(A + 14 * lda + threadIdx.x);
			 r15 = *(A + 15 * lda + threadIdx.x);
			/*
			 r16 = *(A + 16 * lda + threadIdx.x);
			 r17 = *(A + 17 * lda + threadIdx.x);
			 r18 = *(A + 18 * lda + threadIdx.x);
			 r19 = *(A + 19 * lda + threadIdx.x);
			 r20 = *(A + 20 * lda + threadIdx.x);
			 r21 = *(A + 21 * lda + threadIdx.x);
			 r22 = *(A + 22 * lda + threadIdx.x);
			 r23 = *(A + 23 * lda + threadIdx.x);
			 r24 = *(A + 24 * lda + threadIdx.x);
			 r25 = *(A + 25 * lda + threadIdx.x);
			 r26 = *(A + 26 * lda + threadIdx.x);
			 r27 = *(A + 27 * lda + threadIdx.x);
			 r28 = *(A + 28 * lda + threadIdx.x);
			 r29 = *(A + 29 * lda + threadIdx.x);
			 r30 = *(A + 30 * lda + threadIdx.x);
			 r31 = *(A + 31 * lda + threadIdx.x);
			 */


			for (int j = 0; j < B; j++) {
				temp = cache[j][threadIdx.x];
				sum1 += temp;
				sum2 += temp * (i + j + 1);
				
			}
			
			__syncthreads();

			
		}

		//idx += threadIdx.x;

		*(Chk + (blockIdx.y * nb + k + threadIdx.x) * ldchk + blockIdx.x * 2 ) = sum1;
		*(Chk + (blockIdx.y * nb + k + threadIdx.x) * ldchk + blockIdx.x * 2 + 1) = sum2;


		tA += B * lda;
		//if(threadIdx.x == 0)
		//printf("next:%f\n", (*tA));
		A = tA ;
	}
	
}


//N=16 Prefetch - full - Row
__global__ void
chkenc_kernel3_P_FR(double * A, int lda, int nb, double * Chk , int ldchk)
{

    //blockIdx.x: determin the column to process
    //int idx = blockIdx.x;


    double sum1 = 0;
    double sum2 = 0;

    double temp = 0;

	A = A + blockIdx.x * nb + blockIdx.y * nb * lda;

	

	__shared__ double cache[B][B];

	double r0 = 0;
	double r1 = 0;
	
	double r2 = 0;
	double r3 = 0;
	double r4 = 0;
	double r5 = 0;
	double r6 = 0;
	double r7 = 0;
	
	double r8 = 0;
	double r9 = 0;
	double r10 = 0;
	double r11 = 0;
	double r12 = 0;
	double r13 = 0;
	double r14 = 0;
	double r15 = 0;
	/*
	double r16 = 0;
	double r17 = 0;
	double r18 = 0;
	double r19 = 0;
	double r20 = 0;
	double r21 = 0;
	double r22 = 0;
	double r23 = 0;
	double r24 = 0;
	double r25 = 0;
	double r26 = 0;
	double r27 = 0;
	double r28 = 0;
	double r29 = 0;
	double r30 = 0;
	double r31 = 0;
	*/

	double * tA = A;
	for (int k = 0; k < nb; k += B) {
		
		r0 = *(A + 0 * lda + threadIdx.x);
		r1 = *(A + 1 * lda + threadIdx.x);
		
		r2 = *(A + 2 * lda + threadIdx.x);
		r3 = *(A + 3 * lda + threadIdx.x);
		r4 = *(A + 4 * lda + threadIdx.x);
		r5 = *(A + 5 * lda + threadIdx.x);
		r6 = *(A + 6 * lda + threadIdx.x);
		r7 = *(A + 7 * lda + threadIdx.x);
		
		r8 = *(A + 8 * lda + threadIdx.x);
		r9 = *(A + 9 * lda + threadIdx.x);
		r10 = *(A + 10 * lda + threadIdx.x);
		r11 = *(A + 11 * lda + threadIdx.x);
		r12 = *(A + 12 * lda + threadIdx.x);
		r13 = *(A + 13 * lda + threadIdx.x);
		r14 = *(A + 14 * lda + threadIdx.x);
		r15 = *(A + 15 * lda + threadIdx.x);
		/*
		r16 = *(A + 16 * lda + threadIdx.x);
		r17 = *(A + 17 * lda + threadIdx.x);
		r18 = *(A + 18 * lda + threadIdx.x);
		r19 = *(A + 19 * lda + threadIdx.x);
		r20 = *(A + 20 * lda + threadIdx.x);
		r21 = *(A + 21 * lda + threadIdx.x);
		r22 = *(A + 22 * lda + threadIdx.x);
		r23 = *(A + 23 * lda + threadIdx.x);
		r24 = *(A + 24 * lda + threadIdx.x);
		r25 = *(A + 25 * lda + threadIdx.x);
		r26 = *(A + 26 * lda + threadIdx.x);
		r27 = *(A + 27 * lda + threadIdx.x);
		r28 = *(A + 28 * lda + threadIdx.x);
		r29 = *(A + 29 * lda + threadIdx.x);
		r30 = *(A + 30 * lda + threadIdx.x);
		r31 = *(A + 31 * lda + threadIdx.x);
		*/

		sum1 = 0;
		sum2 = 0;
		temp = 0;


		for (int i = 0; i < nb; i += B) {

			//load current register->shared mem.
			cache[0][threadIdx.x] = r0;
			cache[1][threadIdx.x] = r1;
			
			cache[2][threadIdx.x] = r2;
			cache[3][threadIdx.x] = r3;
			cache[4][threadIdx.x] = r4;
			cache[5][threadIdx.x] = r5;
			cache[6][threadIdx.x] = r6;
			cache[7][threadIdx.x] = r7;
			
			cache[8][threadIdx.x] = r8;
			cache[9][threadIdx.x] = r9;
			cache[10][threadIdx.x] = r10;
			cache[11][threadIdx.x] = r11;
			cache[12][threadIdx.x] = r12;
			cache[13][threadIdx.x] = r13;
			cache[14][threadIdx.x] = r14;
			cache[15][threadIdx.x] = r15;
			/*
			cache[threadIdx.x][16] = r16;
			cache[threadIdx.x][17] = r17;
			cache[threadIdx.x][18] = r18;
			cache[threadIdx.x][19] = r19;
			cache[threadIdx.x][20] = r20;
			cache[threadIdx.x][21] = r21;
			cache[threadIdx.x][22] = r22;
			cache[threadIdx.x][23] = r23;
			cache[threadIdx.x][24] = r24;
			cache[threadIdx.x][25] = r25;
			cache[threadIdx.x][26] = r26;
			cache[threadIdx.x][27] = r27;
			cache[threadIdx.x][28] = r28;
			cache[threadIdx.x][29] = r29;
			cache[threadIdx.x][30] = r30;
			cache[threadIdx.x][31] = r31;
			*/

			__syncthreads();

			A = A + B * lda;

			//load a next block to register
			
			 r0 = *(A + 0 * lda + threadIdx.x);
			 r1 = *(A + 1 * lda + threadIdx.x);
			 
			 r2 = *(A + 2 * lda + threadIdx.x);
			 r3 = *(A + 3 * lda + threadIdx.x);
			 r4 = *(A + 4 * lda + threadIdx.x);
			 r5 = *(A + 5 * lda + threadIdx.x);
			 r6 = *(A + 6 * lda + threadIdx.x);
			 r7 = *(A + 7 * lda + threadIdx.x);
			 
			 r8 = *(A + 8 * lda + threadIdx.x);
			 r9 = *(A + 9 * lda + threadIdx.x);
			 r10 = *(A + 10 * lda + threadIdx.x);
			 r11 = *(A + 11 * lda + threadIdx.x);
			 r12 = *(A + 12 * lda + threadIdx.x);
			 r13 = *(A + 13 * lda + threadIdx.x);
			 r14 = *(A + 14 * lda + threadIdx.x);
			 r15 = *(A + 15 * lda + threadIdx.x);
			/*
			 r16 = *(A + 16 * lda + threadIdx.x);
			 r17 = *(A + 17 * lda + threadIdx.x);
			 r18 = *(A + 18 * lda + threadIdx.x);
			 r19 = *(A + 19 * lda + threadIdx.x);
			 r20 = *(A + 20 * lda + threadIdx.x);
			 r21 = *(A + 21 * lda + threadIdx.x);
			 r22 = *(A + 22 * lda + threadIdx.x);
			 r23 = *(A + 23 * lda + threadIdx.x);
			 r24 = *(A + 24 * lda + threadIdx.x);
			 r25 = *(A + 25 * lda + threadIdx.x);
			 r26 = *(A + 26 * lda + threadIdx.x);
			 r27 = *(A + 27 * lda + threadIdx.x);
			 r28 = *(A + 28 * lda + threadIdx.x);
			 r29 = *(A + 29 * lda + threadIdx.x);
			 r30 = *(A + 30 * lda + threadIdx.x);
			 r31 = *(A + 31 * lda + threadIdx.x);
			 */


			for (int j = 0; j < B; j++) {
				temp = cache[j][threadIdx.x];
				sum1 += temp;
				sum2 += temp * (i + j + 1);
				
			}
			
			__syncthreads();

			
		}

		//idx += threadIdx.x;

		*(Chk + (blockIdx.y * 2) * ldchk + blockIdx.x * nb + k + threadIdx.x) = sum1;
		*(Chk + (blockIdx.y * 2 + 1) * ldchk + blockIdx.x * nb + k + threadIdx.x) = sum2;


		tA += B;
		//if(threadIdx.x == 0)
		//printf("next:%f\n", (*tA));
		A = tA ;
	}
	
}


//N=16
__global__ void
chkenc_kernel3_5(double * A, int lda, double * Chk, int ldchk)
{

    //blockIdx.x: determin the column to process
    

    int idx = blockIdx.x * B;

    double sum1 = 0;
    double sum2 = 0;

	A = A + idx * lda;

	__shared__ double cache[B][B]; //B * B

	for (int i = 0; i < NB; i += B) {
		
		//load a block to cache
		cache[threadIdx.y][threadIdx.x] = *(A + threadIdx.y * lda + threadIdx.x);
		__syncthreads();

		int k = B / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + k];
			}
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum1 += cache[threadIdx.y][0];
		}


		cache[threadIdx.y][threadIdx.x] = *(A + threadIdx.y * lda + threadIdx.x) * (i + threadIdx.x + 1);
		__syncthreads();
		k = B / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + k];
			}
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum2 += cache[threadIdx.y][0];
		}
				
		A = A + B;
	}

	idx += threadIdx.y;

	if (threadIdx.x == 0) {
		*(Chk + idx * ldchk) = sum1;
		*(Chk + idx * ldchk+1) = sum2;
	}
	
}


//N=16
__global__ void
chkenc_kernel3_5_P(double * A, int lda, double * Chk, int ldchk)
{

    //blockIdx.x: determin the column to process
    

    int idx = blockIdx.x * B;

    double sum = 0;

	A = A + idx * lda;
	idx += threadIdx.y;

	__shared__ double cache[B][B]; //B * B

	double r = *(A + threadIdx.y * lda + threadIdx.x);


	for (int i = 0; i < NB; i += B) {
		
		//load a block register -> cache
		cache[threadIdx.y][threadIdx.x] = r;
		__syncthreads();

		//load next to register
		r = *(A + i + B + threadIdx.y * lda + threadIdx.x);

		int k = B / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + k];
			}
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum += cache[threadIdx.y][0];
		}

	}

	if (threadIdx.x == 0) {
		*(Chk + idx * ldchk) = sum;
	}

	sum = 0;
	r = *(A + threadIdx.y * lda + threadIdx.x);

	for (int i = 0; i < NB; i += B) {

		cache[threadIdx.y][threadIdx.x] = r * (i + threadIdx.x + 1);
		__syncthreads();

		r = *(A + i + B +  threadIdx.y * lda + threadIdx.x);
		int k = B / 2;
		while (k != 0) {
			if (threadIdx.x < k) {
				cache[threadIdx.y][threadIdx.x] += cache[threadIdx.y][threadIdx.x + k];
			}
			__syncthreads();
			k /= 2;
		}
		if (threadIdx.x == 0) {
			sum += cache[threadIdx.y][0];
		}
	}

	

	if (threadIdx.x == 0) {
		*(Chk + idx * ldchk+1) = sum;
	}
	
}



void col_chkenc(double * A, int lda, int m, int n, int nb, double * chk , int ldchk, magma_queue_t stream) {
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
	//int rb = B;
	//int cb = B;
	dim3 d(m/NB, n/NB, 1);
	
	//for (int i = 0; i < m; i+=NB) {
	//	chkenc_kernel3_P<<<n/B, B, 0, stream>>>(A + i, lda, chk + (i/NB)*2, ldchk);
	//}
	//chkenc_kernel3_P_F<<<d, B, 0, stream>>>(A, lda, chk, ldchk);
	//chkenc_kernel3_P_R<<<m/B, B, 0, stream>>>(A, lda, chk, ldchk);

	chkenc_kernel3_P_F<<<d, B, 0, stream>>>(A, lda, nb, chk, ldchk);
}


void row_chkenc(double * A, int lda, int m, int n, int nb, double * chk , int ldchk, magma_queue_t stream) {
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
	//int rb = B;
	//int cb = B;
	dim3 d(m/nb, n/nb, 1);
	
	//for (int i = 0; i < m; i+=NB) {
	//	chkenc_kernel3_P<<<n/B, B, 0, stream>>>(A + i, lda, chk + (i/NB)*2, ldchk);
	//}
	//chkenc_kernel3_P_F<<<d, B, 0, stream>>>(A, lda, chk, ldchk);
	//chkenc_kernel3_P_R<<<m/B, B, 0, stream>>>(A, lda, chk, ldchk);

	chkenc_kernel3_P_FR<<<d, B, 0, stream>>>(A, lda, nb, chk, ldchk);
}



