/*
    Enhanced Online ABFT
    UC Riverside
    Jieyang Chen
*/
#include "FT.h"
#include "common_magma.h"
#include "magma.h"
#include <stdlib.h>


__global__ void
col_detect_correct_kernel(double * A, int lda, int B, double E,
				double * checksum_update, int checksum_update_ld,
				double * hrz_recal_chk, int hrz_recal_chk_ld)
{
    //determin the block to process
	A = A + blockIdx.x * B + blockIdx.y * B * lda;
	
	//printf("block:%f---blockx=%d, blocky=%d \n",*A,blockIdx.x,blockIdx.y);
	
	
    checksum_update = checksum_update + blockIdx.x * 2  + blockIdx.y * B * checksum_update_ld;
    hrz_recal_chk = hrz_recal_chk + blockIdx.x * 2+ blockIdx.y * B * hrz_recal_chk_ld;

    
    
    //printf("block:%f---blockx=%d, blocky=%d \n",*checksum2_recal,blockIdx.x,blockIdx.y);
    
    //determine the specific colum to process
    A = A + threadIdx.x * lda;
    checksum_update = checksum_update + threadIdx.x * checksum_update_ld;
	hrz_recal_chk = hrz_recal_chk + threadIdx.x * hrz_recal_chk_ld;
	
	double d1 = (*checksum_update) - (*hrz_recal_chk);
	double d2 = (*(checksum_update + 1)) - (*hrz_recal_chk + 1);
	
	//error detected
	if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 - d1) - 1;
		printf("[col check]error detected:%f---%d \n",d1,loc);
		
		//the sum of the rest correct number except the error one
	//	double sum = 0.0;
	//	for (int i = 0; i < B; i++) {
	//		if (i != loc) {
	//			sum +=	*(A + i); 
	//		}
	//	}
	//	//correct the error
	//	*(A + loc) = *checksum_update - sum;
	}
}


__global__ void
row_detect_correct_kernel(double * A, int lda, int B, double E,
				double * checksum_update, int checksum_update_ld,
				double * vrt_recal_chk, int vrt_recal_chk_ld)
{
    //determin the block to process
	A = A + blockIdx.x * B + blockIdx.y * B * lda;
	
	//printf("block:%f---blockx=%d, blocky=%d \n",*A,blockIdx.x,blockIdx.y);
	
	
    checksum_update = checksum_update + blockIdx.x * B + blockIdx.y * 2 * checksum_update_ld;
    vrt_recal_chk = vrt_recal_chk + blockIdx.x * B + blockIdx.y * 2 * vrt_recal_chk_ld;
    
    
    //printf("block:%f---blockx=%d, blocky=%d \n",*checksum2_recal,blockIdx.x,blockIdx.y);
    
    //determine the specific colum to process
    A = A + threadIdx.x;
    checksum_update = checksum_update + threadIdx.x;
	vrt_recal_chk = vrt_recal_chk + threadIdx.x;
	
	double d1 = (*checksum_update) - (*vrt_recal_chk);
	double d2 = (*(checksum_update + checksum_update_ld)) - (*(vrt_recal_chk + vrt_recal_chk_ld));
	
	//error detected
	if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 - d1) - 1;
		printf("[row check]error detected:%f---%d \n",d1,loc);
		
		//the sum of the rest correct number except the error one
		//double sum = 0.0;
		//for (int i = 0; i < B; i++) {
		//	if (i != loc) {
		//		sum +=	*(A + i); 
		//	}
		//}
		//correct the error
		//*(A + loc) = *checksum_update - sum;
	}
}

/*
 * B: block size
 * m: # of row
 * n: # of column
 */
void col_detect_correct(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * hrz_recal_chk, int hrz_recal_chk_ld,
		cudaStream_t stream) 
{
	//printf("col_detect_correct called \n");
	//error threshold 
	/*double E = 1e-5;
	
	col_detect_correct_kernel<<<dim3(m/B, n/B), dim3(B), 0, stream>>>(A, lda, B, E,
					checksum_update, checksum_update_ld,
					hrz_recal_chk, hrz_recal_chk_ld);

	cudaStreamSynchronize(stream);
	*/				
}


/*
 * B: block size
 * m: # of row
 * n: # of column
 */
void row_detect_correct(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * vrt_recal_chk, int vrt_recal_chk_ld,
		cudaStream_t stream) 
{
	//printf("row_detect_correct called \n");
	//error threshold 
	/*
	double E = 1e-5;
	
	row_detect_correct_kernel<<<dim3(m/B, n/B), dim3(B), 0, stream>>>(A, lda, B, E,
					checksum_update, checksum_update_ld,
					vrt_recal_chk, vrt_recal_chk_ld);
	*/				
}


void col_debug(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld, 
		cudaStream_t stream) {

		double E = 1e-5;
		double * update_host = new double[(m/B)*2 * n]();
		double * chk1_host = new double[(m/B) * n]();
		double * chk2_host = new double[(m/B) * n]();

		magma_dgetmatrix((m/B)*2, n, checksum_update, checksum_update_ld, update_host, (m/B)*2);
		magma_dgetmatrix((m/B), n, checksum1_recal, checksum1_recal_ld, chk1_host, (m/B));
		magma_dgetmatrix((m/B), n, checksum2_recal, checksum2_recal_ld, chk2_host, (m/B));

		printMatrix_gpu(A, lda, 16, 16, 4, 4);

		for (int i = 0; i < m/B; i++) {
			for (int j =0 ; j < n; j++) {
				double u1 = *(update_host + j * (m/B)*2 + i * 2);
				double u2 = *(update_host + j * (m/B)*2 + i * 2 + 1);
				double r1 = *(chk1_host + j * (m/B) + i);
				double r2 = *(chk2_host + j * (m/B) + i);
			if (i < 16 && j < 16) {
				//if (fabs(u1-r1) > E) {

					printf("%d,%d,%10.10f-%10.10f=%10.10f(%f)error1\n", i, j, u1, r1, u1 - r1, fabs(u1-r1) );
				//}
				//if (fabs(u2-r2) > E) {
					printf("%d,%d,%10.10f-%10.10f=%10.10f(%f)error1\n", i, j, u2, r2, u2 - r2, fabs(u2-r2) );
				//}
			}
			}
		}
		delete [] update_host;
		delete [] chk1_host;
		delete [] chk2_host;
}









void ErrorDetectAndCorrectHost(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld)
{
	double E = 1e-5;
	//check one column by one column
	for (int c = 0; c < B; c++) {
		double d1 = *(checksum_update + checksum_update_ld * c) - *(checksum1_recal + checksum1_recal_ld * c);
		double d2 = *(checksum_update + 1 + checksum_update_ld * c) - *(checksum2_recal + checksum2_recal_ld * c);
		if(fabs(d1) > E) {
			int loc = round(d2 - d1) - 1;
			printf("error detected:%f---%d \n",d1,loc);
			
			//the sum of the rest correct number except the error one
			double sum = 0.0;
			for (int i = 0; i < B; i++) {
				if (i != loc) {
					sum +=	*(A + lda * c + i); 
				}
			}
			//correct the error
			*(A + lda * c + loc) = *(checksum_update + checksum_update_ld * c) - sum;
		}
		
	}
}

void LowerGenerator(double * A, int lda, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (j <= i) {
				*(A + j * lda + i) = rand()%20 + 1;
			} else {
				*(A + j * lda + i) = 0.0;
			}
		}
	}
}

void UpperGenerator(double * A, int lda, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (j >= i) {
				*(A + j * lda + i) = rand()/1.0 + 1;
			} else {
				*(A + j * lda + i) = 0.0;
			}
		}
	}
}

void CholeskyGenerator(double * A, int lda, int n) {
	double * L = new double[n * n];
	LowerGenerator(L, n, n, n);
	char N = 'N';
	char T = 'T';
	double one = 1;
	double zero = 0;

	blasf77_dgemm(  &N, &T,
                    &n, &n, &n,
                    &one,
                    L, &n,
                    L, &n,
                    &zero,
                    A, &lda );

}


void LUGenerator(double * A, int lda, int n) {
	double * L = new double[n * n];
	LowerGenerator(L, n, n, n);
	double * U = new double[n * n];
	UpperGenerator(U, n, n, n);
	char N = 'N';
	char T = 'T';
	double one = 1;
	double zero = 0;

	blasf77_dgemm(  &N, &T,
                    &n, &n, &n,
                    &one,
                    L, &n,
                    U, &n,
                    &zero,
                    A, &lda );

}






