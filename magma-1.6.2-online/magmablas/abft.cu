/*
    Enhanced Online ABFT
    UC Riverside
    Jieyang Chen
*/
#include "FT.h"
#include "common_magma.h"


__global__ void
DetectAndCorrectKernel(double * A, int lda, int B, double E,
				double * checksum_update, int checksum_update_ld,
				double * checksum1_recal, int checksum1_recal_ld,
				double * checksum2_recal, int checksum2_recal_ld)
{
    //determin the block to process
	A = A + blockIdx.x * B + blockIdx.y * B * lda;
	
	//printf("block:%f---blockx=%d, blocky=%d \n",*A,blockIdx.x,blockIdx.y);
	
	
    checksum_update = checksum_update + blockIdx.x * 2  + blockIdx.y * B * checksum_update_ld;
    checksum1_recal = checksum1_recal + blockIdx.x      + blockIdx.y * B * checksum1_recal_ld;
    checksum2_recal = checksum2_recal + blockIdx.x      + blockIdx.y * B * checksum2_recal_ld;
    
    //determine the specific colum to process
    A = A + threadIdx.x * lda;
    checksum_update = checksum_update + threadIdx.x * checksum_update_ld;
	checksum1_recal = checksum1_recal + threadIdx.x * checksum1_recal_ld;
	checksum2_recal = checksum2_recal + threadIdx.x * checksum2_recal_ld;
	
	double d1 = (*checksum_update)       - (*checksum1_recal);
	double d2 = (*(checksum_update + 1)) - (*checksum2_recal);
	
	//error detected
	if(fabs(d1) > E) {
		//locate the error
		int loc = round(d2 / d1) - 1;
		printf("error detected:%f---%d \n",d1,loc);
		
		//the sum of the rest correct number except the error one
		double sum = 0.0;
		for (int i = 0; i < B; i++) {
			if (i != loc) {
				sum +=	*(A + i); 
			}
		}
		//correct the error
		*(A + loc) = *checksum_update - sum;
	}
}

/*
 * B: block size
 * m: # of row
 * n: # of column
 */


void ErrorDetectAndCorrect(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld, 
		cudaStream_t stream) 
{
	//error threshold 
	double E = 1e-10;
	
	DetectAndCorrectKernel<<<dim3(m/B, n/B), dim3(B), 0, stream>>>(A, lda, B, E,
					checksum_update, checksum_update_ld,
					checksum1_recal, checksum1_recal_ld,
					checksum2_recal, checksum2_recal_ld);
}

void ErrorDetectAndCorrectHost(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld)
{
	double E = 1e-10;
	//check one column by one column
	for (int c = 0; c < B; c++) {
		double d1 = *(checksum_update + checksum_update_ld * c) - *(checksum1_recal + checksum1_recal_ld * c);
		double d2 = *(checksum_update + 1 + checksum_update_ld * c) - *(checksum2_recal + checksum2_recal_ld * c);
		if(fabs(d1) > E) {
			int loc = round(d2 / d1) - 1;
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


