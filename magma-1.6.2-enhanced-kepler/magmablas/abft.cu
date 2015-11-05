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
DetectAndCorrect(double * A, int lda, int B, double E,
				double * checksum_update, double * checksum_update_ld,
				double * checksum1_recal, double * checksum1_recal_ld,
				double * checksum2_recal, double * checksum2_recal_ld)
{
    //determin the block to process
	A = A + blockIdx.x * B * ld + blockIdx.y * B;
    checksum_update = checksum_update + blockIdx.x * B * checksum_update_ld + blockIdx.y * 2;
    checksum1_recal = checksum1_recal + blockIdx.x * B * checksum1_recal_ld + blockIdx.y;
    checksum2_recal = checksum2_recal + blockIdx.x * B * checksum2_recal_ld + blockIdx.y;
    
    //determine the specific colum to process
    A = A + threadIdx.x * ld;
    checksum_update = checksum_update + threadIdx.x * checksum_update_ld;
	checksum1_recal = checksum_update + threadIdx.x * checksum1_recal_ld;
	checksum2_recal = checksum_update + threadIdx.x * checksum2_recal_ld;
	
	double d1 = (*checksum_update) - (*checksum1_recal);
	double d2 = (*(checksum_update + 1)) - (*checksum2_recal);
	
	//error detected
	if(fabs(d1) > E) {
		int loc = round(d2 - d1);
		printf("error detected:%f---%d \n",d1,loc);
	}
}



// ----------------------------------------------------------------------
// adds   x += r (including conversion to double)  --and--
// copies w = b
void
test_abft(double * A, int lda, int B, int n, int m,
		double * checksum_update, double * checksum_update_ld,
		double * checksum1_recal, double * checksum1_recal_ld,
		double * checksum2_recal, double * checksum2_recal_ld) 
{
	
	double E = 1e-10;
	
	_DetectAndCorrect<<<dim3(m/B, n/B), dim3(B)>>>(A, lda, B, E,
					checksum_update, checksum_update_ld,
					checksum1_recal, checksum1_recal_ld,
					checksum2_recal, checksum2_recal_ld)
}

