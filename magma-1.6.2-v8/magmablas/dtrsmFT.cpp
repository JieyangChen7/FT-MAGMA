#include"FT.h"
#include<iostream>
using namespace std;
//TRSM with FT on GPU using cuBLAS
/*
__global__ void detectAndCorrectForTrsm(double * B, int ldb, int n,
		double * chksumB1, int incB1, double * chksumB2, int incB2,
		double * chkB1, int incB1_2, double * chkB2, int incB2_2) {
	//determin the reponsisble column 
	int block = blockIdx.x;
	int col = threadIdx.x;
	double diff = abs(
			*(chkB1 + block + col * incB1_2)
					- *(chksumB1 + block + col * incB1));
	if (diff > 0.1) {
		double diff2 = abs(
				*(chkB2 + block + col * incB2_2)
						- *(chksumB2 + block + col * incB2));
		int row = (int) round(diff2 / diff) - 1;
		*(B + n * block + row + col * ldb) += *(chksumB1 + block + col * incB1)
				- *(chkB1 + block + col * incB1_2);
	}
}
*/
/*
 * m: number of row of B
 * n: number of col of B
 */

void dtrsmFT(int m, int n, double * A, int lda,
		double * B, int ldb, 
		int K, 
		double * checksumB, int checksumB_ld,
		double * vd, int vd_ld,
		double * chk, int chk_ld, 
		double * work, int work_ld, 
		bool FT, bool DEBUG, magma_queue_t stream1, magma_queue_t stream2, magma_queue_t stream3) {

	double negone = -1;
	double one = 1;
	double zero = 0;
	
	if (FT) {
		//verify B before use
		//recalculate checksums on GPU
		double beta = 0;
		for (int i = 0; i < m; i += n) {
			for (int j = 0; j < K; j++) {
				if (j % 2 == 0) {
					magmablasSetKernelStream(stream2);
					magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
							B + i, ldb, vd + j, vd_ld, MAGMA_D_ZERO, chk + (i / n) * K + j, chk_ld );
				} else {
					magmablasSetKernelStream(stream3);
					magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
							B + i, ldb, vd + j, vd_ld, MAGMA_D_ZERO, chk + (i / n) * K + j, chk_ld );			
				}
			}
		}
		magmablasSetKernelStream(stream1);	
		//handle error - to be finished
		
		if (DEBUG) {
			cout<<"recalculated checksum of B before dtrsm:"<<endl;
			printMatrix_gpu(chk,chk_ld, (m / n) * K, n);
			cout<<"updated checksum of B before dtrsm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, (m / n) * K, n);
		}		
	}

	magma_dtrsm(MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
	                                m, n,
	                                MAGMA_D_ONE, A, lda,
	                                       B, ldb);
	if (FT) {
		//update checksums asynchronized on CPU 
		char R = 'R';
		char L = 'L';
		char T = 'T';
		char N = 'N';
		int m2 = (m / n) * K;
		int n2 = n;
		blasf77_dtrsm(&R, &L, &T, &N,
					 &m2, &n2,
					 &one,
					 work, &work_ld,
					 checksumB, &checksumB_ld);		
	}
}