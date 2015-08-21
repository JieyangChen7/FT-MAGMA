#include"FT.h"
#include<iostream>
using namespace std;
//dgemm with FT
/*
__global__ void detectAndCorrectForGemm(double * C, int ldc, int n,
		double * chksumC1, int incC1, double * chksumC2, int incC2,
		double * chkC1, int incC1_2, double * chkC2, int incC2_2){
	//determin the reponsisble column 
	int block = blockIdx.x;
	int col = threadIdx.x;
	double diff = abs(*(chkC1+block+col*incC1_2)-*(chksumC1+block+col*incC1));
	if(diff>0.1){
		double diff2=abs(*(chkC2+block+col*incC2_2)-*(chksumC2+block+col*incC2));
		int row = (int)round(diff2/diff)-1;
		*(C+n*block+row+col*ldc) += *(chksumC1+block+col*incC1)-*(chkC1+block+col*incC1_2);
	}
}
*/
/**
 * m: number of row of A (N-i-B)
 * n: number of row of B (B)
 * k: number of col of A / col of B (i)
 */
void dgemmFT(int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc,
		int K,
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double ** chk1, int * chk_ld,
		double * temp, int temp_ld,
		magma_queue_t stream0, magma_queue_t stream1, magma_queue_t stream2, magma_queue_t stream3,
		bool FT, bool DEBUG) {

	double negone = -1;
	double one = 1;
	double zero = 0;
	
	if (FT) {
		
		magma_dgetmatrix_async( n, k,
								B, ldb,
								temp, temp_ld,
								stream0 );							
		//verify B before use
		//reclaculate checksums of B on GPU
		
		for (int i = 0; i < K; i++) {
			if (i % 2 == 0) {
				magmablasSetKernelStream(stream2);
				magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
						B, ldb, vd + i, vd_ld, MAGMA_D_ZERO, chk[i], chk_ld[i] );
			} else {
				magmablasSetKernelStream(stream3);
				magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
						B, ldb, vd + i, vd_ld, MAGMA_D_ZERO, chk[i], chk_ld[i] );
			}
		}
		//handle error - to be finished
		
		
		if (DEBUG) {
			cout<<"recalculated checksum of B before dgemm:"<<endl;
			for (int i = 0; i < K; i++) {
				printMatrix_gpu(chk[i], chk_ld[i], 1, k);
			}
			cout<<"updated checksum of B before dgemm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, K, k);
		}
		
		
	
		//verify A before use
		for (int i = 0; i < m; i += n) {
			for (int j = 0; j < K; j++) {
					if (j % 2 == 0) {
						magmablasSetKernelStream(stream2);
						magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
								A + i, ldb, vd + j, vd_ld, MAGMA_D_ZERO, chk[j], chk_ld[j] );
					} else {
						magmablasSetKernelStream(stream3);
						magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
								A + i, ldb, vd + j, vd_ld, MAGMA_D_ZERO, chk[j], chk_ld[j] );
					}
			}
		}
		//handle error - to be finished
		magmablasSetKernelStream(stream1);

		if (DEBUG) {	
			cout<<"recalculated checksum of A before dgemm:"<<endl;
			for (int i = 0; i < K; i++) {
				printMatrix_gpu(chk[i], chk_ld[i], m / n, k);
			}
			cout<<"updated checksum of A before dgemm:"<<endl;
			printMatrix_host(checksumA, checksumA_ld, (m / n) * K, k);
		}
		
	}
	
	
	magma_dgemm(
				MagmaNoTrans, MagmaTrans,
				m, n, k,
				MAGMA_D_ONE * (-1),
				A, lda, B, ldb,
				MAGMA_D_ONE,
				C, ldc );
	
	if(FT){	
		magma_queue_sync( stream0 );
		//update checksums on CPU
		char N = 'N';
		char T = 'T';
		int m2 = (m / n) * K;
		int n2 = n;
		int k2 = k;
				
		blasf77_dgemm(  &N, &T,
						&m2, &n2, &k2,
						&negone,
						checksumA, &checksumA_ld,
						temp, &temp_ld,
						&one,
						checksumC, &checksumC_ld );
	}
}