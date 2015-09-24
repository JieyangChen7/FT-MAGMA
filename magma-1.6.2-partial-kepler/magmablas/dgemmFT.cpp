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
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams,
		bool FT, bool DEBUG, bool VERIFY) {

	double negone = -1;
	double one = 1;
	double zero = 0;
	
	if (FT && VERIFY) {
						
		//verify B before use
		//reclaculate checksums of B on GPU
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
				B, lda, vd, vd_ld, MAGMA_D_ZERO, chk1, chk1_ld );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
				B, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2, chk2_ld );
		//handle error - to be finished
		
		
		if (DEBUG) {
			cout<<"recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, 1, k);
			printMatrix_gpu(chk2, chk2_ld, 1, k);
		
			cout<<"updated checksum of B before dgemm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, 2, k);
		}
		
		
	
		//verify A before use
		for (int i = 0; i < m; i += n) {
			magmablasSetKernelStream(streams[2]);
			magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
					A + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
					A + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );
		}
		//handle error - to be finished
		

		
		
		
		
		if (DEBUG) {	
			cout<<"recalculated checksum of A before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, m / n, k);
			printMatrix_gpu(chk2, chk2_ld, m / n, k);
		
			cout<<"updated checksum of A before dgemm:"<<endl;
			printMatrix_host(checksumA, checksumA_ld, (m / n) * 2, k);
		}
		
	}
	
	magmablasSetKernelStream(streams[1]);
	magma_dgemm(
				MagmaNoTrans, MagmaTrans,
				m, n, k,
				MAGMA_D_ONE * (-1),
				A, lda, B, ldb,
				MAGMA_D_ONE,
				C, ldc );
	
	if(FT){	
		magmablasSetKernelStream(streams[4]);
		magma_dgemm(
					MagmaNoTrans, MagmaTrans,
					(m / n) * 2, n, k,
					MAGMA_D_ONE * (-1),
					checksumA, checksumA_ld, B, ldb,
					MAGMA_D_ONE,
					checksumC, checksumC_ld );
	}
}