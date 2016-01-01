#include"FT.h"
#include<iostream>
using namespace std;
//dgemm with FT

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
	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[4]);
	if (FT && VERIFY) {
						
		//verify B before use
		//reclaculate checksums of B on GPU
		//magmablasSetKernelStream(streams[1]);
		
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
				B, lda, vd, vd_ld, MAGMA_D_ZERO, chk1, chk1_ld );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
				B, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2, chk2_ld );
		cudaStreamSynchronize(streams[2]);
		cudaStreamSynchronize(streams[3]);
		//handle error 
//		ErrorDetectAndCorrect(B, ldb,
//							n, n, k, 
//							checksumB, checksumB_ld, 
//							chk1, chk1_ld, 
//							chk2, chk2_ld,
//							streams[1]);
//		
		
		if (DEBUG) {
			cout<<"recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, 1, k);
			printMatrix_gpu(chk2, chk2_ld, 1, k);
		
			cout<<"updated checksum of B before dgemm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, 2, k);
		}
		
		
	
		//verify A before use
		//magmablasSetKernelStream(streams[1]);
		for (int i = 0; i < m; i += n) {
			
			magmablasSetKernelStream(streams[2]);
			magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
					A + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
					A + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );
		}
		cudaStreamSynchronize(streams[2]);
		cudaStreamSynchronize(streams[3]);
		//handle error
//		ErrorDetectAndCorrect(A, lda,
//							n, m, k, 
//							checksumA, checksumA_ld, 
//							chk1, chk1_ld, 
//							chk2, chk2_ld,
//							streams[1]);
		
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
		//magmablasSetKernelStream(streams[1]);
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