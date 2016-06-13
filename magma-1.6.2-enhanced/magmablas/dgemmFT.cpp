#include"FT.h"
#include<iostream>
using namespace std;
//dgemm with FT

/**
 * m: number of row of A (N-i-B)
 * n: number of row of B (B)
 * k: number of col of A / col of B (i)
 */
void dgemmFT( magma_trans_t transA, magma_trans_t transB,
	    int m, int n, int k, 
	    double alpha, 
	    double * A, int lda,
		double * B, int ldb, 
		double beta, 
		double * C, int ldc, 
		int chk_nb,
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * streams) {

	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[4]);
	if (FT && VERIFY) {
						
		//verify B before use
		//reclaculate checksums of B on GPU
		//magmablasSetKernelStream(streams[1]);
		
		// magmablasSetKernelStream(streams[2]);
		// magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
		// 		B, lda, vd, vd_ld, MAGMA_D_ZERO, chk1, chk1_ld );
		// magmablasSetKernelStream(streams[3]);
		// magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
		// 		B, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2, chk2_ld );


		int mem_row = 0; // number of row and col of B stored in memory(no trans operation)
		int mem_col = 0;
		if (transB == MagmaNoTrans) {
			mem_row = k;
			mem_col = n;
		} else if (transB == MagmaTrans) {
			mem_row = n;
			mem_col = k;
		}
		recalculateChecksum(B, ldb,
							mem_row, mem_col,
							chk_nb,
							vd, vd_ld,
							chk1, chk2_ld,
							chk2, chk2_ld,
							streams);
		if (DEBUG) {
			cout<<"recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
			printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
			cout<<"updated checksum of B before dgemm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, (mem_row / chk_nb) * 2, mem_col);
		}


		//handle error 
//		ErrorDetectAndCorrect(B, ldb,
//							n, n, k, 
//							checksumB, checksumB_ld, 
//							chk1, chk1_ld, 
//							chk2, chk2_ld,
//							streams[1]);
//		
		
		
		
		
	
		//verify A before use
		//magmablasSetKernelStream(streams[1]);
		// for (int i = 0; i < m; i += n) {
			
		// 	magmablasSetKernelStream(streams[2]);
		// 	magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
		// 			A + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
		// 	magmablasSetKernelStream(streams[3]);
		// 	magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
		// 			A + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );
		// }
		// cudaStreamSynchronize(streams[2]);
		// cudaStreamSynchronize(streams[3]);
		//handle error
//		ErrorDetectAndCorrect(A, lda,
//							n, m, k, 
//							checksumA, checksumA_ld, 
//							chk1, chk1_ld, 
//							chk2, chk2_ld,
//							streams[1]);
		
		// if (DEBUG) {	
		// 	cout<<"recalculated checksum of A before dgemm:"<<endl;
		// 	printMatrix_gpu(chk1, chk1_ld, m / n, k);
		// 	printMatrix_gpu(chk2, chk2_ld, m / n, k);
		
		// 	cout<<"updated checksum of A before dgemm:"<<endl;
		// 	printMatrix_host(checksumA, checksumA_ld, (m / n) * 2, k);
		// }

		// number of row and col of A stored in memory(no trans operation)
		if (transA == MagmaNoTrans) {
			mem_row = m;
			mem_col = k;
		} else if (transA == MagmaTrans) {
			mem_row = k;
			mem_col = m;
		}
		recalculateChecksum(A, lda,
							mem_row, mem_col,
							chk_nb,
							vd, vd_ld,
							chk1, chk2_ld,
							chk2, chk2_ld,
							streams);
		if (DEBUG) {
			cout<<"recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
			printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
			cout<<"updated checksum of B before dgemm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, (mem_row / chk_nb) * 2, mem_col);
		}


		
	}
	
	magmablasSetKernelStream(streams[1]);
	//[Cholesky] MagmaNoTrans, MagmaTrans, MAGMA_D_ONE * (-1)， MAGMA_D_ONE
	magma_dgemm(transA, transB,
				m, n, k,
				alpha,
				A, lda, B, ldb,
				beta,
				C, ldc );
	
	if(FT){	
		//magmablasSetKernelStream(streams[1]);
		magmablasSetKernelStream(streams[4]);
		//this only works if A is not trans, B can be trans or not trans
		//we can further work on this to support trans A.
		magma_dgemm(transA, transB,
					(m / n) * 2, n, k,
					alpha,
					checksumA, checksumA_ld, B, ldb,
					beta,
					checksumC, checksumC_ld );
	}
}