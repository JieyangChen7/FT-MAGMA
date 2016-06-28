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
		int nb,
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * vd2, int vd2_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		double * chk21, int chk21_ld, 
	    double * chk22, int chk22_ld, 
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * streams,
		int * mapping, int mapping_ld) {

	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[4]);

	int mem_row = 0; // number of row and col of B stored in memory(no trans operation)
	int mem_col = 0;
	if (FT && VERIFY) {
						
		//verify B before use
		if (transB == MagmaNoTrans) {
			mem_row = k;
			mem_col = n;
		} else if (transB == MagmaTrans) {
			mem_row = n;
			mem_col = k;
		}
		// recalculateChecksum2(B, ldb,
		// 					mem_row, mem_col,
		// 					chk_nb,
		// 					vd, vd_ld,
		// 					chk1, chk1_ld,
		// 					chk2, chk2_ld,
		// 					streams);

		AutoTuneChecksumRecal(B, ldb,
				   mem_row, mem_col, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   mapping, mapping_ld);

		if (DEBUG) {
			cudaStreamSynchronize(streams[1]);
			cout<<"[dgemm] B before dgemm:"<<endl;

			printMatrix_gpu(B, ldb, mem_row, mem_col);


			cout<<"[dgemm] recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
			printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
			cout<<"[dgemm] updated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(checksumB, checksumB_ld, (mem_row / chk_nb) * 2, mem_col);
		}



		// number of row and col of A stored in memory(no trans operation)
		if (transA == MagmaNoTrans) {
			mem_row = m;
			mem_col = k;
		} else if (transA == MagmaTrans) {
			mem_row = k;
			mem_col = m;
		}
		// recalculateChecksum2(A, lda,
		// 					mem_row, mem_col,
		// 					chk_nb,
		// 					vd, vd_ld,
		// 					chk1, chk1_ld,
		// 					chk2, chk2_ld,
		// 					streams);

		AutoTuneChecksumRecal(A, lda,
				   mem_row, mem_col, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   mapping, mapping_ld);

		if (DEBUG) {

			cout<<"[dgemm] A before dgemm:"<<endl;

			printMatrix_gpu(A, lda, mem_row, mem_col);

			cout<<"[dgemm] recalculated checksum of A before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
			printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
			cout<<"[dgemm] updated checksum of A before dgemm:"<<endl;
			printMatrix_gpu(checksumA, checksumA_ld, (mem_row / chk_nb) * 2, mem_col);
		}


		
		mem_row = m;
		mem_col = n;
		
		// recalculateChecksum2(C, ldc,
		// 					mem_row, mem_col,
		// 					chk_nb,
		// 					vd, vd_ld,
		// 					chk1, chk1_ld,
		// 					chk2, chk2_ld,
		// 					streams);

		AutoTuneChecksumRecal(C, ldc,
				   mem_row, mem_col, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   mapping, mapping_ld);

		if (DEBUG) {

			cout<<"[dgemm] C before dgemm:"<<endl;

			printMatrix_gpu(C, ldc, mem_row, mem_col);

			cout<<"[dgemm] recalculated checksum of C before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
			printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
			cout<<"[dgemm] updated checksum of C before dgemm:"<<endl;
			printMatrix_gpu(checksumC, checksumC_ld, (mem_row / chk_nb) * 2, mem_col);
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
		//magmablasSetKernelStream(streams[4]);
		//this only works if A is not trans, B can be trans or not trans
		//we can further work on this to support trans A.
		magma_dgemm(transA, transB,
					(m / nb) * 2, n, k,
					alpha,
					checksumA, checksumA_ld, B, ldb,
					beta,
					checksumC, checksumC_ld );

		// cudaStreamSynchronize(streams[1]);
		// cudaStreamSynchronize(streams[4]);

		// mem_row = m;
		// mem_col = n;
		
		// recalculateChecksum(C, ldc,
		// 					mem_row, mem_col,
		// 					chk_nb,
		// 					vd, vd_ld,
		// 					chk1, chk1_ld,
		// 					chk2, chk2_ld,
		// 					streams);
		// if (DEBUG) {

		// 	cout<<"[dgemm] C before dgemm:"<<endl;

		// 	printMatrix_gpu(C, ldc, mem_row, mem_col);

		// 	cout<<"[dgemm] recalculated checksum of C after dgemm:"<<endl;
		// 	printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
		// 	printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
		// 	cout<<"[dgemm] updated checksum of C after dgemm:"<<endl;
		// 	printMatrix_gpu(checksumC, checksumC_ld, (mem_row / chk_nb) * 2, mem_col);
		// }



	}
}