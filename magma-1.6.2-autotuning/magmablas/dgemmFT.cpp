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
		ABFTEnv * abftEnv,
		double * col_chkA, int col_chkA_ld,
		double * row_chkA, int row_chkA_ld,		
		double * col_chkB, int col_chkB_ld,
		double * row_chkB, int row_chkB_ld,
		double * col_chkC, int col_chkC_ld,  
		double * row_chkC, int row_chkC_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream) {

	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[4]);

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

		AutoTuneChecksumRecal(abftEnv, B, ldb, mem_row, mem_col, stream);

		if (DEBUG) {
			cudaStreamSynchronize(stream[1]);
			cout<<"[dgemm] B before dgemm:"<<endl;

			printMatrix_gpu(B, ldb, mem_row, mem_col, -1, -1);


			cout<<"[dgemm] recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, -1, -1);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, -1, -1);
		
			cout<<"[dgemm] updated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(col_chkB, col_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, -1, -1);
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

		AutoTuneChecksumRecal(abftEnv, A, lda, mem_row, mem_col, stream);

		if (DEBUG) {

			cout<<"[dgemm] A before dgemm:"<<endl;

			printMatrix_gpu(A, lda, mem_row, mem_col, -1, -1);

			cout<<"[dgemm] recalculated checksum of A before dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, -1, -1);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, -1, -1);
		
			cout<<"[dgemm] updated checksum of A before dgemm:"<<endl;
			printMatrix_gpu(col_chkA, col_chkA_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, -1, -1);
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

		AutoTuneChecksumRecal(abftEnv, A, lda, mem_row, mem_col, stream);

		if (DEBUG) {

			cout<<"[dgemm] C before dgemm:"<<endl;

			printMatrix_gpu(C, ldc, mem_row, mem_col, -1, -1);

			cout<<"[dgemm] recalculated checksum of C before dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, -1, -1);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, -1, -1);
		
			cout<<"[dgemm] updated checksum of C before dgemm:"<<endl;
			printMatrix_gpu(col_chkC, col_chkC_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, -1, -1);
		}



		
	}
	
	magmablasSetKernelStream(stream[1]);
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
					(m / abftEnv->chk_nb) * 2, n, k,
					alpha,
					col_chkA, col_chkA_ld, B, ldb,
					beta,
					col_chkC, col_chkC_ld );

		//we can further work on this to support trans A.
		magma_dgemm(transA, transB,
					m , (n / abftEnv->chk_nb) * 2, k,
					alpha,
					A, lda,
					row_chkB, row_chkB_ld,
					beta,
					row_chkC, row_chkC_ld );

		cudaStreamSynchronize(stream[1]);
		cudaStreamSynchronize(stream[4]);

		mem_row = m;
		mem_col = n;
		
		// recalculateChecksum(C, ldc,
		// 					mem_row, mem_col,
		// 					chk_nb,
		// 					vd, vd_ld,
		// 					chk1, chk1_ld,
		// 					chk2, chk2_ld,
		// 					streams);
		// if (DEBUG) {

		 	cout<<"[dgemm] C before dgemm:"<<endl;

		 	printMatrix_gpu(C, ldc, mem_row, mem_col, 4, 4);

		// 	cout<<"[dgemm] recalculated checksum of C after dgemm:"<<endl;
		// 	printMatrix_gpu(chk1, chk1_ld, mem_row / chk_nb, mem_col);
		// 	printMatrix_gpu(chk2, chk2_ld, mem_row / chk_nb, mem_col);
		
			cout<<"[dgemm] updated column checksum of C after dgemm:"<<endl;
			printMatrix_gpu(col_chkC, col_chkC_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);

			cout<<"[dgemm] updated row checksum of C after dgemm:"<<endl;
			printMatrix_gpu(row_chkC, row_chkC_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		// }



	}
}