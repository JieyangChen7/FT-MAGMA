#include"FT.h"
#include<iostream>
using namespace std;
//TRSM with FT on GPU using cuBLAS

/*
 * m: number of row of B
 * n: number of col of B
 */

void dtrsmFT(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
		int m, int n, double alpha, double * A, int lda,
		double * B, int ldb, 
		ABFTEnv * abftEnv,
		double * col_chkA, int col_chkA_ld,
		double * row_chkA, int row_chkA_ld,
		double * col_chkB, int col_chkB_ld,
		double * row_chkB, int row_chkB_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream) {
	cout << "trsm" << endl;
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[4]);
	if (VERIFY) {
		//verify B before use
		int mem_row = m; // number of row and col of B stored in memory(no trans operation)
		int mem_col = n;
		//cout << "trsm verify" << endl;
		at_col_chk_recal(abftEnv, B, ldb, mem_row, mem_col);

		// col_detect_correct(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
  //       					  col_chkB, col_chkB_ld,
  //       					  abftEnv->chk1, abftEnv->chk1_ld,
  //       					  abftEnv->chk2, abftEnv->chk2_ld,
  //       					  abftEnv->stream[1]);

		// col_debug(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
  //       					  col_chkB, col_chkB_ld,
  //       					  abftEnv->chk1, abftEnv->chk1_ld,
  //       					  abftEnv->chk2, abftEnv->chk2_ld,
  //       					  abftEnv->stream[1]);


		if (DEBUG) {
			cout<<"[trsm] updated B before trsm:"<<endl;
			printMatrix_gpu(B, ldb, mem_row, mem_col, 4, 4);

			cudaStreamSynchronize(stream[1]);
			cout<<"[trsm] recalculated column checksum of B before trsm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
		
			cout<<"[trsm] updated column checksum of B before trsm:"<<endl;
			printMatrix_gpu(col_chkB, col_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		}

	}
	magmablasSetKernelStream(stream[1]);	
	//[Cholesky]MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit, MAGMA_D_ONE
	//printMatrix_gpu(B, ldb, 16, 16, 4, 4);

	magma_dtrsm(side, uplo, trans, diag,
				m, n,
				alpha, A, lda,
			    B, ldb);


	if (FT) {
		//update checksums
		//magmablasSetKernelStream(stream[1]);	
		
		magmablasSetKernelStream(stream[4]);	
		magma_dtrsm(side, uplo, trans, diag,
                    (m / abftEnv->chk_nb) * 2, n,
                    alpha, A, lda,
                    col_chkB, col_chkB_ld);
	}


	if (VERIFY) {

		cudaStreamSynchronize(stream[1]);
		cudaStreamSynchronize(stream[4]);
		
		int mem_row = m; // number of row and col of B stored in memory(no trans operation)
		int mem_col = n;		
					
		at_col_chk_recal(abftEnv, B, ldb, mem_row, mem_col);
		col_detect_correct(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
        					  col_chkB, col_chkB_ld,
        					  abftEnv->chk1, abftEnv->chk1_ld,
        					  abftEnv->chk2, abftEnv->chk2_ld,
        					  abftEnv->stream[1]);
		
		if (DEBUG) {
			cout<<"[trsm] updated B after trsm:"<<endl;
			printMatrix_gpu(B, ldb, mem_row, mem_col, 4, 4);

			cout<<"[trsm] recalculated checksum of B after trsm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			

			cout<<"[trsm] updated column checksum of B after trsm:"<<endl;
			printMatrix_gpu(col_chkB, col_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		}
	}
}