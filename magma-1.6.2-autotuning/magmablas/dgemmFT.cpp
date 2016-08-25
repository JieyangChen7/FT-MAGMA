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

	// if (FT & VERIFY) {

	// 	// number of row and col of A stored in memory(no trans operation)
	// 	if (transA == MagmaNoTrans) {
	// 		mem_row = m;
	// 		mem_col = k;

	// 		at_col_chk_recal(abftEnv, A, lda, mem_row, mem_col);
	// 		col_detect_correct(A, lda, abftEnv->chk_nb, mem_row, mem_col,
 //        					  col_chkA, col_chkA_ld,
 //        					  abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld,
 //        					  abftEnv->stream[1]);

	// 		if (DEBUG) {
	// 			cout<<"[DGEMM-BEFORE] matrix A:"<<endl;
	// 			printMatrix_gpu(A, lda, mem_row, mem_col, 4, 4);
	// 			cout<<"[DGEMM-BEFORE] recalculated column checksum of A:"<<endl;
	// 			printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
	// 			cout<<"[DGEMM-BEFORE] updated column checksum of A:"<<endl;
	// 			printMatrix_gpu(col_chkA, col_chkA_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
	// 		}

	// 	} else if (transA == MagmaTrans) {
	// 		mem_row = k;
	// 		mem_col = m;

	// 		at_row_chk_recal(abftEnv, A, lda, mem_row, mem_col);
	// 		row_detect_correct(A, lda, abftEnv->chk_nb, mem_row, mem_col,
 //        					  row_chkA, row_chkA_ld,
 //        					  abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld,
 //        					  abftEnv->stream[1]);

	// 		if (DEBUG) {
	// 			cout<<"[DGEMM-BEFORE] matrix A:"<<endl;
	// 			printMatrix_gpu(A, lda, mem_row, mem_col, 4, 4);
	// 			cout<<"[DGEMM-BEFORE] recalculated row checksum of A:"<<endl;
	// 			printMatrix_gpu(abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld, mem_row , (mem_col / abftEnv->chk_nb) * 2, 4, 2);
	// 			cout<<"[DGEMM-BEFORE] updated row checksum of A:"<<endl;
	// 			printMatrix_gpu(row_chkA, row_chkA_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
	// 		}

	// 	}
		


						
	// 	//verify B before use
		

	// 	if (transB == MagmaNoTrans) {
	// 		mem_row = k;
	// 		mem_col = n;
	// 		at_row_chk_recal(abftEnv, B, ldb, mem_row, mem_col);

	// 		row_detect_correct(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
 //        					  row_chkB, row_chkB_ld,
 //        					  abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld,
 //        					  abftEnv->stream[1]);

	// 		if (DEBUG) {
	// 			cout<<"[DGEMM-BEFORE] matrix B:"<<endl;
	// 			printMatrix_gpu(B, ldb, mem_row, mem_col, 4, 4);
	// 			cout<<"[DGEMM-BEFORE] recalculated row checksum of B:"<<endl;
	// 			printMatrix_gpu(abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
	// 			cout<<"[DGEMM-BEFORE] updated row checksum of B:"<<endl;
	// 			printMatrix_gpu(row_chkB, row_chkB_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
	// 		}

	// 	} else if (transB == MagmaTrans) {
	// 		mem_row = n;
	// 		mem_col = k;
	// 		at_col_chk_recal(abftEnv, B, ldb, mem_row, mem_col);

	// 		col_detect_correct(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
 //        					  col_chkB, col_chkB_ld,
 //        					  abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld,
 //        					  abftEnv->stream[1]);

	// 		if (DEBUG) {
	// 			cout<<"[DGEMM-BEFORE] matrix B:"<<endl;
	// 			printMatrix_gpu(B, ldb, mem_row, mem_col, 4, 4);
	// 			cout<<"[DGEMM-BEFORE] recalculated column checksum of B:"<<endl;
	// 			printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
	// 			cout<<"[DGEMM-BEFORE] updated column checksum of B:"<<endl;
	// 			printMatrix_gpu(row_chkB, row_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
	// 		}
	// 	}
		
		

	// 	mem_row = m;
	// 	mem_col = n;
		
	// 	at_col_chk_recal(abftEnv, C, ldc, mem_row, mem_col);

	// 	col_detect_correct(C, ldc, abftEnv->chk_nb, mem_row, mem_col,
 //        					  col_chkC, col_chkC_ld,
 //        					  abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld,
 //        					  abftEnv->stream[1]);

	// 	at_row_chk_recal(abftEnv, C, ldc, mem_row, mem_col);

	// 	row_detect_correct(C, ldc, abftEnv->chk_nb, mem_row, mem_col,
 //        					  row_chkC, row_chkC_ld,
 //        					  abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld,
 //        					  abftEnv->stream[1]);

	// 	if (DEBUG) {

	// 		cout<<"[DGEMM-BEFORE] matrix C:"<<endl;
	// 		printMatrix_gpu(C, ldc, mem_row, mem_col, 4, 4);

	// 		cout<<"[DGEMM-BEFORE] recalculated column checksum of C:"<<endl;
	// 		printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		
	// 		cout<<"[DGEMM-BEFORE] updated column checksum of C:"<<endl;
	// 		printMatrix_gpu(col_chkC, col_chkC_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);

	// 		cout<<"[DGEMM-BEFORE] recalculated row checksum of C:"<<endl;
	// 		printMatrix_gpu(abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		
	// 		cout<<"[DGEMM-BEFORE] updated row checksum of C:"<<endl;
	// 		printMatrix_gpu(row_chkC, row_chkC_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
	// 	}
		
	// }

	
	magmablasSetKernelStream(stream[1]);
	//[Cholesky] MagmaNoTrans, MagmaTrans, MAGMA_D_ONE * (-1)， MAGMA_D_ONE
	magma_dgemm(transA, transB,
				m, n, k,
				alpha,
				A, lda, B, ldb,
				beta,
				C, ldc );
	
	if(FT){	
		magmablasSetKernelStream(stream[4]);

		if (transA == MagmaNoTrans) {
			magma_dgemm(transA, transB,
					(m / abftEnv->chk_nb) * 2, n, k,
					alpha,
					col_chkA, col_chkA_ld, B, ldb,
					beta,
					col_chkC, col_chkC_ld );
		} else {
			magma_dgemm(transA, transB,
					(m / abftEnv->chk_nb) * 2, n, k,
					alpha,
					row_chkA, row_chkA_ld, B, ldb,
					beta,
					col_chkC, col_chkC_ld );
		}

		if (transB == MagmaNoTrans) {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
					m , (n / abftEnv->chk_nb) * 2, k,
					alpha,
					A, lda,
					row_chkB, row_chkB_ld,
					beta,
					row_chkC, row_chkC_ld );
		} else {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
					m , (n / abftEnv->chk_nb) * 2, k,
					alpha,
					A, lda,
					col_chkB, col_chkB_ld,
					beta,
					row_chkC, row_chkC_ld );
		}
		
	}


	if (FT & VERIFY) {
		cudaStreamSynchronize(stream[1]);
		cudaStreamSynchronize(stream[4]);

		// number of row and col of A stored in memory(no trans operation)
		// if (transA == MagmaNoTrans) {
		// 	mem_row = m;
		// 	mem_col = k;

		// 	at_col_chk_recal(abftEnv, A, lda, mem_row, mem_col);
		// 	col_detect_correct(A, lda, abftEnv->chk_nb, mem_row, mem_col,
  //       					  col_chkA, col_chkA_ld,
  //       					  abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld,
  //       					  abftEnv->stream[1]);

		// 	if (DEBUG) {
		// 		cout<<"[DGEMM-AFTER] matrix A:"<<endl;
		// 		printMatrix_gpu(A, lda, mem_row, mem_col, 4, 4);
		// 		cout<<"[DGEMM-AFTER] recalculated column checksum of A:"<<endl;
		// 		printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		// 		cout<<"[DGEMM-AFTER] updated column checksum of A:"<<endl;
		// 		printMatrix_gpu(col_chkA, col_chkA_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		// 	}

		// } else if (transA == MagmaTrans) {
		// 	mem_row = k;
		// 	mem_col = m;

		// 	at_row_chk_recal(abftEnv, A, lda, mem_row, mem_col);
		// 	row_detect_correct(A, lda, abftEnv->chk_nb, mem_row, mem_col,
  //       					  row_chkA, row_chkA_ld,
  //       					  abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld,
  //       					  abftEnv->stream[1]);

		// 	if (DEBUG) {
		// 		cout<<"[DGEMM-AFTER] matrix A:"<<endl;
		// 		printMatrix_gpu(A, lda, mem_row, mem_col, 4, 4);
		// 		cout<<"[DGEMM-AFTER] recalculated row checksum of A:"<<endl;
		// 		printMatrix_gpu(abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld, mem_row , (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		// 		cout<<"[DGEMM-AFTER] updated column checksum of A:"<<endl;
		// 		printMatrix_gpu(row_chkA, row_chkA_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		// 	}

		// }
		


						
		// //verify B after use
		

		// if (transB == MagmaNoTrans) {
		// 	mem_row = k;
		// 	mem_col = n;
		// 	at_row_chk_recal(abftEnv, B, ldb, mem_row, mem_col);

		// 	row_detect_correct(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
  //       					  row_chkB, row_chkB_ld,
  //       					  abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld,
  //       					  abftEnv->stream[1]);

		// 	if (DEBUG) {
		// 		cout<<"[DGEMM-AFTER] matrix B:"<<endl;
		// 		printMatrix_gpu(B, ldb, mem_row, mem_col, 4, 4);
		// 		cout<<"[DGEMM-AFTER] recalculated row checksum of B:"<<endl;
		// 		printMatrix_gpu(abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		// 		cout<<"[DGEMM-AFTER] updated row checksum of B:"<<endl;
		// 		printMatrix_gpu(row_chkB, row_chkB_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		// 	}

		// } else if (transB == MagmaTrans) {
		// 	mem_row = n;
		// 	mem_col = k;
		// 	at_col_chk_recal(abftEnv, B, ldb, mem_row, mem_col);

		// 	col_detect_correct(B, ldb, abftEnv->chk_nb, mem_row, mem_col,
  //       					  col_chkB, col_chkB_ld,
  //       					  abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld,
  //       					  abftEnv->stream[1]);

		// 	if (DEBUG) {
		// 		cout<<"[DGEMM-AFTER] matrix B:"<<endl;
		// 		printMatrix_gpu(B, ldb, mem_row, mem_col, 4, 4);
		// 		cout<<"[DGEMM-AFTER] recalculated column checksum of B:"<<endl;
		// 		printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		// 		cout<<"[DGEMM-AFTER] updated column checksum of B:"<<endl;
		// 		printMatrix_gpu(row_chkB, row_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		// 	}
		// }
		
		

		mem_row = m;
		mem_col = n;
		
		at_col_chk_recal(abftEnv, C, ldc, mem_row, mem_col);

		col_detect_correct(C, ldc, abftEnv->chk_nb, mem_row, mem_col,
        					  col_chkC, col_chkC_ld,
        					  abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld,
        					  abftEnv->stream[1]);

		at_row_chk_recal(abftEnv, C, ldc, mem_row, mem_col);

		row_detect_correct(C, ldc, abftEnv->chk_nb, mem_row, mem_col,
        					  row_chkC, row_chkC_ld,
        					  abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld,
        					  abftEnv->stream[1]);

		if (DEBUG) {

			cout<<"[DGEMM-AFTER] matrix C:"<<endl;
			printMatrix_gpu(C, ldc, mem_row, mem_col, 4, 4);

			cout<<"[DGEMM-AFTER] recalculated column checksum of C:"<<endl;
			printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		
			cout<<"[DGEMM-AFTER] updated column checksum of C:"<<endl;
			printMatrix_gpu(col_chkC, col_chkC_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);

			cout<<"[DGEMM-AFTER] recalculated row checksum of C:"<<endl;
			printMatrix_gpu(abftEnv->vrt_recal_chk, abftEnv->vrt_recal_chk_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		
			cout<<"[DGEMM-AFTER] updated row checksum of C:"<<endl;
			printMatrix_gpu(row_chkC, row_chkC_ld, mem_row, (mem_col / abftEnv->chk_nb) * 2, 4, 2);
		}

	}
}