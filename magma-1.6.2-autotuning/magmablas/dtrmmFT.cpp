#include"FT.h"
#include<iostream>
using namespace std;


void dtrmmFT( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    		int m, int n,
    		double alpha,
    		double * dA, int ldda,
    		double * dB, int lddb,
    		ABFTEnv * abftEnv,
			double * col_chkA, int col_chkA_ld,
			double * row_chkA, int row_chkA_ld,		
			double * col_chkB, int col_chkB_ld,
			double * row_chkB, int row_chkB_ld, 
			bool FT, bool DEBUG, bool VERIFY, 
			magma_queue_t * stream) {

	int mem_row = 0; // number of row and col stored in memory(no trans operation)
	int mem_col = 0;

	if (VERIFY) {
		mem_row = n;
		mem_col = n;
		at_col_chk_recal(abftEnv, dA, ldda, mem_row, mem_col);


		if (DEBUG) {
			
			cout<<"[DTRMM] A before dgemm:"<<endl;
			printMatrix_gpu(dA, ldda, mem_row, mem_col, 4, 4);

			cout<<"[DTRMM] recalculated column checksum of A before dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
		
			cout<<"[DTRMM] updated column checksum of A before dgemm:"<<endl;
			printMatrix_gpu(col_chkA, col_chkA_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		}

		mem_row = m;
		mem_col = n;
		at_col_chk_recal(abftEnv, dB, lddb, mem_row, mem_col);


		if (DEBUG) {
			
			cout<<"[DTRMM] B before dgemm:"<<endl;
			printMatrix_gpu(dB, lddb, mem_row, mem_col, 4, 4);

			cout<<"[DTRMM] recalculated column checksum of B before dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
		
			cout<<"[DTRMM] updated column checksum of B before dgemm:"<<endl;
			printMatrix_gpu(col_chkB, col_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		}


	}


	//update column checksum
	dtrmmFT( side, uplo, trans, diag,
             (m / abftEnv->chk_nb) * 2, n,
             alpha, dA, ldda,
             col_chkB, col_chkB_ld);


	//update row checksum
	magma_dgemm(MagmaNoTrans, MagmaNoTrans,
				m, (n / abftEnv->chk_nb) * 2, n,
				alpha,
				dB, lddb, row_chkA, row_chkA_ld,
				0,
				row_chkB, row_chkB_ld );

	dtrmmFT( side, uplo, trans, diag,
                m, n,
                alpha, dA,  ldda,
                dB, lddb);



	if (VERIFY) {
		mem_row = n;
		mem_col = n;
		at_col_chk_recal(abftEnv, dA, ldda, mem_row, mem_col);


		if (DEBUG) {
			
			cout<<"[DTRMM] A after dgemm:"<<endl;
			printMatrix_gpu(dA, ldda, mem_row, mem_col, 4, 4);

			cout<<"[DTRMM] recalculated column checksum of A after dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
		
			cout<<"[DTRMM] updated column checksum of A after dgemm:"<<endl;
			printMatrix_gpu(col_chkA, col_chkA_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		}

		mem_row = m;
		mem_col = n;
		at_col_chk_recal(abftEnv, dB, lddb, mem_row, mem_col);


		if (DEBUG) {
			
			cout<<"[DTRMM] B after dgemm:"<<endl;
			printMatrix_gpu(dB, lddb, mem_row, mem_col, 4, 4);

			cout<<"[DTRMM] recalculated column checksum of B after dgemm:"<<endl;
			printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
			printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld, mem_row / abftEnv->chk_nb, mem_col, 1, 4);
		
			cout<<"[DTRMM] updated column checksum of B after dgemm:"<<endl;
			printMatrix_gpu(col_chkB, col_chkB_ld, (mem_row / abftEnv->chk_nb) * 2, mem_col, 2, 4);
		}


	}

	
}