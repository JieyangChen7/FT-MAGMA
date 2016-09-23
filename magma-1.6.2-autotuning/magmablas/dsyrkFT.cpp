#include"FT.h"
#include<iostream>
using namespace std;
//dsyrk with FT

/**
 * n: number of row of A
 * m: number of col of A
 */
void dsyrkFT(magma_uplo_t uplo, magma_trans_t trans,
		int n, int m, 
		double alpha,
		double * A, int lda,
		double beta,
		double * C, int ldc,
		ABFTEnv * abftEnv,
		double * col_chkA, int col_chkA_ld, bool MEM_CHECK_A,
		double * col_chkC, int col_chkC_ld, bool MEM_CHECK_C, bool COM_CHECK_C,
		bool FT, bool DEBUG, bool VERIFY,
		magma_queue_t * stream){
	
	/*		   m				n
	 * ******************   *********
	 * *		A		* =>*	C	* n
	 * *				* 	*		*
	 * ******************	*********
	 */
	//
	if (DEBUG) {
		cout << "syrk" << endl;
	}
	
	
	if (FT) { 
		if (MEM_CHECK_A) {
			cudaStreamSynchronize(stream[1]);
			cudaStreamSynchronize(stream[4]);
		
			//verify A before use
			//reclaculate checksums of A on GPU
			at_col_chk_recal(abftEnv, A, lda, n, m);
			//handle error 
			col_detect_correct(A, lda,
								abftEnv->chk_nb, n, m, 
								col_chkA, col_chkA_ld, 
								abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 
								stream[1]);
			
			if (DEBUG) {
				cudaStreamSynchronize(stream[1]);
				cout<<"[DSYRK-BEFORE]matrix A:"<<endl;
				printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 2, m, -1, -1);
			
				cout<<"[DSYRK-BEFORE]updated checksum of A:"<<endl;
				printMatrix_gpu(col_chkA, col_chkA_ld, 2, m, -1, -1);
			}
		}
		if (MEM_CHECK_C) {
			//verify C before use
			//reclaculate checksums of C on GPU
			at_col_chk_recal(abftEnv, C, ldc, n, n);
			//handle error 
			col_detect_correct(C, ldc,
								abftEnv->chk_nb, n, m, 
								col_chkC, col_chkC_ld, 
								abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 
								stream[1]);
			
			if (DEBUG) {
				cudaStreamSynchronize(stream[1]);
				cout<<"[DSYRK-BEFORE]matrix C:"<<endl;
				printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 2, n, -1, -1);
			
				cout<<"[DSYRK-BEFORE]updated checksum of C:"<<endl;
				printMatrix_gpu(col_chkA, col_chkA_ld, 2, n, -1, -1);
			}	
		}
	}

	//if (FT) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemm(
				MagmaNoTrans, MagmaTrans,
				n, n, m,
				MAGMA_D_ONE * (-1),
				A, lda, A, lda,
				MAGMA_D_ONE,
				C, ldc );
//	} else {
//		magma_dsyrk(uplo, trans, n, m,
//					alpha, A, lda,
//					beta,     C, ldc);
//	}
	
	if(FT){
		//update checksums on GPU
		magmablasSetKernelStream(stream[4]);
		magma_dgemm(
					MagmaNoTrans, MagmaTrans,
					2, n, m,
					MAGMA_D_ONE * (-1),
					col_chkA, col_chkA_ld, A, lda,
					MAGMA_D_ONE,
					col_chkC, col_chkC_ld );
	}


	if (FT) {
		if (COM_CHECK_C) {
			cudaStreamSynchronize(stream[1]);
			cudaStreamSynchronize(stream[4]);
		
			//verify C after use
			//reclaculate checksums of C on GPU
			at_col_chk_recal(abftEnv, C, ldc, n, n);
			//handle error 
			col_detect_correct(C, ldc,
								abftEnv->chk_nb, n, m, 
								col_chkC, col_chkC_ld, 
								abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 
								stream[1]);
			
			if (DEBUG) {
				cudaStreamSynchronize(stream[1]);
				cout<<"[DSYRK-AFTER]matrix C:"<<endl;
				printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 2, n, -1, -1);
			
				cout<<"[DSYRK-AFTER]updated checksum of C:"<<endl;
				printMatrix_gpu(col_chkC, col_chkC_ld, 2, n, -1, -1);
			}
		}
	}


}