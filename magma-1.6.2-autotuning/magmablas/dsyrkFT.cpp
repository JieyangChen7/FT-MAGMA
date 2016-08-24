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
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream){
	
	/*		   m				n
	 * ******************   *********
	 * *		A		* =>*	C	* n
	 * *				* 	*		*
	 * ******************	*********
	 */
	//cout << "syrk" << endl;
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[4]);
	
	
	if (FT && VERIFY) {
		//verify A before use
		//reclaculate checksums of A on GPU
		at_col_chk_recal(abftEnv, A, lda, n, m);
		//handle error 
		col_detect_correct(A, lda,
							abftEnv->chk_nb, n, m, 
							checksumA, checksumA_ld, 
							abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 
							stream[1]);
		
		if (DEBUG) {
			cudaStreamSynchronize(stream[1]);
			cout<<"recalculated checksum of A before dsyrk:"<<endl;
			printMatrix_gpu(abftEnv->hrz_recal_chk, abftEnv->hrz_recal_chk_ld, 2, m, -1, -1);
		
			cout<<"updated checksum of A before dsyrk:"<<endl;
			printMatrix_gpu(checksumA, checksumA_ld, 2, m, -1, -1);
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
		//magmablasSetKernelStream(streams[1]);
		magmablasSetKernelStream(stream[4]);
		magma_dgemm(
					MagmaNoTrans, MagmaTrans,
					2, n, m,
					MAGMA_D_ONE * (-1),
					checksumA, checksumA_ld, A, lda,
					MAGMA_D_ONE,
					checksumC, checksumC_ld );
	}
}