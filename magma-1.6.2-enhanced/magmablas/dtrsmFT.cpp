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
		double * B, int ldb, double * checksumB, int checksumB_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		bool FT, bool DEBUG, bool VERIFY, magma_queue_t * streams) {

	cout << "test0" << endl;
	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[4]);
	if (FT && VERIFY) {
		//verify B before use
		//recalculate checksums on GPU

		for (int i = 0; i < m; i += n) {
			magmablasSetKernelStream(streams[2]);
			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
					B + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
					B + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );			
		}
		cudaStreamSynchronize(streams[2]);
		cudaStreamSynchronize(streams[3]);
//		ErrorDetectAndCorrect(B, ldb,
//							n, m, n, 
//							checksumB, checksumB_ld, 
//							chk1, chk1_ld, 
//							chk2, chk2_ld,
//							streams[1]);
		//handle error - to be finished
		
		if (DEBUG) {
			cout<<"recalculated checksum of B before dtrsm:"<<endl;
			printMatrix_gpu(chk1,chk1_ld, (m / n), n);
			printMatrix_gpu(chk2,chk2_ld, (m / n), n);

			cout<<"updated checksum of B before dtrsm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, (m / n) * 2, n);
		}		
	}
	magmablasSetKernelStream(streams[1]);	
	//[Cholesky]MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit, MAGMA_D_ONE

	cout << "test1" << endl;
	magma_dtrsm(side, uplo, trans, diag,
				m, n,
				alpha, A, lda,
			    B, ldb);
	cout << "test2" << endl;


	if (FT) {
		//update checksums
		//magmablasSetKernelStream(streams[1]);	
		
		magmablasSetKernelStream(streams[4]);	
		magma_dtrsm(side, uplo, trans, diag,
                    (m / n) * 2, n,
                    alpha, A, lda,
                    checksumB, checksumB_ld);
	}
}