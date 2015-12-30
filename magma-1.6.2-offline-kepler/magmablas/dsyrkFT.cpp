#include"FT.h"
#include<iostream>
using namespace std;
//dsyrk with FT

/**
 * n: number of row of A
 * m: number of col of A
 */
void dsyrkFT(int n, int m, double * A, int lda, double * C, int ldc,
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chk1, int chk1_ld,
		double * chk2, int chk2_ld,
		magma_queue_t * streams,
		bool FT, bool DEBUG){
	
	double negone = -1;
	double one = 1;
	double zero = 0;
	
//	if (FT) {
	magmablasSetKernelStream(streams[1]);
		magma_dgemm(
				MagmaNoTrans, MagmaTrans,
				n, n, m,
				MAGMA_D_ONE * (-1),
				A, lda, A, lda,
				MAGMA_D_ONE,
				C, ldc );
//	} else {
//		magma_dsyrk(MagmaLower, MagmaNoTrans, n, m,
//						MAGMA_D_ONE * (-1), A, lda,
//						MAGMA_D_ONE,     C, ldc);
//	}

	
	if(FT){
		magma_queue_sync( stream[1] );
		//update checksums
		magmablasSetKernelStream(streams[4]);
		magma_dgemm(
					MagmaNoTrans, MagmaTrans,
					2, n, m,
					MAGMA_D_ONE * (-1),
					checksumA, checksumA_ld, A, lda,
					MAGMA_D_ONE,
					checksumC, checksumC_ld );
 		
		if (DEBUG) {
			
			
			cout<<"recalculated checksum of C after dsyrk:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, 1, n);
			printMatrix_gpu(chk2, chk2_ld, 1, n);
		
			cout<<"updated checksum of C after dsyrk:"<<endl;
			printMatrix_host(checksumC, checksumC_ld, 2, n);
		}

		
	}
}