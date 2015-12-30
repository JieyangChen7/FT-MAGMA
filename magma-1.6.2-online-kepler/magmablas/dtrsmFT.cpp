#include"FT.h"
#include<iostream>
using namespace std;
//TRSM with FT on GPU using cuBLAS

/*
 * m: number of row of B
 * n: number of col of B
 */

void dtrsmFT(int m, int n, double * A, int lda,
		double * B, int ldb, double * checksumB, int checksumB_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		bool FT, bool DEBUG, magma_queue_t * streams) {


	double negone = -1;
	double one = 1;
	double zero = 0;
	magmablasSetKernelStream(streams[1]);
	magma_dtrsm(MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
	                                m, n,
	                                MAGMA_D_ONE, A, lda,
	                                       B, ldb);
	if (FT) {
		//update checksums 
		magmablasSetKernelStream(streams[4]);
		magma_dtrsm(MagmaRight, MagmaLower, MagmaTrans, MagmaNonUnit,
					(m / n) * 2, n,
					MAGMA_D_ONE, A, lda,
					checksumB, checksumB_ld);
		
		
		//recalculate checksums
		//magma_queue_sync( streams[1] );
		magmablasSetKernelStream(streams[1]);
		double beta = 0;
		for (int i = 0; i < m; i += n) {
			//magmablasSetKernelStream(streams[2]);
			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
				B + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
			//magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
				B + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );	
		}

		if (DEBUG) {
			cout<<"recalculated checksum of B after dtrsm:"<<endl;
			printMatrix_gpu(chk1,chk1_ld, (m / n), n);
			printMatrix_gpu(chk2,chk2_ld, (m / n), n);

			cout<<"updated checksum of B after dtrsm:"<<endl;
			printMatrix_gpu(checksumB, checksumB_ld, (m / n) * 2, n);
		}
		
		magma_queue_sync( streams[2] );
		magma_queue_sync( streams[3] );
		magma_queue_sync( streams[4] );
		cout << "trsm" <<endl;
		ErrorDetectAndCorrect(B, ldb, n, m, n, 
				checksumB, checksumB_ld,
				chk1,chk1_ld,
				chk2,chk2_ld,
				streams[1]);
	}
}