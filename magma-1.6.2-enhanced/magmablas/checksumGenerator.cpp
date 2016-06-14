#include "magma.h"
#include"FT.h"
#include<iostream>
using namespace std;
//initialize checksum
//M: number of rows
//N: numner of cols
void initializeChecksum(double * matrix, int ld,
		int M, int N, int B,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chksum, int chksum_ld, magma_queue_t * streams) {


	
	for (int i = 0; i < N; i += B) {		
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, N, B,
					//2, B + i, B,
					MAGMA_D_ONE, vd, vd_ld,
					matrix + i, ld,
					MAGMA_D_ZERO, chksum + (i / B) * 2, chksum_ld);			
	}
	
	
//	
//	double * chk1d;
//	double * chk2d;
//	size_t chk1d_pitch;
//	size_t chk2d_pitch;
//	int chk1d_ld;
//	int chk2d_ld;
//	
//	//allocate space for reclaculated checksum on GPU (vertical)
//	chk1d_pitch = magma_roundup((N / B) * sizeof(double), 32);
//	chk1d_ld = chk1d_pitch / sizeof(double);
//	magma_dmalloc(&chk1d, chk1d_pitch * N);
//	
//	chk2d_pitch = magma_roundup((N / B) * sizeof(double), 32);
//	chk2d_ld = chk2d_pitch / sizeof(double);
//	magma_dmalloc(&chk2d, chk2d_pitch * N);
//	
//	
//	
//	
//	for (int i = 0; i < N; i += B) {		
//			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
//						//2, i + B, B,
//						1, N, B,
//						MAGMA_D_ONE, vd, vd_ld,
//						matrix + i, ld,
//						MAGMA_D_ZERO, chk1d + (i / B), chk1d_ld);			
//		}
//	
//	
//	for (int i = 0; i < N; i += B) {		
//			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
//						//2, i + B, B,
//						1, N, B,
//						MAGMA_D_ONE, vd + 1, vd_ld,
//						matrix + i, ld,
//						MAGMA_D_ZERO, chk2d + (i / B), chk2d_ld);			
//		}
//	
//	
//	cout << "Matrix:" << endl;
//	printMatrix_gpu(matrix, ld, N, N);
//	
//	cout << "checksum:" << endl;
//	printMatrix_gpu(chksum, chksum_ld, (N / B) * 2, N);	
//	
//	cout << "checksum 1:" << endl;
//	printMatrix_gpu(chk1d, chk1d_ld, N / B, N);	
//	
//	cout << "checksum 2:" << endl;
//	printMatrix_gpu(chk2d, chk2d_ld, N / B, N);	
	
	//test_abft(matrix, ld, B, N, N, chksum, chksum_ld, chk1d, chk1d_ld, chk2d, chk2d_ld);

}


//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
void recalculateChecksum(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, chk_nb, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
	}
	
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);


}
