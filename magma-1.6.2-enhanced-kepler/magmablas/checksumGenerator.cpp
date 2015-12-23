#include "magma.h"
#include"FT.h"
#include<iostream>
using namespace std;
//initialize checksum
void initializeChecksum(double * matrix, int ld,
		int N, int B,
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
	
	
	
	double * chk1d;
	double * chk2d;
	size_t chk1d_pitch;
	size_t chk2d_pitch;
	int chk1d_ld;
	int chk2d_ld;
	
	//allocate space for reclaculated checksum on GPU (vertical)
	chk1d_pitch = magma_roundup((N / B) * sizeof(double), 32);
	chk1d_ld = chk1d_pitch / sizeof(double);
	magma_dmalloc(&chk1d, chk1d_pitch * N);
	
	chk2d_pitch = magma_roundup((N / B) * sizeof(double), 32);
	chk2d_ld = chk2d_pitch / sizeof(double);
	magma_dmalloc(&chk2d, chk2d_pitch * N);
	
	
	
	
	for (int i = 0; i < N; i += B) {		
			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
						//2, i + B, B,
						1, N, B,
						MAGMA_D_ONE, vd, vd_ld,
						matrix + i, ld,
						MAGMA_D_ZERO, chk1d + (i / B), chk1d_ld);			
		}
	
	
	for (int i = 0; i < N; i += B) {		
			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
						//2, i + B, B,
						1, N, B,
						MAGMA_D_ONE, vd + 1, vd_ld,
						matrix + i, ld,
						MAGMA_D_ZERO, chk2d + (i / B), chk2d_ld);			
		}
	
	
	cout << "Matrix:" << endl;
	printMatrix_gpu(matrix, ld, N, N);
	
	cout << "checksum:" << endl;
	printMatrix_gpu(chksum, chksum_ld, (N / B) * 2, N);	
	
	cout << "checksum 1:" << endl;
	printMatrix_gpu(chk1d, chk1d_ld, N / B, N);	
	
	cout << "checksum 2:" << endl;
	printMatrix_gpu(chk2d, chk2d_ld, N / B, N);	
	
//	test_abft(B, ldb, n, m, n, checksumB, checksumB_ld, chk1, chk1_ld, chk2, chk2_ld);

}