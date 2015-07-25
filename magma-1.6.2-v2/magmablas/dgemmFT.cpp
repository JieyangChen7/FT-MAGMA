#include"FT.h"
#include<iostream>
using namespace std;
//dgemm with FT
/*
__global__ void detectAndCorrectForGemm(double * C, int ldc, int n,
		double * chksumC1, int incC1, double * chksumC2, int incC2,
		double * chkC1, int incC1_2, double * chkC2, int incC2_2){
	//determin the reponsisble column 
	int block = blockIdx.x;
	int col = threadIdx.x;
	double diff = abs(*(chkC1+block+col*incC1_2)-*(chksumC1+block+col*incC1));
	if(diff>0.1){
		double diff2=abs(*(chkC2+block+col*incC2_2)-*(chksumC2+block+col*incC2));
		int row = (int)round(diff2/diff)-1;
		*(C+n*block+row+col*ldc) += *(chksumC1+block+col*incC1)-*(chkC1+block+col*incC1_2);
	}
}
*/
/**
 * m: number of row of A
 * n: number of col of B
 * k: number of col of A / row of B
 */
void dgemmFT(int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, 
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk, int chk_ld, bool FT, bool DEBUG) {

	/*cout<<"checksum1 of A before dgemm:"<<endl;
	printMatrix_gpu(checksumA1, incA1*sizeof(double), m/n,k);
	cout<<"checksum2 of A before dgemm:"<<endl;
	printMatrix_gpu(checksumA2, incA2*sizeof(double), m/n,k);
	
	cout<<"checksum1 of C before dgemm:"<<endl;
	printMatrix_gpu(checksumC1, incC1*sizeof(double), m/n,n);
	cout<<"checksum2 of C before dgemm:"<<endl;
	printMatrix_gpu(checksumC2, incC2*sizeof(double), m/n,n);
	*/
	double negone = -1;
	double one = 1;
	double zero = 0;
	
//	magma_dgemm(
//				MagmaNoTrans, MagmaTrans,
//				m, n, k,
//				MAGMA_D_ONE * (-1),
//				A, lda, B, ldb,
//				MAGMA_D_ONE,
//				C, ldc );
	
//	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &negone, A, lda, B,
//			ldb, &one, C, ldc);

	if(FT){
		
		//recalculate checksum1 and checksum2
		for (int i = 0; i < m; i += n) {
			magma_dgemm(
						MagmaTrans, MagmaNoTrans,
						2, n, n,
						MAGMA_D_ONE,
						vd, vd_ld, C + i, ldc,
						MAGMA_D_ZERO,
						chk + (i / n) * 2, chk_ld );
		}
		
		//update checksum1 and checksum2
//		magma_dgemm(
//					MagmaNoTrans, MagmaTrans,
//					(m / n) * 2, n, k,
//					MAGMA_D_ONE * (-1),
//					checksumA, checksumA_ld, B, ldb,
//					MAGMA_D_ONE,
//					checksumC, checksumC_ld );
		
		if (DEBUG) {
			cout<<"recalculated checksum of C after dgemm:"<<endl;
			printMatrix_gpu(chk, chk_ld, (m / n) * 2, n);
		
			cout<<"updated checksum of C after dgemm:"<<endl;
			printMatrix_gpu(checksumC, checksumC_ld, (m / n) * 2, n);
		}
		
		//error detection and error correction
	//	detectAndCorrectForGemm<<<dim3(m/n),dim3(n)>>>(C, ldc, n,
	//			checksumC1, incC1, checksumC2, incC2,
	//			chk1, chk1_ld, chk2, chk2_ld);
				
		
	}
}