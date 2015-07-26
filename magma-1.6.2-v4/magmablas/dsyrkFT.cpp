#include"FT.h"
#include<iostream>
using namespace std;
//dsyrk with FT
/*
__global__ void detectAndCorrectForSyrk(double * C, int ldc,
		double * chksumC1, int incC1, double * chksumC2, int incC2,
		double * chkC1, int incC1_2, double * chkC2, int incC2_2){
	//determin the reponsisble column 
	int col = threadIdx.x;
	double diff = abs(*(chkC1+col*incC1_2)-*(chksumC1+col*incC1));
	if(diff>0.1){
		double diff2=abs(*(chkC2+col*incC2_2)-*(chksumC2+col*incC2));
		int row = (int)round(diff2/diff)-1;
		*(C+row+col*ldc) += *(chksumC1+col*incC1)-*(chkC1+col*incC1_2);
	}
}
*/

/**
 * n: number of row of A
 * m: number of col of A
 */
void dsyrkFT(int n, int m, double * A, int lda, double * C, int ldc,
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld,
		double * chk2, int chk2_ld,
		double * temp, int temp_ld, 
		bool FT, bool DEBUG){
	
//	cout<<"checksum1 of A before dsyrk:"<<endl;
//	printMatrix_gpu(checksumA1, incA1, 1,m);
//	cout<<"checksum2 of A before dsyrk:"<<endl;
//	printMatrix_gpu(checksumA2, incA2, 1,m);
//	cout<<"checksum1 of C before dsyrk:"<<endl;
//	printMatrix_gpu(checksum1 + (j / jb) + j * checksum1_ld, checksum1_ld, 1,jb);
//	cout<<"checksum2 of C before dsyrk:"<<endl;
//	printMatrix_gpu(checksum2 + (j / jb) + j * checksum2_ld, checksum2_ld, 1,jb);
	
	
	
	double negone = -1;
	double one = 1;
	double zero = 0;
	//cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m, &negone, A, lda, &one, C, ldc);
	
	magma_dgemm(
			MagmaNoTrans, MagmaTrans,
			n, n, m,
			MAGMA_D_ONE * (-1),
			A, lda, A, lda,
			MAGMA_D_ONE,
			C, ldc );
	
//	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &negone, A, lda, A, lda, &one, C, ldc);
	
	if(FT){
		
		//recalculate checksum1 and checksum2
//		magma_dgemm(
//					MagmaTrans, MagmaNoTrans,
//					2, n, n,
//					MAGMA_D_ONE,
//					vd, vd_ld, C, ldc,
//					MAGMA_D_ZERO,
//					chk, chk_ld );
		magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
				C, ldc, vd, 1, MAGMA_D_ZERO, chk1, chk1_ld );
		magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
				C, ldc, vd + vd_ld, 1, MAGMA_D_ZERO, chk2, chk2_ld );
		
		//update checksum1 and checksum2
		char N = 'N';
		char T = 'T';
		int m2 = 2;
		int n2 = n;
		int k2 = m;
		
		
		blasf77_dgemm(  &N, &T,
		                &m2, &n2, &k2,
		                &negone,
		                checksumA, &checksumA_ld,
		                temp, &temp_ld,
		                &MAGMA_D_ONE,
		                checksumC, &checksumC_ld );
		
//		magma_dgemm(
//					MagmaNoTrans, MagmaTrans,
//					2, n, m,
//					MAGMA_D_ONE * (-1),
//					checksumA, checksumA_ld, A, lda,
//					MAGMA_D_ONE,
//					checksumC, checksumC_ld );
		
//		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, n, m, &negone, checksumA1, incA1, A, lda, &one, checksumC1, incC1);
//		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, n, m, &negone, checksumA2, incA2, A, lda, &one, checksumC2, incC2);
		
		if (DEBUG) {
			cout<<"recalculated checksum of C after dsyrk:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, 1, n);
			printMatrix_gpu(chk2, chk2_ld, 1, n);
		
			cout<<"updated checksum of C after dsyrk:"<<endl;
			printMatrix_host(checksumC, 2, n);
		}
		
		//detect error and correct error
	//	detectAndCorrectForSyrk<<<dim3(1),dim3(n)>>>(C, ldc,
	//			checksumC1, incC1, checksumC2, incC2,
	//			 chk1, chk1_ld, chk2, chk2_ld);
		
	}
}