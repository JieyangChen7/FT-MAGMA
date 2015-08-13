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
 * m: number of row of A (N-i-B)
 * n: number of row of B (N-i)
 * k: number of col of A / col of B (B)
 */
void dgemmFT(int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, 
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		double * temp, int temp_ld,
		magma_queue_t stream0, magma_queue_t stream1, magma_queue_t stream2, magma_queue_t stream3,
		bool FT, bool DEBUG) {

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
	
	if (FT) {
		
//		magma_dsetmatrix_async( (m / k) * 2, n,
//								checksumA, checksumA_ld,
//								temp, temp_ld,
//								stream0 );							
		//verify B before use
		for (int i = 0; i < n; i += k) {
			magmablasSetKernelStream(stream2);
			magma_dgemv(MagmaTrans, k, k, MAGMA_D_ONE,
					B + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / k), chk1_ld );
			magmablasSetKernelStream(stream3);
			magma_dgemv(MagmaTrans, k, k, MAGMA_D_ONE,
					B + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / k), chk2_ld );
		}
		//handle error - to be finished
		magmablasSetKernelStream(stream1);
//		
//		
//		
//		if (DEBUG) {	
//			cout<<"recalculated checksum of B before dgemm:"<<endl;
//			printMatrix_gpu(chk1, chk1_ld, n / k, k);
//			printMatrix_gpu(chk2, chk2_ld, n / k, k);
//		
//			cout<<"updated checksum of B before dgemm:"<<endl;
//			printMatrix_host(checksumB, checksumB_ld, (n / k) * 2, k);
//		}
//		
	}
	
	
//	for (int i = 0; i < m; i += k) {
//		magma_dgemm(
//					MagmaNoTrans, MagmaTrans,
//					k, i+2*k, k,
//					MAGMA_D_ONE * (-1),
//					A + i, lda, B, ldb,
//					MAGMA_D_ONE,
//					C + i, ldc );
//	}


	if(FT){	
	//	magma_queue_sync( stream0 );
		//update checksums on CPU
		char N = 'N';
		char T = 'T';
		int m2 = 2;
		
		int k2 = k;
//		for (int i = 0; i < m; i += k) {
//			int n2 = i+2*k;
////			blasf77_dgemm(  &N, &T,
////							&m2, &n2, &k2,
////							&negone,
////							temp + (i/k)*2, &temp_ld,
////							B, &ldb,
////							&one,
////							checksumC + (i/k)*2, &checksumC_ld );
//			magma_dgemm(
//						MagmaNoTrans, MagmaTrans,
//						m2, n2, k2,
//						MAGMA_D_ONE * (-1),
//						temp + (i/k)*2, temp_ld,
//						B, ldb,
//						MAGMA_D_ONE,
//						temp + (i/k)*2 + k * temp_ld, temp_ld );
//		}
//		
//		magma_dgetmatrix_async( (m / k) * 2, n,
//								temp, temp_ld,
//								checksumA, checksumA_ld,
//								stream0 );		
		
	}
}