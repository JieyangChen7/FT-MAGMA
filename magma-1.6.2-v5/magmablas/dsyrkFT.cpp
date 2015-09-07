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
		double * v, int v_ld,
		double * chk1, int chk1_ld,
		double * chk2, int chk2_ld,
		double * chkd_updateA, int chkd_updateA_ld, 
		double * chkd_updateC, int chkd_updateC_ld, 
		magma_queue_t stream0,magma_queue_t stream1,magma_queue_t stream2,magma_queue_t stream3,
		bool FT, bool DEBUG){
	
	

	if (FT) {
		magma_dsetmatrix_async( 2, m,
								checksumA, checksumA_ld,
								chkd_updateA, chkd_updateA_ld, stream0);
		magma_dsetmatrix_async( 2, n,
								checksumC, checksumC_ld, 
								chkd_updateC, chkd_updateC_ld, stream0);
	}
	
	double negone = -1;
	double one = 1;
	double zero = 0;

	
	
//	if (FT) {
//		magma_dgemm(
//				MagmaNoTrans, MagmaTrans,
//				n, n, m,
//				MAGMA_D_ONE * (-1),
//				A, lda, A, lda,
//				MAGMA_D_ONE,
//				C, ldc );
//	} else {
//		magma_dsyrk(MagmaLower, MagmaNoTrans, n, m,
//						MAGMA_D_ONE * (-1), A, lda,
//						MAGMA_D_ONE,     C, ldc);
//	}

	
	if(FT){
//		//update checksums on GPU
//		magmablasSetKernelStream(stream0);
//		magma_dgemm(
//					MagmaNoTrans, MagmaTrans,
//					2, n, m,
//					MAGMA_D_ONE * (-1),
//					chkd_updateA, chkd_updateA_ld, A, lda,
//					MAGMA_D_ONE,
//					chkd_updateC, chkd_updateC_ld );
//		
//		//transfer updated checksum back to CPU
//		magma_dgetmatrix_async( 2, n,
//								chkd_updateC, chkd_updateC_ld,
//								checksumC, checksumC_ld, stream0);
//		
//		//recalculate checksum1 and checksum2
////		magma_queue_sync( stream1 );
//		magmablasSetKernelStream(stream2);
//		magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
//				C, ldc, vd, vd_ld, MAGMA_D_ZERO, chk1, chk1_ld );
//		magmablasSetKernelStream(stream3);
//		magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
//				C, ldc, vd + 1, vd_ld, MAGMA_D_ZERO, chk2, chk2_ld );
//		magmablasSetKernelStream(stream1);
//		
//		
////		//update checksum1 and checksum2
////		char N = 'N';
////		char T = 'T';
////		int m2 = 2;
////		int n2 = n;
////		int k2 = m;
////		
////		
////		blasf77_dgemm(  &N, &T,
////		                &m2, &n2, &k2,
////		                &negone,
////		                checksumA, &checksumA_ld,
////		                temp, &temp_ld,
////		                &one,
////		                checksumC, &checksumC_ld );
////		 
		
		if (DEBUG) {
			
			
			cout<<"recalculated checksum of C after dsyrk:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, 1, n);
			printMatrix_gpu(chk2, chk2_ld, 1, n);
		
			magma_queue_sync( stream0 );
			cout<<"updated checksum of C after dsyrk:"<<endl;
			printMatrix_host(checksumC, checksumC_ld, 2, n);
		}
		
		//detect error and correct error
	//	detectAndCorrectForSyrk<<<dim3(1),dim3(n)>>>(C, ldc,
	//			checksumC1, incC1, checksumC2, incC2,
	//			 chk1, chk1_ld, chk2, chk2_ld);
		
	}
}