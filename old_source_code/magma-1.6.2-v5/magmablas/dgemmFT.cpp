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
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		double * temp, int temp_ld,
		double * chkd_updateA, int chkd_updateA_ld,
		double * chkd_updateC, int chkd_updateC_ld,
		magma_queue_t * streams,
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
	

	magmablasSetKernelStream(streams[1]);
	magma_dgemm(
				MagmaNoTrans, MagmaTrans,
				m, n, k,
				MAGMA_D_ONE * (-1),
				A, lda, B, ldb,
				MAGMA_D_ONE,
				C, ldc );
	if(FT){			
		//recalculate checksum
//		magma_queue_sync( stream1 );
		for (int i = 0; i < m; i += n) {
			magmablasSetKernelStream(streams[2]);
			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
					C + i, ldc, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
					C + i, ldc, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );
		}
		magma_queue_sync( streams[4] );
		//update checksum
		char N = 'N';
		char T = 'T';
		int m2 = (m / n) * 2;
		//int m2 = c_part * 2;
		int n2 = n;
		int k2 = k;		
		blasf77_dgemm(  &N, &T,
						&m2, &n2, &k2,
						&negone,
						checksumA, &checksumA_ld,
						temp, &temp_ld,
						&one,
						checksumC, &checksumC_ld );		
//		if (DEBUG) {
//			cout<<"recalculated checksum of C after dgemm:"<<endl;
//			printMatrix_gpu(chk1, chk1_ld, (m / n), n);
//			printMatrix_gpu(chk2, chk2_ld, (m / n), n);
//		
//			cout<<"updated checksum of C after dgemm:"<<endl;
//			printMatrix_host(checksumC, checksumC_ld, (m / n) * 2, n);
//		}
		
		//error detection and error correction
	//	detectAndCorrectForGemm<<<dim3(m/n),dim3(n)>>>(C, ldc, n,
	//			checksumC1, incC1, checksumC2, incC2,
	//			chk1, chk1_ld, chk2, chk2_ld);
				
		
	}
}