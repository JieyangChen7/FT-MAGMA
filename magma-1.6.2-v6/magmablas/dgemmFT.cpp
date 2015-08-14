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
 * n: number of row of B (B)
 * k: number of col of A / col of B (i)
 */
void dgemmFT(int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, 
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * v, int v_ld,
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
		
//		magma_dgetmatrix_async( n, k,
//								B, ldb,
//								temp, temp_ld,
//								stream0 );							
		//verify B before use
		//reclaculate checksums of B on GPU
//		magmablasSetKernelStream(stream2);
//		magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
//				B, lda, vd, vd_ld, MAGMA_D_ZERO, chk1, chk1_ld );
//		magmablasSetKernelStream(stream3);
//		magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
//				B, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2, chk2_ld );
		//handle error - to be finished
		
		
		if (DEBUG) {
			cout<<"recalculated checksum of B before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, 1, k);
			printMatrix_gpu(chk2, chk2_ld, 1, k);
		
			cout<<"updated checksum of B before dgemm:"<<endl;
			printMatrix_host(checksumB, checksumB_ld, 2, k);
		}
		
		
		//do part of A verify on CPU
//		double r = 0.8;
//		double * temp_cpu;
//		int temp_cpu_ld;
//		int cpu_start_index = (int)((m / n) * r) * n;
//		if (cpu_start_index < m) {
//			magma_dmalloc_pinned(&temp_cpu, (m - cpu_start_index) * k * sizeof(double));
//			temp_cpu_ld = m - cpu_start_index;
//			magma_dgetmatrix_async(m - cpu_start_index, k,
//									A + cpu_start_index, lda,
//									temp_cpu, temp_cpu_ld,
//									stream0);
//		}
//		//verify A before use
//		for (int i = 0; i < m; i += n) {
//			magmablasSetKernelStream(stream2);
//			magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
//					A + i, ldb, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
//			magmablasSetKernelStream(stream3);
//			magma_dgemv(MagmaTrans, n, k, MAGMA_D_ONE,
//					A + i, ldb, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );
//		}
//		//handle error - to be finished
//		magmablasSetKernelStream(stream1);
		
//		if (cpu_start_index < m) {
//			double * chk1 = new double[((m - cpu_start_index) / n) * k];
//			double * chk2 = new double[((m - cpu_start_index) / n) * k];
//			char T = 'T';		
//			
//			int chk1_inc = (m - cpu_start_index) / n;
//			int chk2_inc = (m - cpu_start_index) / n;
//			for (int i = 0; i < m - cpu_start_index; i += n) {
//				blasf77_dgemv(  &T,
//								&n, &k,
//								&one,
//								temp_cpu + i, &temp_cpu_ld,
//								v, &v_ld,
//								&zero,
//								chk1 + (i / n), &chk1_inc );
//				blasf77_dgemv(  &T,
//								&n, &k,
//								&one,
//								temp_cpu + i, &temp_cpu_ld,
//								v + 1, &v_ld,
//								&zero,
//								chk2 + (i / n) , &chk2_inc );
//			}
//		}
		
		
		
		
		if (DEBUG) {	
			cout<<"recalculated checksum of A before dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, m / n, k);
			printMatrix_gpu(chk2, chk2_ld, m / n, k);
		
			cout<<"updated checksum of A before dgemm:"<<endl;
			printMatrix_host(checksumA, checksumA_ld, (m / n) * 2, k);
		}
		
	}
	
	
//	magma_dgemm(
//				MagmaNoTrans, MagmaTrans,
//				m, n, k,
//				MAGMA_D_ONE * (-1),
//				A, lda, B, ldb,
//				MAGMA_D_ONE,
//				C, ldc );
//	
	if(FT){	
//		magma_queue_sync( stream0 );
//		//update checksums on CPU
//		char N = 'N';
//		char T = 'T';
//		int m2 = (m / n) * 2;
//		int n2 = n;
//		int k2 = k;
//				
//		blasf77_dgemm(  &N, &T,
//						&m2, &n2, &k2,
//						&negone,
//						checksumA, &checksumA_ld,
//						temp, &temp_ld,
//						&one,
//						checksumC, &checksumC_ld );
	}
}