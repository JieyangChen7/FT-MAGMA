#include<iostream>
using namespace std;
//dgemm with FT

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

/**
 * m: number of row of A
 * n: number of col of B
 * k: number of col of A / row of B
 */
void dgemmFT(cublasHandle_t handle, int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, 
		double * checksumA, int checksumA_ld, 
		double * checksumC,int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		double * tempB, int tempB_ld, cudaStream_t stream0,
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
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &negone, A, lda, B, \
			ldb, &one, C, ldc);

	
	//cout<<"after dgemm"<<endl;
	//printMatrix_gpu(C, ldb * sizeof(double), m, n);
	if (FT) {
		
		
		//checksum recalculate on GPU
		for (int i = 0; i < m; i += n) {
			//cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2, n, n, &one, vd, vd_ld, C + i, ldc, \
					&zero, chk +(i/n)*2, chk_ld);
			cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C + i, ldc, vd, 1, &zero, chk1 + (i / n), chk1_ld);
			cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C + i, ldc, vd + vd_ld, 1, &zero, chk2 + (i / n), chk2_ld);
		}
		
		//wait for data transfer (tempB/tempA)
		cudaStreamSynchronize(stream0);
		
		
		//cout<<"B on GPU"<<endl;
		//printMatrix_gpu(B, ldb * sizeof(double), n, k);
		//cout<<"B on CPU"<<endl;
		//printMatrix_host(tempB, tempB_ld, n, k);
		
		//checksum update on CPU
		dgemm('N', 'T', (m / n) * 2, n, k, negone, checksumA, checksumA_ld, tempB, tempB_ld, one, checksumC, checksumC_ld);
		
		
		//update checksum1 and checksum2
		//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, (m/n)*2, n, k, &negone, \
				checksumA, checksumA_ld, B, ldb, &one, checksumC, checksumC_ld);
		
		if (DEBUG) {
			cout<<"recalculated checksum of C after dgemm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld* sizeof(double), (m/n),n);
			printMatrix_gpu(chk2, chk2_ld* sizeof(double), (m/n),n);
			
			cout<<"updated checksum of C after dgemm:"<<endl;
			//printMatrix_gpu(checksumC, checksumC_ld*sizeof(double), (m/n)*2,n);
			printMatrix_host(checksumC, checksumC_ld, (m / n) * 2, n);
		}
		
		
		//error detection and error correction
	//	detectAndCorrectForGemm<<<dim3(m/n),dim3(n)>>>(C, ldc, n,
	//			checksumC1, incC1, checksumC2, incC2,
	//			chk1, chk1_ld, chk2, chk2_ld);
		
	}
}