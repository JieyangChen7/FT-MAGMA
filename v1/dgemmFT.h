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
		double * B, int ldb, double * C, int ldc, double * checksumA1,
		int incA1, double * checksumA2, int incA2, double * checksumC1,
		int incC1, double * checksumC2, int incC2,
		double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT) {

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
	//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &negone, A, lda, B,
	//		ldb, &one, C, ldc);

	if(FT){
		
		//recalculate checksum1 and checksum2
		for (int i = 0; i < m; i += n) {
			cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C + i, ldc, v1d, 1,
					&zero, chk1 + (i / n), chk1_ld);
			cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C + i, ldb, v2d, 1,
					&zero, chk2 + (i / n), chk2_ld);
		}
		
		
		/*cout<<"recalculated checksum1 of C after dgemm:"<<endl;
		printMatrix_gpu(chk1, chk1_pitch, m/n,n);
		cout<<"recalculated checksum2 of C after dgemm:"<<endl;
		printMatrix_gpu(chk2, chk2_pitch, m/n,n);
		*/	
		
		
		//update checksum1 and checksum2
		//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m/n, n, k, &negone,
		//		checksumA1, incA1, B, ldb, &one, checksumC1, incC1);
		//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m/n, n, k, &negone,
		//		checksumA2, incA2, B, ldb, &one, checksumC2, incC2);
		
		
		/*cout<<"updated checksum1 of C after dgemm:"<<endl;
		printMatrix_gpu(checksumC1, incC1*sizeof(double), m/n,n);
		cout<<"updated checksum2 of C after dgemm:"<<endl;
		printMatrix_gpu(checksumC2, incC2*sizeof(double), m/n,n);
		*/
		//error detection and error correction
	//	detectAndCorrectForGemm<<<dim3(m/n),dim3(n)>>>(C, ldc, n,
	//			checksumC1, incC1, checksumC2, incC2,
	//			chk1, chk1_ld, chk2, chk2_ld);
				
		
	}
}