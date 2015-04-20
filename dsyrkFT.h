#include<iostream>
using namespace std;
//dsyrk with FT

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


/**
 * n: number of row of A
 * m: number of col of A
 */
void dsyrkFT(cublasHandle_t handle, int n, int m, double * A, int lda, double * C, int ldc,
		double * checksumA1, int incA1, double * checksumA2, int incA2,
		double * checksumC1, int incC1, double * checksumC2, int incC2){
	
	cout<<"checksum1 of A before dsyrk:"<<endl;
	printMatrix_gpu(checksumA1, incA1*sizeof(double), 1,m);
	cout<<"checksum2 of A before dsyrk:"<<endl;
	printMatrix_gpu(checksumA2, incA2*sizeof(double), 1,m);
	
	cout<<"checksum1 of C before dsyrk:"<<endl;
	printMatrix_gpu(checksumC1, incC1*sizeof(double), 1,n);
	cout<<"checksum2 of C before dsyrk:"<<endl;
	printMatrix_gpu(checksumC2, incC2*sizeof(double), 1,n);
	
	
	double negone = -1;
	double one = 1;
	double zero = 0;
	cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m, &negone, A, lda, &one, C, ldc);
	
	//recalculate checksum1 and checksum2
	double * chk1;
	double * chk2;
	size_t chk1_pitch;
	size_t chk2_pitch;
			
	cudaMallocPitch((void**) &chk1, &chk1_pitch, 1 * sizeof(double), n);
	cudaMallocPitch((void**) &chk2, &chk2_pitch, 1 * sizeof(double), n);

	int chk1_ld = chk1_pitch / sizeof(double);
	int chk2_ld = chk2_pitch / sizeof(double);
		
	double * v1 = new double[n];
	double * v2 = new double[n];
	for (int i = 0; i < n; i++) {
			v1[i] = 1;
			v2[i] = i+1;
	}
		
	double * v1d;
	size_t v1d_pitch;
	cudaMallocPitch((void**) &v1d, &v1d_pitch, n * sizeof(double), 1);
	cudaMemcpy2D(v1d, v1d_pitch, v1, n * sizeof(double), n * sizeof(double),
					1, cudaMemcpyHostToDevice);
	double * v2d;
	size_t v2d_pitch;
	cudaMallocPitch((void**) &v2d, &v2d_pitch, n * sizeof(double), 1);
	cudaMemcpy2D(v2d, v2d_pitch, v2, n * sizeof(double), n * sizeof(double),
							1, cudaMemcpyHostToDevice);
	
	cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C, ldc, v1d, 1,
							&zero, chk1, chk1_ld);
	cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C, ldc, v2d, 1,
							&zero, chk2, chk2_ld);
	
	cout<<"recalculated checksum1 of C after dsyrk:"<<endl;
	printMatrix_gpu(chk1, chk1_pitch, 1, n);
	cout<<"recalculated checksum2 of C after dsyrk:"<<endl;
	printMatrix_gpu(chk2, chk2_pitch, 1, n);
	
	//update checksum1 and checksum2
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, m, n, &negone, checksumA1, incA1, A, lda, &one, checksumC1, incC1);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1, m, n, &negone, checksumA2, incA2, A, lda, &one, checksumC2, incC2);
	
	cout<<"updated checksum1 of C after dsyrk:"<<endl;
	printMatrix_gpu(checksumC1, incC1*sizeof(double), 1,n);
	cout<<"updated checksum2 of C after dsyrk:"<<endl;
	printMatrix_gpu(checksumC2, incC2*sizeof(double), 1,n);
	
	//detect error and correct error
	/*detectAndCorrectForSyrk<<<dim3(1),dim3(n)>>>(C, ldc,
			checksumC1, incC1, checksumC2, incC2,
			 chk1, chk1_ld, chk2, chk2_ld);
	*/
}