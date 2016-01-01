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
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld,
		double * chk2, int chk2_ld,
		double * tempB, int tempB_ld, cudaStream_t stream0,
		double * checksumA_dev, int checksumA_dev_ld,
		double * checksumC_dev, int checksumC_dev_ld,
		bool FT, bool DEBUG){
	
	/*cout<<"checksum1 of A before dsyrk:"<<endl;
	printMatrix_gpu(checksumA1, incA1*sizeof(double), 1,m);
	cout<<"checksum2 of A before dsyrk:"<<endl;
	printMatrix_gpu(checksumA2, incA2*sizeof(double), 1,m);
	
	cout<<"checksum1 of C before dsyrk:"<<endl;
	printMatrix_gpu(checksumC1, incC1*sizeof(double), 1,n);
	cout<<"checksum2 of C before dsyrk:"<<endl;
	printMatrix_gpu(checksumC2, incC2*sizeof(double), 1,n);
	*/
	
	double negone = -1;
	double one = 1;
	double zero = 0;
	
	
	
	if (FT) {
			
		    // transfer data needed for updating checksum
			cudaMemcpy2DAsync(checksumA_dev, checksumA_dev_ld * sizeof(double), 
								checksumA, checksumA_ld * sizeof(double), 
								2 * sizeof(double), m,
								cudaMemcpyHostToDevice, stream0);
			cudaMemcpy2DAsync(checksumC_dev, checksumC_dev_ld * sizeof(double), 
										checksumC, checksumC_ld * sizeof(double), 
										2 * sizeof(double), n,
										cudaMemcpyHostToDevice, stream0);
			
	}
	
	
	//cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, m, &negone, A, lda, &one, C, ldc);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, n, m, &negone, A, lda, A, lda, &one, C, ldc);
	//dgemm('N', 'T', n, n, m, negone, tempB, tempB_ld, tempB, tempB_ld, one, tempB, tempB_ld);
	if (FT) {
		
		// wait for data need for checksum update
		cudaStreamSynchronize(stream0);
		
		// checksum update on GPU
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, n, m, &negone, checksumA_dev, checksumA_dev_ld, A, lda, &one, checksumC_dev, checksumC_dev_ld);
		
		// transfer updated result back to CPU memory
		cudaMemcpy2DAsync(checksumC, checksumC_ld * sizeof(double), 
											checksumC_dev, checksumC_dev_ld * sizeof(double), 
											2 * sizeof(double), n,
											cudaMemcpyDeviceToHost, stream0);
		
		//recalculate checksum1 and checksum2
		
		//cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2, n, n, &one, vd, vd_ld, C, ldc, &zero, chk, chk_ld);
	    
		// checksum recalcuate on GPU
	    cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C, ldc, vd, 1, &zero, chk1, chk1_ld);
	    cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C, ldc, vd + vd_ld, 1, &zero, chk2, chk2_ld);
		
		
		
		//dgemv('T', n, n, one, tempB, tempB_ld, tempB, 1, zero, tempB, tempB_ld);
		//dgemv('T', n, n, one, tempB, tempB_ld, tempB, 1, zero, tempB, tempB_ld);
		
		//dgemm('N', 'T', 2, n, m, negone, checksumA, checksumA_ld, tempB, tempB_ld, one, checksumC, checksumC_ld);
		
		//update checksum1 and checksum2
		//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, n, m, &negone, checksumA, checksumA_ld, A, lda, &one, checksumC, checksumC_ld);
		
		if (DEBUG) {
			cout<<"recalculated checksum of C after dsyrk:"<<endl;
			printMatrix_gpu(chk1, chk1_ld * sizeof(double), 1, n);
			printMatrix_gpu(chk2, chk2_ld * sizeof(double), 1, n);
			
			cudaStreamSynchronize(stream0);
			cout<<"updated checksum of C after dsyrk:"<<endl;
			//printMatrix_gpu(checksumC, checksumC_ld * sizeof(double), 2, n);
			printMatrix_host(checksumC, checksumC_ld, 2, n);
		}
		
		
		//detect error and correct error
		//detectAndCorrectForSyrk<<<dim3(1),dim3(n)>>>(C, ldc,
		//			checksumC1, incC1, checksumC2, incC2,
		//			 chk1, chk1_ld, chk2, chk2_ld);
		
	}
}