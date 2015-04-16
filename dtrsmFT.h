//TRSM with FT on GPU using cuBLAS

__global__ detectAndCorrect(double * B, int ldb, int n,
		double * chksumB1, int incB1, double * chksumB2, int incB2,
		double * chkB1, int incB1_2, double * chkB2, int incB2_2){
	//determin the reponsisble column 
	int block = blockIdx.x;
	int col = threadIdx.x;
	double diff = abs(*(chkB1+block+col*incB1_2)-*(chksumB1+block+col*incB1);
	if(diff>0.1){
		double diff2=abs(*(chkB2+block+col*incB2_2)-*(chksumB2+block+col*incB2);
		int row = (int)round(diff2/diff);
		*(B+n*block+row+col*ldb) += *(chksumB1+block+col*incB1)-*(chkB1+block+col*incB1_2);
	}
}

/*
 * m: number of row of B
 * n: number of col of B
 */

void dtrsmFT(cublasHandle_t handle, int m, int n, double * A, int lda, double * B, int ldb, 
	double * chksumA1, int incA1, double * chksumA2, int incA2,
	double * chksumB1, int incB1, double * chksumB2, int incB2) {
	
	double alpha = 1;
	cublasDtrsm(handle, 
			CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
			CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT, 
			m, n, &alpha, 
			A, lda,
			B, ldb);
	
	//recalculate checksum1 and checksum2
	double * chk1;
	double * chk2;
	size_t chk1_pitch;
	size_t chk2_pitch;
		
	cudaMallocPitch((void**) &chk1, &chk1_pitch, (m/n) * sizeof(double), n);
	cudaMallocPitch((void**) &chk2, &chk2_pitch, (m/n) * sizeof(double), n);

	int chk1_ld = chk1_pitch / sizeof(double);
	int chk2_ld = chk2_pitch / sizeof(double);
	
	double * v1 = new double[B];
	double * v2 = new double[B];
	for (int i = 0; i < B; i++) {
			v1[i] = 1;
			v2[i] = i+1;
		}
	
	double * v1d;
	size_t v1d_pitch;
	cudaMallocPitch((void**) &v1d, &v1d_pitch, B * sizeof(double), 1);
	cudaMemcpy2D(v1d, v1d_pitch, v1, B * sizeof(double), B * sizeof(double),
				1, cudaMemcpyHostToDevice);
	double * v2d;
	size_t v2d_pitch;
	cudaMallocPitch((void**) &v2d, &v2d_pitch, B * sizeof(double), 1);
	cudaMemcpy2D(v2d, v2d_pitch, v2, B * sizeof(double), B * sizeof(double),
						1, cudaMemcpyHostToDevice);
	double beta = 0;
	for(int i=0;i<m;i+=n){
		cublasDgemv(handle, CUBLAS_OP_T, n, n, &alpha, B+i, ldb, v1d, 1,
						&beta, chk1 + (i/n), chk1_ld);
		cublasDgemv(handle, CUBLAS_OP_T, n, n, &alpha, B+i, ldb, v2d, 1,
						&beta, chk2 + (i/n), chk2_ld);
	}
	//update checksum1 and checksum2
	cublasDtrsm(handle, 
				CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
				CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT, 
				m/n, n, &alpha, 
				A, lda,
				checksumB1, incB1);
	cublasDtrsm(handle, 
				CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, 
				CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT, 
				m/n, n, &alpha, 
				A, lda,
				checksumB2, incB2);
	
	detectAndCorrect<<<dim3(m/n),dim3(n)>>>(B, ldb, n,
			checksumB1, incB1, checksumB2, incB2,
			chk1, chk1_ld, chk2, chk2_ld);
}