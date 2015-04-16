//dgemm with FT

__global__ detectAndCorrectForGemm(double * C, int ldc, int n,
		double * chksumC1, int incC1, double * chksumC2, int incC2,
		double * chkC1, int incC1_2, double * chkC2, int incC2_2){
	//determin the reponsisble column 
	int block = blockIdx.x;
	int col = threadIdx.x;
	double diff = abs(*(chkC1+block+col*incC1_2)-*(chksumC1+block+col*incC1);
	if(diff>0.1){
		double diff2=abs(*(chkC2+block+col*incC2_2)-*(chksumC2+block+col*incC2);
		int row = (int)round(diff2/diff);
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
		int incC1, double * checksumC2, int incC2) {

	double negone = -1;
	double one = 1;
	double zero = 0;
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &negone, A, lda, B,
			ldb, &one, C, ldc);

	//recalculate checksum1 and checksum2
	double * chk1;
	double * chk2;
	size_t chk1_pitch;
	size_t chk2_pitch;

	cudaMallocPitch((void**) &chk1, &chk1_pitch, (m / n) * sizeof(double), n);
	cudaMallocPitch((void**) &chk2, &chk2_pitch, (m / n) * sizeof(double), n);

	int chk1_ld = chk1_pitch / sizeof(double);
	int chk2_ld = chk2_pitch / sizeof(double);

	double * v1 = new double[B];
	double * v2 = new double[B];
	for (int i = 0; i < B; i++) {
		v1[i] = 1;
		v2[i] = i + 1;
	}

	double * v1d;
	size_t v1d_pitch;
	cudaMallocPitch((void**) &v1d, &v1d_pitch, B * sizeof(double), 1);
	cudaMemcpy2D(v1d, v1d_pitch, v1, B * sizeof(double), B * sizeof(double), 1,
			cudaMemcpyHostToDevice);
	
	double * v2d;
	size_t v2d_pitch;
	cudaMallocPitch((void**) &v2d, &v2d_pitch, B * sizeof(double), 1);
	cudaMemcpy2D(v2d, v2d_pitch, v2, B * sizeof(double), B * sizeof(double), 1,
			cudaMemcpyHostToDevice);
	
	for (int i = 0; i < m; i += n) {
		cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C + i, ldc, v1d, 1,
				&zero, chk1 + (i / n), chk1_ld);
		cublasDgemv(handle, CUBLAS_OP_T, n, n, &one, C + i, ldb, v2d, 1,
				&zero, chk2 + (i / n), chk2_ld);
	}
	//update checksum1 and checksum2
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m/n, n, k, &negone,
			checksumA1, incA1, B, ldb, &one, checksumC1, incC1);
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m/n, n, k, &negone,
			checksumA2, incA2, B, ldb, &one, checksumC2, incC2);
	//error detection and error correction
	detectAndCorrectForGemm<<<dim3(m/n),dim3(n)>>>(C, ldc, n,
			checksumC1, incC1, checksumC2, incC2,
			chk1, chk1_ld, chk2, chk2_ld)

}