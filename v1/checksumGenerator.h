using namespace std;
//initialize checksum
double * initializeChecksum(cublasHandle_t handle, double * matrix, int ld, int N, int B, double * vd, size_t& chksum_pitch) {

	//cout<<"checksum vector on GPU:"<<endl;
	//printVector_gpu(vd,B);
	
	double * chksum;
	cudaMallocPitch((void**) &chksum, &chksum_pitch, (N / B) * sizeof(double), N);
	int chksum_ld = chksum_pitch / sizeof(double);
	//printMatrix_gpu(matrix,ld*sizeof(double),N,N);
	//printMatrix_gpu(matrix,ld*sizeof(double),B,N);
	double alpha = 1;
	double beta = 0;
	for (int i = 0; i < N; i += B) {
		cublasDgemv(handle, CUBLAS_OP_T, B, N, &alpha, matrix + i, ld, vd, 1,
				&beta, chksum + (i / B), chksum_ld);
		//cout<<"i="<<i<<endl;
		//printMatrix_gpu(matrix+i,ld*sizeof(double),B,N);
		//printVector_gpu(vd,B);
		//printMatrix_gpu(chksum + (i / B), chksum_pitch, 1, N);
	}
	return chksum;

}