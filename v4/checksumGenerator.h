using namespace std;
//initialize checksum
double * initializeChecksum(cublasHandle_t handle, double * matrix, int ld, int N, int B, 
		double * vd, int vd_ld) {

	//cout<<"checksum vector on GPU:"<<endl;
	size_t checksum_dev_pitch;
	double * checksum_dev;
	cudaMallocPitch((void**) &checksum_dev, &checksum_dev_pitch, (N / B) * 2 * sizeof(double), N);
	int checksum_dev_ld = checksum_dev_pitch / sizeof(double);
	
	
	//printMatrix_gpu(matrix,ld*sizeof(double),N,N);
	//printMatrix_gpu(matrix,ld*sizeof(double),B,N);
	double one = 1;
	double zero = 0;
	for (int i = 0; i < N; i += B) {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2, N, B, &one, vd, vd_ld, matrix + i, ld,
				&zero, checksum_dev + (i / B) * 2, checksum_dev_ld);
	}
	
	double * checksum_host;
	int checksum_host_ld = (N / B) * 2;
	cudaHostAlloc((void**) &checksum_host, (N / B) * N * 2 * sizeof(double), cudaHostAllocDefault);
	cudaMemcpy2D(checksum_host, checksum_host_ld * sizeof(double), 
			checksum_dev, checksum_dev_pitch, 
			(N / B) * 2 * sizeof(double), N, cudaMemcpyDeviceToHost);
	
	return checksum_host;

}
