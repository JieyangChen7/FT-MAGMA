using namespace std;
//initialize checksum
double * initializeChecksum(cublasHandle_t handle, double * matrix, int ld, int N, int B, 
		double * vd, int vd_ld) {

	//cout<<"checksum vector on GPU:"<<endl;
	//printVector_gpu(vd,B);
	size_t checksum_dev_pitch
	double * checksum_dev;
	cudaMallocPitch((void**) &chksum_dev, &chksum_dev_pitch, (N / B) * 2 * sizeof(double), N);
	int chksum_ld = chksum_pitch / sizeof(double);
	
	
	//printMatrix_gpu(matrix,ld*sizeof(double),N,N);
	//printMatrix_gpu(matrix,ld*sizeof(double),B,N);
	double one = 1;
	double zero = 0;
	for (int i = 0; i < N; i += B) {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2, N, B, &one, vd, vd_ld, matrix + i, ld,
				&zero, chksum + (i / B) * 2, chksum_ld);
	}
	
	double * checksum_host;
	cudaHostAlloc((void**) &checksum_host, (N / B) * N * 2 * sizeof(double), cudaHostAllocDefault);
	
	
	return chksum;

}

double * initializeChecksumCPU(double * checksum_dev, size_t chksum_pitch, int N, int B) {

	//cout<<"checksum vector on GPU:"<<endl;
	//printVector_gpu(vd,B);
	double * chek_host;
	cudaMallocPitch((void**) &chksum, &chksum_pitch, (N / B) * 2 * sizeof(double), N);
	int chksum_ld = chksum_pitch / sizeof(double);
	//printMatrix_gpu(matrix,ld*sizeof(double),N,N);
	//printMatrix_gpu(matrix,ld*sizeof(double),B,N);
	double one = 1;
	double zero = 0;
	for (int i = 0; i < N; i += B) {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2, N, B, &one, vd, vd_ld, matrix + i, ld,
				&zero, chksum + (i / B) * 2, chksum_ld);
		/*cout<<"i="<<i<<endl;
		printMatrix_gpu(matrix+i,ld*sizeof(double),B,N);
		printMatrix_gpu(vd, vd_ld*sizeof(double), B, 2 );
		printMatrix_gpu(chksum + (i / B)*2, chksum_pitch, 2, N);
		cout<<endl;
		*/
	}
	return chksum;

}
