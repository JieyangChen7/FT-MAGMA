using namespace std;
//initialize checksum
double * initializeChecksum(cublasHandle_t handle, double * matrix, int ld, int N, int B, 
		double * vd, int vd_ld, size_t& chksum_pitch) {

	//cout<<"checksum vector on GPU:"<<endl;
	//printVector_gpu(vd,B);
	double * chksum;
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