using namespace std;
//initialize checksum
double * initializeChecksum(cublasHandle_t handle, double * matrix, int ld, int N, int B, double * v, size_t& chksum_pitch) {

	double * vd;
	size_t vd_pitch;
	cudaMallocPitch((void**) &vd, &vd_pitch, B * sizeof(double), 1);
	cudaMemcpy2D(vd, vd_pitch, v, B*sizeof(double), B * sizeof(double),
			1, cudaMemcpyHostToDevice);

	double * chksum;
	//size_t chksum_pitch;
	cudaMallocPitch((void**) &chksum, &chksum_pitch, (N / B) * sizeof(double), N);
	cudaMemset2D((void*) chksum, chksum_pitch, 0, (N / B) * sizeof(double), N);
	int chksum_ld = chksum_pitch / sizeof(double);

	double alpha = 1;
	double beta = 0;
	for (int i = 0; i < N; i += B) {
		cublasDgemv(handle, CUBLAS_OP_T, N, B, &alpha, matrix + i, ld, vd, 1,
				&beta, chksum + (i / B), chksum_ld);
	}
	return chksum;

}