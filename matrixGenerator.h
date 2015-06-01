using namespace std;

//matrix generate tools
__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo,
		double alpha, double beta) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (uplo == 'u') {
		if (row < col + 1) {
			matrix[col * ld + row] = int(matrix[col * ld + row] * alpha + beta);
		} else {
			matrix[col * ld + row] = int(0.0);
		}
	} else {
		if (col < row + 1) {
			matrix[col * ld + row] = int(matrix[col * ld + row] * alpha + beta);
		} else {
			matrix[col * ld + row] = int(0.0);
		}
	}
}

void matrixGenerator_gpu(char uplo, double * matrix, int matrix_ld,
	 int N, int B) {
	double a = 10.0;
	//initialize cublas
	cublasStatus_t cublasStatus;
	cublasHandle_t handle;
	cublasStatus = cublasCreate(&handle);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
		cout << "CUBLAS NOT INITIALIZED(handle1) in matrixGenerator_gpu"
				<< endl;

	//initialize curand
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 10ULL);
	//generate random number in range (0,1] on result using curand
	curandGenerateUniformDouble(gen, matrix, matrix_ld * N);
	cudaDeviceSynchronize();

	matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, matrix_ld, uplo, a,1);
	cudaDeviceSynchronize();

	//do matrix-matrix multiplcation using cublas
	//cudaMemset(matrix, 0, matrix_ld * N * sizeof(double));

	double alpha = 1.0;
	double beta = 1.0;
	if (uplo == 'u') {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, matrix,
				matrix_ld, matrix, matrix_ld, &beta, matrix, matrix_ld);
	} else if (uplo == 'l') {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, matrix,
				matrix_ld, matrix, matrix_ld, &beta, matrix, matrix_ld);
	}
	cudaDeviceSynchronize();

	//matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, matrix_ld, uplo, 1.0,0);
	cudaDeviceSynchronize();
	cublasDestroy(handle);
	
	//print matrix
	//printMatrix_gpu(matrix, matrix_ld * sizeof(double),N, N);
	//print result
	//printMatrix_gpu(result,result_ld*sizeof(double),N,N);
	
}

void matrixGenerator_gpu2(char uplo, double * matrix, int matrix_ld, double * result, int result_ld,
	 int N, int B) {
	double a = 10.0;
	//initialize cublas
	cublasStatus_t cublasStatus;
	cublasHandle_t handle;
	cublasStatus = cublasCreate(&handle);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
		cout << "CUBLAS NOT INITIALIZED(handle1) in matrixGenerator_gpu"
				<< endl;

	//initialize curand
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 10ULL);
	//generate random number in range (0,1] on result using curand
	curandGenerateUniformDouble(gen, result, result_ld * N);
	cudaDeviceSynchronize();

	matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(result, result_ld, uplo, a,1);
	cudaDeviceSynchronize();

	//do matrix-matrix multiplcation using cublas
	//cudaMemset(matrix, 0, matrix_ld * N * sizeof(double));

	double alpha = 1.0;
	double beta = 0.0;
	if (uplo == 'u') {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, result,
				result_ld, result, result_ld, &beta, matrix, matrix_ld);
	} else if (uplo == 'l') {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, result,
				result_ld, result, result_ld, &beta, matrix, matrix_ld);
	}
	cudaDeviceSynchronize();

	//matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, matrix_ld, uplo, 1.0,0);
	cudaDeviceSynchronize();
	cublasDestroy(handle);
	
	//print matrix
	printMatrix_gpu(matrix, matrix_ld * sizeof(double),N, N);
	//print result
	printMatrix_gpu(result,result_ld*sizeof(double),N,N);
	
}

__global__ void resultVerify_gpu_help(double * realResult, int real_ld,
		double * testResult, int test_ld, double * diff, int N) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	diff[col * N + row] = testResult[col * test_ld + row]
			- realResult[col * real_ld + row];
}

bool resultVerify_gpu(double * realResult, int real_ld, double * testResult,
		int test_ld, int N, int B) {
	double * diff;
	cudaMalloc((void**) &diff, N * N * sizeof(double));
	resultVerify_gpu_help<<<dim3(N/B,N/B),dim3(B,B)>>>(realResult,real_ld,testResult,test_ld,diff,N);

	//printMatrix_gpu(realResult,real_ld*sizeof(double),N);
	//printMatrix_gpu(testResult,test_ld*sizeof(double),N,N);

	double * diff_host = new double[N * N]();
	cudaMemcpy(diff_host, diff, N * N * sizeof(double), cudaMemcpyDeviceToHost);
	//  printMatrix(diff_host,N);
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < j+1; i++) {
			if (abs(diff_host[i * N + j]) > 1e-3) {
				//  cout<<"diff:"<<abs(diff_host[i*N+j])<<endl;
				delete[] diff_host;
				cudaFree(diff);
				return false;
			}
		}
	}
	delete[] diff_host;
	cudaFree(diff);
	return true;

}
