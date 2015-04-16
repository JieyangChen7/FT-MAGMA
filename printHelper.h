//printing tools
void printMatrix_host(double * matrix_host, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(5);
			cout.setf(ios::left);
			cout << matrix_host[j * N + i];
		}
		cout << endl;
	}
	cout << endl;
}

void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int N) {
	double * matrix_host = new double[N * N]();
	cudaMemcpy2D(matrix_host, N * sizeof(double), matrix_device, matrix_pitch,
			N * sizeof(double), N, cudaMemcpyDeviceToHost);
	printMatrix_host(matrix_host, N);
	delete[] matrix_host;
}

void printVector_host(double * vector_host, int N) {
	for (int i = 0; i < N; i++) {
		cout.width(5);
		cout.setf(ios::left);
		cout << vector_host[i];
	}
	cout << endl;
}

void printVector_gpu(double * vector_device, int N) {
	double * vector_host = new double[N]();
	cudaMemcpy(vector_host, vector_device, N * sizeof(double),
			cudaMemcpyDeviceToHost);
	printVector_host(vector_host, N);
	delete[] vector_host;
}