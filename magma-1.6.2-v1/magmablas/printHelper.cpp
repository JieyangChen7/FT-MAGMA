#include"FT.h"
#include<iostream>
using namespace std;
//printing tools
void printMatrix_host(double * matrix_host, int M, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(10);
			cout.setf(ios::left);
			cout << matrix_host[j * M + i];
		}
		cout << endl;
	}
	cout << endl;
}
/**
 * M: number of row
 * N: number of col
 */
void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int M, int N) {
	double * matrix_host = new double[M * N]();
//	cudaMemcpy2D(matrix_host, M * sizeof(double), matrix_device, matrix_pitch,
//			M * sizeof(double), N, cudaMemcpyDeviceToHost);
	magma_dgetmatrix(M, N, matrix_device, matrix_pitch / sizeof(double), matrix_host, M);
	printMatrix_host(matrix_host, M, N);
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