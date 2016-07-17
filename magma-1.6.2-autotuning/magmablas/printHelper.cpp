#include "magma.h"
#include"FT.h"
#include<iostream>
using namespace std;
//printing tools
//int n = 4;

/*
 * row_block and col_block control the display block, -1 represents no block
 */
void printMatrix_host(double * matrix_host, int ld,  int M, int N, int row_block, int col_block) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(20);
			cout.setf(ios::left);
			cout << matrix_host[j * ld + i];
			if (col_block != -1 && (j + 1) % col_block == 0) {
				cout << "	";
			}
		}
		cout << endl;
		if (row_block != -1 && (i + 1) % row_block == 0) {
			cout << endl;
		}
	}
	cout << endl;
}
/**
 * M: number of row
 * N: number of col
 */
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N, int row_block, int col_block) {
	double * matrix_host = new double[M * N]();
//	cudaMemcpy2D(matrix_host, M * sizeof(double), matrix_device, matrix_pitch,
//			M * sizeof(double), N, cudaMemcpyDeviceToHost);
	magma_dgetmatrix(M, N, matrix_device, matrix_ld, matrix_host, M);
	printMatrix_host(matrix_host, M, M, N, row_block, col_block);
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