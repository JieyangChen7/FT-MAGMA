#include "FT.h"
#include <iostream>

using namespace std;

//QR Factorization with FT on CPU using ACML
//m: number of row
//n: number of col


void dgeqrfFT( int m, int n, double * A, int lda, double * tau, double * work, int lwork, int *info ) {

	cout << "[DGEQRF] input matrix before factorization" << endl;
	printMatrix_host(A, lda, m, n, 4, 4);

	lapackf77_dgeqrf(&m, &n, A, &lda, tau, work, &lwork, info);

	cout << "[DGEQRF] input matrix after factorization" << endl;
	printMatrix_host(A, lda, m, n, 4, 4);

	cout << "[DGEQRF] TAU after factorization" << endl;
	for (int i = 0; i < n; i++)
		cout << tau[i] << "\t";
	cout << endl;


}