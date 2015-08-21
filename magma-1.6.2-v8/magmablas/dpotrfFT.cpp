#include"FT.h"
#include<iostream>

using namespace std;
//Cholesky Factorization with FT on CPU using ACML
double get(double * matrix, int ld, int n, int i, int j) {
	if (i > ld || j > n)
		cout << "matrix_get_error" << endl;
	return *(matrix + j * ld + i);
}
/**
 * Cholesky factorization with FT on CPU using ACML
 * A: input matrix
 * lda: leading dimension of A
 * n: size of A
 * chksum1: checksum 1
 * inc1: stride between elememts in chksum1
 * chksum2: checksum 2
 * inc2: stride between elememts in chksum2
 */
void dpotrfFT(double * A, int lda, int n, int * info,
				int K,
				double * chksum, int chksum_ld,
				double * v, int v_ld, 
				bool FT , bool DEBUG) {
	
	double one = 1;
	double zero = 0;
	double negone = -1;
	
	char T = 'T';
	char N = 'N';
	char L = 'L';
	
	if (FT) {
		//verify A before use
		
		double * chk = new double[K * n];
		int chk_inc = n;
		for (int i = 0; i < K; i++) {
			blasf77_dgemv(  &T,
							&n, &n,
							&one,
							A, &lda,
							v + i, &v_ld,
							&zero,
							chk + i, &chk_inc );	
		}
		//handle error - to be finished
		
		if (DEBUG) {
			cout<<"recalcuated checksum on CPU before factorization:"<<endl;
			printMatrix_host(chk, chk_inc, K, n);
			cout<<"updated checksum on CPU before factorization:"<<endl;
			printMatrix_host(chksum, chksum_ld, K, n);
		}
	}
	
	//do Choleksy factorization
	char uplo = 'L';
	lapackf77_dpotrf(&uplo, &n, A, &n, info);
	if (FT) {	
		//update checksum1 and checksum2
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < K; j++) {
				*(chksum + i*chksum_ld + j) = *(chksum + i*chksum_ld + j) / get(A, lda, n, i, i);
			}
			int m = n-i-1;
			int ONE = 1;
			blasf77_dgemm(  &N, &T,
							 &K, &m, &ONE,
							 &negone,
							 chksum + i*chksum_ld, &chksum_ld,
							 A + i*lda + i+1, &lda,
							 &one,
							 chksum + (i+1)*chksum_ld, 
							 &chksum_ld );
		}
	}
}