#include "FT.h"
#include <iostream>

using namespace std;

//QR Factorization with FT on CPU using ACML
//m: number of row
//n: number of col


void dgeqrfFT( int m, int n, double * A, int lda, double * tau, double * work, int lwork, int *info ) {


	lapackf77_dgeqrf(&m, &n, A, &lda, tau, work, &lwork, info);
}