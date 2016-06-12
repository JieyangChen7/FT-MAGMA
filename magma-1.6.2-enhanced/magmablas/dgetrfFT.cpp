#include "FT.h"
#include <iostream>

using namespace std;

//LU Factorization with FT on CPU using ACML
//m: number of row
//n: number of col

void row_swap(double * chksum, int chksum_ld, int n, int i, int j) {
	for (int k = 0; k < n; k++) {
		double temp = *(chksum + k * chksum_ld + i);
		*(chksum + k * chksum_ld + i) = *(chksum + k * chksum_ld + j);
		*(chksum + k * chksum_ld + j) = temp;
	}
}

void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * info,
              double * chksum, int chksum_ld,
              double * v, int v_ld,
              bool FT , bool DEBUG, bool VERIFY) {


    if (FT & VERIFY) {
        double * chk1 = new double[m];
        double * chk2 = new double[m];
        int chk1_inc = 1;
        int chk2_inc = 1;

        blasf77_dgemv(  &T,
                        &n, &n,
                        &one,
                        A, &lda,
                        v, &v_ld,
                        &zero,
                        chk1, &chk1_inc );
        blasf77_dgemv(  &T,
                        &n, &n,
                        &one,
                        A, &lda,
                        v + 1, &v_ld,
                        &zero,
                        chk2, &chk2_inc );


    }
    
    lapackf77_dgetrf( &m, &n, A, &lda, ipiv, info);
    
    if (FT) {
        //update checksums
        for (int j = 0; j < n; j++) {
        	//swap row j with ipiv[j]
        	row_swap(chksum, chksum_ld, 2, j, ipiv[j]);
        	double Ajj = *(A + j * lda + j);
        	if (Ajj != 0) {
        		double r = *(chksum + j) * (-1);
        		int inc = 1;
                int t = m - j - 1;
        		blasf77_daxpy(&t, &r, 
        			A + j * lda + j + 1, &inc,
        			chksum + j + 1, &inc);

				r = *(chksum + chksum_ld + j) * (-1);
        		blasf77_daxpy(&t, &r, 
        			A + j * lda + j + 1, &inc,
        			chksum + chksum_ld + j + 1, &inc);
        	}
        }

        if (DEBUG) {
        	cout << "[dgetrf] updated matrix:" << endl;
        	printMatrix_host(A, lda,  m, n);
        	cout << "[dgetrf] updated checksum:" << endl;
        	printMatrix_host(chksum, chksum_ld,  m, 2);
        }
    }
}

