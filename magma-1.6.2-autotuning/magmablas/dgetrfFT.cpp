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
              ABFTEnv * abftEnv,
              bool FT , bool DEBUG, bool VERIFY) {

    double one = 1;
    double zero = 0;
    double negone = -1;

    if (DEBUG) {
        cout << "[dgetrf] to be updated matrix:" << endl;
        printMatrix_host(A, lda,  m, n, -1, -1);
    }


    if (FT & VERIFY) {
    	char N = 'N';
        double * chk1 = new double[m];
        double * chk2 = new double[m];
        int chk1_inc = 1;
        int chk2_inc = 1;


        blasf77_dgemv(  &N,
                        &m, &n,
                        &one,
                        A, &lda,
                        abftEnv->v, &(abftEnv->v_ld),
                        &zero,
                        chk1, &chk1_inc );
        blasf77_dgemv(  &N,
                        &m, &n,
                        &one,
                        A, &lda,
                        abftEnv->v + 1, &(abftEnv->v_ld),
                        &zero,
                        chk2, &chk2_inc );

        if (DEBUG) {
			cout<<"[dgetrf] recalcuated checksum on CPU before factorization:"<<endl;
			printMatrix_host(chk1, 1, m, 1, -1, -1);
			printMatrix_host(chk2, 1, m, 1, -1, -1);
			cout<<"[dgetrf] updated checksum on CPU before factorization:"<<endl;
			printMatrix_host(abftEnv->work_chk, abftEnv->work_chk_ld, m, 2, -1, -1);
		}

    }


    
    lapackf77_dgetrf( &m, &n, A, &lda, ipiv, info);
    
    if (FT) {
        //update checksums
        for (int j = 0; j < n; j++) {
        	//swap row j with ipiv[j]
        	if (ipiv[j] != 0) {
        		row_swap(abftEnv->row_hchk, abftEnv->row_hchk_ld, 2, j, ipiv[j]-1);
        	}

        }
         for (int j = 0; j < n; j++) {
        	double Ajj = *(A + j * lda + j);
        	if (Ajj != 0) {
        		double r = *(abftEnv->row_hchk + j) * (-1);
        		int inc = 1;
                int t = m - j - 1;
        		blasf77_daxpy(&t, &r, 
        			A + j * lda + j + 1, &inc,
        			abftEnv->row_chhk + j + 1, &inc);

				r = *(abftEnv->row_chhk + abftEnv->row_hchk_ld + j) * (-1);
        		blasf77_daxpy(&t, &r, 
        			A + j * lda + j + 1, &inc,
        			abftEnv->row_chhk + abftEnv->row_hchk_ld + j + 1, &inc);
        	}
        }

        if (DEBUG) {
        	cout << "[dgetrf] updated matrix:" << endl;
        	printMatrix_host(A, lda,  m, n, -1, -1);
        	cout << "[dgetrf] updated checksum:" << endl;
        	printMatrix_host(abftEnv->row_chhk, abftEnv->row_hchk_ld,  m, 2, -1, -1);
        }
    }
}

