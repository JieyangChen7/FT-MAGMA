#include "FT.h"
#include <iostream>

using namespace std;

//LU Factorization with FT on CPU using ACML
//m: number of row
//n: number of col

void swap_row_chk(double * chksum, int chksum_ld, int n, int i, int j) {
	for (int k = 0; k < n; k++) {
		double temp = *(chksum + k * chksum_ld + i);
		*(chksum + k * chksum_ld + i) = *(chksum + k * chksum_ld + j);
		*(chksum + k * chksum_ld + j) = temp;
	}
}

void swap_col_chk(ABFTEnv * abftEnv, double * A, int lda, double * chksum, int chksum_ld, int n, int i, int j) {

    //cout << i << "<->" << j << endl;

    int origin_block = i / abftEnv->chk_nb;
    int target_block = j / abftEnv->chk_nb;

    //cout << "origin_block:" << origin_block << endl;
    //cout << "target_block:" << target_block << endl;

    int origin_block_pos = i % abftEnv->chk_nb;
    int target_block_pos = j % abftEnv->chk_nb;

    //cout << "origin_block_pos:" << origin_block_pos << endl;
    //cout << "target_block_pos:" << target_block_pos << endl;

    for (int k = 0; k < n; k++) {
        double origin_element = *(A + k * lda + i);
        double target_element = *(A + k * lda + j);

        //cout << origin_element << "<->" << target_element << "  ";

        *(chksum + k * chksum_ld + origin_block * 2) -= origin_element;
        *(chksum + k * chksum_ld + origin_block * 2) += target_element;

        *(chksum + k * chksum_ld + target_block * 2) -= target_element;
        *(chksum + k * chksum_ld + target_block * 2) += origin_element;

        *(chksum + k * chksum_ld + origin_block * 2 + 1) -= origin_element * (origin_block_pos + 1);
        *(chksum + k * chksum_ld + origin_block * 2 + 1) += target_element * (origin_block_pos + 1);

        *(chksum + k * chksum_ld + target_block * 2 + 1) -= target_element * (target_block_pos + 1);
        *(chksum + k * chksum_ld + target_block * 2 + 1) += origin_element * (target_block_pos + 1);
    }
}

void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * info,
              ABFTEnv * abftEnv,
              bool FT , bool DEBUG, bool VERIFY) {

    double one = 1;
    double zero = 0;
    double negone = -1;
    // swap_row_chk(A, lda, n, 0, 6);
    // swap_row_chk(A, lda, n, 1, 11);
    // swap_row_chk(A, lda, n, 2, 6);
    // swap_row_chk(A, lda, n, 3, 14);
    double * cA = new double[lda * n];
    int ldca = lda;

    memcpy(cA, A, lda*n*sizeof(double));


    if (DEBUG) {
        cout << "[dgetrf] to be updated matrix:" << endl;
        printMatrix_host(A, lda,  m, n, 4, 4);

        // cout << "[dgetrf] to be copy matrix:" << endl;
        // printMatrix_host(cA, ldca,  m, n, 4, 4);
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
			printMatrix_host(abftEnv->row_hchk, abftEnv->row_hchk_ld, m, 2, -1, -1);
		}

    }



    
    lapackf77_dgetrf( &m, &n, A, &lda, ipiv, info);
    


    if (FT) {
        //update row checksums
        for (int j = 0; j < n; j++) {
        	//swap row j with ipiv[j]
        	if (ipiv[j] != 0) {
                //cout << j << "<->" << ipiv[j] << endl;
        		swap_row_chk(abftEnv->row_hchk, abftEnv->row_hchk_ld, 2, j, ipiv[j]-1);
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
        			abftEnv->row_hchk + j + 1, &inc);

				r = *(abftEnv->row_hchk + abftEnv->row_hchk_ld + j) * (-1);
        		blasf77_daxpy(&t, &r, 
        			A + j * lda + j + 1, &inc,
        			abftEnv->row_hchk + abftEnv->row_hchk_ld + j + 1, &inc);
        	}
        }

        if (DEBUG) {
            cout << "[dgetrf] before swap column checksum:" << endl;
            printMatrix_host(abftEnv->col_hchk, abftEnv->col_hchk_ld,  (m / abftEnv->chk_nb) * 2, abftEnv->chk_nb, 2, 4);
        }

        //update column checksums
        for (int j = 0; j < n; j++) {
            //swap row j with ipiv[j]
            if (ipiv[j] != 0) {
                swap_col_chk(abftEnv, cA, ldca, abftEnv->col_hchk, abftEnv->col_hchk_ld, abftEnv->chk_nb, j, ipiv[j]-1);
                swap_row_chk(cA, ldca, n, j, ipiv[j]-1);
            }
        }

        if (DEBUG) {
            cout << "[dgetrf] after swap column checksum:" << endl;
            printMatrix_host(abftEnv->col_hchk, abftEnv->col_hchk_ld,  (m / abftEnv->chk_nb) * 2, abftEnv->chk_nb, 2, 4);
        }

        for (int j = 0; j < n; j++) {
            int chk_m = (m / abftEnv->chk_nb) * 2;
            int chk_n = n - j - 1;
            double negone = -1;
            int incx = 1;
            double * chk = abftEnv->col_hchk + (j + 1) * abftEnv->col_hchk_ld;
            int chk_ld = abftEnv->col_hchk_ld;

            cout << "chk_ld=" << chk_ld << endl;


            double scalar = 1/(*(A + lda * j + j));

            // cout << "[" << j << "]:" << scalar << endl;
            blasf77_dscal(&chk_m, &scalar, abftEnv->col_hchk + j * abftEnv->col_hchk_ld, &incx);
            blasf77_dger(&chk_m, &chk_n, &negone, 
                         abftEnv->col_hchk + j * abftEnv->col_hchk_ld, &incx,
                         A + lda * (j + 1) + j, &lda,
                         chk, &chk_ld);
        }



        if (DEBUG) {
        	cout << "[dgetrf] updated matrix:" << endl;
        	printMatrix_host(A, lda,  m, n, 4, 4);
        	cout << "[dgetrf] updated row checksum:" << endl;
        	printMatrix_host(abftEnv->row_hchk, abftEnv->row_hchk_ld,  m, 2, 4, 2);

            cout << "[dgetrf] updated column checksum:" << endl;
            printMatrix_host(abftEnv->col_hchk, abftEnv->col_hchk_ld,  (m / abftEnv->chk_nb) * 2, abftEnv->chk_nb, 2, 4);
        }
    }
}

