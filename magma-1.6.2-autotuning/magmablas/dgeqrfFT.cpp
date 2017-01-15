#include "FT.h"
#include <iostream>

using namespace std;

//QR Factorization with FT on CPU using ACML
//m: number of row
//n: number of col


void dgeqrfFT( int m, int n, double * A, int lda, double * tau, double * work, int lwork, int * info,
			   ABFTEnv * abftEnv, 
			   bool FT , bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER) {

	if (FT && CHECK_BEFORE) {

	}

	if (DEBUG) {
		cout << "[DGEQRF] input matrix before factorization" << endl;
		printMatrix_host(A, lda, m, n, 4, 4);

		cout << "[DGEQRF] column checksum before factorization" << endl;
		printMatrix_host(abftEnv->col_hchk, abftEnv->col_hchk_ld, (m / abftEnv->chk_nb) * 2, n, 2, 4);

		cout << "[DGEQRF] row checksum before factorization" << endl;
		printMatrix_host(abftEnv->row_hchk, abftEnv->row_hchk_ld, m , (n / abftEnv->chk_nb) * 2, 4, 2);

	}
	cout << "dgeqrf" <<endl;

	if (FT) {

		int k = min(m, n);
		for (int i = 0; i < k; i++) {
			//cout << "i=" << i <<endl;



			int m2 = m - i;
			//int n2 = n - i - 1;
			int n2 = n - i;
			int cm = (m/abftEnv->chk_nb)*2;
			int incx = 1;
			double Aii = *(A + i * lda + i);
			double b = *(A + i * lda + i);
			double * v = new double[m2];
			double * cv = new double[cm];


			if (i > 1) {
				double alpha = -1;
				blasf77_daxpy(&n2,
	              &alpha,
	              A + i * lda + i - 1, &lda,
	              abftEnv->col_hchk + i * abftEnv->col_hchk_ld, &abftEnv->col_hchk_ld);

				alpha = -1 * (i+1);
				blasf77_daxpy(&n2,
	              &alpha,
	              A + i * lda + i - 1, &lda,
	              abftEnv->col_hchk + i * abftEnv->col_hchk_ld + 1, &abftEnv->col_hchk_ld);
			}


			memcpy(v+1, A + i * lda + i + 1, (m2-1)* sizeof(double));
			memcpy(cv, abftEnv->col_hchk + i * abftEnv->col_hchk_ld, cm * sizeof(double));



			//cv[0] = *(abftEnv->col_hchk + i * abftEnv->col_hchk_ld);
			//cv[1] = *(abftEnv->col_hchk + i * abftEnv->col_hchk_ld + 1);





			lapackf77_dlarfg(&m2, &b, v+1, &incx, tau + i);

			v[0] = 1;

			double t = *(tau + i);
			double scale = 1/(-(t * b));

			cv[0] -= 1 * Aii;
			cv[1] -= (i + 1) * Aii;



			//cv[0] *= scale;
			//cv[1] *= scale;

			blasf77_dscal( &cm,
	                       &scale,
	                       cv, &incx );

			cv[0] += 1;
			cv[1] += 1;



			char L = 'L';
			lapackf77_dlarf( &L, &m2, &n2,
	                         v, &incx,
	                         tau + i,
	                         A + i * lda + i, &lda,
	                         work );

			char T = 'T';
			int two = 2;
			int one = 1;
			double d_zero = 0;
			double * cw = new double[2];
			int inccw = 1;
			double neg_tau = *(tau + i) * -1;
			double d_one = 1; 
			blasf77_dgemv( &T,
	                     &m2, &two,
	                     &d_one,
	                     abftEnv->row_hchk + i, &abftEnv->row_hchk_ld,
	                     v, &incx,
	                     &d_zero,
	                     cw, &inccw);

			blasf77_dger( &cm, &n2,
	                      &neg_tau,
	                     cv, &one,
	                     work, &one,
	                     abftEnv->col_hchk + i * abftEnv->row_hchk_ld, &abftEnv->row_hchk_ld);

			blasf77_dger( &m2, &two,
	                      &neg_tau,
	                      v, &incx,
	                      cw, &inccw,
	                      abftEnv->row_hchk + i, &abftEnv->row_hchk_ld);




			cv[0] -= 1;
			cv[1] -= 1;

			memcpy(A + i * lda + i + 1, v+1, (m2-1)* sizeof(double));
			memcpy(abftEnv->col_hchk + i * abftEnv->col_hchk_ld, cv, cm * sizeof(double));


		

		}

		// cout << "after2" << endl;
		// printMatrix_host(A, lda, m, n, 4, 4);

			
		// cout << "column checksum after2" << endl;
		// printMatrix_host(abftEnv->col_hchk, abftEnv->col_hchk_ld, (m / abftEnv->chk_nb) * 2, n, 2, 4);

		// cout << "row checksum after2" << endl;
		// printMatrix_host(abftEnv->row_hchk, abftEnv->row_hchk_ld, m , (n / abftEnv->chk_nb) * 2, 4, 2);


	} else {

		lapackf77_dgeqrf(&m, &n, A, &lda, tau, work, &lwork, info);
	}
	if (FT) {
		int k = min (m, n);
		char L = 'L';

		for (int i = 0; i < k; i++) {

			//update row checksums
			double Aii = *(A + i * lda + i);
			*(A + i * lda + i) = 1;

			int pm = m - i;
			int pn = 2;
			int pincv = 1;

			// lapackf77_dlarf(&L, &pm, &pn,
   //                       	A + i * lda + i, &pincv,
   //                       	tau + i,
   //                       	abftEnv->row_hchk + i, &(abftEnv->row_hchk_ld),
   //                       	work );

			*(A + i * lda + i) = Aii;




			//update column checksums
			double c = 1 / (-1 * (*(tau + i) * Aii));
			int p2n = (m / abftEnv->chk_nb) * 2;
			int pincx = 1;
		//	blasf77_dscal(&p2n, &c, abftEnv->col_hchk + i * abftEnv->col_hchk_ld, &pincx);

			// int p3m = p2n - i;
			// int p3n = n - i - 1;
			// lapackf77_dlarf(&L, &pm, &pn,
   //                       	A + i * lda + i, &pincv,
   //                       	tau + i,
   //                       	abftEnv->row_hchk + i, &(abftEnv->row_hchk_ld),
   //                       	work );
			//construct v with column checksums
			// double * v = new double[m + 2];
			// int j = 0;
			// while (j < i) {
			// 	v[j] = 0.0;
			// 	j++;
			// }
			// v[j] = 1; //j = i
			// j++;
			// while (j < m) {
			// 	v[j] = *(A + i * lda + j);
			// 	j++;
			// }
			// v[j] = *(abftEnv->col_hchk + i);
			// v[j + 1] = *(abftEnv->col_hchk + i + 1);

			// cout << "[DGEQRF] v[" << i << "]:";
			// for (int k = 0; k < m + 2; k++)
			// 	cout << v[k] << "\t";
			// cout << endl;

		}

	}

	if (DEBUG) {
		cout << "[DGEQRF] input matrix after factorization" << endl;
		printMatrix_host(A, lda, m, n, 4, 4);

		cout << "[DGEQRF] TAU after factorization" << endl;
		for (int i = 0; i < n; i++)
			cout << tau[i] << "\t";
		cout << endl;

		cout << "[DGEQRF] column checksum after factorization" << endl;
		printMatrix_host(abftEnv->col_hchk, abftEnv->col_hchk_ld, (m / abftEnv->chk_nb) * 2, n, 2, 4);

		cout << "[DGEQRF] row checksum after factorization" << endl;
		printMatrix_host(abftEnv->row_hchk, abftEnv->row_hchk_ld, m , (n / abftEnv->chk_nb) * 2, 4, 2);
	}

	if (FT && CHECK_AFTER) {

	}



}