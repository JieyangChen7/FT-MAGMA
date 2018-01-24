#include"FT.h"
#include<iostream>
using namespace std;
//dgemm with FT

/**
 * m: number of row of A (N-i-B)
 * n: number of row of B (B)
 * k: number of col of A / col of B (i)
 */
void dgemmFT( magma_trans_t transA, magma_trans_t transB,
			  int m, int n, int k, 
			  double alpha, 
			  double * dA, int ldda,
		      double * dB, int lddb, 
			  double beta, 
			  double * dC, int lddc, 
			  double * col_chkA,   int ld_col_chkA,
			  double * col_chkA_r, int ld_col_chkA_r,
			  double * rowchkA,    int ld_row_chkA,	
			  double * rowchkA_r,  int ld_row_chkA_r,	
			  double * colchkB,    int ld_col_chkB,
			  double * colchkB_r,  int ld_col_chkB_r,
			  double * rowchkB,    int ld_row_chkB, 
			  double * rowchkB_r,  int ld_row_chkB_r,
			  double * colchkC,    int ld_col_chkC,
			  double * colchkC_r,  int ld_col_chkC_r,  
			  double * rowchkC,    int ld_row_chkC,
			  double * rowchkC_r,  int ld_row_chkC_r,
			  double * chk_v, int ld_chk_v, 
			  bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, bool INJECT,
			  magma_queue_t * stream) {

	
	// if (true) {
	// 	cout << "dgemm" << endl;
	// }

	

	int mem_row = 0; // number of row and col of B stored in memory(no trans operation)
	int mem_col = 0;

	if (FT && CHECK_BEFORE) {

		// number of row and col of A stored in memory(no trans operation)
		if (transA == MagmaNoTrans) {
			mem_row = m;
			mem_col = k;
			check_col(mem_row, mem_col, nb,
	                  dA, ldda,
	                  chk_v, ld_chk_v,
	                  colchkA, ld_colchkA, 
	                  colchkA_r, ld_colchkA_r, 
	                  stream[1],
	                  DEBUG, "gemm-before-A");


		} else if (transA == MagmaTrans) {
			mem_row = k;
			mem_col = m;
			check_row(mem_row, mem_col, nb,
	                  dA, ldda,
	                  chk_v, ld_chk_v,
	                  rowchkA, ld_rowchkA, 
	                  rowchkA_r, ld_rowchkA_r, 
	                  stream[1],
	                  DEBUG, "gemm-before-A");
		}
		
		//verify B before use
		if (transB == MagmaNoTrans) {
			mem_row = k;
			mem_col = n;
			check_row(mem_row, mem_col, nb,
	                  dB, lddb,
	                  chk_v, ld_chk_v,
	                  rowchkB, ld_rowchkB, 
	                  rowchkB_r, ld_rowchkB_r, 
	                  stream[1],
	                  DEBUG, "gemm-before-B");

		} else if (transB == MagmaTrans) {
			mem_row = n;
			mem_col = k;
			check_col(mem_row, mem_col, nb,
	                  dB, lddb,
	                  chk_v, ld_chk_v,
	                  colchkB, ld_colchkB, 
	                  colchkB_r, ld_colchkB_r, 
	                  stream[1],
	                  DEBUG, "gemm-before-B");
		}
		
		
		mem_row = m;
		mem_col = n;

		check_col(mem_row, mem_col, nb,
                  dC, lddc,
                  chk_v, ld_chk_v,
                  colchkC, ld_colchkC, 
                  colchkC_r, ld_colchkC_r, 
                  stream[1],
                  DEBUG, "gemm-before-C");

		check_row(mem_row, mem_col, nb,
                  dC, lddc,
                  chk_v, ld_chk_v,
                  rowchkC, ld_rowchkC, 
                  rowchkC_r, ld_rowchkC_r, 
                  stream[1],
                  DEBUG, "gemm-before-C");
	}
		

	
	magmablasSetKernelStream(stream[1]);
	magma_dgemm(transA, transB,
				m, n, k,
				alpha,
				dA, ldda, dB, lddb,
				beta,
				dC, lddc );

	// if (INJECT) {
	// 	magma_dscal( 1, 10000000000, C, 1);
	// }  
	
	if(FT){	
		magmablasSetKernelStream(stream[4]);
		//magmablasSetKernelStream(stream[1]);
		if (transA == MagmaNoTrans) {
			magma_dgemm(transA, transB,
					   (m / nb) * 2, n, k,
					   alpha,
					   colchkA, ld_colchkA, dB, lddb,
					   beta,
					   colchkC, ld_colchkC );
		} else {
			magma_dgemm(transA, transB,
					   (m / nb) * 2, n, k,
					   alpha,
					   rowchkA, ld_rowchkA, dB, lddb,
					   beta,
					   colchkC, ld_colchkC );
		}

		if (transB == MagmaNoTrans) {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
						m , (n / nb) * 2, k,
						alpha,
						dA, ldda,
						rowchkB, ld_rowchkB,
						beta,
						rowchkC, ld_rowchkC );
		} else {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
						m , (n / nb) * 2, k,
						alpha,
						dA, ldda,
						colchkB, ld_colchkB,
						beta,
						rowchkC, ld_rowchkC );
		}
		
	}


	if (FT && CHECK_AFTER) {

		mem_row = m;
		mem_col = n;

		check_col(mem_row, mem_col, nb,
                  dC, lddc,
                  chk_v, ld_chk_v,
                  colchkC, ld_colchkC, 
                  colchkC_r, ld_colchkC_r, 
                  stream[1],
                  DEBUG, "gemm-after-C");

		check_row(mem_row, mem_col, nb,
                  dC, lddc,
                  chk_v, ld_chk_v,
                  rowchkC, ld_rowchkC, 
                  rowchkC_r, ld_rowchkC_r, 
                  stream[1],
                  DEBUG, "gemm-after-C");

	}
}