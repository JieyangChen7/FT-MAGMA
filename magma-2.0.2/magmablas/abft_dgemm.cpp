#include "magma_internal.h"
#include"abft_checker.h"
//dgemm with FT

/**
 * m: number of row of A (N-i-B)
 * n: number of row of B (B)
 * k: number of col of A / col of B (i)
 */
void abft_dgemm( magma_trans_t transA, magma_trans_t transB,
			  int m, int n, int k, 
			  double alpha, 
			  double * dA, int ldda,
		      double * dB, int lddb, 
			  double beta, 
			  double * dC, int lddc,
			  int nb,
			  double * dA_colchk,	int ldda_colchk,
			  double * dA_rowchk,   int ldda_rowchk,
			  double * dA_colchk_r,	int ldda_colchk_r,
			  double * dA_rowchk_r, int ldda_rowchk_r,
			  double * dB_colchk,   int lddb_colchk,
			  double * dB_rowchk,   int lddb_rowchk,
			  double * dB_colchk_r, int lddb_colchk_r,
			  double * dB_rowchk_r, int lddb_rowchk_r,
			  double * dC_colchk,   int lddc_colchk,
			  double * dC_rowchk,   int lddc_rowchk,
			  double * dC_colchk_r, int lddc_colchk_r,
			  double * dC_rowchk_r, int lddc_rowchk_r,
			  double * chk_v, int ld_chk_v, 
			  bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
			  magma_queue_t stream1, magma_queue_t stream2) {

	

	int mem_row = 0; // number of row and col of B stored in memory(no trans operation)
	int mem_col = 0;

	if (FT && CHECK_BEFORE) {

		// number of row and col of A stored in memory(no trans operation)
		if (transA == MagmaNoTrans) {
			mem_row = m;
			mem_col = k;
			abft_checker_colchk(dA, ldda, mem_row, mem_col, nb,
	                            dA_colchk,   ldda_colchk,
	                            dA_colchk_r, ldda_colchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);


		} else if (transA == MagmaTrans) {
			mem_row = k;
			mem_col = m;
			abft_checker_rowchk(dA, ldda, mem_row, mem_col, nb,
	                            dA_rowchk,   ldda_rowchk,
	                            dA_rowchk_r, ldda_rowchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);
		}
		
		//verify B before use
		if (transB == MagmaNoTrans) {
			mem_row = k;
			mem_col = n;
			abft_checker_rowchk(dB, lddb, mem_row, mem_col, nb,
	                            dB_rowchk,   lddb_rowchk,
	                            dB_rowchk_r, lddb_rowchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);

		} else if (transB == MagmaTrans) {
			mem_row = n;
			mem_col = k;
			abft_checker_colchk(dB, lddb, mem_row, mem_col, nb,
	                            dB_colchk,   lddb_colchk,
	                            dB_colchk_r, lddb_colchk_r,
	                            chk_v,       ld_chk_v,
	                            DEBUG,
	                            stream1);
		}
		
		
		mem_row = m;
		mem_col = n;

		abft_checker_colchk(dC, lddc, mem_row, mem_col, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

		abft_checker_rowchk(dC, lddc, mem_row, mem_col, nb,
                            dC_rowchk,   lddc_rowchk,
                            dC_rowchk_r, lddc_rowchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);
	}
		

	magma_dgemm(transA, transB,
				m, n, k,
				alpha,
				dA, ldda, dB, lddb,
				beta,
				dC, lddc,
				stream1);

	// if (INJECT) {
	// 	magma_dscal( 1, 10000000000, C, 1);
	// }  
	
	if(FT){	
		if (transA == MagmaNoTrans) {
			magma_dgemm(transA, transB,
					   (m / nb) * 2, n, k,
					   alpha,
					   dA_colchk, ldda_colchk, dB, lddb,
					   beta,
					   dC_colchk, lddc_colchk,
					   stream2);
		} else {
			magma_dgemm(transA, transB,
					   (m / nb) * 2, n, k,
					   alpha,
					   dA_rowchk, ldda_rowchk, dB, lddb,
					   beta,
					   dC_colchk, lddc_colchk,
					   stream2);
		}

		if (transB == MagmaNoTrans) {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
						m , (n / nb) * 2, k,
						alpha,
						dA, ldda,
						dB_rowchk, lddb_rowchk,
						beta,
						dC_rowchk, lddc_rowchk,
						stream2);
		} else {
			//we can further work on this to support trans A.
			magma_dgemm(transA, transB,
						m , (n / nb) * 2, k,
						alpha,
						dA, ldda,
						dB_colchk, lddb_colchk,
						beta,
						dC_rowchk, lddc_rowchk,
						stream2);
		}
		
	}


	if (FT && CHECK_AFTER) {

		mem_row = m;
		mem_col = n;

		abft_checker_colchk(dC, lddc, mem_row, mem_col, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

		abft_checker_rowchk(dC, lddc, mem_row, mem_col, nb,
                            dC_rowchk,   lddc_rowchk,
                            dC_rowchk_r, lddc_rowchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

	}
}