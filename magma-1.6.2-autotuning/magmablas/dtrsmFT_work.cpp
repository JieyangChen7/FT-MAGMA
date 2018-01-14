#include"FT.h"
#include<iostream>
using namespace std;
//TRSM with FT on GPU using cuBLAS

/*
 * m: number of row of B
 * n: number of col of B
 */

void dtrsmFT_work(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    		      magma_int_t m, magma_int_t n,
    		      double alpha, 
    		      magmaDouble_const_ptr dA, magma_int_t ldda,
        		  magmaDouble_ptr       dB, magma_int_t lddb,
        		  magma_int_t flag, magmaDouble_ptr d_dinvA, magmaDouble_ptr dX,
                  int nb,
    		      double * colchkA, int ld_colchkA,
                  double * colchkA_r, int ld_colchkA_r,
    		      double * rowchkA, int ld_rowchkA, 
                  double * rowchkA_r, int ld_rowchkA_r, 
    		      double * colchkB, int ld_colchkB,
                  double * colchkB_r, int ld_colchkB_r,
    		      double * rowchkB, int ld_rowchkB, 
                  double * rowchkB_r, int ld_rowchkB_r, 
                  double * chk_v, int ld_chk_v, 
    		      bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    		      magma_queue_t * stream) {

	if (FT && CHECK_BEFORE) {
        check_col(m, n, nb,
                  dB, lddb,
                  chk_v, ld_chk_v,
                  colchkB, ld_colchkB, 
                  colchkB_r, ld_colchkB_r, 
                  stream[1],
                  DEBUG, "trsm-before")
    }               
	
	magmablasSetKernelStream(stream[1]);	
	magmablas_dtrsm_work( side, uplo, transA, diag,
                          m, n,
                          alpha,
                          dA, ldda,
                          dB, lddb,
                          flag, d_dinvA, dX );

	if (FT) {
		//update checksums
		//magmablasSetKernelStream(stream[1]);	
		
		magmablasSetKernelStream(stream[1]);	
		magma_dtrsm(side, uplo, trans, diag,
                    (m / nb) * 2, n,
                    alpha, dA, ldda,
                    colchkB, ld_colchkB);
	}

	if (FT && CHECK_AFTER) {
        check_col(m, n, nb,
                  dB, lddb,
                  chk_v, ld_chk_v,
                  colchkB, ld_colchkB, 
                  colchkB_r, ld_colchkB_r, 
                  stream[1],
                  DEBUG, "trsm-after")
        }      
}