#include"FT.h"
#include<iostream>
using namespace std;
//dsyrk with FT

/**
 * n: number of row of A
 * m: number of col of A
 */
void dsyrkFT(magma_uplo_t uplo, magma_trans_t trans,
			 int n, int k, 
			 double alpha,
			 double * dA, int ldda,
			 double beta,
			 double * dC, int lddc,
			 double * colchkA, int ld_colchkA,
			 double * colchkC, int ld_colchkC, 
			 double * chk_v, int ld_chk_v, 
			 bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
			 magma_queue_t * stream){
	
	/*		   k				n
	 * ******************   *********
	 * *		A		* =>*	C	* n
	 * *				* 	*		*
	 * ******************	*********
	 */
	
	if (FT && CHECK_BEFORE) { 
		//verify A before use
		check_col(k, n, nb,
                  dA, ldda,
                  chk_v, ld_chk_v,
                  colchkA, ld_colchkA, 
                  colchkA_r, ld_colchkA_r, 
                  stream[1],
                  DEBUG, "syrk-before-A");

		//verify C before use
		check_col(n, n, nb,
                  dC, lddc,
                  chk_v, ld_chk_v,
                  colchkC, ld_colchkC, 
                  colchkC_r, ld_colchkC_r, 
                  stream[1],
                  DEBUG, "syrk-before-C");		
	}

	if (FT) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemm(
				MagmaNoTrans, MagmaTrans,
				n, n, k,
				MAGMA_D_ONE * (-1),
				dA, ldda, dA, ldda,
				MAGMA_D_ONE,
				dC, lddc );
	} else {
		magma_dsyrk(uplo, trans, n, k,
					alpha, dA, ldda,
					beta,     dC, lddc);
	}
	
	if(FT){
		//update checksums on GPU
		//magmablasSetKernelStream(stream[1]);
		magmablasSetKernelStream(stream[4]);
		magma_dgemm(
					MagmaNoTrans, MagmaTrans,
					2, n, k,
					MAGMA_D_ONE * (-1),
					colchkA, ld_colchkA, dA, ldda,
					MAGMA_D_ONE,
					colchkC, ld_colchkC );
	}


	if (FT && CHECK_AFTER) {
		//verify C after use
		check_col(n, n, nb,
                  dC, lddc,
                  chk_v, ld_chk_v,
                  colchkC, ld_colchkC, 
                  colchkC_r, ld_colchkC_r, 
                  stream[1],
                  DEBUG, "syrk-after-C");


}