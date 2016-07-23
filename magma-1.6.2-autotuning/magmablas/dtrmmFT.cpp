#include"FT.h"
#include<iostream>
using namespace std;


void dtrmmFT( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    		int m, int n,
    		double alpha,
    		double * dA, int ldda,
    		double * dB, int lddb,
    		ABFTEnv * abftEnv,
			double * col_chkA, int col_chkA_ld,
			double * row_chkA, int row_chkA_ld,		
			double * col_chkB, int col_chkB_ld,
			double * row_chkB, int row_chkB_ld, 
			bool FT, bool DEBUG, bool VERIFY, 
			magma_queue_t * stream) {


	//update column checksum
	dtrmmFT( side, uplo, trans, diag,
             (m / abftEnv->chk_nb) * 2, n,
             alpha, dA, ldda,
             col_chkB, col_chkB_ld);


	//update row checksum
	magma_dgemm(MagmaNoTrans, MagmaNoTrans,
				m, (n / abftEnv->chk_nb) * 2, n,
				alpha,
				dB, lddb, row_chkA, row_chkA_ld,
				0,
				row_chkB, row_chkB_ld );

	dtrmmFT( side, uplo, trans, diag,
                m, n,
                alpha, dA,  ldda,
                dB, lddb);

	
}