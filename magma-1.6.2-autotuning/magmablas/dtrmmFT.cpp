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
	
}