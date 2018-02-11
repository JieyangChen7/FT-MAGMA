#include <cstdio>
#include "magma_internal.h"
#include "abft_encoder.h"
#include "abft_printer.h"
#include "abft_corrector.h"
void abft_checker_colchk(double * dA, int ldda, int m, int n, int nb,
						 double * dA_colchk,    int ldda_colchk,
    					 double * dA_colchk_r,  int ldda_colchk_r,
    					 double * dev_chk_v,    int ld_dev_chk_v,
    					 bool DEBUG,
    					 magma_queue_t stream){
	printf("abft_checker_colchk\n");
	col_chk_enc(m, n, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_colchk_r, ldda_colchk_r, 
                stream);
	magma_queue_sync(stream);
	

	if (DEBUG) {
			printf( "input matrix:\n" );
            printMatrix_gpu(dA, ldda, m, n, nb, nb, stream);
            printf( "updated column chk:\n" );
            printMatrix_gpu(dA_colchk, ldda_colchk, (m / nb) * 2, n, 2, nb, stream);
            printf( "recalculated column chk:\n" );
            printMatrix_gpu(dA_colchk_r, ldda_colchk_r, (m / nb) * 2, n, 2, nb, stream);
    }
    colchk_detect_correct(dA, ldda, m, n, nb,
				          dA_colchk,	ldda_colchk,
				          dA_colchk_r, 	ldda_colchk_r,
						  stream);
    magma_queue_sync(stream);
}

void abft_checker_rowchk(double * dA, int ldda, int m, int n, int nb,
						 double * dA_rowchk,    int ldda_rowchk,
    					 double * dA_rowchk_r,  int ldda_rowchk_r,
    					 double * dev_chk_v,    int ld_dev_chk_v,
    					 bool DEBUG,
    					 magma_queue_t stream){
	printf("abft_checker_rowchk\n");
	row_chk_enc(m, n, nb, 
                dA, ldda,  
                dev_chk_v, ld_dev_chk_v, 
                dA_rowchk_r, ldda_rowchk_r, 
                stream);
	magma_queue_sync(stream);

	
	if (DEBUG) {
			printf( "input matrix:\n" );
            printMatrix_gpu(dA, ldda, m, n, nb, nb, stream);
            printf( "updated row chk:\n" );
            printMatrix_gpu(dA_rowchk, ldda_rowchk, m, (n / nb) * 2, nb, 2, stream);
            printf( "recalculated row chk:\n" );
            printMatrix_gpu(dA_rowchk_r, ldda_rowchk_r, m, (n / nb) * 2, nb, 2, stream);
    }
    rowchk_detect_correct(dA, ldda, m, n, nb,
				          dA_rowchk,	ldda_rowchk,
				          dA_rowchk_r, 	ldda_rowchk_r,
						  stream);
    magma_queue_sync(stream);
	
}

void abft_checker_fullchk(double * dA, int ldda, int m, int n, int nb,
						  double * dA_colchk,    int ldda_colchk,
    					  double * dA_colchk_r,  int ldda_colchk_r,
    					  double * dA_rowchk,    int ldda_rowchk,
    					  double * dA_rowchk_r,  int ldda_rowchk_r,
    					  double * dev_chk_v,    int ld_dev_chk_v,
    					  bool DEBUG,
    					  magma_queue_t stream){

	abft_checker_colchk(dA, ldda, m, n, nb,
						dA_colchk,		ldda_colchk,
    					dA_colchk_r, 	ldda_colchk_r,
    					dev_chk_v, 		ld_dev_chk_v,
    					DEBUG,
    					stream);

	abft_checker_rowchk(dA, ldda, m, n, nb,
						dA_rowchk,		ldda_rowchk,
    					dA_rowchk_r, 	ldda_rowchk_r,
    					dev_chk_v, 		ld_dev_chk_v,
    					DEBUG,
    					stream);
	
}