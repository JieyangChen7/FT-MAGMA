#include "FT.h"
#include <iostream>
using namespace std;

//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated
void row_recal_1(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < n; i += chk_nb) {
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaNoTrans, m, chk_nb, MAGMA_D_ONE,
				A + i * lda, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(stream[3]);
		magma_dgemv(MagmaNoTrans, m, chk_nb, MAGMA_D_ONE,
				A + i * lda, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
	}
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
}





void row_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream, int select) {
	switch(select) {
			case 1: row_recal_1(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
	}
}


void at_row_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream){

	// needs to do boundary check first


	//int i = abftEnv->mapping[(m / abftEnv->chk_nb) * abftEnv->mapping_ld + (n / abftEnv->chk_nb)];
	row_chk_recal_select(abftEnv, A, lda, m, n, stream, 1);

}