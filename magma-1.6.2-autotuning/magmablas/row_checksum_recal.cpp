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
void row_recal_1(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	for (int i = 0; i < n; i += abftEnv->chk_nb) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaNoTrans,
					m, abftEnv->chk_nb, 
					MAGMA_D_ONE,
					A + i * lda, lda, 
					abftEnv->vd, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * abftEnv->chk21_ld, 1 );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaNoTrans, 
					m, abftEnv->chk_nb, 
					MAGMA_D_ONE,
					A + i * lda, lda, 
					abftEnv->vd + 1, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk22 + (i / abftEnv->chk_nb) * abftEnv->chk22_ld, 1 );
	}
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
}


//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated － 4 steams
void row_recal_2(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	for (int i = 0; i < n; i += abftEnv->chk_nb * 2) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaNoTrans, 
					m, abftEnv->chk_nb, 
					MAGMA_D_ONE,
					A + i * lda, lda, 
					abftEnv->vd, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * abftEnv->chk21_ld, 1 );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaNoTrans, 
					m, abftEnv->chk_nb, 
					MAGMA_D_ONE,
					A + i * lda, lda, 
					abftEnv->vd + 1, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk22 + (i / abftEnv->chk_nb) * abftEnv->chk22_ld, 1 );

		if (i + abftEnv->chk_nb < n) {
			magmablasSetKernelStream(abftEnv->stream[3]);
			magma_dgemv(MagmaNoTrans, 
						m, abftEnv->chk_nb, 
						MAGMA_D_ONE,
						A + (i + abftEnv->chk_nb) * lda, lda, 
						abftEnv->vd, abftEnv->vd_ld, 
						MAGMA_D_ZERO, 
						abftEnv->chk21 + ((i / abftEnv->chk_nb) + 1) * abftEnv->chk21_ld, 1 );

			magmablasSetKernelStream(abftEnv->stream[4]);
			magma_dgemv(MagmaNoTrans, 
						m, abftEnv->chk_nb, 
						MAGMA_D_ONE,
						A + (i + abftEnv->chk_nb) * lda, lda, 
						abftEnv->vd + 1, abftEnv->vd_ld, 
						MAGMA_D_ZERO, 
						abftEnv->chk22 + ((i / abftEnv->chk_nb) + 1) * abftEnv->chk22_ld, 1 );
		}
	}
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
	cudaStreamSynchronize(abftEnv->stream[3]);
	cudaStreamSynchronize(abftEnv->stream[4]);
}







void row_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, int select) {
	switch(select) {
			case 1: row_recal_1(abftEnv, A, lda, m, n);
					break;
	}
}


void at_row_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n){

	// needs to do boundary check first


	//int i = abftEnv->mapping[(m / abftEnv->chk_nb) * abftEnv->mapping_ld + (n / abftEnv->chk_nb)];
	row_chk_recal_select(abftEnv, A, lda, m, n, 1);

}