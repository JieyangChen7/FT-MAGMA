#include "magma.h"
#include"FT.h"
using namespace std;
//initialize column/row checksum
void initializeChecksum(char ColumnOrRow, double * matrix, int ld, int N, int B, int k, 
		double * vd, int vd_ld, 
		double * v, int v_ld, 
		double * chksum, int chksum_ld) {
	
	if (ColumnOrRow == 'C') { // initialize column checksum
		for (int i = 0; i < N; i += B) {
			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
						k, i + B, B,
						MAGMA_D_ONE, vd, vd_ld,
						matrix + i, ld,
						MAGMA_D_ZERO, chksum + (i / B) * k, chksum_ld);	
		}
	} else if (ColumnOrRow == 'R') { // initialize row checksum 
		for (int i = 0; i < N; i += B) {
			magma_dgemm(MagmaNoTrans, MagmaTrans,
						N - i, k, B,
						MAGMA_D_ONE, vd, vd_ld,
						matrix + i * ld, ld,
						MAGMA_D_ZERO, chksum + (i / B) * k * chksum_ld, chksum_ld);	
		}
	} else {
		return;
	}
}
