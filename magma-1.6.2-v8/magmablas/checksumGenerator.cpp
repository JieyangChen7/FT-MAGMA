#include "magma.h"
#include"FT.h"
using namespace std;
//initialize checksum
void initializeChecksum(double * matrix, int ld, int N, int B, int k, 
		double * vd, int vd_ld, 
		double * v, int v_ld, 
		double * chksum, int chksum_ld) {
	for (int i = 0; i < N; i += B) {
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					k, i + B, B,
					MAGMA_D_ONE, vd, vd_ld,
					matrix + i, ld,
					MAGMA_D_ZERO, chksum + (i / B) * k, chksum_ld);	
	}
}