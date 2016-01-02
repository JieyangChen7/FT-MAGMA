#include "magma.h"
//#include"FT.h"
using namespace std;
//initialize checksum
void initializeChecksum(double * matrix, int ld, int N, int B, double * vd, int vd_ld, double * chksum, int chksum_ld) {

	//cout<<"checksum vector on GPU:"<<endl;
	//printVector_gpu(vd,B);
	
//	double alpha = 1;
//	double beta = 0;
	for (int i = 0; i < N; i += B) {
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, i+B, B,
					MAGMA_D_ONE, vd, vd_ld,
					matrix + i, ld,
					MAGMA_D_ZERO,chksum + (i / B) * 2, chksum_ld);

	}

}