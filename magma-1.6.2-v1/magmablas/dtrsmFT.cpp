#include"FT.h"
#include<iostream>
using namespace std;
//TRSM with FT on GPU using cuBLAS
/*
__global__ void detectAndCorrectForTrsm(double * B, int ldb, int n,
		double * chksumB1, int incB1, double * chksumB2, int incB2,
		double * chkB1, int incB1_2, double * chkB2, int incB2_2) {
	//determin the reponsisble column 
	int block = blockIdx.x;
	int col = threadIdx.x;
	double diff = abs(
			*(chkB1 + block + col * incB1_2)
					- *(chksumB1 + block + col * incB1));
	if (diff > 0.1) {
		double diff2 = abs(
				*(chkB2 + block + col * incB2_2)
						- *(chksumB2 + block + col * incB2));
		int row = (int) round(diff2 / diff) - 1;
		*(B + n * block + row + col * ldb) += *(chksumB1 + block + col * incB1)
				- *(chkB1 + block + col * incB1_2);
	}
}
*/
/*
 * m: number of row of B
 * n: number of col of B
 */

void dtrsmFT(int m, int n, double * A, int lda,
		double * B, int ldb, double * checksumB1, int incB1,
		double * checksumB2, int incB2, double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG) {

	/*cout<<"matrix A before dtrsm:"<<endl;
	 printMatrix_gpu(A,lda*sizeof(double),n,n);
	 
	 cout<<"checksum1 of B before dtrsm:"<<endl;
	 printMatrix_gpu(checksumB1,incB1*sizeof(double),m/n,n);
	 cout<<"checksum2 of B before dtrsm:"<<endl;
	 printMatrix_gpu(checksumB2,incB2*sizeof(double),m/n,n);
	 */

	double alpha = 1;
	magma_dtrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
	                                m, n,
	                                MAGMA_D_ONE, A, lda,
	                                       B, ldb);
//	
//	cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
//			CUBLAS_DIAG_NON_UNIT, m, n, &alpha, A, lda, B, ldb);

	/*cout<<"matrix A after dtrsm:"<<endl;
	 printMatrix_gpu(A,lda*sizeof(double),n,n);
	 */

	if (FT) {
		//recalculate checksum1 and checksum2
		double beta = 0;
//		for (int i = 0; i < m; i += n) {
//			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
//					B + i, ldb, v1d, 1, MAGMA_D_ZERO, chk1 + (i / n), chk1_ld );
//			magma_dgemv(MagmaTrans, n, n, MAGMA_D_ONE,
//					B + i, ldb, v2d, 1, MAGMA_D_ZERO, chk2 + (i / n), chk2_ld );
////			cublasDgemv(handle, CUBLAS_OP_T, n, n, &alpha, B + i, ldb, v1d, 1,
////					&beta, chk1 + (i / n), chk1_ld);
////			cublasDgemv(handle, CUBLAS_OP_T, n, n, &alpha, B + i, ldb, v2d, 1,
////					&beta, chk2 + (i / n), chk2_ld);
//		}
		
		//update checksum1 and checksum2
//		magma_dtrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
//			                                m / n, n,
//			                                MAGMA_D_ONE, A, lda,
//			                                checksumB1, incB1);
//		magma_dtrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
//			                                m / n, n,
//			                                MAGMA_D_ONE, A, lda,
//			                                checksumB2, incB2);
//		cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
//				CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m / n, n, &alpha, A, lda,
//				checksumB1, incB1);
//		cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
//				CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m / n, n, &alpha, A, lda,
//				checksumB2, incB2);
		
		//cudaStream_t stream1;
		//cublasGetStream(handle, &stream1);
		//cudaStreamSynchronize(stream1);

		if (DEBUG) {
			cout<<"recalculated checksum1 of B after dtrsm:"<<endl;
			printMatrix_gpu(chk1, chk1_ld, m / n, n);
			cout<<"recalculated checksum2 of B after dtrsm:"<<endl;
			printMatrix_gpu(chk2, chk2_ld, m / n, n);
					 
			cout<<"updated checksum1 of B after dtrsm:"<<endl;
			printMatrix_gpu(checksumB1, incB1, m / n, n);
			cout<<"updated checksum2 of B after dtrsm:"<<endl;
			printMatrix_gpu(checksumB2, incB2, m / n, n);
		}
		/*detectAndCorrectForTrsm<<<dim3(m/n),dim3(n)>>>(B, ldb, n,
			checksumB1, incB1, checksumB2, incB2,
			chk1, chk1_ld, chk2, chk2_ld);
		*/
	}
}