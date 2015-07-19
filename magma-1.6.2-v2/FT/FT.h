#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
void printMatrix_host(double * matrix_host, int M, int N);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N);
void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

void initializeChecksum(double * matrix, int ld, int N, int B, double * vd, int vd_ld, double * chksum, int chksum_ld);


void dpotrfFT(double * A, int lda, int n, int * info,
				double * chksum, int chksum_ld,
				double * v, double * v_ld, 
				bool FT , bool DEBUG);

void dtrsmFT(int m, int n, double * A, int lda,
		double * B, int ldb, double * checksumB1, int incB1,
		double * checksumB2, int incB2, double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG);

void dsyrkFT(int n, int m, double * A, int lda, double * C, int ldc,
		double * checksumA1, int incA1, double * checksumA2, int incA2,
		double * checksumC1, int incC1, double * checksumC2, int incC2,
		double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG);

void dgemmFT(int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, double * checksumA1,
		int incA1, double * checksumA2, int incA2, double * checksumC1,
		int incC1, double * checksumC2, int incC2,
		double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG);