#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
#include "cublas_v2.h"
//#include "acml.h"
void printMatrix_host(double * matrix_host, int M, int N);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N);
void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

void initializeChecksum(double * matrix, int ld, int N, int B, double * vd, double * chksum, int chksum_ld);


void dpotrfFT(double * A, int lda, int n, int * info,
		double * chksum1, int inc1, double * chksum2, int inc2, 
		double * v1, double * v2, bool FT , bool DEBUG);

void dtrsmFT(int m, int n, double * A, int lda,
		double * B, int ldb, double * checksumB1, int incB1,
		double * checksumB2, int incB2, double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG);

void dsyrkFT(cublasHandle_t handle, int n, int m, double * A, int lda, double * C, int ldc,
		double * checksumA1, int incA1, double * checksumA2, int incA2,
		double * checksumC1, int incC1, double * checksumC2, int incC2,
		double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG);

void dgemmFT(cublasHandle_t handle, int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, double * checksumA1,
		int incA1, double * checksumA2, int incA2, double * checksumC1,
		int incC1, double * checksumC2, int incC2,
		double * v1d, double * v2d,
		double * chk1, int chk1_ld, double * chk2, int chk2_ld, bool FT, bool DEBUG);