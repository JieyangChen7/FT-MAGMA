#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
#include "common_magma.h"
void printMatrix_host(double * matrix_host, int ld,  int M, int N);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N);
void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

void initializeChecksum(double * matrix, int ld,
		int N, int B,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chksum, int chksum_ld, magma_queue_t * streams);


void dpotrfFT(double * A, int lda, int n, int * info,
				double * chksum, int chksum_ld,
				double * v, int v_ld, 
				bool FT , bool DEBUG, bool VERIFY);

void dtrsmFT(int m, int n, double * A, int lda,
		double * B, int ldb, double * checksumB, int checksumB_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		bool FT, bool DEBUG, bool VERIFY, magma_queue_t * streams);

void dsyrkFT(int n, int m, double * A, int lda, double * C, int ldc,
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chk1, int chk1_ld,
		double * chk2, int chk2_ld,
		magma_queue_t * streams,
		bool FT, bool DEBUG, bool VERIFY);

void dgemmFT(int m, int n, int k, double * A, int lda,
		double * B, int ldb, double * C, int ldc, 
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams,
		bool FT, bool DEBUG, bool VERIFY);