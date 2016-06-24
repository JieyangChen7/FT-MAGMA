#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
#include "common_magma.h"
void printMatrix_host(double * matrix_host, int ld,  int M, int N);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N);
void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

void initializeChecksum(double * matrix, int ld,
                        int M, int N, int B,
                        double * vd, int vd_ld,
                        double * v, int v_ld,
                        double * chksum, int chksum_ld, magma_queue_t * streams);

void recalculateChecksum(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams);

void recalculateChecksum2(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams);

void benchmark(double * A, int lda,
			   int m, int n, int chk_nb,
			   double * vd, int vd_ld,
			   double * vd2, int vd2_ld,
			   double * chk1, int chk1_ld, 
			   double * chk2, int chk2_ld, 
			   double * chk21, int chk21_ld, 
			   double * chk22, int chk22_ld, 
			   magma_queue_t * streams
			   ) ;

void dpotrfFT(double * A, int lda, int n, int * info,
				double * chksum, int chksum_ld,
				double * v, int v_ld, 
				bool FT , bool DEBUG, bool VERIFY);

void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * info,
			  int nb,
              double * chksum, int chksum_ld,
              double * v, int v_ld,
              bool FT , bool DEBUG, bool VERIFY);

void dtrsmFT(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
		int m, int n, 
		double alpha,
		double * A, int lda,
		double * B, int ldb, 
		int chk_nb,
		int nb,
		double * checksumB, int checksumB_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * streams);

void dsyrkFT(magma_uplo_t uplo, magma_trans_t trans,
		int n, int m, 
		double alpha,
		double * A, int lda,
		double beta,
		double * C, int ldc,
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld,
		double * chk2, int chk2_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * streams);

void dgemmFT(magma_trans_t transA, magma_trans_t transB,
		int m, int n, int k, 
		double alpha, 
		double * A, int lda,
		double * B, int ldb, 
		double beta, 
		double * C, int ldc, 
		int chk_nb,
		int nb,
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * streams);

void ErrorDetectAndCorrect(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld, 
		cudaStream_t stream);

void ErrorDetectAndCorrectHost(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld);