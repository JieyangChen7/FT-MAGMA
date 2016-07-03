#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
#include "common_magma.h"
#include <ctime>

#define CHK_T(i_, j_) (abftEnv->checksum + (i_)*(abftEnv->chk_nb)*(abftEnv->checksum_ld) + (j_)*2)
#define CHK(i_, j_) (abftEnv->checksum + (j_)*(abftEnv->chk_nb)*(abftEnv->checksum_ld) + (i_)*2)

struct ABFTEnv {
	int gpu_m;
	int gpu_n;
	int cpu_m;
	int cpu_n;


    double * v;
    int v_ld;

    double * v2;
    int v2_ld;

	double * vd;
	int vd_ld;

   	double * vd2;
   	int vd2_ld;

   	double * chk1;
   	int chk1_ld;

   	double * chk2;
   	int chk2_ld;

   	double * chk21;
   	int chk21_ld, 

   	double * chk22;
   	int chk22_ld;

   	double * work_chk;
    int work_chk_ld;

    double * checksum;
    int checksum_ld;

    int * mapping;
    int mapping_ld;

    time_t * lastCheckTime;
    int lastCheckTime_ld;

    int * updatedCounter;
    int updatedCounter_ld;
}


void printMatrix_host(double * matrix_host, int ld,  int M, int N);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N);
void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

void initializeChecksum(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream);

void initializeABFTEnv(ABFTEnv * abftEnv, int chk_nb,
						int gpu_m, int gpu_n,
						int cpu_m, int cpu_n);


void recalculateChecksum(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream);

void recalculateChecksum2(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream);

void ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream);

void benchmark(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream);

void ChecksumRecalSelector(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream, int select);

void AutoTuneChecksumRecal(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream);

void dpotrfFT(double * A, int lda, int n, int * info, ABFTEnv * abftEnv, bool FT , bool DEBUG, bool VERIFY);

void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * info, ABFTEnv * abftEnv, bool FT , bool DEBUG, bool VERIFY);

void dtrsmFT(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
		int m, int n, double alpha, double * A, int lda,
		double * B, int ldb, 
		ABFTEnv * abftEnv,
		double * checksumB, int checksumB_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream);


void dsyrkFT(magma_uplo_t uplo, magma_trans_t trans,
		int n, int m, 
		double alpha,
		double * A, int lda,
		double beta,
		double * C, int ldc,
		ABFTEnv * ABFTEnv,
		double * checksumA, int checksumA_ld,
		double * checksumC, int checksumC_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream);

void dgemmFT( magma_trans_t transA, magma_trans_t transB,
	    int m, int n, int k, 
	    double alpha, 
	    double * A, int lda,
		double * B, int ldb, 
		double beta, 
		double * C, int ldc, 
		ABFTEnv * abftEnv,
		double * checksumA, int checksumA_ld,
		double * checksumB, int checksumB_ld,
		double * checksumC, int checksumC_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream);

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