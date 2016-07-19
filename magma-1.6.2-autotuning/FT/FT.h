#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
#include "common_magma.h"
#include <ctime>

/* i and j are block number */
#define COL_CHK_T(i_, j_) (abftEnv->col_dchk + (i_)*(abftEnv->chk_nb)*(abftEnv->col_dchk_ld) + (j_)*2)
#define COL_CHK(i_, j_) (abftEnv->col_dchk + (j_)*(abftEnv->chk_nb)*(abftEnv->col_dchk_ld) + (i_)*2)

#define ROW_CHK_T(i_, j_) (abftEnv->row_dchk + (i_)*2*(abftEnv->row_dchk_ld) + (j_)*(abftEnv->chk_nb))
#define ROW_CHK(i_, j_) (abftEnv->row_dchk + (j_)*2*(abftEnv->row_dchk_ld) + (i_)*(abftEnv->chk_nb))

struct ABFTEnv {

	/* mode */
	/* 1 ... Cholesky Decomposition
	 * 2 ... LU Decomposition
	 * 3 ... QR Decomposition
	 */
	int mode;


	/* checksum encoding unit, by defualt = nb */
	int chk_nb;

	/* checksum encoding size on CPU and GPU */
	int gpu_row;
	int gpu_col;
	int cpu_row;
	int cpu_col;

	/* checksum vector on CPU */
    double * v;
    int v_ld;

    double * v2;
    int v2_ld;

    /* checksum vector on GPU */
	double * vd;
	int vd_ld;

   	double * vd2;
   	int vd2_ld;

   	/* space allocated for checksum verification */
   	double * chk1;
   	int chk1_ld;

   	double * chk2;
   	int chk2_ld;

   	double * chk21;
   	int chk21_ld;

   	double * chk22;
   	int chk22_ld;

   	/* column checksums on CPU */
   	double * col_hchk;
    int col_hchk_ld;

    /* row checksums on CPU */
    double * row_hchk;
    int row_hchk_ld;

    /* column checksums on GPU */
    double * col_dchk;
    int col_dchk_ld;

    /* row checksums on GPU */
    double * row_dchk;
    int row_dchk_ld;

    /* CUDA streams for computation */
    magma_queue_t * stream;

    /* performance autotuning result */
    int * mapping;
    int mapping_ld;

    /* record the last time a block is checked */
    time_t * lastCheckTime;
    int lastCheckTime_ld;

    /* record the number of times a block is updated since last time checked */
    int * updatedCounter;
    int updatedCounter_ld;

    /* autotuning veriables */
    time_t T;
    int N;
};


void CholeskyGenerator(double * A, int lda, int n);

void printMatrix_host(double * matrix_host, int ld,  int M, int N, int row_block, int col_block);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N, int row_block, int col_block);

void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

void init_col_chk(ABFTEnv * abftEnv, double * A, int lda);
void init_row_chk(ABFTEnv * abftEnv, double * A, int lda);

void initializeABFTEnv(ABFTEnv * abftEnv, int chk_nb, 
						double * A, int lda,
						int gpu_row, int gpu_col,
						int cpu_row, int cpu_col,
						magma_queue_t * stream,
						int mode, 
						bool DEBUG);


void ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream);

void benchmark(ABFTEnv * abftEnv, double * A, int lda);

void col_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, int select);

void row_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, int select);

void at_col_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n);

void at_row_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n);

void ABFTCheck(ABFTEnv * abftEnv, double * A, int lda, int m, int n, double * checksumA, int checksumA_ld);

void MemoryErrorCheck(ABFTEnv * abftEnv, double * A, int lda);

bool updateCounter(ABFTEnv * abftEnv, int row1, int row2, int col1, int col2, int count);

void dpotrfFT(double * A, int lda, int n, int * info, ABFTEnv * abftEnv, bool FT , bool DEBUG, bool VERIFY);

void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * info, ABFTEnv * abftEnv, bool FT , bool DEBUG, bool VERIFY);

void dgeqrfFT( int m, int n, double * A, int lda, double * tau, double * work, int lwork, int *info );

void dtrsmFT(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
		int m, int n, double alpha, double * A, int lda,
		double * B, int ldb, 
		ABFTEnv * abftEnv,
		double * col_chkA, int col_chkA_ld,
		double * row_chkA, int row_chkA_ld,
		double * col_chkB, int col_chkB_ld,
		double * row_chkB, int row_chkB_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream);


void dsyrkFT(magma_uplo_t uplo, magma_trans_t trans,
		int n, int m, 
		double alpha,
		double * A, int lda,
		double beta,
		double * C, int ldc,
		ABFTEnv * abftEnv,
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
		double * col_chkA, int col_chkA_ld,
		double * row_chkA, int row_chkA_ld,
		double * col_chkB, int col_chkB_ld,
		double * row_chkB, int row_chkB_ld,
		double * col_chkC, int col_chkC_ld,
		double * row_chkC, int row_chkC_ld,
		bool FT, bool DEBUG, bool VERIFY, 
		magma_queue_t * stream);

void col_detect_correct(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld, 
		cudaStream_t stream);

void row_detect_correct(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld, 
		cudaStream_t stream);

void col_debug(double * A, int lda, int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld, 
		cudaStream_t stream);

void ErrorDetectAndCorrectHost(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * checksum1_recal, int checksum1_recal_ld,
		double * checksum2_recal, int checksum2_recal_ld);