#include "magma.h"
#include "magma_lapack.h"
#include "magmablas.h"
#include "common_magma.h"
#include  <ctime>

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
    double * hrz_v;
    int hrz_v_ld;

    double * vrt_v;
    int vrt_v_ld;

    /* checksum vector on GPU */
	double * hrz_vd;
	int hrz_vd_ld;

   	double * vrt_vd;
   	int vrt_vd_ld;

   	/* space allocated for checksum verification */
   	double * hrz_recal_chk;
   	int hrz_recal_chk_ld;

   	double * vrt_recal_chk;
   	int vrt_recal_chk_ld;

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
    int * col_mapping;
    int col_mapping_ld;

	int * row_mapping;
    int row_mapping_ld;


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



void compareChk(double * chk1, int ldchk1, double * chk2, int ldchk2, int m, int n);

void col_chkenc(double * A, int lda, int m, int n, int nb, double * Chk , int ldchk, magma_queue_t stream);


void row_chkenc(double * A, int lda, int m, int n, int nb, double * Chk , int ldchk, magma_queue_t stream);

void col_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * dcolchk, int ld_dcolchk, 
                 magma_queue_t stream);

void row_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * drowchk, int ld_drowchk, 
                 magma_queue_t stream);

void check_row(int m, int n, int nb,
               double * A, int lda,
               double * chk_v, int ld_chk_v,
               double * rowchkA, int ld_rowchkA, 
               double * rowchkA_r, int ld_rowchkA_r,
               magma_queue_t stream,
               bool DEBUG, char[] s);

void check_col(int m, int n, int nb,
               double * A, int lda,
               double * chk_v, int ld_chk_v,
               double * colchkA, int ld_colchkA, 
               double * colchkA_r, int ld_colchkA_r,
               magma_queue_t stream,
               bool DEBUG, char[] s);



void row_checksum_kernel_cccs4(int m, int n, int chk_nb, 
			   	 		   double * A, int lda, 
			     		   double * vrt_vd, int vrt_vd_ld, 
			     		   double * vrt_chk, int vrt_chk_ld, 
			     		   magma_queue_t * stream);

void col_checksum_kernel_ccns4(int m, int n, int chk_nb, 
						   double * A, int lda, 
						   double * vrt_vd, int vrt_vd_ld, 
						   double * hrz_chk, int hrz_chk_ld, 
						   magma_queue_t * stream);


void CholeskyGenerator(double * A, int lda, int n);

void printMatrix_host(double * matrix_host, int ld,  int M, int N, int row_block, int col_block);
void printMatrix_host_int(int * matrix_host, int ld,  int M, int N, int row_block, int col_block);
void printMatrix_host_time(time_t * matrix_host, int ld,  int M, int N, int row_block, int col_block);
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


void col_ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda);

void col_benchmark(ABFTEnv * abftEnv, double * A, int lda);
void col_benchmark_single(ABFTEnv * abftEnv, double * A, int lda);


void row_ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda);

void row_benchmark(ABFTEnv * abftEnv, double * A, int lda);
void row_benchmark_single(ABFTEnv * abftEnv, double * A, int lda);


void col_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, int select);

void row_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, int select);

void at_col_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n);

void at_row_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n);

void ABFTCheck(ABFTEnv * abftEnv, double * A, int lda, int m, int n, double * checksumA, int checksumA_ld);

void MemoryErrorCheck(ABFTEnv * abftEnv, double * A, int lda);

bool ComputationCheck(ABFTEnv * abftEnv, int row1, int row2, int col1, int col2, int count);

bool MemoryCheck(ABFTEnv * abftEnv, int row1, int row2, int col1, int col2);

void dpotrfFT(const char uplo, int n, double * A, int lda, int * info, 
			  int nb, 
			  double * colchk, int ld_colchk, 
			  double * rowchk, int ld_rowchk, 
			  double * chk_v, int ld_chk_v, 
			  bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);


void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * info,
              ABFTEnv * abftEnv,
              bool FT , bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);

void dgeqrfFT( int m, int n, double * A, int lda, double * tau, double * work, int lwork, int * info,
			   ABFTEnv * abftEnv, 
			   bool FT , bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);

void dtrsmFT(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
		int m, int n, double alpha, double * A, int lda,
		double * B, int ldb, 
		ABFTEnv * abftEnv,
		double * col_chkA, int col_chkA_ld,
		double * row_chkA, int row_chkA_ld, 
		double * col_chkB, int col_chkB_ld,
		double * row_chkB, int row_chkB_ld, 
		bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
		magma_queue_t * stream);

void dtrsmFT_work(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    		      magma_int_t m, magma_int_t n,
    		      double alpha, 
    		      magmaDouble_const_ptr dA, magma_int_t ldda,
        		  magmaDouble_ptr       dB, magma_int_t lddb,
        		  magma_int_t flag, magmaDouble_ptr d_dinvA, magmaDouble_ptr dX,
                  int nb,
    		      double * colchkA,   int ld_colchkA,
                  double * colchkA_r, int ld_colchkA_r,
    		      double * rowchkA,   int ld_rowchkA, 
                  double * rowchkA_r, int ld_rowchkA_r, 
    		      double * colchkB,   int ld_colchkB,
                  double * colchkB_r, int ld_colchkB_r,
    		      double * rowchkB,   int ld_rowchkB, 
                  double * rowchkB_r, int ld_rowchkB_r, 
                  double * chk_v,     int ld_chk_v, 
    		      bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
    		      magma_queue_t * stream);


void dsyrkFT(magma_uplo_t uplo, magma_trans_t trans,
			 int n, int k, 
			 double alpha,
			 double * dA, int ldda,
			 double beta,
			 double * dC, int lddc,
			 double * colchkA, int ld_colchkA,
			 double * colchkC, int ld_colchkC, 
			 double * chk_v, int ld_chk_v, 
			 bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
			 magma_queue_t * stream);

void dgemmFT( magma_trans_t transA, magma_trans_t transB,
			  int m, int n, int k, 
			  double alpha, 
			  double * dA, int ldda,
		      double * dB, int lddb, 
			  double beta, 
			  double * dC, int lddc, 
			  double * col_chkA,   int ld_col_chkA,
			  double * col_chkA_r, int ld_col_chkA_r,
			  double * rowchkA,    int ld_row_chkA,	
			  double * rowchkA_r,  int ld_row_chkA_r,	
			  double * colchkB,    int ld_col_chkB,
			  double * colchkB_r,  int ld_col_chkB_r,
			  double * rowchkB,    int ld_row_chkB, 
			  double * rowchkB_r,  int ld_row_chkB_r,
			  double * colchkC,    int ld_col_chkC,
			  double * colchkC_r,  int ld_col_chkC_r,  
			  double * rowchkC,    int ld_row_chkC,
			  double * rowchkC_r,  int ld_row_chkC_r,
			  double * chk_v, int ld_chk_v, 
			  bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, bool INJECT,
			  magma_queue_t * stream);

void dtrmmFT( magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    		int m, int n,
    		double alpha,
    		double * dA, int ldda,
    		double * dB, int lddb,
    		ABFTEnv * abftEnv,
			double * col_chkA, int col_chkA_ld,
			double * row_chkA, int row_chkA_ld,			
			double * col_chkB, int col_chkB_ld,
			double * row_chkB, int row_chkB_ld, 
			bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
			magma_queue_t * stream);

int dlarfbFT( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    						int m, int n, int k,
						  	double * dV, int lddv,
						  	double * dT, int lddt,
						  	double * dC, int lddc,
						  	double * dwork, int ldwork,
						  	ABFTEnv * abftEnv,
						  	double * col_chkV, int col_chkV_ld,
							double * row_chkV, int row_chkV_ld, 	
							double * col_chkT, int col_chkT_ld,
							double * row_chkT, int row_chkT_ld, 
							double * col_chkC, int col_chkC_ld,  
							double * row_chkC, int row_chkC_ld, 
							double * col_chkW, int col_chkW_ld,  
							double * row_chkW, int row_chkW_ld,
							bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER, bool INJECT,
							magma_queue_t * stream);

void col_detect_correct(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * hrz_recal_chk, int hrz_recal_chk_ld,
		cudaStream_t stream);

void row_detect_correct(double * A, int lda, 
		int B, int m, int n,
		double * checksum_update, int checksum_update_ld,
		double * vrt_recal_chk, int vrt_recal_chk_ld,
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