#include "FT.h"
#include <iostream>
using namespace std;

//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated
void chk_recal_1(ABFTEnv * abftEnv, double * A, int lda,int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda,
					abftEnv->vd, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk1 + (i / abftEnv->chk_nb), abftEnv->chk1_ld );

		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd + 1, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk2 + (i / abftEnv->chk_nb), abftEnv->chk2_ld );
	}
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
}




//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated - 4 stream
void chk_recal_2(ABFTEnv * abftEnv, double * A, int lda,int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb * 2) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk1 + (i / abftEnv->chk_nb), abftEnv->chk1_ld );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd + 1, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk2 + (i / abftEnv->chk_nb), abftEnv->chk2_ld );

		if (i + abftEnv->chk_nb < m) {			
			magmablasSetKernelStream(abftEnv->stream[3]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd, abftEnv->vd_ld, 
						MAGMA_D_ZERO, 
						abftEnv->chk1 + (i / abftEnv->chk_nb) + 1, abftEnv->chk1_ld );

			magmablasSetKernelStream(abftEnv->stream[4]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd + 1, abftEnv->vd_ld, 
						MAGMA_D_ZERO, 
						abftEnv->chk2 + (i / abftEnv->chk_nb) + 1, abftEnv->chk2_ld );
		}
	}
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
	cudaStreamSynchronize(abftEnv->stream[3]);
	cudaStreamSynchronize(abftEnv->stream[4]);
	
}


//col-read-A
//col-read-B
//non-col-write-C
//separated
void chk_recal_3(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk1 + (i / abftEnv->chk_nb), abftEnv->chk1_ld );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2 + abftEnv->vd2_ld, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk2 + (i / abftEnv->chk_nb), abftEnv->chk2_ld );
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
}

//col-read-A
//col-read-B
//non-col-write-C
//separated - 4 stream
void chk_recal_4(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb * 2) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk1 + (i / abftEnv->chk_nb), abftEnv->chk1_ld );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2 + abftEnv->vd2_ld, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk2 + (i / abftEnv->chk_nb), abftEnv->chk2_ld );

		if (i + abftEnv->chk_nb < m) {
			magmablasSetKernelStream(abftEnv->stream[3]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd2, 1, 
						MAGMA_D_ZERO, 
						abftEnv->chk1 + (i / abftEnv->chk_nb) + 1, abftEnv->chk1_ld );

			magmablasSetKernelStream(abftEnv->stream[4]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd2 + abftEnv->vd2_ld, 1, 
						MAGMA_D_ZERO, 
						abftEnv->chk2 + (i / abftEnv->chk_nb) + 1, abftEnv->chk2_ld );
		}
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
	cudaStreamSynchronize(abftEnv->stream[3]);
	cudaStreamSynchronize(abftEnv->stream[4]);
}

//non-col-read-A
//col-read-B
//col-write-C
//separated
void chk_recal_5(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * abftEnv->chk21_ld, 1 );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd + 1, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk22 + (i / abftEnv->chk_nb) * abftEnv->chk22_ld, 1 );
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
}


//non-col-read-A
//col-read-B
//col-write-C
//separated - 4 stream
void chk_recal_6(ABFTEnv * abftEnv, double * A, int lda,int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb * 2) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * abftEnv->chk21_ld, 1 );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd + 1, abftEnv->vd_ld, 
					MAGMA_D_ZERO, 
					abftEnv->chk22 + (i / abftEnv->chk_nb) * abftEnv->chk22_ld, 1 );

		if (i + abftEnv->chk_nb < m) {
			magmablasSetKernelStream(abftEnv->stream[3]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd, abftEnv->vd_ld, 
						MAGMA_D_ZERO, 
						abftEnv->chk21 + (i / abftEnv->chk_nb + 1) * abftEnv->chk21_ld, 1 );

			magmablasSetKernelStream(abftEnv->stream[4]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd + 1, abftEnv->vd_ld, 
						MAGMA_D_ZERO, 
						abftEnv->chk22 + (i / abftEnv->chk_nb + 1) * abftEnv->chk22_ld, 1 );
		}
	}
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
	cudaStreamSynchronize(abftEnv->stream[3]);
	cudaStreamSynchronize(abftEnv->stream[4]);
}


//col-read-A
//col-read-B
//col-write-C
//separated
void chk_recal_7(ABFTEnv * abftEnv, double * A, int lda,int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * abftEnv->chk21_ld, 1 );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2 + abftEnv->vd2_ld, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk22 + (i / abftEnv->chk_nb) * abftEnv->chk22_ld, 1 );
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
}



//col-read-A
//col-read-B
//col-write-C
//separated - 4 streams
void chk_recal_8(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magmablasSetKernelStream(abftEnv->stream[1]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * abftEnv->chk21_ld, 1 );

		magmablasSetKernelStream(abftEnv->stream[2]);
		magma_dgemv(MagmaTrans, 
					abftEnv->chk_nb, n, 
					MAGMA_D_ONE,
					A + i, lda, 
					abftEnv->vd2 + abftEnv->vd2_ld, 1, 
					MAGMA_D_ZERO, 
					abftEnv->chk22 + (i / abftEnv->chk_nb) * abftEnv->chk22_ld, 1 );

		if (i + abftEnv->chk_nb < m) {
			magmablasSetKernelStream(abftEnv->stream[3]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd2, 1, 
						MAGMA_D_ZERO, 
						abftEnv->chk21 + (i / abftEnv->chk_nb + 1) * abftEnv->chk21_ld, 1 );

			magmablasSetKernelStream(abftEnv->stream[4]);
			magma_dgemv(MagmaTrans, 
						abftEnv->chk_nb, n, 
						MAGMA_D_ONE,
						A + i + abftEnv->chk_nb, lda, 
						abftEnv->vd2 + abftEnv->vd2_ld, 1,
						MAGMA_D_ZERO, 
						abftEnv->chk22 + (i / abftEnv->chk_nb + 1) * abftEnv->chk22_ld, 1 );
		}
	}
	cudaStreamSynchronize(abftEnv->stream[1]);
	cudaStreamSynchronize(abftEnv->stream[2]);
	cudaStreamSynchronize(abftEnv->stream[3]);
	cudaStreamSynchronize(abftEnv->stream[4]);


}



//non-col-read-A
//col-read-B
//non-col-write-C
//combined
void chk_recal_9(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	magmablasSetKernelStream(abftEnv->stream[1]);
	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, n, abftEnv->chk_nb,
					MAGMA_D_ONE, 
					abftEnv->vd, abftEnv->vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, 
					abftEnv->chk1 + (i / abftEnv->chk_nb) * 2, abftEnv->chk1_ld);		
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
}



//col-read-A
//col-read-B
//non-col-write-C
//combined
void chk_recal_10(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {


	// testing
	// cublasHandle_t handle;
	// cublasCreate(&handle);
	// cublasSetStream(handle, stream[2]);

	// double one = 1;
	// double zero = 0;

	// cublasOperation_t T = CUBLAS_OP_T;
	// cublasOperation_t N = CUBLAS_OP_N;
	// cublasStatus_t result;
	// result = cublasDgemm(handle,
	// 			T, N,
	// 			2, 15360, 512,
	// 			&one, A, lda,
	// 			A, lda,
	// 			&zero, A, lda);	
	// if (result == CUBLAS_STATUS_SUCCESS) {
	// 	cout << "cublas ok" << endl;
	// } else if (result == CUBLAS_STATUS_NOT_INITIALIZED) {
	// 	cout << "CUBLAS_STATUS_NOT_INITIALIZED" << endl;
	// } else if (result == CUBLAS_STATUS_INVALID_VALUE) {
	// 	cout << "CUBLAS_STATUS_INVALID_VALUE" << endl;
	// } else if (result == CUBLAS_STATUS_ARCH_MISMATCH) {
	// 	cout << "CUBLAS_STATUS_ARCH_MISMATCH" << endl;
	// } else if (result == CUBLAS_STATUS_EXECUTION_FAILED) {
	// 	cout << "CUBLAS_STATUS_EXECUTION_FAILED" << endl;
	// }
	cout << "size:" << m << "," << n << endl;
	 magmablasSetKernelStream(abftEnv->stream[1]);
	cudaError_t r;
	 for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magma_dgemm(MagmaTrans, MagmaNoTrans,
					2, n, abftEnv->chk_nb,
					MAGMA_D_ONE, 
					abftEnv->vd2, abftEnv->vd2_ld,
					A + i, lda,
					MAGMA_D_ZERO, 
					abftEnv->chk1 + (i / abftEnv->chk_nb) * 2, abftEnv->chk1_ld);		
	 }
	cudaStreamSynchronize(abftEnv->stream[1]);
	r = cudaGetLastError();
	r = cudaStreamSynchronize(abftEnv->stream[1]);
	if (r != cudaSuccess) {
	 	cout << "cuda sync error" << endl;
	 	magma_queue_t newStream;
	 	magma_queue_create( &newStream );
	 	abftEnv->stream[1] = newStream;
	} else {
	 	cout << "sync success" << endl;
	}
}






//non-col-read-A
//col-read-B
//col-write-C
//combined
void chk_recal_11(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	magmablasSetKernelStream(abftEnv->stream[1]);
	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magma_dgemm(MagmaTrans, MagmaTrans,
					n, 2, abftEnv->chk_nb,
					MAGMA_D_ONE, 
					A + i, lda,
					abftEnv->vd, abftEnv->vd_ld,
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * 2 * abftEnv->chk21_ld, abftEnv->chk21_ld);		
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
}

//col-read-A
//col-read-B
//col-write-C
//combined
void chk_recal_12(ABFTEnv * abftEnv, double * A, int lda, int m, int n) {

	magmablasSetKernelStream(abftEnv->stream[1]);
	for (int i = 0; i < m; i += abftEnv->chk_nb) {
		magma_dgemm(MagmaTrans, MagmaNoTrans,
					n, 2, abftEnv->chk_nb,
					MAGMA_D_ONE, 
					A + i, lda,
					abftEnv->vd2, abftEnv->vd2_ld,
					MAGMA_D_ZERO, 
					abftEnv->chk21 + (i / abftEnv->chk_nb) * 2 * abftEnv->chk21_ld, abftEnv->chk21_ld);		
	}
	
	cudaStreamSynchronize(abftEnv->stream[1]);
}


void ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda) {

	double gpu_time = 0.0;
	double min_time = 1000;
	int min_choice = 0;
	int num_choice = 12;
	int num_test = 1;
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {
		cout << "[" << i << "]:	";
		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {
			for (int c = 0; c < num_choice; c++) {
				gpu_time = magma_wtime();
				for (int t = 0; t < num_test; t++) {
					col_chk_recal_select(abftEnv, A, lda, i, j, c);
				}
				gpu_time = magma_wtime() - gpu_time;
				if (min_time > gpu_time) {
					min_time = gpu_time;
					min_choice = c;
				}
			}
			cout << min_choice << "	";
			abftEnv->mapping[(i / abftEnv->chk_nb) * abftEnv->mapping_ld + j /abftEnv->chk_nb] = min_choice;
		}
		cout << endl;
	}
}

void col_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, int select) {

		switch(select) {
			case 1: chk_recal_1(abftEnv, A, lda, m, n);
					break;

			case 2:	chk_recal_2(abftEnv, A, lda, m, n);
					break;

			case 3: chk_recal_3(abftEnv, A, lda, m, n);
					break;

			case 4: chk_recal_4(abftEnv, A, lda, m, n);
					break;

			case 5: chk_recal_5(abftEnv, A, lda, m, n);
					break;
		
			case 6: chk_recal_6(abftEnv, A, lda, m, n);
					break;
		
			case 7: chk_recal_7(abftEnv, A, lda, m, n);
					break;
			
			case 8: chk_recal_8(abftEnv, A, lda, m, n);
					break;

			case 9: chk_recal_9(abftEnv, A, lda, m, n);
					break;
		
			case 10: chk_recal_10(abftEnv, A, lda, m, n);
					break;
		
			case 11: chk_recal_11(abftEnv, A, lda, m, n);
					break;
			
			case 12: chk_recal_12(abftEnv, A, lda, m, n);
					break;

			default: cout << "selecting error" << endl;
		}
}


void at_col_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n){

	// needs to do boundary check first


	//int i = abftEnv->mapping[(m / abftEnv->chk_nb) * abftEnv->mapping_ld + (n / abftEnv->chk_nb)];
	col_chk_recal_select(abftEnv, A, lda, m, n, 1);

}


void benchmark(ABFTEnv * abftEnv, double * A, int lda){
	cout << "start banchmarking:" << endl;
	double benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {

			at_col_chk_recal(abftEnv, A, lda, i, j);
		}

	}
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "auto tuning time: " << benchmark_time << endl;



	benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {

			col_chk_recal_select(abftEnv, A, lda, i, j, 2);
		}

	}
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "naive tuning time: " << benchmark_time << endl;


	benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {

			col_chk_recal_select(abftEnv, A, lda, i, j, 1);
		}

	}
	
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "hand tuning time: " << benchmark_time << endl;

	cout << "done benchmarking" << endl;
}

//check matrix A using checksums
void ABFTCheck(ABFTEnv * abftEnv, double * A, int lda, int m, int n, double * checksumA, int checksumA_ld) {
	at_col_chk_recal(abftEnv, A, lda, m, n);
	//do check here

}