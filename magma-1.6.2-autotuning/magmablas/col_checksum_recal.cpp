#include "FT.h"
#include <iostream>


//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated
void chk_recal_1(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(stream[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
	}
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
}




//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated - 4 stream
void chk_recal_2(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb * 2) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
		if (i + chk_nb < m) {			
			magmablasSetKernelStream(stream[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
			magmablasSetKernelStream(stream[4]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
		}
	}
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
	cudaStreamSynchronize(stream[4]);
	
}


//col-read-A
//col-read-B
//non-col-write-C
//separated
void chk_recal_3(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(stream[1]);
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(stream[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
	}
	//cudaStreamSynchronize(stream[1]);
	
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
}

//col-read-A
//col-read-B
//non-col-write-C
//separated - 4 stream
void chk_recal_4(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb * 2) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
		if (i + chk_nb < m) {
			magmablasSetKernelStream(stream[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
			magmablasSetKernelStream(stream[2]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
		}
	}
	
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
	cudaStreamSynchronize(stream[4]);
}

//non-col-read-A
//col-read-B
//col-write-C
//separated
void chk_recal_5(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(stream[1]);
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(stream[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
	}
	//cudaStreamSynchronize(stream[1]);
	
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
}


//non-col-read-A
//col-read-B
//col-write-C
//separated - 4 stream
void chk_recal_6(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb * 2) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
		if (i + chk_nb < m) {
			magmablasSetKernelStream(stream[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb + 1) * chk1_ld, 1 );
			magmablasSetKernelStream(stream[4]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb + 1) * chk2_ld, 1 );
		}
	}
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
	cudaStreamSynchronize(stream[4]);
}


//col-read-A
//col-read-B
//col-write-C
//separated
void chk_recal_7(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(stream[1]);
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(stream[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
	}
	//cudaStreamSynchronize(stream[1]);
	
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
}



//col-read-A
//col-read-B
//col-write-C
//separated
void chk_recal_8(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
		if (i + chk_nb < m) {
			magmablasSetKernelStream(stream[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb + 1) * chk1_ld, 1 );
			magmablasSetKernelStream(stream[4]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb + 1) * chk2_ld, 1 );
		}
	}
	cudaStreamSynchronize(stream[1]);
	cudaStreamSynchronize(stream[2]);
	cudaStreamSynchronize(stream[3]);
	cudaStreamSynchronize(stream[4]);


}



//non-col-read-A
//col-read-B
//non-col-write-C
//combined
void chk_recal_9(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(stream[1]);
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, n, chk_nb,
					MAGMA_D_ONE, vd, vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2, chk1_ld);		
	}
	
	cudaStreamSynchronize(stream[1]);
}



//col-read-A
//col-read-B
//non-col-write-C
//combined
void chk_recal_10(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {


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
	 magmablasSetKernelStream(stream[1]);
	cudaError_t r;
	 for (int i = 0; i < m; i += chk_nb) {
		magma_dgemm(MagmaTrans, MagmaNoTrans,
					2, n, chk_nb,
					MAGMA_D_ONE, vd, vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2, chk1_ld);		
	 }
	cudaStreamSynchronize(stream[1]);
	r = cudaGetLastError();
	r = cudaStreamSynchronize(stream[1]);
	if (r != cudaSuccess) {
	 	cout << "cuda sync error" << endl;
	 	magma_queue_t newStream;
	 	magma_queue_create( &newStream );
	 	stream[1] = newStream;
	} else {
	 	cout << "sync success" << endl;
	}
}






//non-col-read-A
//col-read-B
//col-write-C
//combined
void chk_recal_11(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	magmablasSetKernelStream(stream[1]);
	for (int i = 0; i < m; i += chk_nb) {
		magma_dgemm(MagmaTrans, MagmaTrans,
					n, 2, chk_nb,
					MAGMA_D_ONE, 
					A + i, lda,
					vd, vd_ld,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2 * chk1_ld, chk1_ld);		
	}
	
	cudaStreamSynchronize(stream[1]);
}

//col-read-A
//col-read-B
//col-write-C
//combined
void recalculateChecksum12(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * stream) {

	magmablasSetKernelStream(stream[1]);
	for (int i = 0; i < m; i += chk_nb) {
		magma_dgemm(MagmaTrans, MagmaNoTrans,
					n, 2, chk_nb,
					MAGMA_D_ONE, 
					A + i, lda,
					vd, vd_ld,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2 * chk1_ld, chk1_ld);		
	}
	
	cudaStreamSynchronize(stream[1]);
}


void ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda, 
						   magma_queue_t * stream) {

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
					col_chk_recal_select(abftEnv, A, lda, i, j, stream, c);
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

void col_chk_recal_select(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream, int select) {

		switch(select) {
			case 1: chk_recal_1(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;

			case 2:	chk_recal_2(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;

			case 3: chk_recal_3(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;

			case 4: chk_recal_4(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;

			case 5: chk_recal_5(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
		
			case 6: chk_recal_6(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
		

			case 7: chk_recal_7(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
			
			case 8: chk_recal_8(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
			case 9: chk_recal_9(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;
		
			case 10: chk_recal_10(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;
		
			case 11: chk_recal_11(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
			
			case 12: chk_recal_12(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;

			default: cout << "selecting error" << endl;
		}
}


void at_col_chk_recal(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream){

	// needs to do boundary check first


	//int i = abftEnv->mapping[(m / abftEnv->chk_nb) * abftEnv->mapping_ld + (n / abftEnv->chk_nb)];
	col_chk_recal_select(abftEnv, A, lda, m, n, stream, 1);

}


void benchmark(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream){
	cout << "start banchmarking:" << endl;
	double benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {

			AutoTuneChecksumRecal(abftEnv, A, lda, i, j, stream);
		}

	}
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "auto tuning time: " << benchmark_time << endl;



	benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {

			ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 2);
		}

	}
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "naive tuning time: " << benchmark_time << endl;


	benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {

			ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 1);
		}

	}
	
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "hand tuning time: " << benchmark_time << endl;

	cout << "done benchmarking" << endl;
}

//check matrix A using checksums
void ABFTCheck(ABFTEnv * abftEnv, double * A, int lda, int m, int n, double * checksumA, int checksumA_ld, magma_queue_t * stream) {
	AutoTuneChecksumRecal(abftEnv, A, lda, m, n, stream);
	//do check here

}