#include "magma.h"
#include "FT.h"
#include <iostream>
#include <cmath>
#include "cuda_profiler_api.h"
#include "cublas_v2.h"
using namespace std;
//initialize checksum
//M: number of rows
//N: numner of cols
void initializeChecksum(double * matrix, int ld,
		int M, int N, int B,
		double * vd, int vd_ld,
		double * v, int v_ld,
		double * chksum, int chksum_ld, magma_queue_t * streams) {


	
	for (int i = 0; i < N; i += B) {		
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, N, B,
					//2, B + i, B,
					MAGMA_D_ONE, vd, vd_ld,
					matrix + i, ld,
					MAGMA_D_ZERO, chksum + (i / B) * 2, chksum_ld);			
	}
	
	
//	
//	double * chk1d;
//	double * chk2d;
//	size_t chk1d_pitch;
//	size_t chk2d_pitch;
//	int chk1d_ld;
//	int chk2d_ld;
//	
//	//allocate space for reclaculated checksum on GPU (vertical)
//	chk1d_pitch = magma_roundup((N / B) * sizeof(double), 32);
//	chk1d_ld = chk1d_pitch / sizeof(double);
//	magma_dmalloc(&chk1d, chk1d_pitch * N);
//	
//	chk2d_pitch = magma_roundup((N / B) * sizeof(double), 32);
//	chk2d_ld = chk2d_pitch / sizeof(double);
//	magma_dmalloc(&chk2d, chk2d_pitch * N);
//	
//	
//	
//	
//	for (int i = 0; i < N; i += B) {		
//			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
//						//2, i + B, B,
//						1, N, B,
//						MAGMA_D_ONE, vd, vd_ld,
//						matrix + i, ld,
//						MAGMA_D_ZERO, chk1d + (i / B), chk1d_ld);			
//		}
//	
//	
//	for (int i = 0; i < N; i += B) {		
//			magma_dgemm(MagmaNoTrans, MagmaNoTrans,
//						//2, i + B, B,
//						1, N, B,
//						MAGMA_D_ONE, vd + 1, vd_ld,
//						matrix + i, ld,
//						MAGMA_D_ZERO, chk2d + (i / B), chk2d_ld);			
//		}
//	
//	
//	cout << "Matrix:" << endl;
//	printMatrix_gpu(matrix, ld, N, N);
//	
//	cout << "checksum:" << endl;
//	printMatrix_gpu(chksum, chksum_ld, (N / B) * 2, N);	
//	
//	cout << "checksum 1:" << endl;
//	printMatrix_gpu(chk1d, chk1d_ld, N / B, N);	
//	
//	cout << "checksum 2:" << endl;
//	printMatrix_gpu(chk2d, chk2d_ld, N / B, N);	
	
	//test_abft(matrix, ld, B, N, N, chksum, chksum_ld, chk1d, chk1d_ld, chk2d, chk2d_ld);

}


//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated
void recalculateChecksum(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(streams[1]);
		magmablasSetKernelStream(streams[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );

		// magmablasSetKernelStream(streams[3]);
		// magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
		// 		A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
		// magmablasSetKernelStream(streams[4]);
		// magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
		// 		A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
	}
	cudaStreamSynchronize(streams[1]);
	
	cudaStreamSynchronize(streams[2]);
	// cudaStreamSynchronize(streams[3]);

	// cudaStreamSynchronize(streams[4]);


}


//non-col-read-A
//col-read-B
//non-col-write-C
//combined
void recalculateChecksum2(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(streams[1]);
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, n, chk_nb,
					MAGMA_D_ONE, vd, vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2, chk1_ld);		
	}
	
	cudaStreamSynchronize(streams[1]);
}





//col-read-A
//col-read-B
//non-col-write-C
//separated
void recalculateChecksum3(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(streams[1]);
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
	}
	//cudaStreamSynchronize(streams[1]);
	
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);


}


//col-read-A
//col-read-B
//non-col-write-C
//combined
void recalculateChecksum4(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {


	// testing
	// cublasHandle_t handle;
	// cublasCreate(&handle);
	// cublasSetStream(handle, streams[2]);

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
	 magmablasSetKernelStream(streams[1]);
	cudaError_t r;
	 for (int i = 0; i < m; i += chk_nb) {
		
		magma_dgemm(MagmaTrans, MagmaNoTrans,
					2, n, chk_nb,
					MAGMA_D_ONE, vd, vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2, chk1_ld);		
	 }
	cudaStreamSynchronize(streams[1]);
	r = cudaGetLastError();
	r = cudaStreamSynchronize(streams[1]);
	if (r != cudaSuccess) {
	 	cout << "cuda sync error" << endl;
	 	magma_queue_t newStream;
	 	int * a = new int[10];
	 	magma_queue_create( &newStream );
	 	streams[1] = newStream;
	} else {
	 	cout << "sync success" << endl;
	}
}



//non-col-read-A
//col-read-B
//col-write-C
//separated
void recalculateChecksum5(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(streams[1]);
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
	}
	//cudaStreamSynchronize(streams[1]);
	
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);


}


//non-col-read-A
//col-read-B
//col-write-C
//combined
void recalculateChecksum6(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	magmablasSetKernelStream(streams[1]);
	for (int i = 0; i < m; i += chk_nb) {
		magma_dgemm(MagmaTrans, MagmaTrans,
					n, 2, chk_nb,
					MAGMA_D_ONE, 
					A + i, lda,
					vd, vd_ld,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2 * chk1_ld, chk1_ld);		
	}
	
	cudaStreamSynchronize(streams[1]);
}





//col-read-A
//col-read-B
//col-write-C
//separated
void recalculateChecksum7(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		//magmablasSetKernelStream(streams[1]);
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
	}
	//cudaStreamSynchronize(streams[1]);
	
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);


}


//col-read-A
//col-read-B
//col-write-C
//combined
void recalculateChecksum8(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	magmablasSetKernelStream(streams[1]);
	for (int i = 0; i < m; i += chk_nb) {
		magma_dgemm(MagmaTrans, MagmaNoTrans,
					n, 2, chk_nb,
					MAGMA_D_ONE, 
					A + i, lda,
					vd, vd_ld,
					MAGMA_D_ZERO, chk1 + (i / chk_nb) * 2 * chk1_ld, chk1_ld);		
	}
	
	cudaStreamSynchronize(streams[1]);
}


void benchmark(double * A, int lda,
			   int m, int n, int chk_nb,
			   double * vd, int vd_ld,
			   double * vd2, int vd2_ld,
			   double * chk1, int chk1_ld, 
			   double * chk2, int chk2_ld, 
			   double * chk21, int chk21_ld, 
			   double * chk22, int chk22_ld, 
			   magma_queue_t * streams,
			   int * mapping, int mapping_ld
			   ) {

	double gpu_time1 = 1000.0;
	double gpu_time2 = 1000.0;
	double gpu_time3 = 1000.0;
	double gpu_time4 = 1000.0;
	double gpu_time5 = 1000.0;
	double gpu_time6 = 1000.0;
	double gpu_time7 = 1000.0;
	double gpu_time8 = 1000.0;
	int K = 5;
	cudaProfilerStart();
	for (int i = chk_nb; i < m; i += chk_nb) {
		cout << "[" << i << "]:	";
		for (int j = chk_nb; j < n; j += chk_nb) {
			gpu_time1 = magma_wtime();
			for (int k = 0; k < K; k ++) {
				ChecksumRecalSelector(A, lda,
				   i, j, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   1);
			}
			gpu_time1 = magma_wtime() - gpu_time1;


			gpu_time2 = magma_wtime();
			for (int k = 0; k < K; k ++) {
				ChecksumRecalSelector(A, lda,
				   i, j, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   2);
			}
			gpu_time2 = magma_wtime() - gpu_time2;


			gpu_time3 = magma_wtime();
			for (int k = 0; k < K; k ++){
			ChecksumRecalSelector(A, lda,
				   i, j, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   3);
			}
			gpu_time3 = magma_wtime() - gpu_time3;


			gpu_time4 = magma_wtime();
			for (int k = 0; k < 1; k ++){
			ChecksumRecalSelector(A, lda,
				   i, j, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   4);
			}
			gpu_time4 = magma_wtime() - gpu_time4;


			gpu_time5 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(A, lda,
					   i, j, chk_nb,
					   vd, vd_ld,
					   vd2, vd2_ld,
					   chk1, chk1_ld, 
					   chk2, chk2_ld, 
					   chk21, chk21_ld, 
					   chk22, chk22_ld, 
					   streams,
					   5);
			}
			gpu_time5 = magma_wtime() - gpu_time5;



			gpu_time6 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(A, lda,
				   i, j, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   6);
			}
			gpu_time6 = magma_wtime() - gpu_time6;


			gpu_time7 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(A, lda,
				   i, j, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   7);
			}
			gpu_time7 = magma_wtime() - gpu_time7;
			


			// gpu_time8 = magma_wtime();
			// for (int k = 0; k < K; k ++){
			// 	ChecksumRecalSelector(A, lda,
			// 	   i, j, chk_nb,
			// 	   vd, vd_ld,
			// 	   vd2, vd2_ld,
			// 	   chk1, chk1_ld, 
			// 	   chk2, chk2_ld, 
			// 	   chk21, chk21_ld, 
			// 	   chk22, chk22_ld, 
			// 	   streams,
			// 	   8);
			// }
			// gpu_time8 = magma_wtime() - gpu_time8;

			double min_time1 = fmin(gpu_time1, fmin(gpu_time2, fmin(gpu_time3, gpu_time4)));
			double min_time2 = fmin(gpu_time5, fmin(gpu_time6, fmin(gpu_time7, gpu_time8)));
			double min_time = fmin(min_time1, min_time2);

			if (min_time == gpu_time1) {
				cout << "1 ";
				mapping[i * mapping_ld + j] = 1;
			} else if (min_time == gpu_time2) {
				cout << "2 ";
				mapping[i * mapping_ld + j] = 2;
			} else if (min_time == gpu_time3) {
				cout << "3 ";
				mapping[i * mapping_ld + j] = 3;
			} else if (min_time == gpu_time4) {
				cout << "4 ";
				mapping[i * mapping_ld + j] = 4;
			} else if (min_time == gpu_time5) {
				cout << "5 ";
				mapping[i * mapping_ld + j] = 5;
			} else if  (min_time == gpu_time6) {
				cout << "6 ";
				mapping[i * mapping_ld + j] = 6;
			} else if  (min_time == gpu_time7) {
				cout << "7 ";
				mapping[i * mapping_ld + j] = 7;
			} else if  (min_time == gpu_time8){
				cout << "8 ";
				mapping[i * mapping_ld + j] = 8;
			}
			// if (gpu_time1 < gpu_time2) cout << "S ";
			// else cout <<"C ";
			// cout << gpu_time1 << " ";
			// cout << gpu_time2 << " ";
			// cout << gpu_time3 << " ";
			// cout << gpu_time4 << " ";
			// cout << gpu_time5 << " ";
			// cout << gpu_time6 << " ";
			// cout << gpu_time7 << " ";
			// cout << gpu_time8 << " ";
			// cout << endl;
		}
		cout << endl;
	}
	cudaProfilerStop();
}

void ChecksumRecalSelector(double * A, int lda,
			   int m, int n, int chk_nb,
			   double * vd, int vd_ld,
			   double * vd2, int vd2_ld,
			   double * chk1, int chk1_ld, 
			   double * chk2, int chk2_ld, 
			   double * chk21, int chk21_ld, 
			   double * chk22, int chk22_ld, 
			   magma_queue_t * streams,
			   int select) {

		switch(select) {
			case 1: recalculateChecksum(A, lda,
						m, n, chk_nb,
						vd, vd_ld,
			   			chk1, chk1_ld, 
			   			chk2, chk2_ld, 
			   			streams);
					break;


			case 2:	recalculateChecksum2(A, lda,
						m, n, chk_nb,
						vd, vd_ld,
			   			chk1, chk1_ld, 
			   			chk2, chk2_ld, 
			   			streams);
					break;

			case 3: recalculateChecksum3(A, lda,
						m, n, chk_nb,
						vd2, vd2_ld,
			   			chk1, chk1_ld, 
			   			chk2, chk2_ld, 
			   			streams);
					break;


			case 4: recalculateChecksum4(A, lda,
						m, n, chk_nb,
						vd2, vd2_ld,
			   			chk1, chk1_ld, 
			   			chk2, chk2_ld, 
			   			streams);
					break;


			case 5: recalculateChecksum5(A, lda,
						m, n, chk_nb,
						vd, vd_ld,
			   			chk21, chk21_ld, 
			   			chk22, chk22_ld, 
			   			streams);
					break;
		
			case 6: recalculateChecksum6(A, lda,
						m, n, chk_nb,
						vd, vd_ld,
			   			chk21, chk21_ld, 
			   			chk22, chk22_ld, 
			   			streams);
					break;
		

			case 7: recalculateChecksum7(A, lda,
						m, n, chk_nb,
						vd2, vd2_ld,
			   			chk21, chk21_ld, 
			   			chk22, chk22_ld, 
			   			streams);
					break;
			
			case 8: recalculateChecksum8(A, lda,
						m, n, chk_nb,
						vd2, vd2_ld,
			   			chk21, chk21_ld, 
			   			chk22, chk22_ld, 
			   			streams);
					break;
		}
}


void AutoTuneChecksumRecal(double * A, int lda,
			   int m, int n, int chk_nb,
			   double * vd, int vd_ld,
			   double * vd2, int vd2_ld,
			   double * chk1, int chk1_ld, 
			   double * chk2, int chk2_ld, 
			   double * chk21, int chk21_ld, 
			   double * chk22, int chk22_ld, 
			   magma_queue_t * streams,
			   int * mapping, int mapping_ld
			   ){

	// needs to do boundary check first


	int i = mapping[m * mapping_ld + n];
	ChecksumRecalSelector(A, lda,
				   m, n, chk_nb,
				   vd, vd_ld,
				   vd2, vd2_ld,
				   chk1, chk1_ld, 
				   chk2, chk2_ld, 
				   chk21, chk21_ld, 
				   chk22, chk22_ld, 
				   streams,
				   i);

}
	


