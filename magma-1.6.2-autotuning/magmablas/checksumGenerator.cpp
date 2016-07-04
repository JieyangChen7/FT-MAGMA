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
void initializeChecksum(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * streams) {

	for (int i = 0; i < abftEnv->gpu_m; i += abftEnv->chk_nb) {		
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, abftEnv->gpu_n, abftEnv->chk_nb,
					MAGMA_D_ONE, abftEnv->vd, abftEnv->vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, abftEnv->checksum + (i / (abftEnv->chk_nb)) * 2, abftEnv->checksum_ld);			
	}
}


void initializeABFTEnv(ABFTEnv * abftEnv, int chk_nb,
						int gpu_m, int gpu_n,
						int cpu_m, int cpu_n) {


	bool DEBUG = false;
	abftEnv->chk_nb = chk_nb;
	abftEnv->gpu_m = gpu_m;
	abftEnv->gpu_n = gpu_n;
	abftEnv->cpu_m = cpu_m;
	abftEnv->cpu_n = cpu_n;

	/* initialize checksum vectors on CPU */
    /* v =
     * 1 1 1 1
     * 1 2 3 4 
     */
    cout << "checksum vectors initialization on CPU......"; 
    magma_dmalloc_pinned(&(abftEnv->v), (abftEnv->chk_nb) * 2 * sizeof(double));
    abftEnv->v_ld = 2;
    for (int i = 0; i < (abftEnv->chk_nb); ++i) {
        *((abftEnv->v) + i * (abftEnv->v_ld)) = 1;
    }
    for (int i = 0; i < (abftEnv->chk_nb); ++i) {
        *((abftEnv->v) + i * (abftEnv->v_ld) + 1) = i+1;
    }
    if(DEBUG) {
        cout << "checksum vector on CPU:" << endl;
        printMatrix_host(abftEnv->v, abftEnv->v_ld, 2, abftEnv->chk_nb);
    }
    cout << "done." << endl;

    /* initialize checksum vectors on CPU */
    /* v2 =
     * 1 1 
     * 1 2  
     * 1 3
     * 1 4
     */
    cout << "checksum vectors initialization on CPU......"; 
    magma_dmalloc_pinned(&(abftEnv->v2), (abftEnv->chk_nb) * 2 * sizeof(double));
    abftEnv->v2_ld = abftEnv->chk_nb;
    for (int i = 0; i < (abftEnv->chk_nb); ++i) {
        *((abftEnv->v2) + i) = 1;
    }
    for (int i = 0; i < (abftEnv->chk_nb); ++i) {
        *((abftEnv->v2) + (abftEnv->v2_ld) + i) = i+1;
    }
    if(DEBUG) {
        cout << "checksum vector on CPU:" << endl;
        printMatrix_host(abftEnv->v2, abftEnv->v2_ld, abftEnv->chk_nb, 2);
    }
    cout << "done." << endl;

    /* initialize checksum vectors on GPU */
    cout << "checksum vectors initialization on GPU......";
    size_t vd_pitch = magma_roundup(2 * sizeof(double), 32);
    abftEnv->vd_ld = vd_pitch / sizeof(double);  
    magma_dmalloc(&(abftEnv->vd), vd_pitch * (abftEnv->chk_nb));
    magma_dsetmatrix(2, abftEnv->chk_nb, 
    				 abftEnv->v, abftEnv->v_ld,
    				 abftEnv->vd, abftEnv->vd_ld);
    if(DEBUG) {
        cout << "checksum vector on GPU:" << endl;
        printMatrix_gpu(abftEnv->vd, abftEnv->vd_ld, abftEnv->chk_nb, 2);
    }
    cout << "done." << endl;


    /* initialize checksum vectors on GPU */
    cout << "checksum vectors initialization on GPU......";
    size_t vd2_pitch = magma_roundup((abftEnv->chk_nb) * sizeof(double), 32);
    abftEnv->vd2_ld = vd2_pitch / sizeof(double);  
    magma_dmalloc(&(abftEnv->vd2), vd2_pitch * 2);
    magma_dsetmatrix(abftEnv->chk_nb, 2,
    				 abftEnv->v2, abftEnv->v2_ld, 
    				 abftEnv->vd2, abftEnv->vd2_ld);
    if(DEBUG) {
        cout << "checksum vector on GPU:" << endl;
        printMatrix_gpu(abftEnv->vd2, abftEnv->vd2_ld, abftEnv->chk_nb, 2);
    }
    cout << "done." << endl;


    cout << "allocate space for recalculated checksum on GPU......";
    /* allocate space for reclaculated checksum on GPU */
    size_t chk1_pitch = magma_roundup(2 * ((abftEnv->gpu_m) / (abftEnv->chk_nb)) * sizeof(double), 32);
    abftEnv->chk1_ld = chk1_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk1), chk1_pitch * (abftEnv->gpu_n));
    
   
    size_t chk2_pitch = magma_roundup(2 * ((abftEnv->gpu_m) / (abftEnv->chk_nb)) * sizeof(double), 32);
    abftEnv->chk2_ld = chk2_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk2), chk2_pitch * (abftEnv->gpu_n));
    cout << "done." << endl;



    cout << "allocate space for recalculated checksum on GPU......";
    /* allocate space for reclaculated checksum on GPU */
    size_t chk21_pitch = magma_roundup( (abftEnv->gpu_n) * sizeof(double), 32);
    abftEnv->chk21_ld = chk21_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk21), chk21_pitch * 2 * ((abftEnv->gpu_m) / (abftEnv->chk_nb)));
    
   
    size_t chk22_pitch = magma_roundup((abftEnv->gpu_n) * sizeof(double), 32);
    abftEnv->chk22_ld = chk22_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk22), chk22_pitch * 2 * ((abftEnv->gpu_m) / (abftEnv->chk_nb)));
    cout << "done." << endl;



    /* allocate space for update checksum on CPU */
    cout << "allocate space for checksum on CPU......";
    magma_dmalloc_pinned(&(abftEnv->work_chk), (abftEnv->cpu_m) * cpu_n * sizeof(double));
    abftEnv->work_chk_ld = abftEnv->cpu_m;
    cout << "done." << endl;


    
    /* allocate space for update checksum on GPU */
    cout << "allocate space for checksum on GPU......";
    size_t checksum_pitch = magma_roundup(((abftEnv->gpu_m) / (abftEnv->chk_nb)) * 2 * sizeof(double), 32);
    abftEnv->checksum_ld = checksum_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->checksum), checksum_pitch * (abftEnv->gpu_n));
    //cudaMemset2D(checksum, checksum_pitch, 0, (n / nb) * 2 * sizeof(double), m);
    cout << "done." << endl;


    cout << "auto tuning mapping initialize" << endl;
    abftEnv->mapping = new int[(abftEnv->gpu_m/abftEnv->chk_nb) * (abftEnv->gpu_n/abftEnv->chk_nb)];
    abftEnv->mapping_ld = abftEnv->gpu_m/abftEnv->chk_nb;
    cout << "done." << endl;

    cout << "lastCheckTime initialize" << endl;
    abftEnv->lastCheckTime = new time_t[(abftEnv->gpu_m/abftEnv->chk_nb) * (abftEnv->gpu_n/abftEnv->chk_nb)];
    abftEnv->lastCheckTime_ld = abftEnv->gpu_m/abftEnv->chk_nb;
    cout << "done." << endl;


	cout << "updatedCounter initialize" << endl;
    abftEnv->updatedCounter = new int[(abftEnv->gpu_m/abftEnv->chk_nb) * (abftEnv->gpu_n/abftEnv->chk_nb)];
    abftEnv->updatedCounter_ld = abftEnv->gpu_m/abftEnv->chk_nb;
    cout << "done." << endl;

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
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(streams[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );

		// magmablasSetKernelStream(streams[3]);
		// magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
		// 		A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
		// magmablasSetKernelStream(streams[4]);
		// magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
		// 		A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
	}
	cudaStreamSynchronize(streams[2]);
	
	cudaStreamSynchronize(streams[3]);
	
}




//recalculate column checksums
//M: number of rows of A
//N: numner of cols of A
//non-col-read-A
//col-read-B
//non-col-write-C
//separated - 4 streams
void recalculateChecksum9(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb * 2) {
		magmablasSetKernelStream(streams[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
		if (i + chk_nb < m) {			
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
			magmablasSetKernelStream(streams[4]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
		}
	}
	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);
	cudaStreamSynchronize(streams[4]);
	
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
//separated - 4 streams
void recalculateChecksum10(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb * 2) {
		magmablasSetKernelStream(streams[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );
		if (i + chk_nb < m) {
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
			magmablasSetKernelStream(streams[2]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
		}
	}
	
	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);
	cudaStreamSynchronize(streams[4]);
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
//separated - 4 streams
void recalculateChecksum11(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb * 2) {
		magmablasSetKernelStream(streams[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
		if (i + chk_nb < m) {
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb + 1) * chk1_ld, 1 );
			magmablasSetKernelStream(streams[4]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb + 1) * chk2_ld, 1 );
		}
	}
	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);
	cudaStreamSynchronize(streams[4]);
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
//separated
void recalculateChecksum12(double * A, int lda,
		int m, int n, int chk_nb,
		double * vd, int vd_ld,
		double * chk1, int chk1_ld, 
		double * chk2, int chk2_ld, 
		magma_queue_t * streams) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(streams[1]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb) * chk1_ld, 1 );
		magmablasSetKernelStream(streams[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb) * chk2_ld, 1 );
		if (i + chk_nb < m) {
			magmablasSetKernelStream(streams[3]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd, 1, MAGMA_D_ZERO, chk1 + (i / chk_nb + 1) * chk1_ld, 1 );
			magmablasSetKernelStream(streams[4]);
			magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
					A + i + chk_nb, lda, vd + vd_ld, 1, MAGMA_D_ZERO, chk2 + (i / chk_nb + 1) * chk2_ld, 1 );
		}
	}
	cudaStreamSynchronize(streams[1]);
	cudaStreamSynchronize(streams[2]);
	cudaStreamSynchronize(streams[3]);
	cudaStreamSynchronize(streams[4]);


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


void ChecksumRecalProfiler(ABFTEnv * abftEnv, double * A, int lda, 
						   magma_queue_t * streams) {

	double gpu_time1 = 1000.0;
	double gpu_time2 = 1000.0;
	double gpu_time3 = 1000.0;
	double gpu_time4 = 1000.0;
	double gpu_time5 = 1000.0;
	double gpu_time6 = 1000.0;
	double gpu_time7 = 1000.0;
	double gpu_time8 = 1000.0;
	double gpu_time9 = 1000.0;
	double gpu_time10 = 1000.0;
	double gpu_time11 = 1000.0;
	double gpu_time12 = 1000.0;
	int K = 1;
	cudaProfilerStart();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_m; i += abftEnv->chk_nb) {
		cout << "[" << i << "]:	";
		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_n; j += abftEnv->chk_nb) {
			gpu_time1 = magma_wtime();
			for (int k = 0; k < K; k ++) {
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 1);
			}
			gpu_time1 = magma_wtime() - gpu_time1;


			gpu_time2 = magma_wtime();
			for (int k = 0; k < K; k ++) {
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 2);
			}
			gpu_time2 = magma_wtime() - gpu_time2;


			gpu_time3 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 3);
			}
			gpu_time3 = magma_wtime() - gpu_time3;


			// gpu_time4 = magma_wtime();
			// for (int k = 0; k < 1; k ++){
			// 		ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 4);
			// }
			// gpu_time4 = magma_wtime() - gpu_time4;


			gpu_time5 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 5);
			}
			gpu_time5 = magma_wtime() - gpu_time5;



			gpu_time6 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 6);
			}
			gpu_time6 = magma_wtime() - gpu_time6;


			gpu_time7 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 7);
			}
			gpu_time7 = magma_wtime() - gpu_time7;
			


			// gpu_time8 = magma_wtime();
			// for (int k = 0; k < K; k ++){
			// 	ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 8);
			// }
			// gpu_time8 = magma_wtime() - gpu_time8;


			gpu_time9 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 9);
			}
			gpu_time9 = magma_wtime() - gpu_time9;


			gpu_time10 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 10);
			}
			gpu_time10 = magma_wtime() - gpu_time10;


			gpu_time11 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 11);
			}
			gpu_time11 = magma_wtime() - gpu_time11;

			gpu_time12 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 12);
			}
			gpu_time12 = magma_wtime() - gpu_time12;

			double min_time1 = fmin(gpu_time1, fmin(gpu_time2, fmin(gpu_time3, gpu_time4)));
			double min_time2 = fmin(gpu_time5, fmin(gpu_time6, fmin(gpu_time7, gpu_time8)));
			double min_time3 = fmin(gpu_time9, fmin(gpu_time10, fmin(gpu_time11, gpu_time12)));
			double min_time = fmin(min_time1, fmin(min_time2, min_time3));

			int best = 0;
			if (min_time == gpu_time1) {
				cout << "1 ";
				best = 1;
			} else if (min_time == gpu_time2) {
				cout << "2 ";
				best = 2;
			} else if (min_time == gpu_time3) {
				cout << "3 ";
				best = 3;
			} else if (min_time == gpu_time4) {
				cout << "4 ";
				best = 4;
			} else if (min_time == gpu_time5) {
				cout << "5 ";
				best = 5;
			} else if  (min_time == gpu_time6) {
				cout << "6 ";
				best = 6;
			} else if  (min_time == gpu_time7) {
				cout << "7 ";
				best = 7;
			} else if  (min_time == gpu_time8){
				cout << "8 ";
				best = 8;
			} else if (min_time == gpu_time9) {
				cout << "9 ";
				best = 9;
			} else if  (min_time == gpu_time10) {
				cout << "10 ";
				best =10;
			} else if  (min_time == gpu_time11) {
				cout << "11 ";
				best =11;
			} else if  (min_time == gpu_time12){
				cout << "12 ";
				best = 12;
			}
			abftEnv->mapping[(i / abftEnv->chk_nb) * abftEnv->mapping_ld + j /abftEnv->chk_nb] = best;
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

void ChecksumRecalSelector(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * streams, int select) {

		switch(select) {
			case 1: recalculateChecksum(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			streams);
					break;


			case 2:	recalculateChecksum2(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			streams);
					break;

			case 3: recalculateChecksum3(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			streams);
					break;


			case 4: recalculateChecksum4(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			streams);
					break;


			case 5: recalculateChecksum5(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			streams);
					break;
		
			case 6: recalculateChecksum6(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			streams);
					break;
		

			case 7: recalculateChecksum7(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			streams);
					break;
			
			case 8: recalculateChecksum8(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			streams);
					break;
			case 9: recalculateChecksum9(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			streams);
					break;
		
			case 10: recalculateChecksum10(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			streams);
					break;
		

			case 11: recalculateChecksum11(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			streams);
					break;
			
			case 12: recalculateChecksum12(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			streams);
					break;

			default: cout << "selecting error" << endl;
		}
}


void AutoTuneChecksumRecal(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * streams){

	// needs to do boundary check first


	int i = abftEnv->mapping[(m / abftEnv->chk_nb) * abftEnv->mapping_ld + (n / abftEnv->chk_nb)];
	ChecksumRecalSelector(abftEnv, A, lda, m, n, streams, i);

}


void benchmark(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * streams){
	cout << "start banchmarking:" << endl;
	double benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_m; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_n; j += abftEnv->chk_nb) {

			AutoTuneChecksumRecal(abftEnv, A, lda, i, j, streams);
		}

	}
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "auto tuning time: " << benchmark_time << endl;



	benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_m; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_n; j += abftEnv->chk_nb) {

			ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 2);
		}

	}
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "naive tuning time: " << benchmark_time << endl;


	benchmark_time = magma_wtime();
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_m; i += abftEnv->chk_nb) {

		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_n; j += abftEnv->chk_nb) {

			ChecksumRecalSelector(abftEnv, A, lda, i, j, streams, 1);
		}

	}
	
	benchmark_time = magma_wtime() - benchmark_time;
	cout << "hand tuning time: " << benchmark_time << endl;

	cout << "done benchmarking" << endl;
}

//check matrix A using checksums
void ABFTCheck(double * A, int lda, 
			   int m, int n, int chk_nb,
			   double * checksumA, int checksumA_ld) {

}

//check each block of data based on last time check
void MemoryErrorCheck(time_t * lastCheckTime, int lastCheckTime_ld, 
					  int m, int n, int chk_nb,
					  double * A, int lda,
					  double * checksumA, int checksumA_ld) {

}
	


