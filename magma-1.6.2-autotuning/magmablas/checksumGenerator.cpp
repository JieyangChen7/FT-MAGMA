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
void init_col_chk(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream) {

	for (int i = 0; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {		
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, abftEnv->gpu_col, abftEnv->chk_nb,
					MAGMA_D_ONE, abftEnv->vd, abftEnv->vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, COL_CHK(i / abftEnv->chk_nb, 0), abftEnv->col_dchk_ld);			
	}
}


void init_row_chk(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream) {

	for (int i = 0; i < abftEnv->gpu_col; i += abftEnv->chk_nb) {		
		magma_dgemm(MagmaNoTrans, MagmaTrans,
					abftEnv->gpu_row, 2, abftEnv->chk_nb,
					MAGMA_D_ONE, abftEnv->vd2, abftEnv->vd2_ld,
					A + i * lda, lda,
					MAGMA_D_ZERO, ROW_CHK(0, i / abftEnv->chk_nb), abftEnv->row_dchk_ld);			
	}
}




void initializeABFTEnv(ABFTEnv * abftEnv, int chk_nb, 
						double * A, int lda,
						int gpu_row, int gpu_col,
						int cpu_row, int cpu_col,
						magma_queue_t * stream) {


	bool DEBUG = true;

	abftEnv->chk_nb = chk_nb;

	abftEnv->gpu_row = gpu_row;
	abftEnv->gpu_col = gpu_col;
	abftEnv->cpu_row = cpu_row;
	abftEnv->cpu_col = cpu_col;

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
        printMatrix_host(abftEnv->v, abftEnv->v_ld, 2, abftEnv->chk_nb, -1, -1);
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
        printMatrix_host(abftEnv->v2, abftEnv->v2_ld, abftEnv->chk_nb, 2, -1, -1);
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
        printMatrix_gpu(abftEnv->vd, abftEnv->vd_ld, abftEnv->chk_nb, 2, -1, -1);
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
        printMatrix_gpu(abftEnv->vd2, abftEnv->vd2_ld, abftEnv->chk_nb, 2, -1, -1);
    }
    cout << "done." << endl;


    cout << "allocate space for recalculated checksum on GPU......";
    /* allocate space for reclaculated checksum on GPU */
    size_t chk1_pitch = magma_roundup(2 * ((abftEnv->gpu_row) / (abftEnv->chk_nb)) * sizeof(double), 32);
    abftEnv->chk1_ld = chk1_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk1), chk1_pitch * (abftEnv->gpu_col));
    
   
    size_t chk2_pitch = magma_roundup(2 * ((abftEnv->gpu_row) / (abftEnv->chk_nb)) * sizeof(double), 32);
    abftEnv->chk2_ld = chk2_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk2), chk2_pitch * (abftEnv->gpu_col));
    cout << "done." << endl;



    cout << "allocate space for recalculated checksum on GPU......";
    /* allocate space for reclaculated checksum on GPU */
    size_t chk21_pitch = magma_roundup( (abftEnv->gpu_col) * sizeof(double), 32);
    abftEnv->chk21_ld = chk21_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk21), chk21_pitch * 2 * ((abftEnv->gpu_row) / (abftEnv->chk_nb)));
    
   
    size_t chk22_pitch = magma_roundup((abftEnv->gpu_col) * sizeof(double), 32);
    abftEnv->chk22_ld = chk22_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->chk22), chk22_pitch * 2 * ((abftEnv->gpu_row) / (abftEnv->chk_nb)));
    cout << "done." << endl;



    /* allocate space for update column checksum on CPU */
    cout << "allocate space for column checksum on CPU......";
    magma_dmalloc_pinned(&(abftEnv->col_hchk), (cpu_row / chk_nb) * cpu_col * sizeof(double));
    abftEnv->col_hchk_ld = cpu_row / chk_nb;
    cout << "done." << endl;

    /* allocate space for update column checksum on CPU */
    cout << "allocate space for row checksum on CPU......";
    magma_dmalloc_pinned(&(abftEnv->row_hchk), cpu_row * (cpu_col / chk_nb) * sizeof(double));
    abftEnv->row_hchk_ld = cpu_row;
    cout << "done." << endl;

    
    /* allocate space for update checksum on GPU */
    cout << "allocate space for column checksums on GPU......";
    size_t col_dchk_pitch = magma_roundup((gpu_row / chk_nb) * 2 * sizeof(double), 32);
    abftEnv->col_dchk_ld = col_dchk_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->col_dchk), col_dchk_pitch * gpu_col);
    //cudaMemset2D(checksum, checksum_pitch, 0, (n / nb) * 2 * sizeof(double), m);
    cout << "done." << endl;


    /* allocate space for update checksum on GPU */
    cout << "allocate space for row checksums on GPU......";
    size_t row_dchk_pitch = magma_roundup(gpu_row * sizeof(double), 32);
    abftEnv->row_dchk_ld = row_dchk_pitch / sizeof(double);
    magma_dmalloc(&(abftEnv->row_dchk), row_dchk_pitch * (gpu_col / chk_nb) * 2);
    //cudaMemset2D(checksum, checksum_pitch, 0, (n / nb) * 2 * sizeof(double), m);
    cout << "done." << endl;

    /* initialize checksums */
    cout << "column checksums initiallization......";
    init_col_chk(abftEnv, A, lda, stream);
    cout << "done." << endl;

    cout << "row checksums initiallization......";
    init_row_chk(abftEnv, A, lda, stream);
    cout << "done." << endl;

    if (DEBUG) {
    	cout << "input matrix:" << endl;
        printMatrix_gpu(dAT, lddat, n, m, 4, 4);
        cout << "column checksum matrix on GPU:" << endl;
        printMatrix_gpu(abftEnv->col_dchk, abftEnv->col_dchk_ld,
        	 			(abftEnv->gpu_row / abftEnv->chk_nb) * 2, abftEnv->gpu_col, 
        	 			2, chk_nb);
        cout << "row checksum matrix on GPU:" << endl;
        printMatrix_gpu(abftEnv->row_dchk, abftEnv->row_dchk_ld,
        	 			abftEnv->gpu_row, (abftEnv->gpu_col / abftEnv->chk_nb) * 2, 
        	 			chk_nb, 2);
    }






    cout << "auto tuning mapping initialize" << endl;
    abftEnv->mapping = new int[(abftEnv->gpu_row/abftEnv->chk_nb) * (abftEnv->gpu_col/abftEnv->chk_nb)];
    abftEnv->mapping_ld = abftEnv->gpu_row/abftEnv->chk_nb;
    cout << "done." << endl;

    cout << "lastCheckTime initialize" << endl;
    abftEnv->lastCheckTime = new time_t[(abftEnv->gpu_row/abftEnv->chk_nb) * (abftEnv->gpu_col/abftEnv->chk_nb)];
    abftEnv->lastCheckTime_ld = abftEnv->gpu_row/abftEnv->chk_nb;
    cout << "done." << endl;


	cout << "updatedCounter initialize" << endl;
    abftEnv->updatedCounter = new int[(abftEnv->gpu_row/abftEnv->chk_nb) * (abftEnv->gpu_col/abftEnv->chk_nb)];
    abftEnv->updatedCounter_ld = abftEnv->gpu_row/abftEnv->chk_nb;
    cout << "done." << endl;

    //to he auto tuned later
    abftEnv->N = 10;
    abftEnv->T = 10;

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
		magma_queue_t * stream) {

	for (int i = 0; i < m; i += chk_nb) {
		magmablasSetKernelStream(stream[2]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb), chk1_ld );
		magmablasSetKernelStream(stream[3]);
		magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
				A + i, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb), chk2_ld );

		// magmablasSetKernelStream(stream[3]);
		// magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
		// 		A + i + chk_nb, lda, vd, vd_ld, MAGMA_D_ZERO, chk1 + (i / chk_nb) + 1, chk1_ld );
		// magmablasSetKernelStream(stream[4]);
		// magma_dgemv(MagmaTrans, chk_nb, n, MAGMA_D_ONE,
		// 		A + i + chk_nb, lda, vd + 1, vd_ld, MAGMA_D_ZERO, chk2 + (i / chk_nb) + 1, chk2_ld );
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
void recalculateChecksum9(double * A, int lda,
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





//non-col-read-A
//col-read-B
//non-col-write-C
//combined
void recalculateChecksum2(double * A, int lda,
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
//separated
void recalculateChecksum3(double * A, int lda,
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
void recalculateChecksum10(double * A, int lda,
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



//col-read-A
//col-read-B
//non-col-write-C
//combined
void recalculateChecksum4(double * A, int lda,
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
//separated
void recalculateChecksum5(double * A, int lda,
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
void recalculateChecksum11(double * A, int lda,
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



//non-col-read-A
//col-read-B
//col-write-C
//combined
void recalculateChecksum6(double * A, int lda,
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
//separated
void recalculateChecksum7(double * A, int lda,
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
void recalculateChecksum12(double * A, int lda,
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


//col-read-A
//col-read-B
//col-write-C
//combined
void recalculateChecksum8(double * A, int lda,
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
	for (int i = abftEnv->chk_nb; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {
		cout << "[" << i << "]:	";
		for (int j = abftEnv->chk_nb; j < abftEnv->gpu_col; j += abftEnv->chk_nb) {
			gpu_time1 = magma_wtime();
			for (int k = 0; k < K; k ++) {
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 1);
			}
			gpu_time1 = magma_wtime() - gpu_time1;


			gpu_time2 = magma_wtime();
			for (int k = 0; k < K; k ++) {
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 2);
			}
			gpu_time2 = magma_wtime() - gpu_time2;


			gpu_time3 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 3);
			}
			gpu_time3 = magma_wtime() - gpu_time3;


			// gpu_time4 = magma_wtime();
			// for (int k = 0; k < 1; k ++){
			// 		ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 4);
			// }
			// gpu_time4 = magma_wtime() - gpu_time4;


			gpu_time5 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 5);
			}
			gpu_time5 = magma_wtime() - gpu_time5;



			gpu_time6 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 6);
			}
			gpu_time6 = magma_wtime() - gpu_time6;


			gpu_time7 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 7);
			}
			gpu_time7 = magma_wtime() - gpu_time7;
			


			// gpu_time8 = magma_wtime();
			// for (int k = 0; k < K; k ++){
			// 	ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 8);
			// }
			// gpu_time8 = magma_wtime() - gpu_time8;


			gpu_time9 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 9);
			}
			gpu_time9 = magma_wtime() - gpu_time9;


			gpu_time10 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 10);
			}
			gpu_time10 = magma_wtime() - gpu_time10;


			gpu_time11 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 11);
			}
			gpu_time11 = magma_wtime() - gpu_time11;

			gpu_time12 = magma_wtime();
			for (int k = 0; k < K; k ++){
				ChecksumRecalSelector(abftEnv, A, lda, i, j, stream, 12);
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

void ChecksumRecalSelector(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream, int select) {

		switch(select) {
			case 1: recalculateChecksum(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;


			case 2:	recalculateChecksum2(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;

			case 3: recalculateChecksum3(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;


			case 4: recalculateChecksum4(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;


			case 5: recalculateChecksum5(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
		
			case 6: recalculateChecksum6(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
		

			case 7: recalculateChecksum7(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
			
			case 8: recalculateChecksum8(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
			case 9: recalculateChecksum9(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;
		
			case 10: recalculateChecksum10(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk1, abftEnv->chk1_ld, 
			   			abftEnv->chk2, abftEnv->chk2_ld, 
			   			stream);
					break;
		

			case 11: recalculateChecksum11(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd, abftEnv->vd_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;
			
			case 12: recalculateChecksum12(A, lda,
						m, n, abftEnv->chk_nb,
						abftEnv->vd2, abftEnv->vd2_ld,
			   			abftEnv->chk21, abftEnv->chk21_ld, 
			   			abftEnv->chk22, abftEnv->chk22_ld, 
			   			stream);
					break;

			default: cout << "selecting error" << endl;
		}
}


void AutoTuneChecksumRecal(ABFTEnv * abftEnv, double * A, int lda, int m, int n, magma_queue_t * stream){

	// needs to do boundary check first


	int i = abftEnv->mapping[(m / abftEnv->chk_nb) * abftEnv->mapping_ld + (n / abftEnv->chk_nb)];
	ChecksumRecalSelector(abftEnv, A, lda, m, n, stream, i);

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

//check each block of data based on last time check
void MemoryErrorCheck(ABFTEnv * abftEnv, double * A, int lda, magma_queue_t * stream) {
	for (int i = 0; i < abftEnv->gpu_row/abftEnv->chk_nb; i++) {
		for (int j = 0; j < abftEnv->gpu_col/abftEnv->chk_nb; j++) {
			time_t temp = *(abftEnv->lastCheckTime + j * abftEnv->lastCheckTime_ld + i);
			if (time(NULL) - temp > abftEnv->T) {
				// we should do check on block[i, j]
				ABFTCheck(abftEnv, A + j*abftEnv->chk_nb*lda + i * abftEnv->chk_nb, lda,
						  abftEnv->chk_nb, abftEnv->chk_nb,
						  COL_CHK(i, j), abftEnv->col_dchk_ld, stream);
				*(abftEnv->lastCheckTime + j * abftEnv->lastCheckTime_ld + i) = time(NULL);
			}
		}
	}
}


//update updating counter and check if necessary
bool updateCounter(ABFTEnv * abftEnv, int row1, int row2, int col1, int col2, int count) {
	bool verify = false;
	for (int i = row1; i <= row2; i++) {
		for (int j = col1; j <= col2; j++) {
			*(abftEnv->updatedCounter + j * abftEnv->updatedCounter_ld + i) += count;

			int temp = *(abftEnv->updatedCounter + j * abftEnv->updatedCounter_ld + i);
			if (temp > abftEnv->N) {
				// we should do check on block[i, j]
				// ABFTCheck(abftEnv, A + j*abftEnv->chk_nb*lda + i * bftEnv->chk_nb, lda,
				// 		  abftEnv->chk_nb, abftEnv->chk_nb,
				// 		  CHK(i, j), stream);
				verify = true;
				*(abftEnv->updatedCounter + j * abftEnv->updatedCounter_ld + i) = 0;
			} 
		}
	}
	return verify;
}






