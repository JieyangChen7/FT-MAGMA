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
void init_col_chk(ABFTEnv * abftEnv, double * A, int lda) {

	for (int i = 0; i < abftEnv->gpu_row; i += abftEnv->chk_nb) {		
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					2, abftEnv->gpu_col, abftEnv->chk_nb,
					MAGMA_D_ONE, abftEnv->vd, abftEnv->vd_ld,
					A + i, lda,
					MAGMA_D_ZERO, COL_CHK(i / abftEnv->chk_nb, 0), abftEnv->col_dchk_ld);			
	}
}


void init_row_chk(ABFTEnv * abftEnv, double * A, int lda) {

	for (int i = 0; i < abftEnv->gpu_col; i += abftEnv->chk_nb) {		
		magma_dgemm(MagmaNoTrans, MagmaNoTrans,
					abftEnv->gpu_row, 2, abftEnv->chk_nb,
					MAGMA_D_ONE, 
					A + i * lda, lda,
					abftEnv->vd2, abftEnv->vd2_ld,
					MAGMA_D_ZERO, 
					ROW_CHK(0, i / abftEnv->chk_nb), abftEnv->row_dchk_ld);			
	}
}




void initializeABFTEnv(ABFTEnv * abftEnv, int chk_nb, 
						double * A, int lda,
						int gpu_row, int gpu_col,
						int cpu_row, int cpu_col,
						magma_queue_t * stream,
						bool DEBUG) {

	abftEnv->chk_nb = chk_nb;

	abftEnv->gpu_row = gpu_row;
	abftEnv->gpu_col = gpu_col;
	abftEnv->cpu_row = cpu_row;
	abftEnv->cpu_col = cpu_col;

	abftEnv->stream = stream;

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
        printMatrix_gpu(abftEnv->vd, abftEnv->vd_ld, 2, abftEnv->chk_nb, -1, -1);
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
    magma_dmalloc_pinned(&(abftEnv->col_hchk), (cpu_row / chk_nb) * 2 * cpu_col * sizeof(double));
    abftEnv->col_hchk_ld = (cpu_row / chk_nb) * 2;
    cout << "done." << endl;

    /* allocate space for update column checksum on CPU */
    cout << "allocate space for row checksum on CPU......";
    magma_dmalloc_pinned(&(abftEnv->row_hchk), cpu_row * (cpu_col / chk_nb) * 2 * sizeof(double));
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
    init_col_chk(abftEnv, A, lda);
    cout << "done." << endl;

    cout << "row checksums initiallization......";
    init_row_chk(abftEnv, A, lda);
    cout << "done." << endl;

    if (true) {
    	// cout << "input matrix:" << endl;
     //    printMatrix_gpu(A, lda, gpu_row, gpu_col, 4, 4);
     //    cout << "column checksum matrix on GPU:" << endl;
     //    printMatrix_gpu(abftEnv->col_dchk, abftEnv->col_dchk_ld,
     //    	 			(abftEnv->gpu_row / abftEnv->chk_nb) * 2, abftEnv->gpu_col, 
     //    	 			2, chk_nb);
     //    cout << "row checksum matrix on GPU:" << endl;
     //    printMatrix_gpu(abftEnv->row_dchk, abftEnv->row_dchk_ld,
     //    	 			abftEnv->gpu_row, (abftEnv->gpu_col / abftEnv->chk_nb) * 2, 
     //    	 			chk_nb, 2);

    	// for (int col = chk_nb; col < gpu_col; col += chk_nb) {
    	// 	for (int row = chk_nb; row < gpu_row; row += chk_nb) {
    	// 		cout << "size: " << row << "-" << col << endl;
    	// 		at_col_chk_recal(abftEnv, A, lda, row, col);
		   //      col_debug(A, lda, chk_nb, row, col,
		   //      					  abftEnv->col_dchk, abftEnv->col_dchk_ld,
		   //      					  abftEnv->chk1, abftEnv->chk1_ld,
		   //      					  abftEnv->chk2, abftEnv->chk2_ld,
		   //      					  stream[1]);

		        // at_row_chk_recal(abftEnv, A, lda, row, col);
		        // col_debug(A, lda, chk_nb, row, col,
		        // 					  abftEnv->row_dchk, abftEnv->row_dchk_ld,
		        // 					  abftEnv->chk21, abftEnv->chk21_ld,
		        // 					  abftEnv->chk22, abftEnv->chk22_ld,
		        // 					  stream[1]);
		        // cout << "update chk:" << endl;
		        // printMatrix_gpu(abftEnv->col_dchk, abftEnv->col_dchk_ld,
        	 // 			16, 16, 
        	 // 			chk_nb, 2);
		        // cout << "chk1:" << endl;
		        // printMatrix_gpu(abftEnv->chk1, abftEnv->chk1_ld,
        	 // 			16, 16, 
        	 // 			chk_nb, 2);
		        // cout << "chk2:" << endl;
		        // printMatrix_gpu(abftEnv->chk2, abftEnv->chk2_ld,
        	 // 			16, 16, 
        	 // 			chk_nb, 2);
    		}
    	}

        // at_col_chk_recal(abftEnv, A, lda, gpu_row, gpu_col);
        // col_detect_correct(A, lda, chk_nb, gpu_row, gpu_col,
        // 					  abftEnv->col_dchk, abftEnv->col_dchk_ld,
        // 					  abftEnv->chk1, abftEnv->chk1_ld,
        // 					  abftEnv->chk2, abftEnv->chk2_ld,
        // 					  stream[1]);

        // at_row_chk_recal(abftEnv, A, lda, gpu_row, gpu_col);
        // row_detect_correct(A, lda, chk_nb, gpu_row, gpu_col,
        // 					  abftEnv->row_dchk, abftEnv->row_dchk_ld,
        // 					  abftEnv->chk21, abftEnv->chk21_ld,
        // 					  abftEnv->chk22, abftEnv->chk22_ld,
        // 					  stream[1]);
       
//    }






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



//check each block of data based on last time check
void MemoryErrorCheck(ABFTEnv * abftEnv, double * A, int lda) {
	for (int i = 0; i < abftEnv->gpu_row/abftEnv->chk_nb; i++) {
		for (int j = 0; j < abftEnv->gpu_col/abftEnv->chk_nb; j++) {
			time_t temp = *(abftEnv->lastCheckTime + j * abftEnv->lastCheckTime_ld + i);
			if (time(NULL) - temp > abftEnv->T) {
				// we should do check on block[i, j]
				ABFTCheck(abftEnv, A + j*abftEnv->chk_nb*lda + i * abftEnv->chk_nb, lda,
						  abftEnv->chk_nb, abftEnv->chk_nb,
						  COL_CHK(i, j), abftEnv->col_dchk_ld);
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









