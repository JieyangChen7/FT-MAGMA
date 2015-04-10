/**
 *
 * @file plasma_ds.h
 *
 *  PLASMA header file for iterative refinement routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.6.0
 * @author Emmanuel Agullo
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @generated ds Tue Jan  7 11:44:39 2014
 *
 **/
#ifndef _PLASMA_DS_H_
#define _PLASMA_DS_H_

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of math functions (LAPACK layout) - alphabetical order
 **/
int PLASMA_dsgesv(int N, int NRHS, double *A, int LDA, int * IPIV, double *B, int LDB, double *X, int LDX, int *ITER);
int PLASMA_dsposv(PLASMA_enum uplo, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *X, int LDX, int *ITER);
/* int PLASMA_dsgels(PLASMA_enum trans, int M, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *X, int LDX, int *ITER); */
int PLASMA_dsungesv(PLASMA_enum trans, int N, int NRHS, double *A, int LDA, double *B, int LDB, double *X, int LDX, int *ITER);

/** ****************************************************************************
 *  Declarations of math functions (tile layout) - alphabetical order
 **/
int PLASMA_dsgesv_Tile(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_desc *X, int *ITER);
int PLASMA_dsposv_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_desc *X, int *ITER);
/* int PLASMA_dsgels_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER); */
int PLASMA_dsungesv_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER);

/** ****************************************************************************
 *  Declarations of math functions (tile layout, asynchronous execution) - alphabetical order
 **/
int PLASMA_dsgesv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_dsposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request);
/* int PLASMA_dsgels_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request); */
int PLASMA_dsungesv_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif
