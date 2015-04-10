/**
 *
 * @file plasma_zc.h
 *
 *  PLASMA header file for iterative refinement routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.6.0
 * @author Emmanuel Agullo
 * @author Mathieu Faverge
 * @date 2010-11-15
 * @precisions mixed zc -> ds
 *
 **/
#ifndef _PLASMA_ZC_H_
#define _PLASMA_ZC_H_

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of math functions (LAPACK layout) - alphabetical order
 **/
int PLASMA_zcgesv(int N, int NRHS, PLASMA_Complex64_t *A, int LDA, int * IPIV, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t *X, int LDX, int *ITER);
int PLASMA_zcposv(PLASMA_enum uplo, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t *X, int LDX, int *ITER);
/* int PLASMA_zcgels(PLASMA_enum trans, int M, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t *X, int LDX, int *ITER); */
int PLASMA_zcungesv(PLASMA_enum trans, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t *X, int LDX, int *ITER);

/** ****************************************************************************
 *  Declarations of math functions (tile layout) - alphabetical order
 **/
int PLASMA_zcgesv_Tile(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_desc *X, int *ITER);
int PLASMA_zcposv_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_desc *X, int *ITER);
/* int PLASMA_zcgels_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER); */
int PLASMA_zcungesv_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER);

/** ****************************************************************************
 *  Declarations of math functions (tile layout, asynchronous execution) - alphabetical order
 **/
int PLASMA_zcgesv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zcposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request);
/* int PLASMA_zcgels_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request); */
int PLASMA_zcungesv_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_desc *X, int *ITER, PLASMA_sequence *sequence, PLASMA_request *request);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif
