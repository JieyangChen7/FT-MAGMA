/**
 *
 * @file plasma_z.h
 *
 *  PLASMA header file for double _Complex routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.6.0
 * @author Jakub Kurzak
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Azzam Haidar
 * @date 2010-11-15
 * @precisions normal z -> c d s
 *
 **/
#ifndef _PLASMA_Z_H_
#define _PLASMA_Z_H_

#undef REAL
#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of math functions (LAPACK layout) - alphabetical order
 **/
int PLASMA_zgebrd(PLASMA_enum jobq, PLASMA_enum jobpt, int M, int N, PLASMA_Complex64_t *A, int LDA, double *D, double *E, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ, PLASMA_Complex64_t *PT, int LDPT);
int PLASMA_zgecon(PLASMA_enum norm, int N, PLASMA_Complex64_t *A, int LDA, double anorm, double *rcond);
int PLASMA_zpocon(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, double anorm, double *rcond);
int PLASMA_zgelqf(int M, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT);
int PLASMA_zgelqs(int M, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zgels(PLASMA_enum trans, int M, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zgemm(PLASMA_enum transA, PLASMA_enum transB, int M, int N, int K, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC);
int PLASMA_zgeqp3( int M, int N, PLASMA_Complex64_t *A, int LDA, int *jpvt, PLASMA_Complex64_t *tau, PLASMA_Complex64_t *work, double *rwork);
int PLASMA_zgeqrf(int M, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT);
int PLASMA_zgeqrs(int M, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zgesv(int N, int NRHS, PLASMA_Complex64_t *A, int LDA, int *IPIV, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zgesv_incpiv(int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descL, int *IPIV, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zgesvd(PLASMA_enum jobu, PLASMA_enum jobvt, int M, int N, PLASMA_Complex64_t *A, int LDA, double *S, PLASMA_desc *descT, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT);
int PLASMA_zgesdd(PLASMA_enum jobu, PLASMA_enum jobvt, int M, int N, PLASMA_Complex64_t *A, int LDA, double *S, PLASMA_desc *descT, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT);
int PLASMA_zgetrf(  int M, int N, PLASMA_Complex64_t *A, int LDA, int *IPIV);
int PLASMA_zgetrf_incpiv(int M, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descL, int *IPIV);
int PLASMA_zgetrf_nopiv( int M, int N, PLASMA_Complex64_t *A, int LDA);
int PLASMA_zgetrf_tntpiv(int M, int N, PLASMA_Complex64_t *A, int LDA, int *IPIV);
int PLASMA_zgetri(int N, PLASMA_Complex64_t *A, int LDA, int *IPIV);
int PLASMA_zgetrs(PLASMA_enum trans, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, const int *IPIV, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zgetrs_incpiv(PLASMA_enum trans, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descL, const int *IPIV, PLASMA_Complex64_t *B, int LDB);
#ifdef COMPLEX
int PLASMA_zhemm(PLASMA_enum side, PLASMA_enum uplo, int M, int N, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC);
int PLASMA_zherk(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, double alpha, PLASMA_Complex64_t *A, int LDA, double beta, PLASMA_Complex64_t *C, int LDC);
int PLASMA_zher2k(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, double beta, PLASMA_Complex64_t *C, int LDC);
#endif
int PLASMA_zheev(PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, double *W, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zheevd(PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, double *W, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zheevr(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, double vl, double vu, int il, int iu, double abstol, int *nbcomputedeig, double *W, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zhegv(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, double *W, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zhegvd(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, double *W, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zhegst(PLASMA_enum itype, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zhetrd(PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, double *D, double *E, PLASMA_desc *descT, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zlacpy(PLASMA_enum uplo, int M, int N, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
double PLASMA_zlange(PLASMA_enum norm, int M, int N, PLASMA_Complex64_t *A, int LDA);
#ifdef COMPLEX
double PLASMA_zlanhe(PLASMA_enum norm, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);
#endif
double PLASMA_zlansy(PLASMA_enum norm, PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);
double PLASMA_zlantr(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N, PLASMA_Complex64_t *A, int LDA);
int PLASMA_zlaset(PLASMA_enum uplo, int M, int N, PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta, PLASMA_Complex64_t *A, int LDA);
int PLASMA_zlaswp( int N, PLASMA_Complex64_t *A, int LDA, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_zlaswpc(int N, PLASMA_Complex64_t *A, int LDA, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_zlauum(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);
#ifdef COMPLEX
int PLASMA_zplghe( double bump, int N, PLASMA_Complex64_t *A, int LDA, unsigned long long int seed);
#endif
int PLASMA_zplgsy( PLASMA_Complex64_t bump, int N, PLASMA_Complex64_t *A, int LDA, unsigned long long int seed);
int PLASMA_zplrnt( int M, int N, PLASMA_Complex64_t *A, int LDA, unsigned long long int seed);
int PLASMA_zpltmg( PLASMA_enum mtxtype, int M, int N, PLASMA_Complex64_t *A, int LDA, unsigned long long int seed);
int PLASMA_zposv(PLASMA_enum uplo, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zpotrf(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);
int PLASMA_zpotri(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);
int PLASMA_zpotrs(PLASMA_enum uplo, int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zsymm(PLASMA_enum side, PLASMA_enum uplo, int M, int N, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC);
int PLASMA_zsyrk(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC);
int PLASMA_zsyr2k(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB, PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int LDC);
int PLASMA_ztrmm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
int PLASMA_ztrsm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
int PLASMA_ztrsmpl(int N, int NRHS, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descL, const int *IPIV, PLASMA_Complex64_t *B, int LDB);
int PLASMA_ztrsmrv(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, PLASMA_Complex64_t alpha, PLASMA_Complex64_t *A, int LDA, PLASMA_Complex64_t *B, int LDB);
int PLASMA_ztrtri(PLASMA_enum uplo, PLASMA_enum diag, int N, PLASMA_Complex64_t *A, int LDA);
int PLASMA_zunglq(int M, int N, int K, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zungqr(int M, int N, int K, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zunmlq(PLASMA_enum side, PLASMA_enum trans, int M, int N, int K, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);
int PLASMA_zunmqr(PLASMA_enum side, PLASMA_enum trans, int M, int N, int K, PLASMA_Complex64_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex64_t *B, int LDB);

int PLASMA_zgecfi(int m, int n, PLASMA_Complex64_t *A, PLASMA_enum fin, int imb, int inb, PLASMA_enum fout, int omb, int onb);
int PLASMA_zgetmi(int m, int n, PLASMA_Complex64_t *A, PLASMA_enum fin, int mb, int nb);

/** ****************************************************************************
 * Declarations of math functions (tile layout) - alphabetical order
 **/
int PLASMA_zgebrd_Tile(PLASMA_enum jobq, PLASMA_enum jobpt, PLASMA_desc *A, double *D, double *E, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ, PLASMA_Complex64_t *PT, int LDPT);
int PLASMA_zgecon_Tile(PLASMA_enum norm, PLASMA_desc *A, double anorm, double *rcond);
int PLASMA_zpocon_Tile(PLASMA_enum uplo, PLASMA_desc *A, double anorm, double *rcond);
int PLASMA_zgelqf_Tile(PLASMA_desc *A, PLASMA_desc *T);
int PLASMA_zgelqs_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_zgels_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_zgemm_Tile(PLASMA_enum transA, PLASMA_enum transB, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C);
int PLASMA_zgeqp3_Tile( PLASMA_desc *A, int *jpvt, PLASMA_Complex64_t *tau, PLASMA_Complex64_t *work, double *rwork);
int PLASMA_zgeqrf_Tile(PLASMA_desc *A, PLASMA_desc *T);
int PLASMA_zgeqrs_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_zgesv_Tile(PLASMA_desc *A, int *IPIV, PLASMA_desc *B);
int PLASMA_zgesv_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_desc *B);
int PLASMA_zgesvd_Tile(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, double *S, PLASMA_desc *T, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT);
int PLASMA_zgesdd_Tile(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, double *S, PLASMA_desc *T, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT);
int PLASMA_zgetrf_Tile(  PLASMA_desc *A, int *IPIV);
int PLASMA_zgetrf_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV);
int PLASMA_zgetrf_nopiv_Tile( PLASMA_desc *A);
int PLASMA_zgetrf_tntpiv_Tile(PLASMA_desc *A, int *IPIV);
int PLASMA_zgetri_Tile(PLASMA_desc *A, int *IPIV);
int PLASMA_zgetrs_Tile(PLASMA_enum trans, PLASMA_desc *A, const int *IPIV, PLASMA_desc *B);
int PLASMA_zgetrs_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B);
#ifdef COMPLEX
int PLASMA_zhemm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C);
int PLASMA_zherk_Tile(PLASMA_enum uplo, PLASMA_enum trans, double alpha, PLASMA_desc *A, double beta, PLASMA_desc *C);
int PLASMA_zher2k_Tile(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, double beta, PLASMA_desc *C);
#endif
int PLASMA_zheev_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, double *W, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zheevd_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, double *W, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zheevr_Tile(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, PLASMA_desc *A, double vl, double vu, int il, int iu, double abstol, int *nbcomputedeig, double *W, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zhegv_Tile( PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, double *W, PLASMA_desc *T, PLASMA_desc *Q);
int PLASMA_zhegvd_Tile(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, double *W, PLASMA_desc *T, PLASMA_desc *Q);
int PLASMA_zhegst_Tile(PLASMA_enum itype, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_zhetrd_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, double *D, double *E, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ);
int PLASMA_zlacpy_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
double PLASMA_zlange_Tile(PLASMA_enum norm, PLASMA_desc *A);
#ifdef COMPLEX
double PLASMA_zlanhe_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A);
#endif
double PLASMA_zlansy_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A);
double PLASMA_zlantr_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A);
int PLASMA_zlaset_Tile(PLASMA_enum uplo, PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta, PLASMA_desc *A);
int PLASMA_zlaswp_Tile(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_zlaswpc_Tile(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_zlauum_Tile(PLASMA_enum uplo, PLASMA_desc *A);
#ifdef COMPLEX
int PLASMA_zplghe_Tile(double bump, PLASMA_desc *A, unsigned long long int seed);
#endif
int PLASMA_zplgsy_Tile(PLASMA_Complex64_t bump, PLASMA_desc *A, unsigned long long int seed);
int PLASMA_zplrnt_Tile(PLASMA_desc *A, unsigned long long int seed);
int PLASMA_zpltmg_Tile(PLASMA_enum mtxtype, PLASMA_desc *A, unsigned long long int seed);
int PLASMA_zposv_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_zpotrf_Tile(PLASMA_enum uplo, PLASMA_desc *A);
int PLASMA_zpotri_Tile(PLASMA_enum uplo, PLASMA_desc *A);
int PLASMA_zpotrs_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_zsymm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C);
int PLASMA_zsyrk_Tile(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_Complex64_t beta, PLASMA_desc *C);
int PLASMA_zsyr2k_Tile(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C);
int PLASMA_ztrmm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ztrsm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ztrsmpl_Tile(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B);
int PLASMA_ztrsmrv_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ztrtri_Tile(PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A);
int PLASMA_zunglq_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_zungqr_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_zunmlq_Tile(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_zunmqr_Tile(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);

/** ****************************************************************************
 * Declarations of math functions (tile layout, asynchronous execution) - alphabetical order
 **/
int PLASMA_zgebrd_Tile_Async(PLASMA_enum jobq, PLASMA_enum jobpt, PLASMA_desc *A, double *S, double *E, PLASMA_desc *T, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgecon_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, double anorm, double *rcond, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zpocon_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, double anorm, double *rcond, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgelqf_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgelqs_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgels_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgeqp3_Tile_Async( PLASMA_desc *A, int *jpvt, PLASMA_Complex64_t *tau, PLASMA_Complex64_t *work, double *rwork, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgeqrf_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgeqrs_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgesv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgesv_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgesvd_Tile_Async(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, double *S, PLASMA_desc *T, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgesdd_Tile_Async(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, double *S, PLASMA_desc *T, PLASMA_Complex64_t *U, int LDU, PLASMA_Complex64_t *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetrf_Tile_Async(  PLASMA_desc *A, int *IPIV, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetrf_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetrf_nopiv_Tile_Async( PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetrf_tntpiv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *W, int *Wpivot, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetri_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *W, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetrs_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetrs_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_zhemm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zherk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, double alpha, PLASMA_desc *A, double beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zher2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, double beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_zheev_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, double *W, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zheevd_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, double *W, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zheevr_Tile_Async(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, PLASMA_desc *A, double vl, double vu, int il, int iu, double abstol, int *nbcomputedeig, double *W, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zhegv_Tile_Async( PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, double *W, PLASMA_desc *T, PLASMA_desc *Q, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zhegvd_Tile_Async(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, double *W, PLASMA_desc *T, PLASMA_desc *Q, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zhegst_Tile_Async(PLASMA_enum itype, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zhetrd_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, double *D, double *E, PLASMA_desc *T, PLASMA_Complex64_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlacpy_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlange_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, double *result, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_zlanhe_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A, double *result, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_zlansy_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A, double *result, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlantr_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A, double *result, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlaset_Tile_Async(PLASMA_enum uplo, PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlaswp_Tile_Async(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlaswpc_Tile_Async(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zlauum_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_zplghe_Tile_Async(double bump, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_zplgsy_Tile_Async(PLASMA_Complex64_t bump, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zplrnt_Tile_Async(PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zpltmg_Tile_Async(PLASMA_enum mtxtype, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zpotrf_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zpotri_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zpotrs_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zsymm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zsyrk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_Complex64_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zsyr2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex64_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ztrmm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ztrsm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ztrsmpl_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ztrsmrv_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex64_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ztrtri_Tile_Async(PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zunglq_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zungqr_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zunmlq_Tile_Async(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zunmqr_Tile_Async(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);

int PLASMA_zgecfi_Async(int m, int n, PLASMA_Complex64_t *A, PLASMA_enum f_in, int imb, int inb, PLASMA_enum f_out, int omb, int onb, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zgetmi_Async(int m, int n, PLASMA_Complex64_t *A, PLASMA_enum f_in, int mb, int inb, PLASMA_sequence *sequence, PLASMA_request *request);

/** ****************************************************************************
 * Declarations of workspace allocation functions (tile layout) - alphabetical order
 **/
int PLASMA_Alloc_Workspace_zgesv_incpiv( int N,        PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_zgetrf_incpiv(int M, int N, PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_zgebrd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgeev( int N,        PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgehrd(int N,        PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgelqf(int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_zgels( int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_zgeqrf(int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_zgesdd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgesvd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zheev( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zheevd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zheevr(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zhegv( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zhegvd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zhetrd(int M, int N, PLASMA_desc **descT);

/** ****************************************************************************
 * Declarations of workspace allocation functions (tile layout, asynchronous execution) - alphabetical order
 **/

/* Workspace required only for asynchronous interface */
int PLASMA_Alloc_Workspace_zgetrf_tntpiv_Tile(PLASMA_desc *A, PLASMA_desc *W, int **Wpivot);
int PLASMA_Alloc_Workspace_zgetri_Tile_Async( PLASMA_desc *A, PLASMA_desc *W);

/* Warning: Those functions are deprecated */
int PLASMA_Alloc_Workspace_zgelqf_Tile(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgels_Tile( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgeqrf_Tile(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_zgesv_incpiv_Tile (int N, PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_zgetrf_incpiv_Tile(int N, PLASMA_desc **descL, int **IPIV);

/** ****************************************************************************
 * Auxiliary function prototypes
 **/
int PLASMA_zLapack_to_Tile(PLASMA_Complex64_t *Af77, int LDA, PLASMA_desc *A);
int PLASMA_zTile_to_Lapack(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int LDA);
int PLASMA_zLapack_to_Tile_Async(PLASMA_Complex64_t *Af77, int LDA, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_zTile_to_Lapack_Async(PLASMA_desc *A, PLASMA_Complex64_t *Af77, int LDA, PLASMA_sequence *sequence, PLASMA_request *request);

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
