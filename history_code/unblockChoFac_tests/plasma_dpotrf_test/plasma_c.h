/**
 *
 * @file plasma_c.h
 *
 *  PLASMA header file for float _Complex routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.6.0
 * @author Jakub Kurzak
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Azzam Haidar
 * @date 2010-11-15
 * @generated c Tue Jan  7 11:44:39 2014
 *
 **/
#ifndef _PLASMA_C_H_
#define _PLASMA_C_H_

#undef REAL
#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of math functions (LAPACK layout) - alphabetical order
 **/
int PLASMA_cgebrd(PLASMA_enum jobq, PLASMA_enum jobpt, int M, int N, PLASMA_Complex32_t *A, int LDA, float *D, float *E, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ, PLASMA_Complex32_t *PT, int LDPT);
int PLASMA_cgecon(PLASMA_enum norm, int N, PLASMA_Complex32_t *A, int LDA, float anorm, float *rcond);
int PLASMA_cpocon(PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, float anorm, float *rcond);
int PLASMA_cgelqf(int M, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT);
int PLASMA_cgelqs(int M, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cgels(PLASMA_enum trans, int M, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cgemm(PLASMA_enum transA, PLASMA_enum transB, int M, int N, int K, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int LDC);
int PLASMA_cgeqp3( int M, int N, PLASMA_Complex32_t *A, int LDA, int *jpvt, PLASMA_Complex32_t *tau, PLASMA_Complex32_t *work, float *rwork);
int PLASMA_cgeqrf(int M, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT);
int PLASMA_cgeqrs(int M, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cgesv(int N, int NRHS, PLASMA_Complex32_t *A, int LDA, int *IPIV, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cgesv_incpiv(int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descL, int *IPIV, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cgesvd(PLASMA_enum jobu, PLASMA_enum jobvt, int M, int N, PLASMA_Complex32_t *A, int LDA, float *S, PLASMA_desc *descT, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT);
int PLASMA_cgesdd(PLASMA_enum jobu, PLASMA_enum jobvt, int M, int N, PLASMA_Complex32_t *A, int LDA, float *S, PLASMA_desc *descT, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT);
int PLASMA_cgetrf(  int M, int N, PLASMA_Complex32_t *A, int LDA, int *IPIV);
int PLASMA_cgetrf_incpiv(int M, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descL, int *IPIV);
int PLASMA_cgetrf_nopiv( int M, int N, PLASMA_Complex32_t *A, int LDA);
int PLASMA_cgetrf_tntpiv(int M, int N, PLASMA_Complex32_t *A, int LDA, int *IPIV);
int PLASMA_cgetri(int N, PLASMA_Complex32_t *A, int LDA, int *IPIV);
int PLASMA_cgetrs(PLASMA_enum trans, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, const int *IPIV, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cgetrs_incpiv(PLASMA_enum trans, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descL, const int *IPIV, PLASMA_Complex32_t *B, int LDB);
#ifdef COMPLEX
int PLASMA_chemm(PLASMA_enum side, PLASMA_enum uplo, int M, int N, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int LDC);
int PLASMA_cherk(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, float alpha, PLASMA_Complex32_t *A, int LDA, float beta, PLASMA_Complex32_t *C, int LDC);
int PLASMA_cher2k(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, float beta, PLASMA_Complex32_t *C, int LDC);
#endif
int PLASMA_cheev(PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, float *W, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_cheevd(PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, float *W, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_cheevr(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, float vl, float vu, int il, int iu, float abstol, int *nbcomputedeig, float *W, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_chegv(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, float *W, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_chegvd(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, float *W, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_chegst(PLASMA_enum itype, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
int PLASMA_chetrd(PLASMA_enum jobz, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, float *D, float *E, PLASMA_desc *descT, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_clacpy(PLASMA_enum uplo, int M, int N, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
float PLASMA_clange(PLASMA_enum norm, int M, int N, PLASMA_Complex32_t *A, int LDA);
#ifdef COMPLEX
float PLASMA_clanhe(PLASMA_enum norm, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA);
#endif
float PLASMA_clansy(PLASMA_enum norm, PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA);
float PLASMA_clantr(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N, PLASMA_Complex32_t *A, int LDA);
int PLASMA_claset(PLASMA_enum uplo, int M, int N, PLASMA_Complex32_t alpha, PLASMA_Complex32_t beta, PLASMA_Complex32_t *A, int LDA);
int PLASMA_claswp( int N, PLASMA_Complex32_t *A, int LDA, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_claswpc(int N, PLASMA_Complex32_t *A, int LDA, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_clauum(PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA);
#ifdef COMPLEX
int PLASMA_cplghe( float bump, int N, PLASMA_Complex32_t *A, int LDA, unsigned long long int seed);
#endif
int PLASMA_cplgsy( PLASMA_Complex32_t bump, int N, PLASMA_Complex32_t *A, int LDA, unsigned long long int seed);
int PLASMA_cplrnt( int M, int N, PLASMA_Complex32_t *A, int LDA, unsigned long long int seed);
int PLASMA_cpltmg( PLASMA_enum mtxtype, int M, int N, PLASMA_Complex32_t *A, int LDA, unsigned long long int seed);
int PLASMA_cposv(PLASMA_enum uplo, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cpotrf(PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA);
int PLASMA_cpotri(PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA);
int PLASMA_cpotrs(PLASMA_enum uplo, int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
int PLASMA_csymm(PLASMA_enum side, PLASMA_enum uplo, int M, int N, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int LDC);
int PLASMA_csyrk(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int LDC);
int PLASMA_csyr2k(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB, PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int LDC);
int PLASMA_ctrmm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
int PLASMA_ctrsm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
int PLASMA_ctrsmpl(int N, int NRHS, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descL, const int *IPIV, PLASMA_Complex32_t *B, int LDB);
int PLASMA_ctrsmrv(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, PLASMA_Complex32_t alpha, PLASMA_Complex32_t *A, int LDA, PLASMA_Complex32_t *B, int LDB);
int PLASMA_ctrtri(PLASMA_enum uplo, PLASMA_enum diag, int N, PLASMA_Complex32_t *A, int LDA);
int PLASMA_cunglq(int M, int N, int K, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cungqr(int M, int N, int K, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cunmlq(PLASMA_enum side, PLASMA_enum trans, int M, int N, int K, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);
int PLASMA_cunmqr(PLASMA_enum side, PLASMA_enum trans, int M, int N, int K, PLASMA_Complex32_t *A, int LDA, PLASMA_desc *descT, PLASMA_Complex32_t *B, int LDB);

int PLASMA_cgecfi(int m, int n, PLASMA_Complex32_t *A, PLASMA_enum fin, int imb, int inb, PLASMA_enum fout, int omb, int onb);
int PLASMA_cgetmi(int m, int n, PLASMA_Complex32_t *A, PLASMA_enum fin, int mb, int nb);

/** ****************************************************************************
 * Declarations of math functions (tile layout) - alphabetical order
 **/
int PLASMA_cgebrd_Tile(PLASMA_enum jobq, PLASMA_enum jobpt, PLASMA_desc *A, float *D, float *E, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ, PLASMA_Complex32_t *PT, int LDPT);
int PLASMA_cgecon_Tile(PLASMA_enum norm, PLASMA_desc *A, float anorm, float *rcond);
int PLASMA_cpocon_Tile(PLASMA_enum uplo, PLASMA_desc *A, float anorm, float *rcond);
int PLASMA_cgelqf_Tile(PLASMA_desc *A, PLASMA_desc *T);
int PLASMA_cgelqs_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_cgels_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_cgemm_Tile(PLASMA_enum transA, PLASMA_enum transB, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C);
int PLASMA_cgeqp3_Tile( PLASMA_desc *A, int *jpvt, PLASMA_Complex32_t *tau, PLASMA_Complex32_t *work, float *rwork);
int PLASMA_cgeqrf_Tile(PLASMA_desc *A, PLASMA_desc *T);
int PLASMA_cgeqrs_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_cgesv_Tile(PLASMA_desc *A, int *IPIV, PLASMA_desc *B);
int PLASMA_cgesv_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_desc *B);
int PLASMA_cgesvd_Tile(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT);
int PLASMA_cgesdd_Tile(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT);
int PLASMA_cgetrf_Tile(  PLASMA_desc *A, int *IPIV);
int PLASMA_cgetrf_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV);
int PLASMA_cgetrf_nopiv_Tile( PLASMA_desc *A);
int PLASMA_cgetrf_tntpiv_Tile(PLASMA_desc *A, int *IPIV);
int PLASMA_cgetri_Tile(PLASMA_desc *A, int *IPIV);
int PLASMA_cgetrs_Tile(PLASMA_enum trans, PLASMA_desc *A, const int *IPIV, PLASMA_desc *B);
int PLASMA_cgetrs_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B);
#ifdef COMPLEX
int PLASMA_chemm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C);
int PLASMA_cherk_Tile(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, float beta, PLASMA_desc *C);
int PLASMA_cher2k_Tile(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C);
#endif
int PLASMA_cheev_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_cheevd_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_cheevr_Tile(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, PLASMA_desc *A, float vl, float vu, int il, int iu, float abstol, int *nbcomputedeig, float *W, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_chegv_Tile( PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q);
int PLASMA_chegvd_Tile(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q);
int PLASMA_chegst_Tile(PLASMA_enum itype, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_chetrd_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *D, float *E, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ);
int PLASMA_clacpy_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
float PLASMA_clange_Tile(PLASMA_enum norm, PLASMA_desc *A);
#ifdef COMPLEX
float PLASMA_clanhe_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A);
#endif
float PLASMA_clansy_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A);
float PLASMA_clantr_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A);
int PLASMA_claset_Tile(PLASMA_enum uplo, PLASMA_Complex32_t alpha, PLASMA_Complex32_t beta, PLASMA_desc *A);
int PLASMA_claswp_Tile(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_claswpc_Tile(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_clauum_Tile(PLASMA_enum uplo, PLASMA_desc *A);
#ifdef COMPLEX
int PLASMA_cplghe_Tile(float bump, PLASMA_desc *A, unsigned long long int seed);
#endif
int PLASMA_cplgsy_Tile(PLASMA_Complex32_t bump, PLASMA_desc *A, unsigned long long int seed);
int PLASMA_cplrnt_Tile(PLASMA_desc *A, unsigned long long int seed);
int PLASMA_cpltmg_Tile(PLASMA_enum mtxtype, PLASMA_desc *A, unsigned long long int seed);
int PLASMA_cposv_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_cpotrf_Tile(PLASMA_enum uplo, PLASMA_desc *A);
int PLASMA_cpotri_Tile(PLASMA_enum uplo, PLASMA_desc *A);
int PLASMA_cpotrs_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_csymm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C);
int PLASMA_csyrk_Tile(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_Complex32_t beta, PLASMA_desc *C);
int PLASMA_csyr2k_Tile(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C);
int PLASMA_ctrmm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ctrsm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ctrsmpl_Tile(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B);
int PLASMA_ctrsmrv_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ctrtri_Tile(PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A);
int PLASMA_cunglq_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_cungqr_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_cunmlq_Tile(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_cunmqr_Tile(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);

/** ****************************************************************************
 * Declarations of math functions (tile layout, asynchronous execution) - alphabetical order
 **/
int PLASMA_cgebrd_Tile_Async(PLASMA_enum jobq, PLASMA_enum jobpt, PLASMA_desc *A, float *S, float *E, PLASMA_desc *T, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgecon_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, float anorm, float *rcond, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cpocon_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, float anorm, float *rcond, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgelqf_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgelqs_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgels_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgeqp3_Tile_Async( PLASMA_desc *A, int *jpvt, PLASMA_Complex32_t *tau, PLASMA_Complex32_t *work, float *rwork, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgeqrf_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgeqrs_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgesv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgesv_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgesvd_Tile_Async(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgesdd_Tile_Async(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, PLASMA_Complex32_t *U, int LDU, PLASMA_Complex32_t *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetrf_Tile_Async(  PLASMA_desc *A, int *IPIV, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetrf_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetrf_nopiv_Tile_Async( PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetrf_tntpiv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *W, int *Wpivot, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetri_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *W, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetrs_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetrs_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_chemm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cherk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cher2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_cheev_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cheevd_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cheevr_Tile_Async(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, PLASMA_desc *A, float vl, float vu, int il, int iu, float abstol, int *nbcomputedeig, float *W, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_chegv_Tile_Async( PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_chegvd_Tile_Async(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_chegst_Tile_Async(PLASMA_enum itype, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_chetrd_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *D, float *E, PLASMA_desc *T, PLASMA_Complex32_t *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_clacpy_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_clange_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_clanhe_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_clansy_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_clantr_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_claset_Tile_Async(PLASMA_enum uplo, PLASMA_Complex32_t alpha, PLASMA_Complex32_t beta, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_claswp_Tile_Async(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_claswpc_Tile_Async(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_clauum_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_cplghe_Tile_Async(float bump, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_cplgsy_Tile_Async(PLASMA_Complex32_t bump, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cplrnt_Tile_Async(PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cpltmg_Tile_Async(PLASMA_enum mtxtype, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cpotrf_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cpotri_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cpotrs_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_csymm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_csyrk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_Complex32_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_csyr2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_Complex32_t beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ctrmm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ctrsm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ctrsmpl_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ctrsmrv_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, PLASMA_Complex32_t alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ctrtri_Tile_Async(PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cunglq_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cungqr_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cunmlq_Tile_Async(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cunmqr_Tile_Async(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);

int PLASMA_cgecfi_Async(int m, int n, PLASMA_Complex32_t *A, PLASMA_enum f_in, int imb, int inb, PLASMA_enum f_out, int omb, int onb, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cgetmi_Async(int m, int n, PLASMA_Complex32_t *A, PLASMA_enum f_in, int mb, int inb, PLASMA_sequence *sequence, PLASMA_request *request);

/** ****************************************************************************
 * Declarations of workspace allocation functions (tile layout) - alphabetical order
 **/
int PLASMA_Alloc_Workspace_cgesv_incpiv( int N,        PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_cgetrf_incpiv(int M, int N, PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_cgebrd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgeev( int N,        PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgehrd(int N,        PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgelqf(int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_cgels( int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_cgeqrf(int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_cgesdd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgesvd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cheev( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cheevd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cheevr(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_chegv( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_chegvd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_chetrd(int M, int N, PLASMA_desc **descT);

/** ****************************************************************************
 * Declarations of workspace allocation functions (tile layout, asynchronous execution) - alphabetical order
 **/

/* Workspace required only for asynchronous interface */
int PLASMA_Alloc_Workspace_cgetrf_tntpiv_Tile(PLASMA_desc *A, PLASMA_desc *W, int **Wpivot);
int PLASMA_Alloc_Workspace_cgetri_Tile_Async( PLASMA_desc *A, PLASMA_desc *W);

/* Warning: Those functions are deprecated */
int PLASMA_Alloc_Workspace_cgelqf_Tile(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgels_Tile( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgeqrf_Tile(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_cgesv_incpiv_Tile (int N, PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_cgetrf_incpiv_Tile(int N, PLASMA_desc **descL, int **IPIV);

/** ****************************************************************************
 * Auxiliary function prototypes
 **/
int PLASMA_cLapack_to_Tile(PLASMA_Complex32_t *Af77, int LDA, PLASMA_desc *A);
int PLASMA_cTile_to_Lapack(PLASMA_desc *A, PLASMA_Complex32_t *Af77, int LDA);
int PLASMA_cLapack_to_Tile_Async(PLASMA_Complex32_t *Af77, int LDA, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_cTile_to_Lapack_Async(PLASMA_desc *A, PLASMA_Complex32_t *Af77, int LDA, PLASMA_sequence *sequence, PLASMA_request *request);

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
