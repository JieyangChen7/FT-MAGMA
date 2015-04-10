/**
 *
 * @file plasma_s.h
 *
 *  PLASMA header file for float routines
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.6.0
 * @author Jakub Kurzak
 * @author Hatem Ltaief
 * @author Mathieu Faverge
 * @author Azzam Haidar
 * @date 2010-11-15
 * @generated s Tue Jan  7 11:44:39 2014
 *
 **/
#ifndef _PLASMA_S_H_
#define _PLASMA_S_H_

#undef COMPLEX
#define REAL

#ifdef __cplusplus
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of math functions (LAPACK layout) - alphabetical order
 **/
int PLASMA_sgebrd(PLASMA_enum jobq, PLASMA_enum jobpt, int M, int N, float *A, int LDA, float *D, float *E, PLASMA_desc *descT, float *Q, int LDQ, float *PT, int LDPT);
int PLASMA_sgecon(PLASMA_enum norm, int N, float *A, int LDA, float anorm, float *rcond);
int PLASMA_spocon(PLASMA_enum uplo, int N, float *A, int LDA, float anorm, float *rcond);
int PLASMA_sgelqf(int M, int N, float *A, int LDA, PLASMA_desc *descT);
int PLASMA_sgelqs(int M, int N, int NRHS, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);
int PLASMA_sgels(PLASMA_enum trans, int M, int N, int NRHS, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);
int PLASMA_sgemm(PLASMA_enum transA, PLASMA_enum transB, int M, int N, int K, float alpha, float *A, int LDA, float *B, int LDB, float beta, float *C, int LDC);
int PLASMA_sgeqp3( int M, int N, float *A, int LDA, int *jpvt, float *tau, float *work, float *rwork);
int PLASMA_sgeqrf(int M, int N, float *A, int LDA, PLASMA_desc *descT);
int PLASMA_sgeqrs(int M, int N, int NRHS, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);
int PLASMA_sgesv(int N, int NRHS, float *A, int LDA, int *IPIV, float *B, int LDB);
int PLASMA_sgesv_incpiv(int N, int NRHS, float *A, int LDA, PLASMA_desc *descL, int *IPIV, float *B, int LDB);
int PLASMA_sgesvd(PLASMA_enum jobu, PLASMA_enum jobvt, int M, int N, float *A, int LDA, float *S, PLASMA_desc *descT, float *U, int LDU, float *VT, int LDVT);
int PLASMA_sgesdd(PLASMA_enum jobu, PLASMA_enum jobvt, int M, int N, float *A, int LDA, float *S, PLASMA_desc *descT, float *U, int LDU, float *VT, int LDVT);
int PLASMA_sgetrf(  int M, int N, float *A, int LDA, int *IPIV);
int PLASMA_sgetrf_incpiv(int M, int N, float *A, int LDA, PLASMA_desc *descL, int *IPIV);
int PLASMA_sgetrf_nopiv( int M, int N, float *A, int LDA);
int PLASMA_sgetrf_tntpiv(int M, int N, float *A, int LDA, int *IPIV);
int PLASMA_sgetri(int N, float *A, int LDA, int *IPIV);
int PLASMA_sgetrs(PLASMA_enum trans, int N, int NRHS, float *A, int LDA, const int *IPIV, float *B, int LDB);
int PLASMA_sgetrs_incpiv(PLASMA_enum trans, int N, int NRHS, float *A, int LDA, PLASMA_desc *descL, const int *IPIV, float *B, int LDB);
#ifdef COMPLEX
int PLASMA_ssymm(PLASMA_enum side, PLASMA_enum uplo, int M, int N, float alpha, float *A, int LDA, float *B, int LDB, float beta, float *C, int LDC);
int PLASMA_ssyrk(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, float alpha, float *A, int LDA, float beta, float *C, int LDC);
int PLASMA_ssyr2k(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, float alpha, float *A, int LDA, float *B, int LDB, float beta, float *C, int LDC);
#endif
int PLASMA_ssyev(PLASMA_enum jobz, PLASMA_enum uplo, int N, float *A, int LDA, float *W, PLASMA_desc *descT, float *Q, int LDQ);
int PLASMA_ssyevd(PLASMA_enum jobz, PLASMA_enum uplo, int N, float *A, int LDA, float *W, PLASMA_desc *descT, float *Q, int LDQ);
int PLASMA_ssyevr(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, int N, float *A, int LDA, float vl, float vu, int il, int iu, float abstol, int *nbcomputedeig, float *W, PLASMA_desc *descT, float *Q, int LDQ);
int PLASMA_ssygv(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, int N, float *A, int LDA, float *B, int LDB, float *W, PLASMA_desc *descT, float *Q, int LDQ);
int PLASMA_ssygvd(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, int N, float *A, int LDA, float *B, int LDB, float *W, PLASMA_desc *descT, float *Q, int LDQ);
int PLASMA_ssygst(PLASMA_enum itype, PLASMA_enum uplo, int N, float *A, int LDA, float *B, int LDB);
int PLASMA_ssytrd(PLASMA_enum jobz, PLASMA_enum uplo, int N, float *A, int LDA, float *D, float *E, PLASMA_desc *descT, float *Q, int LDQ);
int PLASMA_slacpy(PLASMA_enum uplo, int M, int N, float *A, int LDA, float *B, int LDB);
float PLASMA_slange(PLASMA_enum norm, int M, int N, float *A, int LDA);
#ifdef COMPLEX
float PLASMA_slansy(PLASMA_enum norm, PLASMA_enum uplo, int N, float *A, int LDA);
#endif
float PLASMA_slansy(PLASMA_enum norm, PLASMA_enum uplo, int N, float *A, int LDA);
float PLASMA_slantr(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N, float *A, int LDA);
int PLASMA_slaset(PLASMA_enum uplo, int M, int N, float alpha, float beta, float *A, int LDA);
int PLASMA_slaswp( int N, float *A, int LDA, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_slaswpc(int N, float *A, int LDA, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_slauum(PLASMA_enum uplo, int N, float *A, int LDA);
#ifdef COMPLEX
int PLASMA_splgsy( float bump, int N, float *A, int LDA, unsigned long long int seed);
#endif
int PLASMA_splgsy( float bump, int N, float *A, int LDA, unsigned long long int seed);
int PLASMA_splrnt( int M, int N, float *A, int LDA, unsigned long long int seed);
int PLASMA_spltmg( PLASMA_enum mtxtype, int M, int N, float *A, int LDA, unsigned long long int seed);
int PLASMA_sposv(PLASMA_enum uplo, int N, int NRHS, float *A, int LDA, float *B, int LDB);
int PLASMA_spotrf(PLASMA_enum uplo, int N, float *A, int LDA);
int PLASMA_spotri(PLASMA_enum uplo, int N, float *A, int LDA);
int PLASMA_spotrs(PLASMA_enum uplo, int N, int NRHS, float *A, int LDA, float *B, int LDB);
int PLASMA_ssymm(PLASMA_enum side, PLASMA_enum uplo, int M, int N, float alpha, float *A, int LDA, float *B, int LDB, float beta, float *C, int LDC);
int PLASMA_ssyrk(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, float alpha, float *A, int LDA, float beta, float *C, int LDC);
int PLASMA_ssyr2k(PLASMA_enum uplo, PLASMA_enum trans, int N, int K, float alpha, float *A, int LDA, float *B, int LDB, float beta, float *C, int LDC);
int PLASMA_strmm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, float alpha, float *A, int LDA, float *B, int LDB);
int PLASMA_strsm(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, float alpha, float *A, int LDA, float *B, int LDB);
int PLASMA_strsmpl(int N, int NRHS, float *A, int LDA, PLASMA_desc *descL, const int *IPIV, float *B, int LDB);
int PLASMA_strsmrv(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, int N, int NRHS, float alpha, float *A, int LDA, float *B, int LDB);
int PLASMA_strtri(PLASMA_enum uplo, PLASMA_enum diag, int N, float *A, int LDA);
int PLASMA_sorglq(int M, int N, int K, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);
int PLASMA_sorgqr(int M, int N, int K, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);
int PLASMA_sormlq(PLASMA_enum side, PLASMA_enum trans, int M, int N, int K, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);
int PLASMA_sormqr(PLASMA_enum side, PLASMA_enum trans, int M, int N, int K, float *A, int LDA, PLASMA_desc *descT, float *B, int LDB);

int PLASMA_sgecfi(int m, int n, float *A, PLASMA_enum fin, int imb, int inb, PLASMA_enum fout, int omb, int onb);
int PLASMA_sgetmi(int m, int n, float *A, PLASMA_enum fin, int mb, int nb);

/** ****************************************************************************
 * Declarations of math functions (tile layout) - alphabetical order
 **/
int PLASMA_sgebrd_Tile(PLASMA_enum jobq, PLASMA_enum jobpt, PLASMA_desc *A, float *D, float *E, PLASMA_desc *T, float *Q, int LDQ, float *PT, int LDPT);
int PLASMA_sgecon_Tile(PLASMA_enum norm, PLASMA_desc *A, float anorm, float *rcond);
int PLASMA_spocon_Tile(PLASMA_enum uplo, PLASMA_desc *A, float anorm, float *rcond);
int PLASMA_sgelqf_Tile(PLASMA_desc *A, PLASMA_desc *T);
int PLASMA_sgelqs_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_sgels_Tile(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_sgemm_Tile(PLASMA_enum transA, PLASMA_enum transB, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C);
int PLASMA_sgeqp3_Tile( PLASMA_desc *A, int *jpvt, float *tau, float *work, float *rwork);
int PLASMA_sgeqrf_Tile(PLASMA_desc *A, PLASMA_desc *T);
int PLASMA_sgeqrs_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_sgesv_Tile(PLASMA_desc *A, int *IPIV, PLASMA_desc *B);
int PLASMA_sgesv_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_desc *B);
int PLASMA_sgesvd_Tile(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, float *U, int LDU, float *VT, int LDVT);
int PLASMA_sgesdd_Tile(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, float *U, int LDU, float *VT, int LDVT);
int PLASMA_sgetrf_Tile(  PLASMA_desc *A, int *IPIV);
int PLASMA_sgetrf_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, int *IPIV);
int PLASMA_sgetrf_nopiv_Tile( PLASMA_desc *A);
int PLASMA_sgetrf_tntpiv_Tile(PLASMA_desc *A, int *IPIV);
int PLASMA_sgetri_Tile(PLASMA_desc *A, int *IPIV);
int PLASMA_sgetrs_Tile(PLASMA_enum trans, PLASMA_desc *A, const int *IPIV, PLASMA_desc *B);
int PLASMA_sgetrs_incpiv_Tile(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B);
#ifdef COMPLEX
int PLASMA_ssymm_Tile(PLASMA_enum side, PLASMA_enum uplo, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C);
int PLASMA_ssyrk_Tile(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, float beta, PLASMA_desc *C);
int PLASMA_ssyr2k_Tile(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C);
#endif
int PLASMA_ssyev_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, float *Q, int LDQ);
int PLASMA_ssyevd_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, float *Q, int LDQ);
int PLASMA_ssyevr_Tile(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, PLASMA_desc *A, float vl, float vu, int il, int iu, float abstol, int *nbcomputedeig, float *W, PLASMA_desc *T, float *Q, int LDQ);
int PLASMA_ssygv_Tile( PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q);
int PLASMA_ssygvd_Tile(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q);
int PLASMA_ssygst_Tile(PLASMA_enum itype, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ssytrd_Tile(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *D, float *E, PLASMA_desc *T, float *Q, int LDQ);
int PLASMA_slacpy_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
float PLASMA_slange_Tile(PLASMA_enum norm, PLASMA_desc *A);
#ifdef COMPLEX
float PLASMA_slansy_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A);
#endif
float PLASMA_slansy_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A);
float PLASMA_slantr_Tile(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A);
int PLASMA_slaset_Tile(PLASMA_enum uplo, float alpha, float beta, PLASMA_desc *A);
int PLASMA_slaswp_Tile(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_slaswpc_Tile(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX);
int PLASMA_slauum_Tile(PLASMA_enum uplo, PLASMA_desc *A);
#ifdef COMPLEX
int PLASMA_splgsy_Tile(float bump, PLASMA_desc *A, unsigned long long int seed);
#endif
int PLASMA_splgsy_Tile(float bump, PLASMA_desc *A, unsigned long long int seed);
int PLASMA_splrnt_Tile(PLASMA_desc *A, unsigned long long int seed);
int PLASMA_spltmg_Tile(PLASMA_enum mtxtype, PLASMA_desc *A, unsigned long long int seed);
int PLASMA_sposv_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_spotrf_Tile(PLASMA_enum uplo, PLASMA_desc *A);
int PLASMA_spotri_Tile(PLASMA_enum uplo, PLASMA_desc *A);
int PLASMA_spotrs_Tile(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_ssymm_Tile(PLASMA_enum side, PLASMA_enum uplo, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C);
int PLASMA_ssyrk_Tile(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, float beta, PLASMA_desc *C);
int PLASMA_ssyr2k_Tile(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C);
int PLASMA_strmm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, float alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_strsm_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, float alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_strsmpl_Tile(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B);
int PLASMA_strsmrv_Tile(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, float alpha, PLASMA_desc *A, PLASMA_desc *B);
int PLASMA_strtri_Tile(PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A);
int PLASMA_sorglq_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_sorgqr_Tile(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_sormlq_Tile(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);
int PLASMA_sormqr_Tile(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B);

/** ****************************************************************************
 * Declarations of math functions (tile layout, asynchronous execution) - alphabetical order
 **/
int PLASMA_sgebrd_Tile_Async(PLASMA_enum jobq, PLASMA_enum jobpt, PLASMA_desc *A, float *S, float *E, PLASMA_desc *T, float *U, int LDU, float *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgecon_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, float anorm, float *rcond, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_spocon_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, float anorm, float *rcond, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgelqf_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgelqs_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgels_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgemm_Tile_Async(PLASMA_enum transA, PLASMA_enum transB, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgeqp3_Tile_Async( PLASMA_desc *A, int *jpvt, float *tau, float *work, float *rwork, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgeqrf_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgeqrs_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgesv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgesv_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgesvd_Tile_Async(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, float *U, int LDU, float *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgesdd_Tile_Async(PLASMA_enum jobu, PLASMA_enum jobvt, PLASMA_desc *A, float *S, PLASMA_desc *T, float *U, int LDU, float *VT, int LDVT, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetrf_Tile_Async(  PLASMA_desc *A, int *IPIV, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetrf_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, int *IPIV, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetrf_nopiv_Tile_Async( PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetrf_tntpiv_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *W, int *Wpivot, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetri_Tile_Async(PLASMA_desc *A, int *IPIV, PLASMA_desc *W, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetrs_Tile_Async(PLASMA_enum trans, PLASMA_desc *A, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetrs_incpiv_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_ssymm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssyrk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssyr2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_ssyev_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, float *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssyevd_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *W, PLASMA_desc *T, float *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssyevr_Tile_Async(PLASMA_enum jobz, PLASMA_enum range, PLASMA_enum uplo, PLASMA_desc *A, float vl, float vu, int il, int iu, float abstol, int *nbcomputedeig, float *W, PLASMA_desc *T, float *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssygv_Tile_Async( PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssygvd_Tile_Async(PLASMA_enum itype, PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, float *W, PLASMA_desc *T, PLASMA_desc *Q, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssygst_Tile_Async(PLASMA_enum itype, PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssytrd_Tile_Async(PLASMA_enum jobz, PLASMA_enum uplo, PLASMA_desc *A, float *D, float *E, PLASMA_desc *T, float *Q, int LDQ, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slacpy_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slange_Tile_Async(PLASMA_enum norm, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_slansy_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_slansy_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slantr_Tile_Async(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A, float *result, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slaset_Tile_Async(PLASMA_enum uplo, float alpha, float beta, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slaswp_Tile_Async(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slaswpc_Tile_Async(PLASMA_desc *A, int K1, int K2, const int *IPIV, int INCX, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_slauum_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
#ifdef COMPLEX
int PLASMA_splgsy_Tile_Async(float bump, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
#endif
int PLASMA_splgsy_Tile_Async(float bump, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_splrnt_Tile_Async(PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_spltmg_Tile_Async(PLASMA_enum mtxtype, PLASMA_desc *A, unsigned long long int seed, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sposv_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_spotrf_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_spotri_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_spotrs_Tile_Async(PLASMA_enum uplo, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssymm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssyrk_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_ssyr2k_Tile_Async(PLASMA_enum uplo, PLASMA_enum trans, float alpha, PLASMA_desc *A, PLASMA_desc *B, float beta, PLASMA_desc *C, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_strmm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, float alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_strsm_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, float alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_strsmpl_Tile_Async(PLASMA_desc *A, PLASMA_desc *L, const int *IPIV, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_strsmrv_Tile_Async(PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag, float alpha, PLASMA_desc *A, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_strtri_Tile_Async(PLASMA_enum uplo, PLASMA_enum diag, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sorglq_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sorgqr_Tile_Async(PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sormlq_Tile_Async(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sormqr_Tile_Async(PLASMA_enum side, PLASMA_enum trans, PLASMA_desc *A, PLASMA_desc *T, PLASMA_desc *B, PLASMA_sequence *sequence, PLASMA_request *request);

int PLASMA_sgecfi_Async(int m, int n, float *A, PLASMA_enum f_in, int imb, int inb, PLASMA_enum f_out, int omb, int onb, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sgetmi_Async(int m, int n, float *A, PLASMA_enum f_in, int mb, int inb, PLASMA_sequence *sequence, PLASMA_request *request);

/** ****************************************************************************
 * Declarations of workspace allocation functions (tile layout) - alphabetical order
 **/
int PLASMA_Alloc_Workspace_sgesv_incpiv( int N,        PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_sgetrf_incpiv(int M, int N, PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_sgebrd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgeev( int N,        PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgehrd(int N,        PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgelqf(int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_sgels( int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_sgeqrf(int M, int N, PLASMA_desc **T);
int PLASMA_Alloc_Workspace_sgesdd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgesvd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_ssyev( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_ssyevd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_ssyevr(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_ssygv( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_ssygvd(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_ssytrd(int M, int N, PLASMA_desc **descT);

/** ****************************************************************************
 * Declarations of workspace allocation functions (tile layout, asynchronous execution) - alphabetical order
 **/

/* Workspace required only for asynchronous interface */
int PLASMA_Alloc_Workspace_sgetrf_tntpiv_Tile(PLASMA_desc *A, PLASMA_desc *W, int **Wpivot);
int PLASMA_Alloc_Workspace_sgetri_Tile_Async( PLASMA_desc *A, PLASMA_desc *W);

/* Warning: Those functions are deprecated */
int PLASMA_Alloc_Workspace_sgelqf_Tile(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgels_Tile( int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgeqrf_Tile(int M, int N, PLASMA_desc **descT);
int PLASMA_Alloc_Workspace_sgesv_incpiv_Tile (int N, PLASMA_desc **descL, int **IPIV);
int PLASMA_Alloc_Workspace_sgetrf_incpiv_Tile(int N, PLASMA_desc **descL, int **IPIV);

/** ****************************************************************************
 * Auxiliary function prototypes
 **/
int PLASMA_sLapack_to_Tile(float *Af77, int LDA, PLASMA_desc *A);
int PLASMA_sTile_to_Lapack(PLASMA_desc *A, float *Af77, int LDA);
int PLASMA_sLapack_to_Tile_Async(float *Af77, int LDA, PLASMA_desc *A, PLASMA_sequence *sequence, PLASMA_request *request);
int PLASMA_sTile_to_Lapack_Async(PLASMA_desc *A, float *Af77, int LDA, PLASMA_sequence *sequence, PLASMA_request *request);

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
