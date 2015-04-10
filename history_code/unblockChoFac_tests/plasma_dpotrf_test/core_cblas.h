/**
 *
 * @file core_cblas.h
 *
 *  PLASMA auxiliary routines
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
#ifndef _PLASMA_CORE_CBLAS_H_
#define _PLASMA_CORE_CBLAS_H_

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of serial kernels - alphabetical order
 **/
void CORE_scasum(int storev, PLASMA_enum uplo, int M, int N,
                 const PLASMA_Complex32_t *A, int lda, float *work);
void CORE_cbrdalg1(     PLASMA_enum uplo,
                        int n,
                        int nb,
                        PLASMA_Complex32_t *A,
                        int lda,
                        PLASMA_Complex32_t *VQ,
                        PLASMA_Complex32_t *TAUQ,
                        PLASMA_Complex32_t *VP,
                        PLASMA_Complex32_t *TAUP,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        PLASMA_Complex32_t *work);
int CORE_cgbelr(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                int st, int ed, int eltsize);
int CORE_cgbrce(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                int st, int ed, int eltsize);
int CORE_cgblrx(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                int st, int ed, int eltsize);
int CORE_cgeadd(int M, int N, PLASMA_Complex32_t alpha,
                const PLASMA_Complex32_t *A, int LDA,
                      PLASMA_Complex32_t *B, int LDB);
int  CORE_cgelqt(int M, int N, int IB,
                 PLASMA_Complex32_t *A, int LDA,
                 PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *TAU,
                 PLASMA_Complex32_t *WORK);
void CORE_cgemm(PLASMA_enum transA, PLASMA_enum transB,
                int M, int N, int K,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                          const PLASMA_Complex32_t *B, int LDB,
                PLASMA_Complex32_t beta,        PLASMA_Complex32_t *C, int LDC);
void CORE_cgemv(PLASMA_enum trans, int M, int N,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                          const PLASMA_Complex32_t *x, int incx,
                PLASMA_Complex32_t beta,        PLASMA_Complex32_t *y, int incy);
void CORE_cgeqp3_init( int n, int *jpvt );
void CORE_cgeqp3_larfg( PLASMA_desc A, int ii, int jj, int i, int j,
                        PLASMA_Complex32_t *tau, PLASMA_Complex32_t *beta );
void CORE_cgeqp3_norms( PLASMA_desc A, int ioff, int joff, float *norms1, float *norms2 );
void CORE_cgeqp3_pivot( PLASMA_desc A, PLASMA_Complex32_t *F, int ldf,
                        int jj, int k, int *jpvt,
                        float *norms1, float *norms2, int *info );
int  CORE_cgeqp3_tntpiv(int m, int n,
                        PLASMA_Complex32_t *A, int lda,
                        int *IPIV, PLASMA_Complex32_t *tau,
                        int *iwork);
void CORE_cgeqp3_update( const PLASMA_Complex32_t *Ajj, int lda1,
                         PLASMA_Complex32_t       *Ajk, int lda2,
                         const PLASMA_Complex32_t *Fk,  int ldf,
                         int joff, int k, int koff, int nb,
                         float *norms1, float *norms2,
                         int *info );
int  CORE_cgeqrt(int M, int N, int IB,
                 PLASMA_Complex32_t *A, int LDA,
                 PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *TAU, PLASMA_Complex32_t *WORK);
int  CORE_cgessm(int M, int N, int K, int IB,
                 const int *IPIV,
                 const PLASMA_Complex32_t *L, int LDL,
                 PLASMA_Complex32_t *A, int LDA);
int  CORE_cgessq(int M, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *scale, float *sumsq);
int  CORE_cgetf2_nopiv(int m, int n,
                      PLASMA_Complex32_t *A, int lda);
int  CORE_cgetrf(int M, int N,
                 PLASMA_Complex32_t *A, int LDA,
                 int *IPIV, int *INFO);
int  CORE_cgetrf_incpiv(int M, int N, int IB,
                        PLASMA_Complex32_t *A, int LDA,
                        int *IPIV, int *INFO);
int  CORE_cgetrf_nopiv(int m, int n, int ib,
                      PLASMA_Complex32_t *A, int lda);
int  CORE_cgetrf_reclap(int M, int N,
                        PLASMA_Complex32_t *A, int LDA,
                        int *IPIV, int *info);
void CORE_cgetrf_reclap_init(void);
int  CORE_cgetrf_rectil(const PLASMA_desc A, int *IPIV, int *info);
void CORE_cgetrf_rectil_init(void);
void CORE_cgetrip(int m, int n, PLASMA_Complex32_t *A,
                  PLASMA_Complex32_t *work);
int CORE_chbelr(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                int st, int ed, int eltsize);
int CORE_chblrx(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                int st, int ed, int eltsize);
int CORE_chbrce(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                int st, int ed, int eltsize);
void CORE_chbtype1cb(int N, int NB,
                     PLASMA_Complex32_t *A, int LDA,
                     PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     PLASMA_Complex32_t *WORK);
void CORE_chbtype2cb(int N, int NB,
                     PLASMA_Complex32_t *A, int LDA,
                     PLASMA_Complex32_t *V, PLASMA_Complex32_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     PLASMA_Complex32_t *WORK);
void CORE_chbtype3cb(int N, int NB,
                     PLASMA_Complex32_t *A, int LDA,
                     const PLASMA_Complex32_t *V, const PLASMA_Complex32_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     PLASMA_Complex32_t *WORK);
void CORE_cgbtype1cb(PLASMA_enum uplo, int N, int NB,
                PLASMA_Complex32_t *A, int LDA,
                PLASMA_Complex32_t *VQ, PLASMA_Complex32_t *TAUQ,
                PLASMA_Complex32_t *VP, PLASMA_Complex32_t *TAUP,
                int st, int ed, int sweep, int Vblksiz, int WANTZ,
                PLASMA_Complex32_t *WORK);
void CORE_cgbtype2cb(PLASMA_enum uplo, int N, int NB,
                PLASMA_Complex32_t *A, int LDA,
                PLASMA_Complex32_t *VQ, PLASMA_Complex32_t *TAUQ,
                PLASMA_Complex32_t *VP, PLASMA_Complex32_t *TAUP,
                int st, int ed, int sweep, int Vblksiz, int WANTZ,
                PLASMA_Complex32_t *WORK);
void CORE_cgbtype3cb(PLASMA_enum uplo, int N, int NB,
                PLASMA_Complex32_t *A, int LDA,
                PLASMA_Complex32_t *VQ, PLASMA_Complex32_t *TAUQ,
                PLASMA_Complex32_t *VP, PLASMA_Complex32_t *TAUP,
                int st, int ed, int sweep, int Vblksiz, int WANTZ,
                PLASMA_Complex32_t *WORK);
void CORE_chegst(int itype, PLASMA_enum uplo, int N,
                 PLASMA_Complex32_t *A, int LDA,
                 PLASMA_Complex32_t *B, int LDB, int *INFO);
#ifdef COMPLEX
void CORE_chemm(PLASMA_enum side, PLASMA_enum uplo,
                int M, int N,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                          const PLASMA_Complex32_t *B, int LDB,
                PLASMA_Complex32_t beta,        PLASMA_Complex32_t *C, int LDC);
void CORE_cherk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K,
                float alpha, const PLASMA_Complex32_t *A, int LDA,
                float beta,        PLASMA_Complex32_t *C, int LDC);
void CORE_cher2k(PLASMA_enum uplo, PLASMA_enum trans,
                 int N, int K,
                 PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                           const PLASMA_Complex32_t *B, int LDB,
                 float beta,                    PLASMA_Complex32_t *C, int LDC);
int  CORE_chessq(PLASMA_enum uplo, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *scale, float *sumsq);
#endif
int  CORE_cherfb(PLASMA_enum uplo, int N, int K, int IB, int NB,
                 const PLASMA_Complex32_t *A,    int LDA,
                 const PLASMA_Complex32_t *T,    int LDT,
                       PLASMA_Complex32_t *C,    int LDC,
                       PLASMA_Complex32_t *WORK, int LDWORK);
void CORE_clacpy(PLASMA_enum uplo, int M, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                       PLASMA_Complex32_t *B, int LDB);
int CORE_clacpy_pivot( const PLASMA_desc descA,
                       PLASMA_enum direct,
                       int k1, int k2, const int *ipiv,
                       int *rankin, int *rankout,
                       PLASMA_Complex32_t *A, int lda,
                       int init);
void CORE_clange(int norm, int M, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *work, float *normA);
#ifdef COMPLEX
void CORE_clanhe(int norm, PLASMA_enum uplo, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *work, float *normA);
#endif
void CORE_clansy(int norm, PLASMA_enum uplo, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *work, float *normA);
void CORE_clantr(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag,
                 int M, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *work, float *normA);
int CORE_clarfb_gemm(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
                     int M, int N, int K,
                     const PLASMA_Complex32_t *V, int LDV,
                     const PLASMA_Complex32_t *T, int LDT,
                           PLASMA_Complex32_t *C, int LDC,
                           PLASMA_Complex32_t *WORK, int LDWORK);
int CORE_clarfx2(PLASMA_enum side, int N,
                 PLASMA_Complex32_t V,
                 PLASMA_Complex32_t TAU,
                 PLASMA_Complex32_t *C1, int LDC1,
                 PLASMA_Complex32_t *C2, int LDC2);
int CORE_clarfx2c(PLASMA_enum uplo,
                  PLASMA_Complex32_t V,
                  PLASMA_Complex32_t TAU,
                  PLASMA_Complex32_t *C1,
                  PLASMA_Complex32_t *C2,
                  PLASMA_Complex32_t *C3);
int CORE_clarfx2ce(PLASMA_enum uplo,
                   PLASMA_Complex32_t *V,
                   PLASMA_Complex32_t *TAU,
                   PLASMA_Complex32_t *C1,
                   PLASMA_Complex32_t *C2,
                   PLASMA_Complex32_t *C3);
void CORE_clarfy(int N,
                 PLASMA_Complex32_t *A, int LDA,
                 const PLASMA_Complex32_t *V,
                 const PLASMA_Complex32_t *TAU,
                 PLASMA_Complex32_t *WORK);
void CORE_claset(PLASMA_enum uplo, int n1, int n2,
                 PLASMA_Complex32_t alpha, PLASMA_Complex32_t beta,
                 PLASMA_Complex32_t *tileA, int ldtilea);
void CORE_claset2(PLASMA_enum uplo, int n1, int n2, PLASMA_Complex32_t alpha,
                  PLASMA_Complex32_t *tileA, int ldtilea);
void CORE_claswp(int N, PLASMA_Complex32_t *A, int LDA,
                 int I1,  int I2, const int *IPIV, int INC);
int  CORE_claswp_ontile( PLASMA_desc descA, int i1, int i2, const int *ipiv, int inc);
int  CORE_claswpc_ontile(PLASMA_desc descA, int i1, int i2, const int *ipiv, int inc);
int  CORE_clatro(PLASMA_enum uplo, PLASMA_enum trans,
                 int M, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                       PLASMA_Complex32_t *B, int LDB);
void CORE_clauum(PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA);
int CORE_cpamm(int op, PLASMA_enum side, PLASMA_enum storev,
               int M, int N, int K, int L,
               const PLASMA_Complex32_t *A1, int LDA1,
                     PLASMA_Complex32_t *A2, int LDA2,
               const PLASMA_Complex32_t *V, int LDV,
                     PLASMA_Complex32_t *W, int LDW);
int  CORE_cparfb(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
                 int M1, int N1, int M2, int N2, int K, int L,
                       PLASMA_Complex32_t *A1, int LDA1,
                       PLASMA_Complex32_t *A2, int LDA2,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                       PLASMA_Complex32_t *WORK, int LDWORK);
int CORE_cpemv(PLASMA_enum trans, PLASMA_enum storev,
               int M, int N, int L,
               PLASMA_Complex32_t ALPHA,
               const PLASMA_Complex32_t *A, int LDA,
               const PLASMA_Complex32_t *X, int INCX,
               PLASMA_Complex32_t BETA,
               PLASMA_Complex32_t *Y, int INCY,
               PLASMA_Complex32_t *WORK);
void CORE_cplghe(float bump, int m, int n, PLASMA_Complex32_t *A, int lda,
                 int bigM, int m0, int n0, unsigned long long int seed );
void CORE_cplgsy(PLASMA_Complex32_t bump, int m, int n, PLASMA_Complex32_t *A, int lda,
                 int bigM, int m0, int n0, unsigned long long int seed );
void CORE_cplrnt(int m, int n, PLASMA_Complex32_t *A, int lda,
                 int bigM, int m0, int n0, unsigned long long int seed );
int  CORE_cpltmg(PLASMA_enum mtxtype, int m, int n, PLASMA_Complex32_t *A, int lda,
                  int gM, int gN, int m0, int n0, unsigned long long int seed );
int  CORE_cpltmg_chebvand( int M, int N, PLASMA_Complex32_t *A, int LDA,
                           int gN, int m0, int n0,
                           PLASMA_Complex32_t *W );
int  CORE_cpltmg_circul( int M, int N, PLASMA_Complex32_t *A, int LDA,
                         int gM, int m0, int n0,
                         const PLASMA_Complex32_t *V );
void CORE_cpltmg_condexq( int M, int N, PLASMA_Complex32_t *Q, int LDQ );
void CORE_cpltmg_fiedler(int m, int n,
                         const PLASMA_Complex32_t *X, int incX,
                         const PLASMA_Complex32_t *Y, int incY,
                         PLASMA_Complex32_t *A, int lda);
int  CORE_cpltmg_hankel( PLASMA_enum uplo, int M, int N, PLASMA_Complex32_t *A, int LDA,
                         int m0, int n0, int nb,
                         const PLASMA_Complex32_t *V1,
                         const PLASMA_Complex32_t *V2 );
void CORE_cpltmg_toeppd1( int gM, int m0, int M, PLASMA_Complex32_t *W,
                          unsigned long long int seed );
void CORE_cpltmg_toeppd2( int M, int N, int K, int m0, int n0,
                          const PLASMA_Complex32_t *W,
                          PLASMA_Complex32_t *A, int LDA );
void CORE_cpotrf(PLASMA_enum uplo, int N, PLASMA_Complex32_t *A, int LDA, int *INFO);
void CORE_csetvar(const PLASMA_Complex32_t *alpha, PLASMA_Complex32_t *x);
void CORE_cshift(int s, int m, int n, int L,
                 PLASMA_Complex32_t *A);
void CORE_cshiftw(int s, int cl, int m, int n, int L,
                  PLASMA_Complex32_t *A, PLASMA_Complex32_t *W);
int  CORE_cssssm(int M1, int N1, int M2, int N2, int K, int IB,
                       PLASMA_Complex32_t *A1, int LDA1,
                       PLASMA_Complex32_t *A2, int LDA2,
                 const PLASMA_Complex32_t *L1, int LDL1,
                 const PLASMA_Complex32_t *L2, int LDL2,
                 const int *IPIV);
void CORE_csymm(PLASMA_enum side, PLASMA_enum uplo,
                int M, int N,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                          const PLASMA_Complex32_t *B, int LDB,
                PLASMA_Complex32_t beta,        PLASMA_Complex32_t *C, int LDC);
void CORE_csyrk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                PLASMA_Complex32_t beta,        PLASMA_Complex32_t *C, int LDC);
void CORE_csyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                 int N, int K,
                 PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                           const PLASMA_Complex32_t *B, int LDB,
                 PLASMA_Complex32_t beta,        PLASMA_Complex32_t *C, int LDC);
int  CORE_csyssq(PLASMA_enum uplo, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *scale, float *sumsq);
void CORE_cswpab(int i, int n1, int n2,
                 PLASMA_Complex32_t *A, PLASMA_Complex32_t *work);
int  CORE_cswptr_ontile(PLASMA_desc descA, int i1, int i2, const int *ipiv, int inc,
                        const PLASMA_Complex32_t *Akk, int ldak);
void CORE_ctrasm(PLASMA_enum storev, PLASMA_enum uplo, PLASMA_enum diag,
                 int M, int N, const PLASMA_Complex32_t *A, int lda, float *work);
void CORE_ctrdalg1(int n,
                        int nb,
                        PLASMA_Complex32_t *A,
                        int lda,
                        PLASMA_Complex32_t *V,
                        PLASMA_Complex32_t *TAU,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        PLASMA_Complex32_t *work);
void CORE_ctrmm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int M, int N,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                                PLASMA_Complex32_t *B, int LDB);
void CORE_ctrsm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int M, int N,
                PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int LDA,
                                                PLASMA_Complex32_t *B, int LDB);
int  CORE_ctrssq(PLASMA_enum uplo, PLASMA_enum diag, int M, int N,
                 const PLASMA_Complex32_t *A, int LDA,
                 float *scale, float *sumsq);
void CORE_ctrtri(PLASMA_enum uplo, PLASMA_enum diag, int N,
                 PLASMA_Complex32_t *A, int LDA, int *info);
int  CORE_ctslqt(int M, int N, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *TAU, PLASMA_Complex32_t *WORK);
int  CORE_ctsmlq(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *WORK, int LDWORK);
int CORE_ctsmlq_corner( int m1, int n1, int m2, int n2, int m3, int n3,
                        int k, int ib, int nb,
                        PLASMA_Complex32_t *A1, int lda1,
                        PLASMA_Complex32_t *A2, int lda2,
                        PLASMA_Complex32_t *A3, int lda3,
                        const PLASMA_Complex32_t *V, int ldv,
                        const PLASMA_Complex32_t *T, int ldt,
                        PLASMA_Complex32_t *WORK, int ldwork);
int CORE_ctsmlq_hetra1( PLASMA_enum side, PLASMA_enum trans,
                        int m1, int n1, int m2, int n2,
                        int k, int ib,
                        PLASMA_Complex32_t *A1, int lda1,
                        PLASMA_Complex32_t *A2, int lda2,
                        const PLASMA_Complex32_t *V, int ldv,
                        const PLASMA_Complex32_t *T, int ldt,
                        PLASMA_Complex32_t *WORK, int ldwork);
int  CORE_ctsmqr(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *WORK, int LDWORK);
int CORE_ctsmqr_corner( int m1, int n1, int m2, int n2, int m3, int n3,
                        int k, int ib, int nb,
                        PLASMA_Complex32_t *A1, int lda1,
                        PLASMA_Complex32_t *A2, int lda2,
                        PLASMA_Complex32_t *A3, int lda3,
                        const PLASMA_Complex32_t *V, int ldv,
                        const PLASMA_Complex32_t *T, int ldt,
                        PLASMA_Complex32_t *WORK, int ldwork);
int CORE_ctsmqr_hetra1( PLASMA_enum side, PLASMA_enum trans,
                        int m1, int n1, int m2, int n2,
                        int k, int ib,
                        PLASMA_Complex32_t *A1, int lda1,
                        PLASMA_Complex32_t *A2, int lda2,
                        const PLASMA_Complex32_t *V, int ldv,
                        const PLASMA_Complex32_t *T, int ldt,
                        PLASMA_Complex32_t *WORK, int ldwork);
int  CORE_ctsqrt(int M, int N, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *TAU, PLASMA_Complex32_t *WORK);
int  CORE_ctstrf(int M, int N, int IB, int NB,
                 PLASMA_Complex32_t *U, int LDU,
                 PLASMA_Complex32_t *A, int LDA,
                 PLASMA_Complex32_t *L, int LDL,
                 int *IPIV, PLASMA_Complex32_t *WORK,
                 int LDWORK, int *INFO);
int  CORE_cttmqr(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *WORK, int LDWORK);
int  CORE_cttqrt(int M, int N, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *TAU,
                 PLASMA_Complex32_t *WORK);
int  CORE_cttmlq(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *WORK, int LDWORK);
int  CORE_cttlqt(int M, int N, int IB,
                 PLASMA_Complex32_t *A1, int LDA1,
                 PLASMA_Complex32_t *A2, int LDA2,
                 PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *TAU,
                 PLASMA_Complex32_t *WORK);
int  CORE_cunmlq(PLASMA_enum side, PLASMA_enum trans,
                 int M, int N, int IB, int K,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *C, int LDC,
                 PLASMA_Complex32_t *WORK, int LDWORK);
int  CORE_cunmqr(PLASMA_enum side, PLASMA_enum trans,
                 int M, int N, int K, int IB,
                 const PLASMA_Complex32_t *V, int LDV,
                 const PLASMA_Complex32_t *T, int LDT,
                 PLASMA_Complex32_t *C, int LDC,
                 PLASMA_Complex32_t *WORK, int LDWORK);

#if defined(QUARK_H)
/** ****************************************************************************
 *  Declarations of QUARK wrappers (called by PLASMA) - alphabetical order
 **/
void QUARK_CORE_scasum(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum storev, PLASMA_enum uplo, int m, int n,
                       const PLASMA_Complex32_t *A, int lda, int szeA,
                       float *work, int szeW);
void QUARK_CORE_scasum_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum storev, PLASMA_enum uplo, int m, int n,
                          const PLASMA_Complex32_t *A, int lda, int szeA,
                          float *work, int szeW,
                          float *fake, int szeF);
void QUARK_CORE_cgeadd(Quark *quark, Quark_Task_Flags *task_flags,
                      int m, int n, int nb, PLASMA_Complex32_t alpha,
                      const PLASMA_Complex32_t *A, int lda,
                      PLASMA_Complex32_t *B, int ldb);
void QUARK_CORE_cbrdalg1(Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum uplo,
                        int n, int nb,
                        PLASMA_Complex32_t *A,
                        int lda,
                        PLASMA_Complex32_t *VQ,
                        PLASMA_Complex32_t *TAUQ,
                        PLASMA_Complex32_t *VP,
                        PLASMA_Complex32_t *TAUP,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        int *PCOL, int *ACOL, int *MCOL);
void QUARK_CORE_cgelqt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *A, int lda,
                       PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_cgemm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum transA, PLASMA_enum transB,
                      int m, int n, int k, int nb,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                      const PLASMA_Complex32_t *B, int ldb,
                      PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_cgemm2( Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum transA, PLASMA_enum transB,
                        int m, int n, int k, int nb,
                        PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                        const PLASMA_Complex32_t *B, int ldb,
                        PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_cgemm_f2(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum transA, PLASMA_enum transB,
                         int m, int n, int k, int nb,
                         PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                         const PLASMA_Complex32_t *B, int ldb,
                         PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc,
                         PLASMA_Complex32_t *fake1, int szefake1, int flag1,
                         PLASMA_Complex32_t *fake2, int szefake2, int flag2);
void QUARK_CORE_cgemm_p2(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum transA, PLASMA_enum transB,
                         int m, int n, int k, int nb,
                         PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                         const PLASMA_Complex32_t **B, int ldb,
                         PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_cgemm_p2f1(Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum transA, PLASMA_enum transB,
                           int m, int n, int k, int nb,
                           PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                           const PLASMA_Complex32_t **B, int ldb,
                           PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc,
                           PLASMA_Complex32_t *fake1, int szefake1, int flag1);
void QUARK_CORE_cgemm_p3(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum transA, PLASMA_enum transB,
                         int m, int n, int k, int nb,
                         PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                         const PLASMA_Complex32_t *B, int ldb,
                         PLASMA_Complex32_t beta, PLASMA_Complex32_t **C, int ldc);
void QUARK_CORE_cgemm_tile(Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum transA, PLASMA_enum transB,
                           int m, int n, int k, int nb,
                           const PLASMA_Complex32_t *alpha, const PLASMA_Complex32_t *A, int lda,
                                                            const PLASMA_Complex32_t *B, int ldb,
                           const PLASMA_Complex32_t *beta,        PLASMA_Complex32_t *C, int ldc,
                           const PLASMA_Complex32_t *Alock,
                           const PLASMA_Complex32_t *Block,
                           const PLASMA_Complex32_t *Clock);
void QUARK_CORE_cgemv(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum trans, int m, int n,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                                                const PLASMA_Complex32_t *x, int incx,
                      PLASMA_Complex32_t beta,        PLASMA_Complex32_t *y, int incy);
void QUARK_CORE_cgemv_tile(Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum trans,
                           int m, int n,
                           const PLASMA_Complex32_t *alpha, const PLASMA_Complex32_t *A, int lda,
                                                            const PLASMA_Complex32_t *x, int incx,
                           const PLASMA_Complex32_t *beta,        PLASMA_Complex32_t *y, int incy,
                           const PLASMA_Complex32_t *Alock,
                           const PLASMA_Complex32_t *xlock,
                           const PLASMA_Complex32_t *ylock);
void QUARK_CORE_cgeqp3_init( Quark *quark, Quark_Task_Flags *task_flags,
                             int n, int *jpvt );
void QUARK_CORE_cgeqp3_larfg(Quark *quark, Quark_Task_Flags *task_flags,
                             PLASMA_desc A, int ii, int jj, int i, int j,
                             PLASMA_Complex32_t *tau, PLASMA_Complex32_t *beta );
void QUARK_CORE_cgeqp3_norms( Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc A, int ioff, int joff, float *norms1, float *norms2 );
void QUARK_CORE_cgeqp3_pivot( Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc A,
                              PLASMA_Complex32_t *F, int ldf,
                              int jj, int k, int *jpvt,
                              float *norms1, float *norms2, int *info );
void QUARK_CORE_cgeqp3_tntpiv(Quark *quark, Quark_Task_Flags *task_flags,
                              int m, int n, int nb,
                              PLASMA_Complex32_t *A, int lda,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo);
void QUARK_CORE_cgeqp3_update( Quark *quark, Quark_Task_Flags *task_flags,
                               PLASMA_Complex32_t *Ajj, int lda1,
                               PLASMA_Complex32_t *Ajk, int lda2,
                               PLASMA_Complex32_t *Fk,  int ldf,
                               int joff, int k, int koff, int nb,
                               float *norms1, float *norms2, int *info );
void QUARK_CORE_cgeqrt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *A, int lda,
                       PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_cgessm(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int k, int ib, int nb,
                       const int *IPIV,
                       const PLASMA_Complex32_t *L, int ldl,
                       PLASMA_Complex32_t *A, int lda);
void QUARK_CORE_cgessq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           int m, int n, const PLASMA_Complex32_t *A, int lda,
                           float *scale, float *sumsq,
                           float *fake, int szeF, int paramF );
void QUARK_CORE_cgetrf(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int nb,
                       PLASMA_Complex32_t *A, int lda,
                       int *IPIV,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       PLASMA_bool check_info, int iinfo);
void QUARK_CORE_cgetrf_incpiv(Quark *quark, Quark_Task_Flags *task_flags,
                              int m, int n, int ib, int nb,
                              PLASMA_Complex32_t *A, int lda,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo);
void QUARK_CORE_cgetrf_nopiv(Quark *quark, Quark_Task_Flags *task_flags,
                             int m, int n, int ib, int nb,
                             PLASMA_Complex32_t *A, int lda,
                             PLASMA_sequence *sequence, PLASMA_request *request,
                             int iinfo);
void QUARK_CORE_cgetrf_reclap(Quark *quark, Quark_Task_Flags *task_flags,
                              int m, int n, int nb,
                              PLASMA_Complex32_t *A, int lda,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo,
                              int nbthread);
void QUARK_CORE_cgetrf_rectil(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc A, PLASMA_Complex32_t *Amn, int size,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo,
                              int nbthread);
void QUARK_CORE_cgetrip(Quark *quark, Quark_Task_Flags *task_flags,
                        int m, int n, PLASMA_Complex32_t *A, int szeA);
void QUARK_CORE_cgetrip_f1(Quark *quark, Quark_Task_Flags *task_flags,
                           int m, int n, PLASMA_Complex32_t *A, int szeA,
                           PLASMA_Complex32_t *fake, int szeF, int paramF);
void QUARK_CORE_cgetrip_f2(Quark *quark, Quark_Task_Flags *task_flags,
                           int m, int n, PLASMA_Complex32_t *A, int szeA,
                           PLASMA_Complex32_t *fake1, int szeF1, int paramF1,
                           PLASMA_Complex32_t *fake2, int szeF2, int paramF2);
void QUARK_CORE_chemm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo,
                      int m, int n, int nb,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                      const PLASMA_Complex32_t *B, int ldb,
                      PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_chegst(Quark *quark, Quark_Task_Flags *task_flags,
                       int itype, PLASMA_enum uplo, int N,
                       PLASMA_Complex32_t *A, int LDA,
                       PLASMA_Complex32_t *B, int LDB,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       int iinfo);
void QUARK_CORE_cherk(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum uplo, PLASMA_enum trans,
                      int n, int k, int nb,
                      float alpha, const PLASMA_Complex32_t *A, int lda,
                      float beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_cher2k(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum trans,
                       int n, int k, int nb,
                       PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                       const PLASMA_Complex32_t *B, int LDB,
                       float beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_cherfb(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo,
                       int n, int k, int ib, int nb,
                       const PLASMA_Complex32_t *A, int lda,
                       const PLASMA_Complex32_t *T, int ldt,
                       PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_chessq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum uplo, int n, const PLASMA_Complex32_t *A, int lda,
                           float *scale, float *sumsq,
                           float *fake, int szeF, int paramF );
void QUARK_CORE_clacpy(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int m, int n, int mb,
                       const PLASMA_Complex32_t *A, int lda,
                       PLASMA_Complex32_t *B, int ldb);
void QUARK_CORE_clacpy_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum uplo, int m, int n, int nb,
                          const PLASMA_Complex32_t *A, int lda,
                          PLASMA_Complex32_t *B, int ldb,
                          PLASMA_Complex32_t *fake1, int szefake1, int flag1);
void QUARK_CORE_clacpy_pivot(Quark *quark, Quark_Task_Flags *task_flags,
                             const PLASMA_desc descA,
                             PLASMA_enum direct,
                             int k1, int k2, const int *ipiv,
                             int *rankin, int *rankout,
                             PLASMA_Complex32_t *A, int lda,
                             int pos, int init);
void QUARK_CORE_clange(Quark *quark, Quark_Task_Flags *task_flags,
                       int norm, int M, int N,
                       const PLASMA_Complex32_t *A, int LDA, int szeA,
                       int szeW, float *result);
void QUARK_CORE_clange_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          int norm, int M, int N,
                          const PLASMA_Complex32_t *A, int LDA, int szeA,
                          int szeW, float *result,
                          float *fake, int szeF);
#ifdef COMPLEX
void QUARK_CORE_clanhe(Quark *quark, Quark_Task_Flags *task_flags,
                       int norm, PLASMA_enum uplo, int N,
                       const PLASMA_Complex32_t *A, int LDA, int szeA,
                       int szeW, float *result);
void QUARK_CORE_clanhe_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          int norm, PLASMA_enum uplo, int N,
                          const PLASMA_Complex32_t *A, int LDA, int szeA,
                          int szeW, float *result,
                          float *fake, int szeF);
#endif
void QUARK_CORE_clansy(Quark *quark, Quark_Task_Flags *task_flags,
                       int norm, PLASMA_enum uplo, int N,
                       const PLASMA_Complex32_t *A, int LDA, int szeA,
                       int szeW, float *result);
void QUARK_CORE_clansy_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          int norm, PLASMA_enum uplo, int N,
                          const PLASMA_Complex32_t *A, int LDA, int szeA,
                          int szeW, float *result,
                          float *fake, int szeF);
void QUARK_CORE_clantr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N,
                       const PLASMA_Complex32_t *A, int LDA, int szeA,
                       int szeW, float *result);
void QUARK_CORE_clantr_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N,
                          const PLASMA_Complex32_t *A, int LDA, int szeA,
                          int szeW, float *result,
                          float *fake, int szeF);
void QUARK_CORE_claset(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int n1, int n2, PLASMA_Complex32_t alpha,
                       PLASMA_Complex32_t beta, PLASMA_Complex32_t *tileA, int ldtilea);
void QUARK_CORE_claset2(Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum uplo, int n1, int n2, PLASMA_Complex32_t alpha,
                        PLASMA_Complex32_t *tileA, int ldtilea);
void QUARK_CORE_claswp(Quark *quark, Quark_Task_Flags *task_flags,
                       int n, PLASMA_Complex32_t *A, int lda,
                       int i1,  int i2, const int *ipiv, int inc);
void QUARK_CORE_claswp_f2(Quark *quark, Quark_Task_Flags *task_flags,
                          int n, PLASMA_Complex32_t *A, int lda,
                          int i1,  int i2, const int *ipiv, int inc,
                          PLASMA_Complex32_t *fake1, int szefake1, int flag1,
                          PLASMA_Complex32_t *fake2, int szefake2, int flag2);
void QUARK_CORE_claswp_ontile(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc descA, PLASMA_Complex32_t *A,
                              int i1,  int i2, const int *ipiv, int inc, PLASMA_Complex32_t *fakepanel);
void QUARK_CORE_claswp_ontile_f2(Quark *quark, Quark_Task_Flags *task_flags,
                                 PLASMA_desc descA, PLASMA_Complex32_t *A,
                                 int i1,  int i2, const int *ipiv, int inc,
                                 PLASMA_Complex32_t *fake1, int szefake1, int flag1,
                                 PLASMA_Complex32_t *fake2, int szefake2, int flag2);
void QUARK_CORE_claswpc_ontile(Quark *quark, Quark_Task_Flags *task_flags,
                               PLASMA_desc descA, PLASMA_Complex32_t *A,
                               int i1,  int i2, const int *ipiv, int inc, PLASMA_Complex32_t *fakepanel);
void QUARK_CORE_clatro(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum trans, int m, int n, int mb,
                       const PLASMA_Complex32_t *A, int lda,
                       PLASMA_Complex32_t *B, int ldb);
void QUARK_CORE_clatro_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum uplo, PLASMA_enum trans, int m, int n, int mb,
                          const PLASMA_Complex32_t *A, int lda,
                                PLASMA_Complex32_t *B, int ldb,
                          PLASMA_Complex32_t *fake1, int szefake1, int flag1);
void QUARK_CORE_clauum(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int n, int nb,
                       PLASMA_Complex32_t *A, int lda);
void QUARK_CORE_cplghe(Quark *quark, Quark_Task_Flags *task_flags,
                       float bump, int m, int n, PLASMA_Complex32_t *A, int lda,
                       int bigM, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_cplgsy(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_Complex32_t bump, int m, int n, PLASMA_Complex32_t *A, int lda,
                       int bigM, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_cplrnt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, PLASMA_Complex32_t *A, int lda,
                       int bigM, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_cpltmg(Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum mtxtype, int m, int n, PLASMA_Complex32_t *A, int lda,
                        int gM, int gN, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_cpltmg_chebvand( Quark *quark, Quark_Task_Flags *task_flags,
                                 int M, int N, PLASMA_Complex32_t *A, int LDA,
                                 int gN, int m0, int n0,
                                 PLASMA_Complex32_t *W );
void QUARK_CORE_cpltmg_circul( Quark *quark, Quark_Task_Flags *task_flags,
                               int M, int N, PLASMA_Complex32_t *A, int LDA,
                               int gM, int m0, int n0,
                               const PLASMA_Complex32_t *W );
void QUARK_CORE_cpltmg_fiedler(Quark *quark, Quark_Task_Flags *task_flags,
                               int m, int n,
                               const PLASMA_Complex32_t *X, int incX,
                               const PLASMA_Complex32_t *Y, int incY,
                               PLASMA_Complex32_t *A, int lda);
void QUARK_CORE_cpltmg_hankel( Quark *quark, Quark_Task_Flags *task_flags,
                               PLASMA_enum uplo, int M, int N, PLASMA_Complex32_t *A, int LDA,
                               int m0, int n0, int nb,
                               const PLASMA_Complex32_t *V1,
                               const PLASMA_Complex32_t *V2);
void QUARK_CORE_cpltmg_toeppd1(Quark *quark, Quark_Task_Flags *task_flags,
                               int gM, int m0, int M,
                               PLASMA_Complex32_t *W,
                               unsigned long long int seed);
void QUARK_CORE_cpltmg_toeppd2(Quark *quark, Quark_Task_Flags *task_flags,
                               int M, int N, int K, int m0, int n0,
                               const PLASMA_Complex32_t *W,
                               PLASMA_Complex32_t *A, int LDA );
void QUARK_CORE_cpotrf(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int n, int nb,
                       PLASMA_Complex32_t *A, int lda,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       int iinfo);
void QUARK_CORE_csetvar(Quark *quark, Quark_Task_Flags *task_flags,
                        const PLASMA_Complex32_t *alpha, PLASMA_Complex32_t *x,
                        PLASMA_Complex32_t *Alock);
void QUARK_CORE_cshift( Quark *quark, Quark_Task_Flags *task_flags,
                        int s, int m, int n, int L,
                        PLASMA_Complex32_t *A);
void QUARK_CORE_cshiftw(Quark *quark, Quark_Task_Flags *task_flags,
                        int s, int cl, int m, int n, int L,
                        PLASMA_Complex32_t *A, PLASMA_Complex32_t *W);
void QUARK_CORE_cssssm(Quark *quark, Quark_Task_Flags *task_flags,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       const PLASMA_Complex32_t *L1, int ldl1,
                       const PLASMA_Complex32_t *L2, int ldl2,
                       const int *IPIV);
void QUARK_CORE_csymm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo,
                      int m, int n, int nb,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                      const PLASMA_Complex32_t *B, int ldb,
                      PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_csyrk(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum uplo, PLASMA_enum trans,
                      int n, int k, int nb,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                      PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_csyr2k(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum trans,
                       int n, int k, int nb,
                       PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                       const PLASMA_Complex32_t *B, int LDB,
                       PLASMA_Complex32_t beta, PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_csyssq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum uplo, int n, const PLASMA_Complex32_t *A, int lda,
                           float *scale, float *sumsq,
                           float *fake, int szeF, int paramF );
void QUARK_CORE_cswpab(Quark *quark, Quark_Task_Flags *task_flags,
                       int i, int n1, int n2,
                       PLASMA_Complex32_t *A, int szeA);
void QUARK_CORE_cswptr_ontile(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc descA, PLASMA_Complex32_t *Aij,
                              int i1,  int i2, const int *ipiv, int inc,
                              const PLASMA_Complex32_t *Akk, int ldak);
void QUARK_CORE_ctrasm(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum storev, PLASMA_enum uplo, PLASMA_enum diag, int m, int n,
                       const PLASMA_Complex32_t *A, int lda, int szeA,
                       float *work, int szeW);
void QUARK_CORE_ctrasm_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum storev, PLASMA_enum uplo, PLASMA_enum diag, int m, int n,
                          const PLASMA_Complex32_t *A, int lda, int szeA,
                          float *work, int szeW,
                          float *fake, int szeF);
void QUARK_CORE_ctrdalg1(Quark *quark, Quark_Task_Flags *task_flags,
                        int n,
                        int nb,
                        PLASMA_Complex32_t *A,
                        int lda,
                        PLASMA_Complex32_t *V,
                        PLASMA_Complex32_t *TAU,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        int *PCOL, int *ACOL, int *MCOL);
void QUARK_CORE_ctrmm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
                      int m, int n, int nb,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                      PLASMA_Complex32_t *B, int ldb);
void QUARK_CORE_ctrmm_p2(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
                         int m, int n, int nb,
                         PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                         PLASMA_Complex32_t **B, int ldb);
void QUARK_CORE_ctrsm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
                      int m, int n, int nb,
                      PLASMA_Complex32_t alpha, const PLASMA_Complex32_t *A, int lda,
                      PLASMA_Complex32_t *B, int ldb);
void QUARK_CORE_ctrssq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum uplo, PLASMA_enum diag,
                           int m, int n, const PLASMA_Complex32_t *A, int lda,
                           float *scale, float *sumsq,
                           float *fake, int szeF, int paramF );
void QUARK_CORE_ctrtri(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum diag, int n, int nb,
                       PLASMA_Complex32_t *A, int lda,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       int iinfo);
void QUARK_CORE_ctslqt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsmlq(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       const PLASMA_Complex32_t *V, int ldv,
                       const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsmlq_hetra1(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_enum side, PLASMA_enum trans,
                              int m1, int n1, int m2, int n2, int k, int ib, int nb,
                              PLASMA_Complex32_t *A1, int lda1,
                              PLASMA_Complex32_t *A2, int lda2,
                              const PLASMA_Complex32_t *V, int ldv,
                              const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsmlq_corner(Quark *quark, Quark_Task_Flags *task_flags,
                              int m1, int n1, int m2, int n2, int m3, int n3, int k, int ib, int nb,
                              PLASMA_Complex32_t *A1, int lda1,
                              PLASMA_Complex32_t *A2, int lda2,
                              PLASMA_Complex32_t *A3, int lda3,
                              const PLASMA_Complex32_t *V, int ldv,
                              const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsmqr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       const PLASMA_Complex32_t *V, int ldv,
                       const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsmqr_hetra1(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_enum side, PLASMA_enum trans,
                              int m1, int n1, int m2, int n2, int k, int ib, int nb,
                              PLASMA_Complex32_t *A1, int lda1,
                              PLASMA_Complex32_t *A2, int lda2,
                              const PLASMA_Complex32_t *V, int ldv,
                              const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsmqr_corner(Quark *quark, Quark_Task_Flags *task_flags,
                              int m1, int n1, int m2, int n2, int m3, int n3, int k, int ib, int nb,
                              PLASMA_Complex32_t *A1, int lda1,
                              PLASMA_Complex32_t *A2, int lda2,
                              PLASMA_Complex32_t *A3, int lda3,
                              const PLASMA_Complex32_t *V, int ldv,
                              const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctsqrt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_ctstrf(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *U, int ldu,
                       PLASMA_Complex32_t *A, int lda,
                       PLASMA_Complex32_t *L, int ldl,
                       int *IPIV,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       PLASMA_bool check_info, int iinfo);
void QUARK_CORE_cttmqr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       const PLASMA_Complex32_t *V, int ldv,
                       const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_cttqrt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_cttmlq(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       const PLASMA_Complex32_t *V, int ldv,
                       const PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_cttlqt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       PLASMA_Complex32_t *T, int ldt);
void QUARK_CORE_cpamm(Quark *quark, Quark_Task_Flags *task_flags,
                       int op, PLASMA_enum side, PLASMA_enum storev,
                       int m, int n, int k, int l,
                       const PLASMA_Complex32_t *A1, int lda1,
                       PLASMA_Complex32_t *A2, int lda2,
                       const PLASMA_Complex32_t *V, int ldv,
                       PLASMA_Complex32_t *W, int ldw);
void QUARK_CORE_cplssq( Quark *quark, Quark_Task_Flags *task_flags,
                        int m, const float *A, float *result );
void QUARK_CORE_cunmlq(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m, int n, int ib,  int nb, int k,
                       const PLASMA_Complex32_t *A, int lda,
                       const PLASMA_Complex32_t *T, int ldt,
                       PLASMA_Complex32_t *C, int ldc);
void QUARK_CORE_cunmqr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m, int n, int k, int ib, int nb,
                       const PLASMA_Complex32_t *A, int lda,
                       const PLASMA_Complex32_t *T, int ldt,
                       PLASMA_Complex32_t *C, int ldc);

/** ****************************************************************************
 *  Declarations of QUARK wrappers (called by QUARK) - alphabetical order
 **/
void CORE_scasum_quark(Quark *quark);
void CORE_scasum_f1_quark(Quark *quark);
void CORE_cgeadd_quark(Quark *quark);
void CORE_cbrdalg1_quark(Quark *quark);
void CORE_cgelqt_quark(Quark *quark);
void CORE_cgemm_quark(Quark *quark);
void CORE_cgemm_tile_quark(Quark *quark);
void CORE_cgemv_quark(Quark *quark);
void CORE_cgemv_tile_quark(Quark *quark);
void CORE_cgeqp3_init_quark(Quark *quark);
void CORE_cgeqp3_larfg_quark(Quark *quark);
void CORE_cgeqp3_norms_quark(Quark *quark);
void CORE_cgeqp3_pivot_quark(Quark *quark);
void CORE_cgeqp3_tntpiv_quark(Quark *quark);
void CORE_cgeqp3_update_quark(Quark *quark);
void CORE_cgeqrt_quark(Quark *quark);
void CORE_cgessm_quark(Quark *quark);
void CORE_cgessq_quark(Quark *quark);
void CORE_cgessq_f1_quark(Quark *quark);
void CORE_cgetrf_quark(Quark *quark);
void CORE_cgetrf_incpiv_quark(Quark *quark);
void CORE_cgetrf_nopiv_quark(Quark* quark);
void CORE_cgetrf_reclap_quark(Quark *quark);
void CORE_cgetrf_rectil_quark(Quark* quark);
void CORE_cgetrip_quark(Quark *quark);
void CORE_cgetrip_f1_quark(Quark *quark);
void CORE_cgetrip_f2_quark(Quark *quark);
#ifdef COMPLEX
void CORE_chemm_quark(Quark *quark);
void CORE_cherk_quark(Quark *quark);
void CORE_cher2k_quark(Quark *quark);
#endif
void CORE_chegst_quark(Quark *quark);
void CORE_cherfb_quark(Quark *quark);
void CORE_chessq_quark(Quark *quark);
void CORE_chessq_f1_quark(Quark *quark);
void CORE_clacpy_quark(Quark *quark);
void CORE_clacpy_f1_quark(Quark *quark);
void CORE_clacpy_pivot_quark(Quark *quark);
void CORE_clatro_quark(Quark *quark);
void CORE_clatro_f1_quark(Quark *quark);
void CORE_clange_quark(Quark *quark);
void CORE_clange_f1_quark(Quark *quark);
#ifdef COMPLEX
void CORE_clanhe_quark(Quark *quark);
void CORE_clanhe_f1_quark(Quark *quark);
#endif
void CORE_clansy_quark(Quark *quark);
void CORE_clansy_f1_quark(Quark *quark);
void CORE_claset_quark(Quark *quark);
void CORE_claset2_quark(Quark *quark);
void CORE_clatro_quark(Quark *quark);
void CORE_clauum_quark(Quark *quark);
void CORE_cpamm_quark(Quark *quark);
void CORE_cplghe_quark(Quark *quark);
void CORE_cplgsy_quark(Quark *quark);
void CORE_cplrnt_quark(Quark *quark);
void CORE_cpltmg_quark(Quark *quark);
void CORE_cplssq_quark(Quark *quark);
void CORE_cpotrf_quark(Quark *quark);
void CORE_csetvar_quark(Quark *quark);
void CORE_cshift_quark(Quark *quark);
void CORE_cshiftw_quark(Quark *quark);
void CORE_cssssm_quark(Quark *quark);
void CORE_csymm_quark(Quark *quark);
void CORE_csyrk_quark(Quark *quark);
void CORE_csyr2k_quark(Quark *quark);
void CORE_csyssq_quark(Quark *quark);
void CORE_csyssq_f1_quark(Quark *quark);
void CORE_cswpab_quark(Quark *quark);
void CORE_cswptr_ontile_quark(Quark *quark);
void CORE_ctrdalg1_quark(Quark *quark);
void CORE_ctrmm_quark(Quark *quark);
void CORE_ctrsm_quark(Quark *quark);
void CORE_ctrtri_quark(Quark *quark);
void CORE_ctslqt_quark(Quark *quark);
void CORE_ctsmlq_quark(Quark *quark);
void CORE_ctsmlq_hetra1_quark(Quark *quark);
void CORE_ctsmlq_corner_quark(Quark *quark);
void CORE_ctsmqr_quark(Quark *quark);
void CORE_ctsmqr_hetra1_quark(Quark *quark);
void CORE_ctsmqr_corner_quark(Quark *quark);
void CORE_ctsqrt_quark(Quark *quark);
void CORE_ctstrf_quark(Quark *quark);
void CORE_cttmqr_quark(Quark *quark);
void CORE_cttqrt_quark(Quark *quark);
void CORE_cttmlq_quark(Quark *quark);
void CORE_cttlqt_quark(Quark *quark);
void CORE_cunmlq_quark(Quark *quark);
void CORE_cunmqr_quark(Quark *quark);
void CORE_claswp_quark(Quark* quark);
void CORE_claswp_f2_quark(Quark* quark);
void CORE_claswp_ontile_quark(Quark *quark);
void CORE_claswp_ontile_f2_quark(Quark *quark);
void CORE_claswpc_ontile_quark(Quark *quark);
void CORE_ctrmm_p2_quark(Quark* quark);
void CORE_cgemm_f2_quark(Quark* quark);
void CORE_cgemm_p2_quark(Quark* quark);
void CORE_cgemm_p2f1_quark(Quark* quark);
void CORE_cgemm_p3_quark(Quark* quark);

#endif /* defined(QUARK_H) */

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
