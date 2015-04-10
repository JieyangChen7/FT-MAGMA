/**
 *
 * @file core_zblas.h
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
 * @precisions normal z -> c d s
 *
 **/
#ifndef _PLASMA_CORE_ZBLAS_H_
#define _PLASMA_CORE_ZBLAS_H_

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

/** ****************************************************************************
 *  Declarations of serial kernels - alphabetical order
 **/
void CORE_dzasum(int storev, PLASMA_enum uplo, int M, int N,
                 const PLASMA_Complex64_t *A, int lda, double *work);
void CORE_zbrdalg1(     PLASMA_enum uplo,
                        int n,
                        int nb,
                        PLASMA_Complex64_t *A,
                        int lda,
                        PLASMA_Complex64_t *VQ,
                        PLASMA_Complex64_t *TAUQ,
                        PLASMA_Complex64_t *VP,
                        PLASMA_Complex64_t *TAUP,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        PLASMA_Complex64_t *work);
int CORE_zgbelr(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                int st, int ed, int eltsize);
int CORE_zgbrce(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                int st, int ed, int eltsize);
int CORE_zgblrx(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                int st, int ed, int eltsize);
int CORE_zgeadd(int M, int N, PLASMA_Complex64_t alpha,
                const PLASMA_Complex64_t *A, int LDA,
                      PLASMA_Complex64_t *B, int LDB);
int  CORE_zgelqt(int M, int N, int IB,
                 PLASMA_Complex64_t *A, int LDA,
                 PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK);
void CORE_zgemm(PLASMA_enum transA, PLASMA_enum transB,
                int M, int N, int K,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                          const PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int LDC);
void CORE_zgemv(PLASMA_enum trans, int M, int N,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                          const PLASMA_Complex64_t *x, int incx,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *y, int incy);
void CORE_zgeqp3_init( int n, int *jpvt );
void CORE_zgeqp3_larfg( PLASMA_desc A, int ii, int jj, int i, int j,
                        PLASMA_Complex64_t *tau, PLASMA_Complex64_t *beta );
void CORE_zgeqp3_norms( PLASMA_desc A, int ioff, int joff, double *norms1, double *norms2 );
void CORE_zgeqp3_pivot( PLASMA_desc A, PLASMA_Complex64_t *F, int ldf,
                        int jj, int k, int *jpvt,
                        double *norms1, double *norms2, int *info );
int  CORE_zgeqp3_tntpiv(int m, int n,
                        PLASMA_Complex64_t *A, int lda,
                        int *IPIV, PLASMA_Complex64_t *tau,
                        int *iwork);
void CORE_zgeqp3_update( const PLASMA_Complex64_t *Ajj, int lda1,
                         PLASMA_Complex64_t       *Ajk, int lda2,
                         const PLASMA_Complex64_t *Fk,  int ldf,
                         int joff, int k, int koff, int nb,
                         double *norms1, double *norms2,
                         int *info );
int  CORE_zgeqrt(int M, int N, int IB,
                 PLASMA_Complex64_t *A, int LDA,
                 PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *TAU, PLASMA_Complex64_t *WORK);
int  CORE_zgessm(int M, int N, int K, int IB,
                 const int *IPIV,
                 const PLASMA_Complex64_t *L, int LDL,
                 PLASMA_Complex64_t *A, int LDA);
int  CORE_zgessq(int M, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *scale, double *sumsq);
int  CORE_zgetf2_nopiv(int m, int n,
                      PLASMA_Complex64_t *A, int lda);
int  CORE_zgetrf(int M, int N,
                 PLASMA_Complex64_t *A, int LDA,
                 int *IPIV, int *INFO);
int  CORE_zgetrf_incpiv(int M, int N, int IB,
                        PLASMA_Complex64_t *A, int LDA,
                        int *IPIV, int *INFO);
int  CORE_zgetrf_nopiv(int m, int n, int ib,
                      PLASMA_Complex64_t *A, int lda);
int  CORE_zgetrf_reclap(int M, int N,
                        PLASMA_Complex64_t *A, int LDA,
                        int *IPIV, int *info);
void CORE_zgetrf_reclap_init(void);
int  CORE_zgetrf_rectil(const PLASMA_desc A, int *IPIV, int *info);
void CORE_zgetrf_rectil_init(void);
void CORE_zgetrip(int m, int n, PLASMA_Complex64_t *A,
                  PLASMA_Complex64_t *work);
int CORE_zhbelr(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                int st, int ed, int eltsize);
int CORE_zhblrx(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                int st, int ed, int eltsize);
int CORE_zhbrce(PLASMA_enum uplo, int N,
                PLASMA_desc *A, PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                int st, int ed, int eltsize);
void CORE_zhbtype1cb(int N, int NB,
                     PLASMA_Complex64_t *A, int LDA,
                     PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     PLASMA_Complex64_t *WORK);
void CORE_zhbtype2cb(int N, int NB,
                     PLASMA_Complex64_t *A, int LDA,
                     PLASMA_Complex64_t *V, PLASMA_Complex64_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     PLASMA_Complex64_t *WORK);
void CORE_zhbtype3cb(int N, int NB,
                     PLASMA_Complex64_t *A, int LDA,
                     const PLASMA_Complex64_t *V, const PLASMA_Complex64_t *TAU,
                     int st, int ed, int sweep, int Vblksiz, int WANTZ,
                     PLASMA_Complex64_t *WORK);
void CORE_zgbtype1cb(PLASMA_enum uplo, int N, int NB,
                PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t *VQ, PLASMA_Complex64_t *TAUQ,
                PLASMA_Complex64_t *VP, PLASMA_Complex64_t *TAUP,
                int st, int ed, int sweep, int Vblksiz, int WANTZ,
                PLASMA_Complex64_t *WORK);
void CORE_zgbtype2cb(PLASMA_enum uplo, int N, int NB,
                PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t *VQ, PLASMA_Complex64_t *TAUQ,
                PLASMA_Complex64_t *VP, PLASMA_Complex64_t *TAUP,
                int st, int ed, int sweep, int Vblksiz, int WANTZ,
                PLASMA_Complex64_t *WORK);
void CORE_zgbtype3cb(PLASMA_enum uplo, int N, int NB,
                PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t *VQ, PLASMA_Complex64_t *TAUQ,
                PLASMA_Complex64_t *VP, PLASMA_Complex64_t *TAUP,
                int st, int ed, int sweep, int Vblksiz, int WANTZ,
                PLASMA_Complex64_t *WORK);
void CORE_zhegst(int itype, PLASMA_enum uplo, int N,
                 PLASMA_Complex64_t *A, int LDA,
                 PLASMA_Complex64_t *B, int LDB, int *INFO);
#ifdef COMPLEX
void CORE_zhemm(PLASMA_enum side, PLASMA_enum uplo,
                int M, int N,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                          const PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int LDC);
void CORE_zherk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K,
                double alpha, const PLASMA_Complex64_t *A, int LDA,
                double beta,        PLASMA_Complex64_t *C, int LDC);
void CORE_zher2k(PLASMA_enum uplo, PLASMA_enum trans,
                 int N, int K,
                 PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                           const PLASMA_Complex64_t *B, int LDB,
                 double beta,                    PLASMA_Complex64_t *C, int LDC);
int  CORE_zhessq(PLASMA_enum uplo, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *scale, double *sumsq);
#endif
int  CORE_zherfb(PLASMA_enum uplo, int N, int K, int IB, int NB,
                 const PLASMA_Complex64_t *A,    int LDA,
                 const PLASMA_Complex64_t *T,    int LDT,
                       PLASMA_Complex64_t *C,    int LDC,
                       PLASMA_Complex64_t *WORK, int LDWORK);
void CORE_zlacpy(PLASMA_enum uplo, int M, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                       PLASMA_Complex64_t *B, int LDB);
int CORE_zlacpy_pivot( const PLASMA_desc descA,
                       PLASMA_enum direct,
                       int k1, int k2, const int *ipiv,
                       int *rankin, int *rankout,
                       PLASMA_Complex64_t *A, int lda,
                       int init);
void CORE_zlange(int norm, int M, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *work, double *normA);
#ifdef COMPLEX
void CORE_zlanhe(int norm, PLASMA_enum uplo, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *work, double *normA);
#endif
void CORE_zlansy(int norm, PLASMA_enum uplo, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *work, double *normA);
void CORE_zlantr(PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag,
                 int M, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *work, double *normA);
int CORE_zlarfb_gemm(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
                     int M, int N, int K,
                     const PLASMA_Complex64_t *V, int LDV,
                     const PLASMA_Complex64_t *T, int LDT,
                           PLASMA_Complex64_t *C, int LDC,
                           PLASMA_Complex64_t *WORK, int LDWORK);
int CORE_zlarfx2(PLASMA_enum side, int N,
                 PLASMA_Complex64_t V,
                 PLASMA_Complex64_t TAU,
                 PLASMA_Complex64_t *C1, int LDC1,
                 PLASMA_Complex64_t *C2, int LDC2);
int CORE_zlarfx2c(PLASMA_enum uplo,
                  PLASMA_Complex64_t V,
                  PLASMA_Complex64_t TAU,
                  PLASMA_Complex64_t *C1,
                  PLASMA_Complex64_t *C2,
                  PLASMA_Complex64_t *C3);
int CORE_zlarfx2ce(PLASMA_enum uplo,
                   PLASMA_Complex64_t *V,
                   PLASMA_Complex64_t *TAU,
                   PLASMA_Complex64_t *C1,
                   PLASMA_Complex64_t *C2,
                   PLASMA_Complex64_t *C3);
void CORE_zlarfy(int N,
                 PLASMA_Complex64_t *A, int LDA,
                 const PLASMA_Complex64_t *V,
                 const PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK);
void CORE_zlaset(PLASMA_enum uplo, int n1, int n2,
                 PLASMA_Complex64_t alpha, PLASMA_Complex64_t beta,
                 PLASMA_Complex64_t *tileA, int ldtilea);
void CORE_zlaset2(PLASMA_enum uplo, int n1, int n2, PLASMA_Complex64_t alpha,
                  PLASMA_Complex64_t *tileA, int ldtilea);
void CORE_zlaswp(int N, PLASMA_Complex64_t *A, int LDA,
                 int I1,  int I2, const int *IPIV, int INC);
int  CORE_zlaswp_ontile( PLASMA_desc descA, int i1, int i2, const int *ipiv, int inc);
int  CORE_zlaswpc_ontile(PLASMA_desc descA, int i1, int i2, const int *ipiv, int inc);
int  CORE_zlatro(PLASMA_enum uplo, PLASMA_enum trans,
                 int M, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                       PLASMA_Complex64_t *B, int LDB);
void CORE_zlauum(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA);
int CORE_zpamm(int op, PLASMA_enum side, PLASMA_enum storev,
               int M, int N, int K, int L,
               const PLASMA_Complex64_t *A1, int LDA1,
                     PLASMA_Complex64_t *A2, int LDA2,
               const PLASMA_Complex64_t *V, int LDV,
                     PLASMA_Complex64_t *W, int LDW);
int  CORE_zparfb(PLASMA_enum side, PLASMA_enum trans, PLASMA_enum direct, PLASMA_enum storev,
                 int M1, int N1, int M2, int N2, int K, int L,
                       PLASMA_Complex64_t *A1, int LDA1,
                       PLASMA_Complex64_t *A2, int LDA2,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                       PLASMA_Complex64_t *WORK, int LDWORK);
int CORE_zpemv(PLASMA_enum trans, PLASMA_enum storev,
               int M, int N, int L,
               PLASMA_Complex64_t ALPHA,
               const PLASMA_Complex64_t *A, int LDA,
               const PLASMA_Complex64_t *X, int INCX,
               PLASMA_Complex64_t BETA,
               PLASMA_Complex64_t *Y, int INCY,
               PLASMA_Complex64_t *WORK);
void CORE_zplghe(double bump, int m, int n, PLASMA_Complex64_t *A, int lda,
                 int bigM, int m0, int n0, unsigned long long int seed );
void CORE_zplgsy(PLASMA_Complex64_t bump, int m, int n, PLASMA_Complex64_t *A, int lda,
                 int bigM, int m0, int n0, unsigned long long int seed );
void CORE_zplrnt(int m, int n, PLASMA_Complex64_t *A, int lda,
                 int bigM, int m0, int n0, unsigned long long int seed );
int  CORE_zpltmg(PLASMA_enum mtxtype, int m, int n, PLASMA_Complex64_t *A, int lda,
                  int gM, int gN, int m0, int n0, unsigned long long int seed );
int  CORE_zpltmg_chebvand( int M, int N, PLASMA_Complex64_t *A, int LDA,
                           int gN, int m0, int n0,
                           PLASMA_Complex64_t *W );
int  CORE_zpltmg_circul( int M, int N, PLASMA_Complex64_t *A, int LDA,
                         int gM, int m0, int n0,
                         const PLASMA_Complex64_t *V );
void CORE_zpltmg_condexq( int M, int N, PLASMA_Complex64_t *Q, int LDQ );
void CORE_zpltmg_fiedler(int m, int n,
                         const PLASMA_Complex64_t *X, int incX,
                         const PLASMA_Complex64_t *Y, int incY,
                         PLASMA_Complex64_t *A, int lda);
int  CORE_zpltmg_hankel( PLASMA_enum uplo, int M, int N, PLASMA_Complex64_t *A, int LDA,
                         int m0, int n0, int nb,
                         const PLASMA_Complex64_t *V1,
                         const PLASMA_Complex64_t *V2 );
void CORE_zpltmg_toeppd1( int gM, int m0, int M, PLASMA_Complex64_t *W,
                          unsigned long long int seed );
void CORE_zpltmg_toeppd2( int M, int N, int K, int m0, int n0,
                          const PLASMA_Complex64_t *W,
                          PLASMA_Complex64_t *A, int LDA );
void CORE_zpotrf(PLASMA_enum uplo, int N, PLASMA_Complex64_t *A, int LDA, int *INFO);
void CORE_zsetvar(const PLASMA_Complex64_t *alpha, PLASMA_Complex64_t *x);
void CORE_zshift(int s, int m, int n, int L,
                 PLASMA_Complex64_t *A);
void CORE_zshiftw(int s, int cl, int m, int n, int L,
                  PLASMA_Complex64_t *A, PLASMA_Complex64_t *W);
int  CORE_zssssm(int M1, int N1, int M2, int N2, int K, int IB,
                       PLASMA_Complex64_t *A1, int LDA1,
                       PLASMA_Complex64_t *A2, int LDA2,
                 const PLASMA_Complex64_t *L1, int LDL1,
                 const PLASMA_Complex64_t *L2, int LDL2,
                 const int *IPIV);
void CORE_zsymm(PLASMA_enum side, PLASMA_enum uplo,
                int M, int N,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                          const PLASMA_Complex64_t *B, int LDB,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int LDC);
void CORE_zsyrk(PLASMA_enum uplo, PLASMA_enum trans,
                int N, int K,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int LDC);
void CORE_zsyr2k(PLASMA_enum uplo, PLASMA_enum trans,
                 int N, int K,
                 PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                           const PLASMA_Complex64_t *B, int LDB,
                 PLASMA_Complex64_t beta,        PLASMA_Complex64_t *C, int LDC);
int  CORE_zsyssq(PLASMA_enum uplo, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *scale, double *sumsq);
void CORE_zswpab(int i, int n1, int n2,
                 PLASMA_Complex64_t *A, PLASMA_Complex64_t *work);
int  CORE_zswptr_ontile(PLASMA_desc descA, int i1, int i2, const int *ipiv, int inc,
                        const PLASMA_Complex64_t *Akk, int ldak);
void CORE_ztrasm(PLASMA_enum storev, PLASMA_enum uplo, PLASMA_enum diag,
                 int M, int N, const PLASMA_Complex64_t *A, int lda, double *work);
void CORE_ztrdalg1(int n,
                        int nb,
                        PLASMA_Complex64_t *A,
                        int lda,
                        PLASMA_Complex64_t *V,
                        PLASMA_Complex64_t *TAU,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        PLASMA_Complex64_t *work);
void CORE_ztrmm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int M, int N,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                                PLASMA_Complex64_t *B, int LDB);
void CORE_ztrsm(PLASMA_enum side, PLASMA_enum uplo,
                PLASMA_enum transA, PLASMA_enum diag,
                int M, int N,
                PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int LDA,
                                                PLASMA_Complex64_t *B, int LDB);
int  CORE_ztrssq(PLASMA_enum uplo, PLASMA_enum diag, int M, int N,
                 const PLASMA_Complex64_t *A, int LDA,
                 double *scale, double *sumsq);
void CORE_ztrtri(PLASMA_enum uplo, PLASMA_enum diag, int N,
                 PLASMA_Complex64_t *A, int LDA, int *info);
int  CORE_ztslqt(int M, int N, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *TAU, PLASMA_Complex64_t *WORK);
int  CORE_ztsmlq(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *WORK, int LDWORK);
int CORE_ztsmlq_corner( int m1, int n1, int m2, int n2, int m3, int n3,
                        int k, int ib, int nb,
                        PLASMA_Complex64_t *A1, int lda1,
                        PLASMA_Complex64_t *A2, int lda2,
                        PLASMA_Complex64_t *A3, int lda3,
                        const PLASMA_Complex64_t *V, int ldv,
                        const PLASMA_Complex64_t *T, int ldt,
                        PLASMA_Complex64_t *WORK, int ldwork);
int CORE_ztsmlq_hetra1( PLASMA_enum side, PLASMA_enum trans,
                        int m1, int n1, int m2, int n2,
                        int k, int ib,
                        PLASMA_Complex64_t *A1, int lda1,
                        PLASMA_Complex64_t *A2, int lda2,
                        const PLASMA_Complex64_t *V, int ldv,
                        const PLASMA_Complex64_t *T, int ldt,
                        PLASMA_Complex64_t *WORK, int ldwork);
int  CORE_ztsmqr(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *WORK, int LDWORK);
int CORE_ztsmqr_corner( int m1, int n1, int m2, int n2, int m3, int n3,
                        int k, int ib, int nb,
                        PLASMA_Complex64_t *A1, int lda1,
                        PLASMA_Complex64_t *A2, int lda2,
                        PLASMA_Complex64_t *A3, int lda3,
                        const PLASMA_Complex64_t *V, int ldv,
                        const PLASMA_Complex64_t *T, int ldt,
                        PLASMA_Complex64_t *WORK, int ldwork);
int CORE_ztsmqr_hetra1( PLASMA_enum side, PLASMA_enum trans,
                        int m1, int n1, int m2, int n2,
                        int k, int ib,
                        PLASMA_Complex64_t *A1, int lda1,
                        PLASMA_Complex64_t *A2, int lda2,
                        const PLASMA_Complex64_t *V, int ldv,
                        const PLASMA_Complex64_t *T, int ldt,
                        PLASMA_Complex64_t *WORK, int ldwork);
int  CORE_ztsqrt(int M, int N, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *TAU, PLASMA_Complex64_t *WORK);
int  CORE_ztstrf(int M, int N, int IB, int NB,
                 PLASMA_Complex64_t *U, int LDU,
                 PLASMA_Complex64_t *A, int LDA,
                 PLASMA_Complex64_t *L, int LDL,
                 int *IPIV, PLASMA_Complex64_t *WORK,
                 int LDWORK, int *INFO);
int  CORE_zttmqr(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *WORK, int LDWORK);
int  CORE_zttqrt(int M, int N, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK);
int  CORE_zttmlq(PLASMA_enum side, PLASMA_enum trans,
                 int M1, int N1, int M2, int N2, int K, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *WORK, int LDWORK);
int  CORE_zttlqt(int M, int N, int IB,
                 PLASMA_Complex64_t *A1, int LDA1,
                 PLASMA_Complex64_t *A2, int LDA2,
                 PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *TAU,
                 PLASMA_Complex64_t *WORK);
int  CORE_zunmlq(PLASMA_enum side, PLASMA_enum trans,
                 int M, int N, int IB, int K,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *C, int LDC,
                 PLASMA_Complex64_t *WORK, int LDWORK);
int  CORE_zunmqr(PLASMA_enum side, PLASMA_enum trans,
                 int M, int N, int K, int IB,
                 const PLASMA_Complex64_t *V, int LDV,
                 const PLASMA_Complex64_t *T, int LDT,
                 PLASMA_Complex64_t *C, int LDC,
                 PLASMA_Complex64_t *WORK, int LDWORK);

#if defined(QUARK_H)
/** ****************************************************************************
 *  Declarations of QUARK wrappers (called by PLASMA) - alphabetical order
 **/
void QUARK_CORE_dzasum(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum storev, PLASMA_enum uplo, int m, int n,
                       const PLASMA_Complex64_t *A, int lda, int szeA,
                       double *work, int szeW);
void QUARK_CORE_dzasum_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum storev, PLASMA_enum uplo, int m, int n,
                          const PLASMA_Complex64_t *A, int lda, int szeA,
                          double *work, int szeW,
                          double *fake, int szeF);
void QUARK_CORE_zgeadd(Quark *quark, Quark_Task_Flags *task_flags,
                      int m, int n, int nb, PLASMA_Complex64_t alpha,
                      const PLASMA_Complex64_t *A, int lda,
                      PLASMA_Complex64_t *B, int ldb);
void QUARK_CORE_zbrdalg1(Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum uplo,
                        int n, int nb,
                        PLASMA_Complex64_t *A,
                        int lda,
                        PLASMA_Complex64_t *VQ,
                        PLASMA_Complex64_t *TAUQ,
                        PLASMA_Complex64_t *VP,
                        PLASMA_Complex64_t *TAUP,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        int *PCOL, int *ACOL, int *MCOL);
void QUARK_CORE_zgelqt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_zgemm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum transA, PLASMA_enum transB,
                      int m, int n, int k, int nb,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                      const PLASMA_Complex64_t *B, int ldb,
                      PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zgemm2( Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum transA, PLASMA_enum transB,
                        int m, int n, int k, int nb,
                        PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                        const PLASMA_Complex64_t *B, int ldb,
                        PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zgemm_f2(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum transA, PLASMA_enum transB,
                         int m, int n, int k, int nb,
                         PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                         const PLASMA_Complex64_t *B, int ldb,
                         PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc,
                         PLASMA_Complex64_t *fake1, int szefake1, int flag1,
                         PLASMA_Complex64_t *fake2, int szefake2, int flag2);
void QUARK_CORE_zgemm_p2(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum transA, PLASMA_enum transB,
                         int m, int n, int k, int nb,
                         PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                         const PLASMA_Complex64_t **B, int ldb,
                         PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zgemm_p2f1(Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum transA, PLASMA_enum transB,
                           int m, int n, int k, int nb,
                           PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                           const PLASMA_Complex64_t **B, int ldb,
                           PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc,
                           PLASMA_Complex64_t *fake1, int szefake1, int flag1);
void QUARK_CORE_zgemm_p3(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum transA, PLASMA_enum transB,
                         int m, int n, int k, int nb,
                         PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                         const PLASMA_Complex64_t *B, int ldb,
                         PLASMA_Complex64_t beta, PLASMA_Complex64_t **C, int ldc);
void QUARK_CORE_zgemm_tile(Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum transA, PLASMA_enum transB,
                           int m, int n, int k, int nb,
                           const PLASMA_Complex64_t *alpha, const PLASMA_Complex64_t *A, int lda,
                                                            const PLASMA_Complex64_t *B, int ldb,
                           const PLASMA_Complex64_t *beta,        PLASMA_Complex64_t *C, int ldc,
                           const PLASMA_Complex64_t *Alock,
                           const PLASMA_Complex64_t *Block,
                           const PLASMA_Complex64_t *Clock);
void QUARK_CORE_zgemv(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum trans, int m, int n,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                                                const PLASMA_Complex64_t *x, int incx,
                      PLASMA_Complex64_t beta,        PLASMA_Complex64_t *y, int incy);
void QUARK_CORE_zgemv_tile(Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum trans,
                           int m, int n,
                           const PLASMA_Complex64_t *alpha, const PLASMA_Complex64_t *A, int lda,
                                                            const PLASMA_Complex64_t *x, int incx,
                           const PLASMA_Complex64_t *beta,        PLASMA_Complex64_t *y, int incy,
                           const PLASMA_Complex64_t *Alock,
                           const PLASMA_Complex64_t *xlock,
                           const PLASMA_Complex64_t *ylock);
void QUARK_CORE_zgeqp3_init( Quark *quark, Quark_Task_Flags *task_flags,
                             int n, int *jpvt );
void QUARK_CORE_zgeqp3_larfg(Quark *quark, Quark_Task_Flags *task_flags,
                             PLASMA_desc A, int ii, int jj, int i, int j,
                             PLASMA_Complex64_t *tau, PLASMA_Complex64_t *beta );
void QUARK_CORE_zgeqp3_norms( Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc A, int ioff, int joff, double *norms1, double *norms2 );
void QUARK_CORE_zgeqp3_pivot( Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc A,
                              PLASMA_Complex64_t *F, int ldf,
                              int jj, int k, int *jpvt,
                              double *norms1, double *norms2, int *info );
void QUARK_CORE_zgeqp3_tntpiv(Quark *quark, Quark_Task_Flags *task_flags,
                              int m, int n, int nb,
                              PLASMA_Complex64_t *A, int lda,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo);
void QUARK_CORE_zgeqp3_update( Quark *quark, Quark_Task_Flags *task_flags,
                               PLASMA_Complex64_t *Ajj, int lda1,
                               PLASMA_Complex64_t *Ajk, int lda2,
                               PLASMA_Complex64_t *Fk,  int ldf,
                               int joff, int k, int koff, int nb,
                               double *norms1, double *norms2, int *info );
void QUARK_CORE_zgeqrt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_zgessm(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int k, int ib, int nb,
                       const int *IPIV,
                       const PLASMA_Complex64_t *L, int ldl,
                       PLASMA_Complex64_t *A, int lda);
void QUARK_CORE_zgessq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           int m, int n, const PLASMA_Complex64_t *A, int lda,
                           double *scale, double *sumsq,
                           double *fake, int szeF, int paramF );
void QUARK_CORE_zgetrf(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int nb,
                       PLASMA_Complex64_t *A, int lda,
                       int *IPIV,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       PLASMA_bool check_info, int iinfo);
void QUARK_CORE_zgetrf_incpiv(Quark *quark, Quark_Task_Flags *task_flags,
                              int m, int n, int ib, int nb,
                              PLASMA_Complex64_t *A, int lda,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo);
void QUARK_CORE_zgetrf_nopiv(Quark *quark, Quark_Task_Flags *task_flags,
                             int m, int n, int ib, int nb,
                             PLASMA_Complex64_t *A, int lda,
                             PLASMA_sequence *sequence, PLASMA_request *request,
                             int iinfo);
void QUARK_CORE_zgetrf_reclap(Quark *quark, Quark_Task_Flags *task_flags,
                              int m, int n, int nb,
                              PLASMA_Complex64_t *A, int lda,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo,
                              int nbthread);
void QUARK_CORE_zgetrf_rectil(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc A, PLASMA_Complex64_t *Amn, int size,
                              int *IPIV,
                              PLASMA_sequence *sequence, PLASMA_request *request,
                              PLASMA_bool check_info, int iinfo,
                              int nbthread);
void QUARK_CORE_zgetrip(Quark *quark, Quark_Task_Flags *task_flags,
                        int m, int n, PLASMA_Complex64_t *A, int szeA);
void QUARK_CORE_zgetrip_f1(Quark *quark, Quark_Task_Flags *task_flags,
                           int m, int n, PLASMA_Complex64_t *A, int szeA,
                           PLASMA_Complex64_t *fake, int szeF, int paramF);
void QUARK_CORE_zgetrip_f2(Quark *quark, Quark_Task_Flags *task_flags,
                           int m, int n, PLASMA_Complex64_t *A, int szeA,
                           PLASMA_Complex64_t *fake1, int szeF1, int paramF1,
                           PLASMA_Complex64_t *fake2, int szeF2, int paramF2);
void QUARK_CORE_zhemm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo,
                      int m, int n, int nb,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                      const PLASMA_Complex64_t *B, int ldb,
                      PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zhegst(Quark *quark, Quark_Task_Flags *task_flags,
                       int itype, PLASMA_enum uplo, int N,
                       PLASMA_Complex64_t *A, int LDA,
                       PLASMA_Complex64_t *B, int LDB,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       int iinfo);
void QUARK_CORE_zherk(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum uplo, PLASMA_enum trans,
                      int n, int k, int nb,
                      double alpha, const PLASMA_Complex64_t *A, int lda,
                      double beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zher2k(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum trans,
                       int n, int k, int nb,
                       PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                       const PLASMA_Complex64_t *B, int LDB,
                       double beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zherfb(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo,
                       int n, int k, int ib, int nb,
                       const PLASMA_Complex64_t *A, int lda,
                       const PLASMA_Complex64_t *T, int ldt,
                       PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zhessq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum uplo, int n, const PLASMA_Complex64_t *A, int lda,
                           double *scale, double *sumsq,
                           double *fake, int szeF, int paramF );
void QUARK_CORE_zlacpy(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int m, int n, int mb,
                       const PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *B, int ldb);
void QUARK_CORE_zlacpy_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum uplo, int m, int n, int nb,
                          const PLASMA_Complex64_t *A, int lda,
                          PLASMA_Complex64_t *B, int ldb,
                          PLASMA_Complex64_t *fake1, int szefake1, int flag1);
void QUARK_CORE_zlacpy_pivot(Quark *quark, Quark_Task_Flags *task_flags,
                             const PLASMA_desc descA,
                             PLASMA_enum direct,
                             int k1, int k2, const int *ipiv,
                             int *rankin, int *rankout,
                             PLASMA_Complex64_t *A, int lda,
                             int pos, int init);
void QUARK_CORE_zlange(Quark *quark, Quark_Task_Flags *task_flags,
                       int norm, int M, int N,
                       const PLASMA_Complex64_t *A, int LDA, int szeA,
                       int szeW, double *result);
void QUARK_CORE_zlange_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          int norm, int M, int N,
                          const PLASMA_Complex64_t *A, int LDA, int szeA,
                          int szeW, double *result,
                          double *fake, int szeF);
#ifdef COMPLEX
void QUARK_CORE_zlanhe(Quark *quark, Quark_Task_Flags *task_flags,
                       int norm, PLASMA_enum uplo, int N,
                       const PLASMA_Complex64_t *A, int LDA, int szeA,
                       int szeW, double *result);
void QUARK_CORE_zlanhe_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          int norm, PLASMA_enum uplo, int N,
                          const PLASMA_Complex64_t *A, int LDA, int szeA,
                          int szeW, double *result,
                          double *fake, int szeF);
#endif
void QUARK_CORE_zlansy(Quark *quark, Quark_Task_Flags *task_flags,
                       int norm, PLASMA_enum uplo, int N,
                       const PLASMA_Complex64_t *A, int LDA, int szeA,
                       int szeW, double *result);
void QUARK_CORE_zlansy_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          int norm, PLASMA_enum uplo, int N,
                          const PLASMA_Complex64_t *A, int LDA, int szeA,
                          int szeW, double *result,
                          double *fake, int szeF);
void QUARK_CORE_zlantr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N,
                       const PLASMA_Complex64_t *A, int LDA, int szeA,
                       int szeW, double *result);
void QUARK_CORE_zlantr_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum norm, PLASMA_enum uplo, PLASMA_enum diag, int M, int N,
                          const PLASMA_Complex64_t *A, int LDA, int szeA,
                          int szeW, double *result,
                          double *fake, int szeF);
void QUARK_CORE_zlaset(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int n1, int n2, PLASMA_Complex64_t alpha,
                       PLASMA_Complex64_t beta, PLASMA_Complex64_t *tileA, int ldtilea);
void QUARK_CORE_zlaset2(Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum uplo, int n1, int n2, PLASMA_Complex64_t alpha,
                        PLASMA_Complex64_t *tileA, int ldtilea);
void QUARK_CORE_zlaswp(Quark *quark, Quark_Task_Flags *task_flags,
                       int n, PLASMA_Complex64_t *A, int lda,
                       int i1,  int i2, const int *ipiv, int inc);
void QUARK_CORE_zlaswp_f2(Quark *quark, Quark_Task_Flags *task_flags,
                          int n, PLASMA_Complex64_t *A, int lda,
                          int i1,  int i2, const int *ipiv, int inc,
                          PLASMA_Complex64_t *fake1, int szefake1, int flag1,
                          PLASMA_Complex64_t *fake2, int szefake2, int flag2);
void QUARK_CORE_zlaswp_ontile(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc descA, PLASMA_Complex64_t *A,
                              int i1,  int i2, const int *ipiv, int inc, PLASMA_Complex64_t *fakepanel);
void QUARK_CORE_zlaswp_ontile_f2(Quark *quark, Quark_Task_Flags *task_flags,
                                 PLASMA_desc descA, PLASMA_Complex64_t *A,
                                 int i1,  int i2, const int *ipiv, int inc,
                                 PLASMA_Complex64_t *fake1, int szefake1, int flag1,
                                 PLASMA_Complex64_t *fake2, int szefake2, int flag2);
void QUARK_CORE_zlaswpc_ontile(Quark *quark, Quark_Task_Flags *task_flags,
                               PLASMA_desc descA, PLASMA_Complex64_t *A,
                               int i1,  int i2, const int *ipiv, int inc, PLASMA_Complex64_t *fakepanel);
void QUARK_CORE_zlatro(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum trans, int m, int n, int mb,
                       const PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *B, int ldb);
void QUARK_CORE_zlatro_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum uplo, PLASMA_enum trans, int m, int n, int mb,
                          const PLASMA_Complex64_t *A, int lda,
                                PLASMA_Complex64_t *B, int ldb,
                          PLASMA_Complex64_t *fake1, int szefake1, int flag1);
void QUARK_CORE_zlauum(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int n, int nb,
                       PLASMA_Complex64_t *A, int lda);
void QUARK_CORE_zplghe(Quark *quark, Quark_Task_Flags *task_flags,
                       double bump, int m, int n, PLASMA_Complex64_t *A, int lda,
                       int bigM, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_zplgsy(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_Complex64_t bump, int m, int n, PLASMA_Complex64_t *A, int lda,
                       int bigM, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_zplrnt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, PLASMA_Complex64_t *A, int lda,
                       int bigM, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_zpltmg(Quark *quark, Quark_Task_Flags *task_flags,
                        PLASMA_enum mtxtype, int m, int n, PLASMA_Complex64_t *A, int lda,
                        int gM, int gN, int m0, int n0, unsigned long long int seed );
void QUARK_CORE_zpltmg_chebvand( Quark *quark, Quark_Task_Flags *task_flags,
                                 int M, int N, PLASMA_Complex64_t *A, int LDA,
                                 int gN, int m0, int n0,
                                 PLASMA_Complex64_t *W );
void QUARK_CORE_zpltmg_circul( Quark *quark, Quark_Task_Flags *task_flags,
                               int M, int N, PLASMA_Complex64_t *A, int LDA,
                               int gM, int m0, int n0,
                               const PLASMA_Complex64_t *W );
void QUARK_CORE_zpltmg_fiedler(Quark *quark, Quark_Task_Flags *task_flags,
                               int m, int n,
                               const PLASMA_Complex64_t *X, int incX,
                               const PLASMA_Complex64_t *Y, int incY,
                               PLASMA_Complex64_t *A, int lda);
void QUARK_CORE_zpltmg_hankel( Quark *quark, Quark_Task_Flags *task_flags,
                               PLASMA_enum uplo, int M, int N, PLASMA_Complex64_t *A, int LDA,
                               int m0, int n0, int nb,
                               const PLASMA_Complex64_t *V1,
                               const PLASMA_Complex64_t *V2);
void QUARK_CORE_zpltmg_toeppd1(Quark *quark, Quark_Task_Flags *task_flags,
                               int gM, int m0, int M,
                               PLASMA_Complex64_t *W,
                               unsigned long long int seed);
void QUARK_CORE_zpltmg_toeppd2(Quark *quark, Quark_Task_Flags *task_flags,
                               int M, int N, int K, int m0, int n0,
                               const PLASMA_Complex64_t *W,
                               PLASMA_Complex64_t *A, int LDA );
void QUARK_CORE_zpotrf(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, int n, int nb,
                       PLASMA_Complex64_t *A, int lda,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       int iinfo);
void QUARK_CORE_zsetvar(Quark *quark, Quark_Task_Flags *task_flags,
                        const PLASMA_Complex64_t *alpha, PLASMA_Complex64_t *x,
                        PLASMA_Complex64_t *Alock);
void QUARK_CORE_zshift( Quark *quark, Quark_Task_Flags *task_flags,
                        int s, int m, int n, int L,
                        PLASMA_Complex64_t *A);
void QUARK_CORE_zshiftw(Quark *quark, Quark_Task_Flags *task_flags,
                        int s, int cl, int m, int n, int L,
                        PLASMA_Complex64_t *A, PLASMA_Complex64_t *W);
void QUARK_CORE_zssssm(Quark *quark, Quark_Task_Flags *task_flags,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       const PLASMA_Complex64_t *L1, int ldl1,
                       const PLASMA_Complex64_t *L2, int ldl2,
                       const int *IPIV);
void QUARK_CORE_zsymm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo,
                      int m, int n, int nb,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                      const PLASMA_Complex64_t *B, int ldb,
                      PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zsyrk(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum uplo, PLASMA_enum trans,
                      int n, int k, int nb,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                      PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zsyr2k(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum trans,
                       int n, int k, int nb,
                       PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                       const PLASMA_Complex64_t *B, int LDB,
                       PLASMA_Complex64_t beta, PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zsyssq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum uplo, int n, const PLASMA_Complex64_t *A, int lda,
                           double *scale, double *sumsq,
                           double *fake, int szeF, int paramF );
void QUARK_CORE_zswpab(Quark *quark, Quark_Task_Flags *task_flags,
                       int i, int n1, int n2,
                       PLASMA_Complex64_t *A, int szeA);
void QUARK_CORE_zswptr_ontile(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_desc descA, PLASMA_Complex64_t *Aij,
                              int i1,  int i2, const int *ipiv, int inc,
                              const PLASMA_Complex64_t *Akk, int ldak);
void QUARK_CORE_ztrasm(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum storev, PLASMA_enum uplo, PLASMA_enum diag, int m, int n,
                       const PLASMA_Complex64_t *A, int lda, int szeA,
                       double *work, int szeW);
void QUARK_CORE_ztrasm_f1(Quark *quark, Quark_Task_Flags *task_flags,
                          PLASMA_enum storev, PLASMA_enum uplo, PLASMA_enum diag, int m, int n,
                          const PLASMA_Complex64_t *A, int lda, int szeA,
                          double *work, int szeW,
                          double *fake, int szeF);
void QUARK_CORE_ztrdalg1(Quark *quark, Quark_Task_Flags *task_flags,
                        int n,
                        int nb,
                        PLASMA_Complex64_t *A,
                        int lda,
                        PLASMA_Complex64_t *V,
                        PLASMA_Complex64_t *TAU,
                        int Vblksiz, int wantz,
                        int i, int sweepid, int m, int grsiz,
                        int *PCOL, int *ACOL, int *MCOL);
void QUARK_CORE_ztrmm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
                      int m, int n, int nb,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                      PLASMA_Complex64_t *B, int ldb);
void QUARK_CORE_ztrmm_p2(Quark *quark, Quark_Task_Flags *task_flags,
                         PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
                         int m, int n, int nb,
                         PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                         PLASMA_Complex64_t **B, int ldb);
void QUARK_CORE_ztrsm(Quark *quark, Quark_Task_Flags *task_flags,
                      PLASMA_enum side, PLASMA_enum uplo, PLASMA_enum transA, PLASMA_enum diag,
                      int m, int n, int nb,
                      PLASMA_Complex64_t alpha, const PLASMA_Complex64_t *A, int lda,
                      PLASMA_Complex64_t *B, int ldb);
void QUARK_CORE_ztrssq_f1( Quark *quark, Quark_Task_Flags *task_flags,
                           PLASMA_enum uplo, PLASMA_enum diag,
                           int m, int n, const PLASMA_Complex64_t *A, int lda,
                           double *scale, double *sumsq,
                           double *fake, int szeF, int paramF );
void QUARK_CORE_ztrtri(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum uplo, PLASMA_enum diag, int n, int nb,
                       PLASMA_Complex64_t *A, int lda,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       int iinfo);
void QUARK_CORE_ztslqt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsmlq(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       const PLASMA_Complex64_t *V, int ldv,
                       const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsmlq_hetra1(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_enum side, PLASMA_enum trans,
                              int m1, int n1, int m2, int n2, int k, int ib, int nb,
                              PLASMA_Complex64_t *A1, int lda1,
                              PLASMA_Complex64_t *A2, int lda2,
                              const PLASMA_Complex64_t *V, int ldv,
                              const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsmlq_corner(Quark *quark, Quark_Task_Flags *task_flags,
                              int m1, int n1, int m2, int n2, int m3, int n3, int k, int ib, int nb,
                              PLASMA_Complex64_t *A1, int lda1,
                              PLASMA_Complex64_t *A2, int lda2,
                              PLASMA_Complex64_t *A3, int lda3,
                              const PLASMA_Complex64_t *V, int ldv,
                              const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsmqr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       const PLASMA_Complex64_t *V, int ldv,
                       const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsmqr_hetra1(Quark *quark, Quark_Task_Flags *task_flags,
                              PLASMA_enum side, PLASMA_enum trans,
                              int m1, int n1, int m2, int n2, int k, int ib, int nb,
                              PLASMA_Complex64_t *A1, int lda1,
                              PLASMA_Complex64_t *A2, int lda2,
                              const PLASMA_Complex64_t *V, int ldv,
                              const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsmqr_corner(Quark *quark, Quark_Task_Flags *task_flags,
                              int m1, int n1, int m2, int n2, int m3, int n3, int k, int ib, int nb,
                              PLASMA_Complex64_t *A1, int lda1,
                              PLASMA_Complex64_t *A2, int lda2,
                              PLASMA_Complex64_t *A3, int lda3,
                              const PLASMA_Complex64_t *V, int ldv,
                              const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztsqrt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_ztstrf(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *U, int ldu,
                       PLASMA_Complex64_t *A, int lda,
                       PLASMA_Complex64_t *L, int ldl,
                       int *IPIV,
                       PLASMA_sequence *sequence, PLASMA_request *request,
                       PLASMA_bool check_info, int iinfo);
void QUARK_CORE_zttmqr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       const PLASMA_Complex64_t *V, int ldv,
                       const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_zttqrt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_zttmlq(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m1, int n1, int m2, int n2, int k, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       const PLASMA_Complex64_t *V, int ldv,
                       const PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_zttlqt(Quark *quark, Quark_Task_Flags *task_flags,
                       int m, int n, int ib, int nb,
                       PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       PLASMA_Complex64_t *T, int ldt);
void QUARK_CORE_zpamm(Quark *quark, Quark_Task_Flags *task_flags,
                       int op, PLASMA_enum side, PLASMA_enum storev,
                       int m, int n, int k, int l,
                       const PLASMA_Complex64_t *A1, int lda1,
                       PLASMA_Complex64_t *A2, int lda2,
                       const PLASMA_Complex64_t *V, int ldv,
                       PLASMA_Complex64_t *W, int ldw);
void QUARK_CORE_zplssq( Quark *quark, Quark_Task_Flags *task_flags,
                        int m, const double *A, double *result );
void QUARK_CORE_zunmlq(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m, int n, int ib,  int nb, int k,
                       const PLASMA_Complex64_t *A, int lda,
                       const PLASMA_Complex64_t *T, int ldt,
                       PLASMA_Complex64_t *C, int ldc);
void QUARK_CORE_zunmqr(Quark *quark, Quark_Task_Flags *task_flags,
                       PLASMA_enum side, PLASMA_enum trans,
                       int m, int n, int k, int ib, int nb,
                       const PLASMA_Complex64_t *A, int lda,
                       const PLASMA_Complex64_t *T, int ldt,
                       PLASMA_Complex64_t *C, int ldc);

/** ****************************************************************************
 *  Declarations of QUARK wrappers (called by QUARK) - alphabetical order
 **/
void CORE_dzasum_quark(Quark *quark);
void CORE_dzasum_f1_quark(Quark *quark);
void CORE_zgeadd_quark(Quark *quark);
void CORE_zbrdalg1_quark(Quark *quark);
void CORE_zgelqt_quark(Quark *quark);
void CORE_zgemm_quark(Quark *quark);
void CORE_zgemm_tile_quark(Quark *quark);
void CORE_zgemv_quark(Quark *quark);
void CORE_zgemv_tile_quark(Quark *quark);
void CORE_zgeqp3_init_quark(Quark *quark);
void CORE_zgeqp3_larfg_quark(Quark *quark);
void CORE_zgeqp3_norms_quark(Quark *quark);
void CORE_zgeqp3_pivot_quark(Quark *quark);
void CORE_zgeqp3_tntpiv_quark(Quark *quark);
void CORE_zgeqp3_update_quark(Quark *quark);
void CORE_zgeqrt_quark(Quark *quark);
void CORE_zgessm_quark(Quark *quark);
void CORE_zgessq_quark(Quark *quark);
void CORE_zgessq_f1_quark(Quark *quark);
void CORE_zgetrf_quark(Quark *quark);
void CORE_zgetrf_incpiv_quark(Quark *quark);
void CORE_zgetrf_nopiv_quark(Quark* quark);
void CORE_zgetrf_reclap_quark(Quark *quark);
void CORE_zgetrf_rectil_quark(Quark* quark);
void CORE_zgetrip_quark(Quark *quark);
void CORE_zgetrip_f1_quark(Quark *quark);
void CORE_zgetrip_f2_quark(Quark *quark);
#ifdef COMPLEX
void CORE_zhemm_quark(Quark *quark);
void CORE_zherk_quark(Quark *quark);
void CORE_zher2k_quark(Quark *quark);
#endif
void CORE_zhegst_quark(Quark *quark);
void CORE_zherfb_quark(Quark *quark);
void CORE_zhessq_quark(Quark *quark);
void CORE_zhessq_f1_quark(Quark *quark);
void CORE_zlacpy_quark(Quark *quark);
void CORE_zlacpy_f1_quark(Quark *quark);
void CORE_zlacpy_pivot_quark(Quark *quark);
void CORE_zlatro_quark(Quark *quark);
void CORE_zlatro_f1_quark(Quark *quark);
void CORE_zlange_quark(Quark *quark);
void CORE_zlange_f1_quark(Quark *quark);
#ifdef COMPLEX
void CORE_zlanhe_quark(Quark *quark);
void CORE_zlanhe_f1_quark(Quark *quark);
#endif
void CORE_zlansy_quark(Quark *quark);
void CORE_zlansy_f1_quark(Quark *quark);
void CORE_zlaset_quark(Quark *quark);
void CORE_zlaset2_quark(Quark *quark);
void CORE_zlatro_quark(Quark *quark);
void CORE_zlauum_quark(Quark *quark);
void CORE_zpamm_quark(Quark *quark);
void CORE_zplghe_quark(Quark *quark);
void CORE_zplgsy_quark(Quark *quark);
void CORE_zplrnt_quark(Quark *quark);
void CORE_zpltmg_quark(Quark *quark);
void CORE_zplssq_quark(Quark *quark);
void CORE_zpotrf_quark(Quark *quark);
void CORE_zsetvar_quark(Quark *quark);
void CORE_zshift_quark(Quark *quark);
void CORE_zshiftw_quark(Quark *quark);
void CORE_zssssm_quark(Quark *quark);
void CORE_zsymm_quark(Quark *quark);
void CORE_zsyrk_quark(Quark *quark);
void CORE_zsyr2k_quark(Quark *quark);
void CORE_zsyssq_quark(Quark *quark);
void CORE_zsyssq_f1_quark(Quark *quark);
void CORE_zswpab_quark(Quark *quark);
void CORE_zswptr_ontile_quark(Quark *quark);
void CORE_ztrdalg1_quark(Quark *quark);
void CORE_ztrmm_quark(Quark *quark);
void CORE_ztrsm_quark(Quark *quark);
void CORE_ztrtri_quark(Quark *quark);
void CORE_ztslqt_quark(Quark *quark);
void CORE_ztsmlq_quark(Quark *quark);
void CORE_ztsmlq_hetra1_quark(Quark *quark);
void CORE_ztsmlq_corner_quark(Quark *quark);
void CORE_ztsmqr_quark(Quark *quark);
void CORE_ztsmqr_hetra1_quark(Quark *quark);
void CORE_ztsmqr_corner_quark(Quark *quark);
void CORE_ztsqrt_quark(Quark *quark);
void CORE_ztstrf_quark(Quark *quark);
void CORE_zttmqr_quark(Quark *quark);
void CORE_zttqrt_quark(Quark *quark);
void CORE_zttmlq_quark(Quark *quark);
void CORE_zttlqt_quark(Quark *quark);
void CORE_zunmlq_quark(Quark *quark);
void CORE_zunmqr_quark(Quark *quark);
void CORE_zlaswp_quark(Quark* quark);
void CORE_zlaswp_f2_quark(Quark* quark);
void CORE_zlaswp_ontile_quark(Quark *quark);
void CORE_zlaswp_ontile_f2_quark(Quark *quark);
void CORE_zlaswpc_ontile_quark(Quark *quark);
void CORE_ztrmm_p2_quark(Quark* quark);
void CORE_zgemm_f2_quark(Quark* quark);
void CORE_zgemm_p2_quark(Quark* quark);
void CORE_zgemm_p2f1_quark(Quark* quark);
void CORE_zgemm_p3_quark(Quark* quark);

#endif /* defined(QUARK_H) */

#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif
