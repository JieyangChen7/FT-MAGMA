#ifndef __EMP_CULA_LAPACK_H__
#define __EMP_CULA_LAPACK_H__

/*
 * Copyright (C) 2009-2012 EM Photonics, Inc.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to EM Photonics ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code may
 * not redistribute this code without the express written consent of EM
 * Photonics, Inc.
 *
 * EM PHOTONICS MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  EM PHOTONICS DISCLAIMS ALL WARRANTIES WITH REGARD TO
 * THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL EM
 * PHOTONICS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as that
 * term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of "commercial
 * computer  software"  and "commercial computer software documentation" as
 * such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the
 * U.S. Government only as a commercial end item.  Consistent with 48
 * C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the source code with only those rights set
 * forth herein. 
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code, the
 * above Disclaimer and U.S. Government End Users Notice.
 *
 */

#include "cula_status.h"
#include "cula_types.h"

#ifdef __cplusplus
extern "C" {
#endif

culaStatus culaCbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaFloat* d, culaFloat* e, culaFloatComplex* vt, int ldvt, culaFloatComplex* u, int ldu, culaFloatComplex* c, int ldc);
culaStatus culaCgbtrf(int m, int n, int kl, int ku, culaFloatComplex* a, int lda, culaInt* ipiv);
culaStatus culaCgeConjugate(int m, int n, culaFloatComplex* a, int lda);
culaStatus culaCgeNancheck(int m, int n, culaFloatComplex* a, int lda);
culaStatus culaCgeTranspose(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaCgeTransposeConjugate(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaCgeTransposeConjugateInplace(int n, culaFloatComplex* a, int lda);
culaStatus culaCgeTransposeInplace(int n, culaFloatComplex* a, int lda);
culaStatus culaCgebrd(int m, int n, culaFloatComplex* a, int lda, culaFloat* d, culaFloat* e, culaFloatComplex* tauq, culaFloatComplex* taup);
culaStatus culaCgeev(char jobvl, char jobvr, int n, culaFloatComplex* a, int lda, culaFloatComplex* w, culaFloatComplex* vl, int ldvl, culaFloatComplex* vr, int ldvr);
culaStatus culaCgehrd(int n, int ilo, int ihi, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCgelqf(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCgels(char trans, int m, int n, int nrhs, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaCgeqlf(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCgeqrf(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCgeqrfp(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCgeqrs(int m, int n, int nrhs, culaFloatComplex* a, int lda, culaFloatComplex* tau, culaFloatComplex* b, int ldb);
culaStatus culaCgerqf(int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCgesdd(char jobz, int m, int n, culaFloatComplex* a, int lda, culaFloat* s, culaFloatComplex* u, int ldu, culaFloatComplex* vt, int ldvt);
culaStatus culaCgesv(int n, int nrhs, culaFloatComplex* a, int lda, culaInt* ipiv, culaFloatComplex* b, int ldb);
culaStatus culaCgesvd(char jobu, char jobvt, int m, int n, culaFloatComplex* a, int lda, culaFloat* s, culaFloatComplex* u, int ldu, culaFloatComplex* vt, int ldvt);
culaStatus culaCgetrf(int m, int n, culaFloatComplex* a, int lda, culaInt* ipiv);
culaStatus culaCgetri(int n, culaFloatComplex* a, int lda, culaInt* ipiv);
culaStatus culaCgetrs(char trans, int n, int nrhs, culaFloatComplex* a, int lda, culaInt* ipiv, culaFloatComplex* b, int ldb);
culaStatus culaCgglse(int m, int n, int p, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb, culaFloatComplex* c, culaFloatComplex* d, culaFloatComplex* x);
culaStatus culaCggrqf(int m, int p, int n, culaFloatComplex* a, int lda, culaFloatComplex* taua, culaFloatComplex* b, int ldb, culaFloatComplex* taub);
culaStatus culaCheev(char jobz, char uplo, int n, culaFloatComplex* a, int lda, culaFloat* w);
culaStatus culaCheevx(char jobz, char range, char uplo, int n, culaFloatComplex* a, int lda, culaFloat vl, culaFloat vu, int il, int iu, culaFloat abstol, culaInt* m, culaFloat* w, culaFloatComplex* z, int ldz, culaInt* ifail);
culaStatus culaChegv(int itype, char jobz, char uplo, int n, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb, culaFloat* w);
culaStatus culaCherdb(char jobz, char uplo, int n, int kd, culaFloatComplex* a, int lda, culaFloat* d, culaFloat* e, culaFloatComplex* tau, culaFloatComplex* z, int ldz);
culaStatus culaClacpy(char uplo, int m, int n, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaClag2z(int m, int n, culaFloatComplex* a, int lda, culaDoubleComplex* sa, int ldsa);
culaStatus culaClar2v(int n, culaFloatComplex* x, culaFloatComplex* y, culaFloatComplex* z, int incx, culaFloat* c, culaFloatComplex* s, int incc);
culaStatus culaClarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaFloatComplex* v, int ldv, culaFloatComplex* t, int ldt, culaFloatComplex* c, int ldc);
culaStatus culaClarfg(int n, culaFloatComplex* alpha, culaFloatComplex* x, int incx, culaFloatComplex* tau);
culaStatus culaClargv(int n, culaFloatComplex* x, int incx, culaFloatComplex* y, int incy, culaFloat* c, int incc);
culaStatus culaClartv(int n, culaFloatComplex* x, int incx, culaFloatComplex* y, int incy, culaFloat* c, culaFloatComplex* s, int incc);
culaStatus culaClascl(char type, int kl, int ku, culaFloat cfrom, culaFloat cto, int m, int n, culaFloatComplex* a, int lda);
culaStatus culaClaset(char uplo, int m, int n, culaFloatComplex alpha, culaFloatComplex beta, culaFloatComplex* a, int lda);
culaStatus culaClasr(char side, char pivot, char direct, int m, int n, culaFloat* c, culaFloat* s, culaFloatComplex* a, int lda);
culaStatus culaClat2z(char uplo, int n, culaFloatComplex* a, int lda, culaDoubleComplex* sa, int ldsa);
culaStatus culaCpbtrf(char uplo, int n, int kd, culaFloatComplex* ab, int ldab);
culaStatus culaCposv(char uplo, int n, int nrhs, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaCpotrf(char uplo, int n, culaFloatComplex* a, int lda);
culaStatus culaCpotri(char uplo, int n, culaFloatComplex* a, int lda);
culaStatus culaCpotrs(char uplo, int n, int nrhs, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaCsteqr(char compz, int n, culaFloat* d, culaFloat* e, culaFloatComplex* z, int ldz);
culaStatus culaCtrConjugate(char uplo, char diag, int m, int n, culaFloatComplex* a, int lda);
culaStatus culaCtrtri(char uplo, char diag, int n, culaFloatComplex* a, int lda);
culaStatus culaCtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus culaCungbr(char vect, int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCunghr(int n, int ilo, int ihi, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCunglq(int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCungql(int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCungqr(int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCungrq(int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau);
culaStatus culaCunmlq(char side, char trans, int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau, culaFloatComplex* c, int ldc);
culaStatus culaCunmql(char side, char trans, int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau, culaFloatComplex* c, int ldc);
culaStatus culaCunmqr(char side, char trans, int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau, culaFloatComplex* c, int ldc);
culaStatus culaCunmrq(char side, char trans, int m, int n, int k, culaFloatComplex* a, int lda, culaFloatComplex* tau, culaFloatComplex* c, int ldc);
culaStatus culaDbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDouble* d, culaDouble* e, culaDouble* vt, int ldvt, culaDouble* u, int ldu, culaDouble* c, int ldc);
culaStatus culaDgbtrf(int m, int n, int kl, int ku, culaDouble* a, int lda, culaInt* ipiv);
culaStatus culaDgeNancheck(int m, int n, culaDouble* a, int lda);
culaStatus culaDgeTranspose(int m, int n, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus culaDgeTransposeInplace(int n, culaDouble* a, int lda);
culaStatus culaDgebrd(int m, int n, culaDouble* a, int lda, culaDouble* d, culaDouble* e, culaDouble* tauq, culaDouble* taup);
culaStatus culaDgeev(char jobvl, char jobvr, int n, culaDouble* a, int lda, culaDouble* wr, culaDouble* wi, culaDouble* vl, int ldvl, culaDouble* vr, int ldvr);
culaStatus culaDgehrd(int n, int ilo, int ihi, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDgelqf(int m, int n, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDgels(char trans, int m, int n, int nrhs, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus culaDgeqlf(int m, int n, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDgeqrf(int m, int n, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDgeqrfp(int m, int n, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDgeqrs(int m, int n, int nrhs, culaDouble* a, int lda, culaDouble* tau, culaDouble* b, int ldb);
culaStatus culaDgerqf(int m, int n, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDgesdd(char jobz, int m, int n, culaDouble* a, int lda, culaDouble* s, culaDouble* u, int ldu, culaDouble* vt, int ldvt);
culaStatus culaDgesv(int n, int nrhs, culaDouble* a, int lda, culaInt* ipiv, culaDouble* b, int ldb);
culaStatus culaDgesvd(char jobu, char jobvt, int m, int n, culaDouble* a, int lda, culaDouble* s, culaDouble* u, int ldu, culaDouble* vt, int ldvt);
culaStatus culaDgetrf(int m, int n, culaDouble* a, int lda, culaInt* ipiv);
culaStatus culaDgetri(int n, culaDouble* a, int lda, culaInt* ipiv);
culaStatus culaDgetrs(char trans, int n, int nrhs, culaDouble* a, int lda, culaInt* ipiv, culaDouble* b, int ldb);
culaStatus culaDgglse(int m, int n, int p, culaDouble* a, int lda, culaDouble* b, int ldb, culaDouble* c, culaDouble* d, culaDouble* x);
culaStatus culaDggrqf(int m, int p, int n, culaDouble* a, int lda, culaDouble* taua, culaDouble* b, int ldb, culaDouble* taub);
culaStatus culaDlacpy(char uplo, int m, int n, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus culaDlag2s(int m, int n, culaDouble* a, int lda, culaFloat* sa, int ldsa);
culaStatus culaDlar2v(int n, culaDouble* x, culaDouble* y, culaDouble* z, int incx, culaDouble* c, culaDouble* s, int incc);
culaStatus culaDlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDouble* v, int ldv, culaDouble* t, int ldt, culaDouble* c, int ldc);
culaStatus culaDlarfg(int n, culaDouble* alpha, culaDouble* x, int incx, culaDouble* tau);
culaStatus culaDlargv(int n, culaDouble* x, int incx, culaDouble* y, int incy, culaDouble* c, int incc);
culaStatus culaDlartv(int n, culaDouble* x, int incx, culaDouble* y, int incy, culaDouble* c, culaDouble* s, int incc);
culaStatus culaDlascl(char type, int kl, int ku, culaDouble cfrom, culaDouble cto, int m, int n, culaDouble* a, int lda);
culaStatus culaDlaset(char uplo, int m, int n, culaDouble alpha, culaDouble beta, culaDouble* a, int lda);
culaStatus culaDlasr(char side, char pivot, char direct, int m, int n, culaDouble* c, culaDouble* s, culaDouble* a, int lda);
culaStatus culaDlat2s(char uplo, int n, culaDouble* a, int lda, culaFloat* sa, int ldsa);
culaStatus culaDorgbr(char vect, int m, int n, int k, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDorghr(int n, int ilo, int ihi, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDorglq(int m, int n, int k, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDorgql(int m, int n, int k, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDorgqr(int m, int n, int k, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDorgrq(int m, int n, int k, culaDouble* a, int lda, culaDouble* tau);
culaStatus culaDormlq(char side, char trans, int m, int n, int k, culaDouble* a, int lda, culaDouble* tau, culaDouble* c, int ldc);
culaStatus culaDormql(char side, char trans, int m, int n, int k, culaDouble* a, int lda, culaDouble* tau, culaDouble* c, int ldc);
culaStatus culaDormqr(char side, char trans, int m, int n, int k, culaDouble* a, int lda, culaDouble* tau, culaDouble* c, int ldc);
culaStatus culaDormrq(char side, char trans, int m, int n, int k, culaDouble* a, int lda, culaDouble* tau, culaDouble* c, int ldc);
culaStatus culaDpbtrf(char uplo, int n, int kd, culaDouble* ab, int ldab);
culaStatus culaDposv(char uplo, int n, int nrhs, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus culaDpotrf(char uplo, int n, culaDouble* a, int lda);
culaStatus culaDpotri(char uplo, int n, culaDouble* a, int lda);
culaStatus culaDpotrs(char uplo, int n, int nrhs, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus culaDsgesv(int n, int nrhs, culaDouble* a, int lda, culaInt* ipiv, culaDouble* b, int ldb, culaDouble* x, int ldx, int* iter);
culaStatus culaDsposv(char uplo, int n, int nrhs, culaDouble* a, int lda, culaDouble* b, int ldb, culaDouble* x, int ldx, int* iter);
culaStatus culaDstebz(char range, char order, int n, double vl, double vu, int il, int iu, double abstol, culaDouble* d, culaDouble* e, int* m, int* nsplit, culaDouble* w, culaInt* isplit, culaInt* iblock);
culaStatus culaDsteqr(char compz, int n, culaDouble* d, culaDouble* e, culaDouble* z, int ldz);
culaStatus culaDsyev(char jobz, char uplo, int n, culaDouble* a, int lda, culaDouble* w);
culaStatus culaDsyevx(char jobz, char range, char uplo, int n, culaDouble* a, int lda, culaDouble vl, culaDouble vu, int il, int iu, culaDouble abstol, culaInt* m, culaDouble* w, culaDouble* z, int ldz, culaInt* ifail);
culaStatus culaDsygv(int itype, char jobz, char uplo, int n, culaDouble* a, int lda, culaDouble* b, int ldb, culaDouble* w);
culaStatus culaDsyrdb(char jobz, char uplo, int n, int kd, culaDouble* a, int lda, culaDouble* d, culaDouble* e, culaDouble* tau, culaDouble* z, int ldz);
culaStatus culaDtrtri(char uplo, char diag, int n, culaDouble* a, int lda);
culaStatus culaDtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus culaSbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaFloat* d, culaFloat* e, culaFloat* vt, int ldvt, culaFloat* u, int ldu, culaFloat* c, int ldc);
culaStatus culaSgbtrf(int m, int n, int kl, int ku, culaFloat* a, int lda, culaInt* ipiv);
culaStatus culaSgeNancheck(int m, int n, culaFloat* a, int lda);
culaStatus culaSgeTranspose(int m, int n, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus culaSgeTransposeInplace(int n, culaFloat* a, int lda);
culaStatus culaSgebrd(int m, int n, culaFloat* a, int lda, culaFloat* d, culaFloat* e, culaFloat* tauq, culaFloat* taup);
culaStatus culaSgeev(char jobvl, char jobvr, int n, culaFloat* a, int lda, culaFloat* wr, culaFloat* wi, culaFloat* vl, int ldvl, culaFloat* vr, int ldvr);
culaStatus culaSgehrd(int n, int ilo, int ihi, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSgelqf(int m, int n, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSgels(char trans, int m, int n, int nrhs, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus culaSgeqlf(int m, int n, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSgeqrf(int m, int n, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSgeqrfp(int m, int n, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSgeqrs(int m, int n, int nrhs, culaFloat* a, int lda, culaFloat* tau, culaFloat* b, int ldb);
culaStatus culaSgerqf(int m, int n, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSgesdd(char jobz, int m, int n, culaFloat* a, int lda, culaFloat* s, culaFloat* u, int ldu, culaFloat* vt, int ldvt);
culaStatus culaSgesv(int n, int nrhs, culaFloat* a, int lda, culaInt* ipiv, culaFloat* b, int ldb);
culaStatus culaSgesvd(char jobu, char jobvt, int m, int n, culaFloat* a, int lda, culaFloat* s, culaFloat* u, int ldu, culaFloat* vt, int ldvt);
culaStatus culaSgetrf(int m, int n, culaFloat* a, int lda, culaInt* ipiv);
culaStatus culaSgetri(int n, culaFloat* a, int lda, culaInt* ipiv);
culaStatus culaSgetrs(char trans, int n, int nrhs, culaFloat* a, int lda, culaInt* ipiv, culaFloat* b, int ldb);
culaStatus culaSgglse(int m, int n, int p, culaFloat* a, int lda, culaFloat* b, int ldb, culaFloat* c, culaFloat* d, culaFloat* x);
culaStatus culaSggrqf(int m, int p, int n, culaFloat* a, int lda, culaFloat* taua, culaFloat* b, int ldb, culaFloat* taub);
culaStatus culaSlacpy(char uplo, int m, int n, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus culaSlag2d(int m, int n, culaFloat* a, int lda, culaDouble* sa, int ldsa);
culaStatus culaSlar2v(int n, culaFloat* x, culaFloat* y, culaFloat* z, int incx, culaFloat* c, culaFloat* s, int incc);
culaStatus culaSlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaFloat* v, int ldv, culaFloat* t, int ldt, culaFloat* c, int ldc);
culaStatus culaSlarfg(int n, culaFloat* alpha, culaFloat* x, int incx, culaFloat* tau);
culaStatus culaSlargv(int n, culaFloat* x, int incx, culaFloat* y, int incy, culaFloat* c, int incc);
culaStatus culaSlartv(int n, culaFloat* x, int incx, culaFloat* y, int incy, culaFloat* c, culaFloat* s, int incc);
culaStatus culaSlascl(char type, int kl, int ku, culaFloat cfrom, culaFloat cto, int m, int n, culaFloat* a, int lda);
culaStatus culaSlaset(char uplo, int m, int n, culaFloat alpha, culaFloat beta, culaFloat* a, int lda);
culaStatus culaSlasr(char side, char pivot, char direct, int m, int n, culaFloat* c, culaFloat* s, culaFloat* a, int lda);
culaStatus culaSlat2d(char uplo, int n, culaFloat* a, int lda, culaDouble* sa, int ldsa);
culaStatus culaSorgbr(char vect, int m, int n, int k, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSorghr(int n, int ilo, int ihi, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSorglq(int m, int n, int k, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSorgql(int m, int n, int k, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSorgqr(int m, int n, int k, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSorgrq(int m, int n, int k, culaFloat* a, int lda, culaFloat* tau);
culaStatus culaSormlq(char side, char trans, int m, int n, int k, culaFloat* a, int lda, culaFloat* tau, culaFloat* c, int ldc);
culaStatus culaSormql(char side, char trans, int m, int n, int k, culaFloat* a, int lda, culaFloat* tau, culaFloat* c, int ldc);
culaStatus culaSormqr(char side, char trans, int m, int n, int k, culaFloat* a, int lda, culaFloat* tau, culaFloat* c, int ldc);
culaStatus culaSormrq(char side, char trans, int m, int n, int k, culaFloat* a, int lda, culaFloat* tau, culaFloat* c, int ldc);
culaStatus culaSpbtrf(char uplo, int n, int kd, culaFloat* ab, int ldab);
culaStatus culaSposv(char uplo, int n, int nrhs, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus culaSpotrf(char uplo, int n, culaFloat* a, int lda);
culaStatus culaSpotri(char uplo, int n, culaFloat* a, int lda);
culaStatus culaSpotrs(char uplo, int n, int nrhs, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus culaSstebz(char range, char order, int n, float vl, float vu, int il, int iu, float abstol, culaFloat* d, culaFloat* e, int* m, int* nsplit, culaFloat* w, culaInt* isplit, culaInt* iblock);
culaStatus culaSsteqr(char compz, int n, culaFloat* d, culaFloat* e, culaFloat* z, int ldz);
culaStatus culaSsyev(char jobz, char uplo, int n, culaFloat* a, int lda, culaFloat* w);
culaStatus culaSsyevx(char jobz, char range, char uplo, int n, culaFloat* a, int lda, culaFloat vl, culaFloat vu, int il, int iu, culaFloat abstol, culaInt* m, culaFloat* w, culaFloat* z, int ldz, culaInt* ifail);
culaStatus culaSsygv(int itype, char jobz, char uplo, int n, culaFloat* a, int lda, culaFloat* b, int ldb, culaFloat* w);
culaStatus culaSsyrdb(char jobz, char uplo, int n, int kd, culaFloat* a, int lda, culaFloat* d, culaFloat* e, culaFloat* tau, culaFloat* z, int ldz);
culaStatus culaStrtri(char uplo, char diag, int n, culaFloat* a, int lda);
culaStatus culaStrtrs(char uplo, char trans, char diag, int n, int nrhs, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus culaZbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDouble* d, culaDouble* e, culaDoubleComplex* vt, int ldvt, culaDoubleComplex* u, int ldu, culaDoubleComplex* c, int ldc);
culaStatus culaZcgesv(int n, int nrhs, culaDoubleComplex* a, int lda, culaInt* ipiv, culaDoubleComplex* b, int ldb, culaDoubleComplex* x, int ldx, int* iter);
culaStatus culaZcposv(char uplo, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb, culaDoubleComplex* x, int ldx, int* iter);
culaStatus culaZgbtrf(int m, int n, int kl, int ku, culaDoubleComplex* a, int lda, culaInt* ipiv);
culaStatus culaZgeConjugate(int m, int n, culaDoubleComplex* a, int lda);
culaStatus culaZgeNancheck(int m, int n, culaDoubleComplex* a, int lda);
culaStatus culaZgeTranspose(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZgeTransposeConjugate(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZgeTransposeConjugateInplace(int n, culaDoubleComplex* a, int lda);
culaStatus culaZgeTransposeInplace(int n, culaDoubleComplex* a, int lda);
culaStatus culaZgebrd(int m, int n, culaDoubleComplex* a, int lda, culaDouble* d, culaDouble* e, culaDoubleComplex* tauq, culaDoubleComplex* taup);
culaStatus culaZgeev(char jobvl, char jobvr, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* w, culaDoubleComplex* vl, int ldvl, culaDoubleComplex* vr, int ldvr);
culaStatus culaZgehrd(int n, int ilo, int ihi, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZgelqf(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZgels(char trans, int m, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZgeqlf(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZgeqrf(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZgeqrfp(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZgeqrs(int m, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* tau, culaDoubleComplex* b, int ldb);
culaStatus culaZgerqf(int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZgesdd(char jobz, int m, int n, culaDoubleComplex* a, int lda, culaDouble* s, culaDoubleComplex* u, int ldu, culaDoubleComplex* vt, int ldvt);
culaStatus culaZgesv(int n, int nrhs, culaDoubleComplex* a, int lda, culaInt* ipiv, culaDoubleComplex* b, int ldb);
culaStatus culaZgesvd(char jobu, char jobvt, int m, int n, culaDoubleComplex* a, int lda, culaDouble* s, culaDoubleComplex* u, int ldu, culaDoubleComplex* vt, int ldvt);
culaStatus culaZgetrf(int m, int n, culaDoubleComplex* a, int lda, culaInt* ipiv);
culaStatus culaZgetri(int n, culaDoubleComplex* a, int lda, culaInt* ipiv);
culaStatus culaZgetrs(char trans, int n, int nrhs, culaDoubleComplex* a, int lda, culaInt* ipiv, culaDoubleComplex* b, int ldb);
culaStatus culaZgglse(int m, int n, int p, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb, culaDoubleComplex* c, culaDoubleComplex* d, culaDoubleComplex* x);
culaStatus culaZggrqf(int m, int p, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* taua, culaDoubleComplex* b, int ldb, culaDoubleComplex* taub);
culaStatus culaZheev(char jobz, char uplo, int n, culaDoubleComplex* a, int lda, culaDouble* w);
culaStatus culaZheevx(char jobz, char range, char uplo, int n, culaDoubleComplex* a, int lda, culaDouble vl, culaDouble vu, int il, int iu, culaDouble abstol, culaInt* m, culaDouble* w, culaDoubleComplex* z, int ldz, culaInt* ifail);
culaStatus culaZhegv(int itype, char jobz, char uplo, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb, culaDouble* w);
culaStatus culaZherdb(char jobz, char uplo, int n, int kd, culaDoubleComplex* a, int lda, culaDouble* d, culaDouble* e, culaDoubleComplex* tau, culaDoubleComplex* z, int ldz);
culaStatus culaZlacpy(char uplo, int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZlag2c(int m, int n, culaDoubleComplex* a, int lda, culaFloatComplex* sa, int ldsa);
culaStatus culaZlar2v(int n, culaDoubleComplex* x, culaDoubleComplex* y, culaDoubleComplex* z, int incx, culaDouble* c, culaDoubleComplex* s, int incc);
culaStatus culaZlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDoubleComplex* v, int ldv, culaDoubleComplex* t, int ldt, culaDoubleComplex* c, int ldc);
culaStatus culaZlarfg(int n, culaDoubleComplex* alpha, culaDoubleComplex* x, int incx, culaDoubleComplex* tau);
culaStatus culaZlargv(int n, culaDoubleComplex* x, int incx, culaDoubleComplex* y, int incy, culaDouble* c, int incc);
culaStatus culaZlartv(int n, culaDoubleComplex* x, int incx, culaDoubleComplex* y, int incy, culaDouble* c, culaDoubleComplex* s, int incc);
culaStatus culaZlascl(char type, int kl, int ku, culaDouble cfrom, culaDouble cto, int m, int n, culaDoubleComplex* a, int lda);
culaStatus culaZlaset(char uplo, int m, int n, culaDoubleComplex alpha, culaDoubleComplex beta, culaDoubleComplex* a, int lda);
culaStatus culaZlasr(char side, char pivot, char direct, int m, int n, culaDouble* c, culaDouble* s, culaDoubleComplex* a, int lda);
culaStatus culaZlat2c(char uplo, int n, culaDoubleComplex* a, int lda, culaFloatComplex* sa, int ldsa);
culaStatus culaZpbtrf(char uplo, int n, int kd, culaDoubleComplex* ab, int ldab);
culaStatus culaZposv(char uplo, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZpotrf(char uplo, int n, culaDoubleComplex* a, int lda);
culaStatus culaZpotri(char uplo, int n, culaDoubleComplex* a, int lda);
culaStatus culaZpotrs(char uplo, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZsteqr(char compz, int n, culaDouble* d, culaDouble* e, culaDoubleComplex* z, int ldz);
culaStatus culaZtrConjugate(char uplo, char diag, int m, int n, culaDoubleComplex* a, int lda);
culaStatus culaZtrtri(char uplo, char diag, int n, culaDoubleComplex* a, int lda);
culaStatus culaZtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);
culaStatus culaZungbr(char vect, int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZunghr(int n, int ilo, int ihi, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZunglq(int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZungql(int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZungqr(int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZungrq(int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);
culaStatus culaZunmlq(char side, char trans, int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau, culaDoubleComplex* c, int ldc);
culaStatus culaZunmql(char side, char trans, int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau, culaDoubleComplex* c, int ldc);
culaStatus culaZunmqr(char side, char trans, int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau, culaDoubleComplex* c, int ldc);
culaStatus culaZunmrq(char side, char trans, int m, int n, int k, culaDoubleComplex* a, int lda, culaDoubleComplex* tau, culaDoubleComplex* c, int ldc);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // __EMP_CULA_LAPACK_H__

