#ifndef __EMP_CULA_LAPACK_DEVICE_H__
#define __EMP_CULA_LAPACK_DEVICE_H__

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
#include "cula_device.h"

#ifdef __cplusplus
extern "C" {
#endif

culaStatus culaDeviceCbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* vt, int ldvt, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCgbtrf(int m, int n, int kl, int ku, culaDeviceFloatComplex* a, int lda, culaInt* ipiv);
culaStatus culaDeviceCgeConjugate(int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgeNancheck(int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgeTranspose(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeTransposeConjugate(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeTransposeConjugateInplace(int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgeTransposeInplace(int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCgebrd(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* tauq, culaDeviceFloatComplex* taup);
culaStatus culaDeviceCgeev(char jobvl, char jobvr, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* w, culaDeviceFloatComplex* vl, int ldvl, culaDeviceFloatComplex* vr, int ldvr);
culaStatus culaDeviceCgehrd(int n, int ilo, int ihi, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgelqf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgels(char trans, int m, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgeqlf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgeqrf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgeqrfp(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgeqrs(int m, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgerqf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCgesdd(char jobz, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* s, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* vt, int ldvt);
culaStatus culaDeviceCgesv(int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgesvd(char jobu, char jobvt, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* s, culaDeviceFloatComplex* u, int ldu, culaDeviceFloatComplex* vt, int ldvt);
culaStatus culaDeviceCgetrf(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceCgetri(int n, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceCgetrs(char trans, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCgglse(int m, int n, int p, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb, culaDeviceFloatComplex* c, culaDeviceFloatComplex* d, culaDeviceFloatComplex* x);
culaStatus culaDeviceCggrqf(int m, int p, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* taua, culaDeviceFloatComplex* b, int ldb, culaDeviceFloatComplex* taub);
culaStatus culaDeviceCheev(char jobz, char uplo, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* w);
culaStatus culaDeviceCheevx(char jobz, char range, char uplo, int n, culaDeviceFloatComplex* a, int lda, culaFloat vl, culaFloat vu, int il, int iu, culaFloat abstol, culaInt* m, culaDeviceFloat* w, culaDeviceFloatComplex* z, int ldz, culaInt* ifail);
culaStatus culaDeviceChegv(int itype, char jobz, char uplo, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb, culaDeviceFloat* w);
culaStatus culaDeviceCherdb(char jobz, char uplo, int n, int kd, culaDeviceFloatComplex* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* z, int ldz);
culaStatus culaDeviceClacpy(char uplo, int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceClag2z(int m, int n, culaDeviceFloatComplex* a, int lda, culaDeviceDoubleComplex* sa, int ldsa);
culaStatus culaDeviceClar2v(int n, culaDeviceFloatComplex* x, culaDeviceFloatComplex* y, culaDeviceFloatComplex* z, int incx, culaDeviceFloat* c, culaDeviceFloatComplex* s, int incc);
culaStatus culaDeviceClarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceFloatComplex* v, int ldv, culaDeviceFloatComplex* t, int ldt, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceClarfg(int n, culaDeviceFloatComplex* alpha, culaDeviceFloatComplex* x, int incx, culaDeviceFloatComplex* tau);
culaStatus culaDeviceClargv(int n, culaDeviceFloatComplex* x, int incx, culaDeviceFloatComplex* y, int incy, culaDeviceFloat* c, int incc);
culaStatus culaDeviceClartv(int n, culaDeviceFloatComplex* x, int incx, culaDeviceFloatComplex* y, int incy, culaDeviceFloat* c, culaDeviceFloatComplex* s, int incc);
culaStatus culaDeviceClascl(char type, int kl, int ku, culaFloat cfrom, culaFloat cto, int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceClaset(char uplo, int m, int n, culaFloatComplex alpha, culaFloatComplex beta, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceClasr(char side, char pivot, char direct, int m, int n, culaDeviceFloat* c, culaDeviceFloat* s, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceClat2z(char uplo, int n, culaDeviceFloatComplex* a, int lda, culaDeviceDoubleComplex* sa, int ldsa);
culaStatus culaDeviceCpbtrf(char uplo, int n, int kd, culaDeviceFloatComplex* ab, int ldab);
culaStatus culaDeviceCposv(char uplo, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCpotrf(char uplo, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCpotri(char uplo, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCpotrs(char uplo, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCsteqr(char compz, int n, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloatComplex* z, int ldz);
culaStatus culaDeviceCtrConjugate(char uplo, char diag, int m, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCtrtri(char uplo, char diag, int n, culaDeviceFloatComplex* a, int lda);
culaStatus culaDeviceCtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* b, int ldb);
culaStatus culaDeviceCungbr(char vect, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCunghr(int n, int ilo, int ihi, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCunglq(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCungql(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCungqr(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCungrq(int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau);
culaStatus culaDeviceCunmlq(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCunmql(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCunmqr(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceCunmrq(char side, char trans, int m, int n, int k, culaDeviceFloatComplex* a, int lda, culaDeviceFloatComplex* tau, culaDeviceFloatComplex* c, int ldc);
culaStatus culaDeviceDbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* vt, int ldvt, culaDeviceDouble* u, int ldu, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDgbtrf(int m, int n, int kl, int ku, culaDeviceDouble* a, int lda, culaInt* ipiv);
culaStatus culaDeviceDgeNancheck(int m, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDgeTranspose(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgeTransposeInplace(int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDgebrd(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* tauq, culaDeviceDouble* taup);
culaStatus culaDeviceDgeev(char jobvl, char jobvr, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* wr, culaDeviceDouble* wi, culaDeviceDouble* vl, int ldvl, culaDeviceDouble* vr, int ldvr);
culaStatus culaDeviceDgehrd(int n, int ilo, int ihi, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgelqf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgels(char trans, int m, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgeqlf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgeqrf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgeqrfp(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgeqrs(int m, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgerqf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDgesdd(char jobz, int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* s, culaDeviceDouble* u, int ldu, culaDeviceDouble* vt, int ldvt);
culaStatus culaDeviceDgesv(int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgesvd(char jobu, char jobvt, int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* s, culaDeviceDouble* u, int ldu, culaDeviceDouble* vt, int ldvt);
culaStatus culaDeviceDgetrf(int m, int n, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceDgetri(int n, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceDgetrs(char trans, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceInt* ipiv, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDgglse(int m, int n, int p, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb, culaDeviceDouble* c, culaDeviceDouble* d, culaDeviceDouble* x);
culaStatus culaDeviceDggrqf(int m, int p, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* taua, culaDeviceDouble* b, int ldb, culaDeviceDouble* taub);
culaStatus culaDeviceDlacpy(char uplo, int m, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDlag2s(int m, int n, culaDeviceDouble* a, int lda, culaDeviceFloat* sa, int ldsa);
culaStatus culaDeviceDlar2v(int n, culaDeviceDouble* x, culaDeviceDouble* y, culaDeviceDouble* z, int incx, culaDeviceDouble* c, culaDeviceDouble* s, int incc);
culaStatus culaDeviceDlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceDouble* v, int ldv, culaDeviceDouble* t, int ldt, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDlarfg(int n, culaDeviceDouble* alpha, culaDeviceDouble* x, int incx, culaDeviceDouble* tau);
culaStatus culaDeviceDlargv(int n, culaDeviceDouble* x, int incx, culaDeviceDouble* y, int incy, culaDeviceDouble* c, int incc);
culaStatus culaDeviceDlartv(int n, culaDeviceDouble* x, int incx, culaDeviceDouble* y, int incy, culaDeviceDouble* c, culaDeviceDouble* s, int incc);
culaStatus culaDeviceDlascl(char type, int kl, int ku, culaDouble cfrom, culaDouble cto, int m, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDlaset(char uplo, int m, int n, culaDouble alpha, culaDouble beta, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDlasr(char side, char pivot, char direct, int m, int n, culaDeviceDouble* c, culaDeviceDouble* s, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDlat2s(char uplo, int n, culaDeviceDouble* a, int lda, culaDeviceFloat* sa, int ldsa);
culaStatus culaDeviceDorgbr(char vect, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorghr(int n, int ilo, int ihi, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorglq(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorgql(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorgqr(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDorgrq(int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau);
culaStatus culaDeviceDormlq(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDormql(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDormqr(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDormrq(char side, char trans, int m, int n, int k, culaDeviceDouble* a, int lda, culaDeviceDouble* tau, culaDeviceDouble* c, int ldc);
culaStatus culaDeviceDpbtrf(char uplo, int n, int kd, culaDeviceDouble* ab, int ldab);
culaStatus culaDeviceDposv(char uplo, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDpotrf(char uplo, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDpotri(char uplo, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDpotrs(char uplo, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceDsgesv(int n, int nrhs, culaDeviceDouble* a, int lda, culaInt* ipiv, culaDeviceDouble* b, int ldb, culaDeviceDouble* x, int ldx, int* iter);
culaStatus culaDeviceDsposv(char uplo, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb, culaDeviceDouble* x, int ldx, int* iter);
culaStatus culaDeviceDstebz(char range, char order, int n, double vl, double vu, int il, int iu, double abstol, culaDeviceDouble* d, culaDeviceDouble* e, int* m, int* nsplit, culaDeviceDouble* w, culaDeviceInt* iblock, culaDeviceInt* isplit);
culaStatus culaDeviceDsteqr(char compz, int n, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* z, int ldz);
culaStatus culaDeviceDsyev(char jobz, char uplo, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* w);
culaStatus culaDeviceDsyevx(char jobz, char range, char uplo, int n, culaDeviceDouble* a, int lda, culaDouble vl, culaDouble vu, int il, int iu, culaDouble abstol, culaInt* m, culaDeviceDouble* w, culaDeviceDouble* z, int ldz, culaInt* ifail);
culaStatus culaDeviceDsygv(int itype, char jobz, char uplo, int n, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb, culaDeviceDouble* w);
culaStatus culaDeviceDsyrdb(char jobz, char uplo, int n, int kd, culaDeviceDouble* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDouble* tau, culaDeviceDouble* z, int ldz);
culaStatus culaDeviceDtrtri(char uplo, char diag, int n, culaDeviceDouble* a, int lda);
culaStatus culaDeviceDtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceDouble* a, int lda, culaDeviceDouble* b, int ldb);
culaStatus culaDeviceSbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* vt, int ldvt, culaDeviceFloat* u, int ldu, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSgbtrf(int m, int n, int kl, int ku, culaDeviceFloat* a, int lda, culaInt* ipiv);
culaStatus culaDeviceSgeNancheck(int m, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSgeTranspose(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgeTransposeInplace(int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSgebrd(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* tauq, culaDeviceFloat* taup);
culaStatus culaDeviceSgeev(char jobvl, char jobvr, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* wr, culaDeviceFloat* wi, culaDeviceFloat* vl, int ldvl, culaDeviceFloat* vr, int ldvr);
culaStatus culaDeviceSgehrd(int n, int ilo, int ihi, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgelqf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgels(char trans, int m, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgeqlf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgeqrf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgeqrfp(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgeqrs(int m, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgerqf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSgesdd(char jobz, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* s, culaDeviceFloat* u, int ldu, culaDeviceFloat* vt, int ldvt);
culaStatus culaDeviceSgesv(int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgesvd(char jobu, char jobvt, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* s, culaDeviceFloat* u, int ldu, culaDeviceFloat* vt, int ldvt);
culaStatus culaDeviceSgetrf(int m, int n, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceSgetri(int n, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceSgetrs(char trans, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceInt* ipiv, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSgglse(int m, int n, int p, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb, culaDeviceFloat* c, culaDeviceFloat* d, culaDeviceFloat* x);
culaStatus culaDeviceSggrqf(int m, int p, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* taua, culaDeviceFloat* b, int ldb, culaDeviceFloat* taub);
culaStatus culaDeviceSlacpy(char uplo, int m, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSlag2d(int m, int n, culaDeviceFloat* a, int lda, culaDeviceDouble* sa, int ldsa);
culaStatus culaDeviceSlar2v(int n, culaDeviceFloat* x, culaDeviceFloat* y, culaDeviceFloat* z, int incx, culaDeviceFloat* c, culaDeviceFloat* s, int incc);
culaStatus culaDeviceSlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceFloat* v, int ldv, culaDeviceFloat* t, int ldt, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSlarfg(int n, culaDeviceFloat* alpha, culaDeviceFloat* x, int incx, culaDeviceFloat* tau);
culaStatus culaDeviceSlargv(int n, culaDeviceFloat* x, int incx, culaDeviceFloat* y, int incy, culaDeviceFloat* c, int incc);
culaStatus culaDeviceSlartv(int n, culaDeviceFloat* x, int incx, culaDeviceFloat* y, int incy, culaDeviceFloat* c, culaDeviceFloat* s, int incc);
culaStatus culaDeviceSlascl(char type, int kl, int ku, culaFloat cfrom, culaFloat cto, int m, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSlaset(char uplo, int m, int n, culaFloat alpha, culaFloat beta, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSlasr(char side, char pivot, char direct, int m, int n, culaDeviceFloat* c, culaDeviceFloat* s, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSlat2d(char uplo, int n, culaDeviceFloat* a, int lda, culaDeviceDouble* sa, int ldsa);
culaStatus culaDeviceSorgbr(char vect, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorghr(int n, int ilo, int ihi, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorglq(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorgql(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorgqr(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSorgrq(int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau);
culaStatus culaDeviceSormlq(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSormql(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSormqr(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSormrq(char side, char trans, int m, int n, int k, culaDeviceFloat* a, int lda, culaDeviceFloat* tau, culaDeviceFloat* c, int ldc);
culaStatus culaDeviceSpbtrf(char uplo, int n, int kd, culaDeviceFloat* ab, int ldab);
culaStatus culaDeviceSposv(char uplo, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSpotrf(char uplo, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSpotri(char uplo, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceSpotrs(char uplo, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceSstebz(char range, char order, int n, float vl, float vu, int il, int iu, float abstol, culaDeviceFloat* d, culaDeviceFloat* e, int* m, int* nsplit, culaDeviceFloat* w, culaDeviceInt* iblock, culaDeviceInt* isplit);
culaStatus culaDeviceSsteqr(char compz, int n, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* z, int ldz);
culaStatus culaDeviceSsyev(char jobz, char uplo, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* w);
culaStatus culaDeviceSsyevx(char jobz, char range, char uplo, int n, culaDeviceFloat* a, int lda, culaFloat vl, culaFloat vu, int il, int iu, culaFloat abstol, culaInt* m, culaDeviceFloat* w, culaDeviceFloat* z, int ldz, culaInt* ifail);
culaStatus culaDeviceSsygv(int itype, char jobz, char uplo, int n, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb, culaDeviceFloat* w);
culaStatus culaDeviceSsyrdb(char jobz, char uplo, int n, int kd, culaDeviceFloat* a, int lda, culaDeviceFloat* d, culaDeviceFloat* e, culaDeviceFloat* tau, culaDeviceFloat* z, int ldz);
culaStatus culaDeviceStrtri(char uplo, char diag, int n, culaDeviceFloat* a, int lda);
culaStatus culaDeviceStrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceFloat* a, int lda, culaDeviceFloat* b, int ldb);
culaStatus culaDeviceZbdsqr(char uplo, int n, int ncvt, int nru, int ncc, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* vt, int ldvt, culaDeviceDoubleComplex* u, int ldu, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZcgesv(int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaInt* ipiv, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* x, int ldx, int* iter);
culaStatus culaDeviceZcposv(char uplo, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* x, int ldx, int* iter);
culaStatus culaDeviceZgbtrf(int m, int n, int kl, int ku, culaDeviceDoubleComplex* a, int lda, culaInt* ipiv);
culaStatus culaDeviceZgeConjugate(int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgeNancheck(int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgeTranspose(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgeTransposeConjugate(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgeTransposeConjugateInplace(int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgeTransposeInplace(int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZgebrd(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* tauq, culaDeviceDoubleComplex* taup);
culaStatus culaDeviceZgeev(char jobvl, char jobvr, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* w, culaDeviceDoubleComplex* vl, int ldvl, culaDeviceDoubleComplex* vr, int ldvr);
culaStatus culaDeviceZgehrd(int n, int ilo, int ihi, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgelqf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgels(char trans, int m, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgeqlf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgeqrf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgeqrfp(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgeqrs(int m, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgerqf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZgesdd(char jobz, int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* s, culaDeviceDoubleComplex* u, int ldu, culaDeviceDoubleComplex* vt, int ldvt);
culaStatus culaDeviceZgesv(int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgesvd(char jobu, char jobvt, int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* s, culaDeviceDoubleComplex* u, int ldu, culaDeviceDoubleComplex* vt, int ldvt);
culaStatus culaDeviceZgetrf(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceZgetri(int n, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv);
culaStatus culaDeviceZgetrs(char trans, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceInt* ipiv, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZgglse(int m, int n, int p, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* c, culaDeviceDoubleComplex* d, culaDeviceDoubleComplex* x);
culaStatus culaDeviceZggrqf(int m, int p, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* taua, culaDeviceDoubleComplex* b, int ldb, culaDeviceDoubleComplex* taub);
culaStatus culaDeviceZheev(char jobz, char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* w);
culaStatus culaDeviceZheevx(char jobz, char range, char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDouble vl, culaDouble vu, int il, int iu, culaDouble abstol, culaInt* m, culaDeviceDouble* w, culaDeviceDoubleComplex* z, int ldz, culaInt* ifail);
culaStatus culaDeviceZhegv(int itype, char jobz, char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb, culaDeviceDouble* w);
culaStatus culaDeviceZherdb(char jobz, char uplo, int n, int kd, culaDeviceDoubleComplex* a, int lda, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* z, int ldz);
culaStatus culaDeviceZlacpy(char uplo, int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZlag2c(int m, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceFloatComplex* sa, int ldsa);
culaStatus culaDeviceZlar2v(int n, culaDeviceDoubleComplex* x, culaDeviceDoubleComplex* y, culaDeviceDoubleComplex* z, int incx, culaDeviceDouble* c, culaDeviceDoubleComplex* s, int incc);
culaStatus culaDeviceZlarfb(char side, char trans, char direct, char storev, int m, int n, int k, culaDeviceDoubleComplex* v, int ldv, culaDeviceDoubleComplex* t, int ldt, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZlarfg(int n, culaDeviceDoubleComplex* alpha, culaDeviceDoubleComplex* x, int incx, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZlargv(int n, culaDeviceDoubleComplex* x, int incx, culaDeviceDoubleComplex* y, int incy, culaDeviceDouble* c, int incc);
culaStatus culaDeviceZlartv(int n, culaDeviceDoubleComplex* x, int incx, culaDeviceDoubleComplex* y, int incy, culaDeviceDouble* c, culaDeviceDoubleComplex* s, int incc);
culaStatus culaDeviceZlascl(char type, int kl, int ku, culaDouble cfrom, culaDouble cto, int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZlaset(char uplo, int m, int n, culaDoubleComplex alpha, culaDoubleComplex beta, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZlasr(char side, char pivot, char direct, int m, int n, culaDeviceDouble* c, culaDeviceDouble* s, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZlat2c(char uplo, int n, culaDeviceDoubleComplex* a, int lda, culaDeviceFloatComplex* sa, int ldsa);
culaStatus culaDeviceZpbtrf(char uplo, int n, int kd, culaDeviceDoubleComplex* ab, int ldab);
culaStatus culaDeviceZposv(char uplo, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZpotrf(char uplo, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZpotri(char uplo, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZpotrs(char uplo, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZsteqr(char compz, int n, culaDeviceDouble* d, culaDeviceDouble* e, culaDeviceDoubleComplex* z, int ldz);
culaStatus culaDeviceZtrConjugate(char uplo, char diag, int m, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZtrtri(char uplo, char diag, int n, culaDeviceDoubleComplex* a, int lda);
culaStatus culaDeviceZtrtrs(char uplo, char trans, char diag, int n, int nrhs, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* b, int ldb);
culaStatus culaDeviceZungbr(char vect, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZunghr(int n, int ilo, int ihi, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZunglq(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZungql(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZungqr(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZungrq(int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau);
culaStatus culaDeviceZunmlq(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZunmql(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZunmqr(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);
culaStatus culaDeviceZunmrq(char side, char trans, int m, int n, int k, culaDeviceDoubleComplex* a, int lda, culaDeviceDoubleComplex* tau, culaDeviceDoubleComplex* c, int ldc);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // __EMP_CULA_LAPACK_DEVICE_H__

