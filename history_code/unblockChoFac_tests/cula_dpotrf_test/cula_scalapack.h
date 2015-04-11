#ifndef __EMP_CULA_SCALAPACK_H__
#define __EMP_CULA_SCALAPACK_H__

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

/**
 * @struct pculaConfig
 * @brief Contains information that steers execution and tuning of a pCula function
 */
typedef struct 
{
    // Specifies the number of CUDA devices to use for a given computation. Set 
    // to zero to use all available CUDA devices.
    int nCuda;

    // Optionally sets a list of devices to be used in pCULA execution. This 
    // array must be the same length as 'nCuda'. Set to zero to use the default 
    // CUDA device ordering. This list is zero indexed.
    int* cudaDeviceList;

    // Sets the per CUDA device maximum memory to use (in MB). This array must be 
    // the same length as nCuda and uses the same ordering specified by the 
    // cudaDeviceList parameter. Set to zero to have pCULA safely manage the 
    // maximum amount of memory usage.
    int* maxCudaMemoryUsage;

    // Save auto-tuning results between subsequent function invocations.
    // By default, this data is saved to an OS specific application data folder
    // that can be changed by setting the PCULA_TUNING_DIR environment 
    // variable.
    int preserveTuningResult;

    // Upon completion of the routine, writes out a .dot file to visualize the 
    // direct acyclic graph of task dependencies to the current working 
    // directory.
    const char* dotFileName;
    
    // Upon completion of the routine, writes out a .svg file to visualize the 
    // specific timing diagram of all devices involved in the computation.
    const char* timelineFileName;

} pculaConfig;

// Initializes the pCULA configuration structure to sensible defaults
culaStatus pculaConfigInit(pculaConfig* config);

/**
 * BLAS Routines
 */

culaStatus pculaSgemm(const pculaConfig* config, char transa, char transb, int m, int n, int k, culaFloat alpha, const culaFloat* A, int lda, const culaFloat* B, int ldb, culaFloat beta, culaFloat* C, int ldc);
culaStatus pculaDgemm(const pculaConfig* config, char transa, char transb, int m, int n, int k, culaDouble alpha, const culaDouble* A, int lda, const culaDouble* B, int ldb, culaDouble beta, culaDouble* C, int ldc);
culaStatus pculaCgemm(const pculaConfig* config, char transa, char transb, int m, int n, int k, culaFloatComplex alpha, const culaFloatComplex* A, int lda, const culaFloatComplex* B, int ldb, culaFloatComplex beta, culaFloatComplex* C, int ldc);
culaStatus pculaZgemm(const pculaConfig* config, char transa, char transb, int m, int n, int k, culaDoubleComplex alpha, const culaDoubleComplex* A, int lda, const culaDoubleComplex* B, int ldb, culaDoubleComplex beta, culaDoubleComplex* C, int ldc);

culaStatus pculaStrsm(const pculaConfig* config, char side, char uplo, char transa, char diag, int m, int n, culaFloat alpha, const culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus pculaDtrsm(const pculaConfig* config, char side, char uplo, char transa, char diag, int m, int n, culaDouble alpha, const culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus pculaCtrsm(const pculaConfig* config, char side, char uplo, char transa, char diag, int m, int n, culaFloatComplex alpha, const culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus pculaZtrsm(const pculaConfig* config, char side, char uplo, char transa, char diag, int m, int n, culaDoubleComplex alpha, const culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);

/**
 * LAPACK Routines
 */

culaStatus pculaSgesv(const pculaConfig* config, int n, int nrhs, culaFloat* a, int lda, culaInt* ipiv, culaFloat* b, int ldb);
culaStatus pculaDgesv(const pculaConfig* config, int n, int nrhs, culaDouble* a, int lda, culaInt* ipiv, culaDouble* b, int ldb);
culaStatus pculaCgesv(const pculaConfig* config, int n, int nrhs, culaFloatComplex* a, int lda, culaInt* ipiv, culaFloatComplex* b, int ldb);
culaStatus pculaZgesv(const pculaConfig* config, int n, int nrhs, culaDoubleComplex* a, int lda, culaInt* ipiv, culaDoubleComplex* b, int ldb);

culaStatus pculaSgetrf(const pculaConfig* config, int m, int n, culaFloat* a, int lda, culaInt* ipiv);
culaStatus pculaDgetrf(const pculaConfig* config, int m, int n, culaDouble* a, int lda, culaInt* ipiv);
culaStatus pculaCgetrf(const pculaConfig* config, int m, int n, culaFloatComplex* a, int lda, culaInt* ipiv);
culaStatus pculaZgetrf(const pculaConfig* config, int m, int n, culaDoubleComplex* a, int lda, culaInt* ipiv);

culaStatus pculaSgetrfNoPiv(const pculaConfig* config, int m, int n, culaFloat* a, int lda);
culaStatus pculaDgetrfNoPiv(const pculaConfig* config, int m, int n, culaDouble* a, int lda);
culaStatus pculaCgetrfNoPiv(const pculaConfig* config, int m, int n, culaFloatComplex* a, int lda);
culaStatus pculaZgetrfNoPiv(const pculaConfig* config, int m, int n, culaDoubleComplex* a, int lda);

culaStatus pculaSgetrs(const pculaConfig* config, char trans, int n, int nrhs, const culaFloat* a, int lda, const culaInt* ipiv, culaFloat* b, int ldb);
culaStatus pculaDgetrs(const pculaConfig* config, char trans, int n, int nrhs, const culaDouble* a, int lda, const culaInt* ipiv, culaDouble* b, int ldb);
culaStatus pculaCgetrs(const pculaConfig* config, char trans, int n, int nrhs, const culaFloatComplex* a, int lda, const culaInt* ipiv, culaFloatComplex* b, int ldb);
culaStatus pculaZgetrs(const pculaConfig* config, char trans, int n, int nrhs, const culaDoubleComplex* a, int lda, const culaInt* ipiv, culaDoubleComplex* b, int ldb);

culaStatus pculaSgeqrf(const pculaConfig* config, int m, int n, culaFloat* a, int lda, culaFloat* tau);
culaStatus pculaDgeqrf(const pculaConfig* config, int m, int n, culaDouble* a, int lda, culaDouble* tau);
culaStatus pculaCgeqrf(const pculaConfig* config, int m, int n, culaFloatComplex*  a, int lda, culaFloatComplex* tau);
culaStatus pculaZgeqrf(const pculaConfig* config, int m, int n, culaDoubleComplex* a, int lda, culaDoubleComplex* tau);

culaStatus pculaSposv(const pculaConfig* config, char uplo, int n, int nrhs, culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus pculaDposv(const pculaConfig* config, char uplo, int n, int nrhs, culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus pculaCposv(const pculaConfig* config, char uplo, int n, int nrhs, culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus pculaZposv(const pculaConfig* config, char uplo, int n, int nrhs, culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);

culaStatus pculaSpotrf(const pculaConfig* config, char uplo, int n, culaFloat* a, int lda);
culaStatus pculaDpotrf(const pculaConfig* config, char uplo, int n, culaDouble* a, int lda);
culaStatus pculaCpotrf(const pculaConfig* config, char uplo, int n, culaFloatComplex* a, int lda);
culaStatus pculaZpotrf(const pculaConfig* config, char uplo, int n, culaDoubleComplex* a, int lda);

culaStatus pculaSpotrs(const pculaConfig* config, char uplo, int n, int nrhs, const culaFloat* a, int lda, culaFloat* b, int ldb);
culaStatus pculaDpotrs(const pculaConfig* config, char uplo, int n, int nrhs, const culaDouble* a, int lda, culaDouble* b, int ldb);
culaStatus pculaCpotrs(const pculaConfig* config, char uplo, int n, int nrhs, const culaFloatComplex* a, int lda, culaFloatComplex* b, int ldb);
culaStatus pculaZpotrs(const pculaConfig* config, char uplo, int n, int nrhs, const culaDoubleComplex* a, int lda, culaDoubleComplex* b, int ldb);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // __EMP_CULA_SCALAPACK_H__

