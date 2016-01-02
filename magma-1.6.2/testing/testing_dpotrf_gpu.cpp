/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zpotrf_gpu.cpp normal z -> d, Fri Jan 30 19:00:24 2015
       @modified by Jieyang Chen
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "papi.h"
using namespace std;
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dpotrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops;
    double *h_A, *h_R;
    magmaDouble_ptr d_A;
    magma_int_t N, n2, lda, ldda, info;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,2};
    magma_int_t     status = 0;

    int Nsize[] = {5120, 7680, 10240, 12800, 15360, 17920, 20480, 23040, 25600, 28160, 30720, 33280, 16};
    printf("Benchmark for Enhanced Online-ABFT Choleksy Decomposition==================================\n");
    for( int itest = 0; itest < 11; ++itest ) {
            N   = Nsize[itest];
            lda = N;
            n2  = lda*N;
            ldda = ((N+31)/32)*32;
            gflops = FLOPS_DPOTRF( N ) / 1e9;
            
            TESTING_MALLOC_CPU( h_A, double, n2     );
            TESTING_MALLOC_PIN( h_R, double, n2     );
            TESTING_MALLOC_DEV( d_A, double, ldda*N );
            
            /* Initialize the matrix */
            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
            magma_dmake_hpd( N, h_A, lda );
            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
            magma_dsetmatrix( N, N, h_A, lda, d_A, ldda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            //gpu_time = magma_wtime();
            float real_time = 0.0;
			float proc_time = 0.0;
			long long flpins = 0.0;
			float mflops = 0.0;
			//timing start***************
			if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
				cout << "PAPI ERROR" << endl;
				return -1;
			}            
            magma_dpotrf_gpu( MagmaLower, N, d_A, ldda, &info );
            if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
 				cout << "PAPI ERROR" << endl;
 				return -1;
 			}
 			cout<<"N="<<N<<"---time:"<<real_time<<"---gflops:"<<(double)gflops/real_time<<endl;
 			PAPI_shutdown();   
         
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            fflush( stdout );
    }

    TESTING_FINALIZE();
    return status;
}
