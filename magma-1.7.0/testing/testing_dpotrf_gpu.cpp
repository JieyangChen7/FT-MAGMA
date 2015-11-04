/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zpotrf_gpu.cpp normal z -> d, Fri Jan 30 19:00:24 2015
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "cula.h"
#include "papi.h"
#include <iostream>

using namespace std;
/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dpotrf
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double *h_A, *h_R;
    magmaDouble_ptr d_A;
    magma_int_t N, n2, lda, ldda, info;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,2};
    double      work[1], error;
    magma_int_t     status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    culaInitialize();
    int Nsize[] = {5120, 7680, 10240, 12800, 15360, 17920, 20480, 23040, 25600, 28160, 30720, 33280, 16};
    printf("uplo = %s\n", lapack_uplo_const(opts.uplo) );
    printf("  N     CPU GFlop/s (sec)   GPU GFlop/s (sec)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("========================================================\n");
    //for( int itest = 0; itest < opts.ntest; ++itest ) {
    for( int itest = 0; itest < 12; ++itest ) {
      //  for( int iter = 0; iter < opts.niter; ++iter ) {
            //N   = opts.nsize[itest];
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
            
            if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
            				cout << "PAPI ERROR" << endl;
            				//return -1;
            }
//            culaStatus status = culaDeviceDpotrf('l', N, d_A, ldda);
//			if (status != culaNoError) {
//				cout<<"CULA ERROR:"<<status<<endl;
//			}
			
			magma_dpotrf_gpu( MagmaLower, N, d_A, ldda, &info );
			
            if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
							cout << "PAPI ERROR" << endl;
							//return -1;
			}
            cout << "CULA["<<N<<"]---time:"<<real_time<<"---gflops:"<<(double)gflops/real_time<<endl;
            //culaShutdown();
            PAPI_shutdown();
            
            
            //magma_dpotrf_gpu( MagmaLower, N, d_A, ldda, &info );
//            gpu_time = magma_wtime() - gpu_time;
//            gpu_perf = gflops / gpu_time;
//            if (info != 0)
//                printf("magma_dpotrf_gpu returned error %d: %s.\n",
//                       (int) info, magma_strerror( info ));
//            
//            if ( opts.lapack ) {
//                /* =====================================================================
//                   Performs operation using LAPACK
//                   =================================================================== */
//                cpu_time = magma_wtime();
//                lapackf77_dpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
//                cpu_time = magma_wtime() - cpu_time;
//                cpu_perf = gflops / cpu_time;
//                if (info != 0)
//                    printf("lapackf77_dpotrf returned error %d: %s.\n",
//                           (int) info, magma_strerror( info ));
//                
//                /* =====================================================================
//                   Check the result compared to LAPACK
//                   =================================================================== */
//                magma_dgetmatrix( N, N, d_A, ldda, h_R, lda );
//                error = lapackf77_dlange("f", &N, &N, h_A, &lda, work);
//                blasf77_daxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
//                error = lapackf77_dlange("f", &N, &N, h_R, &lda, work) / error;
//                
//                printf("%5d   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
//                       (int) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
//                       error, (error < tol ? "ok" : "failed") );
//                status += ! (error < tol);
//            }
//            else {
//                printf("%5d     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
//                       (int) N, gpu_perf, gpu_time );
//            }
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            fflush( stdout );
//        }
//        if ( opts.niter > 1 ) {
//            printf( "\n" );
//        }
    }

    TESTING_FINALIZE();
    return status;
}
