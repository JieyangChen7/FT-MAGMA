
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



int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t   gflops;
    double *h_A, *h_R;
    magmaDouble_ptr d_A;
    magma_int_t N, n2, lda, ldda, info;
    double c_neg_one = MAGMA_D_NEG_ONE;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,2};
    
    
    int Nsize[] = {5120, 7680, 10240, 12800, 15360, 17920, 20480, 23040, 25600, 28160, 30720, 33280, 16};
    for( int itest = 0; itest < 12; ++itest ) {
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
            
            float real_time = 0.0;
			float proc_time = 0.0;
			long long flpins = 0.0;
			float mflops = 0.0;
            
            if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
                    cout << "PAPI ERROR" << endl;
                    //return -1;
            }
            culaInitialize();
            culaStatus status = culaDeviceDpotrf('l', N, d_A, ldda);
			if (status != culaNoError) {
				cout<<"CULA ERROR:"<<status<<endl;
			}
			
			//magma_dpotrf_gpu( MagmaLower, N, d_A, ldda, &info );
			
            if (PAPI_flops(&real_time, &proc_time, &flpins, &mflops) < PAPI_OK) {
                    cout << "PAPI ERROR" << endl;
                    //return -1;
			}
            cout << "Size:["<<N<<"]---time:"<<real_time<<"---GFLOPS:"<<(double)gflops/real_time<<endl;
            culaShutdown();
            PAPI_shutdown();
            
            TESTING_FREE_CPU( h_A );
            TESTING_FREE_PIN( h_R );
            TESTING_FREE_DEV( d_A );
            fflush( stdout );
    }

    TESTING_FINALIZE();
    return 0;
}
