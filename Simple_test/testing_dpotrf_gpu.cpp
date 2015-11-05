
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
//#include "magma_lapack.h"
#include "cula.h"
#include "papi.h"
#include <iostream>

#define FMULS_POTRF(n_) ((n_) * (((1. / 6.) * (n_) + 0.5) * (n_) + (1. / 3.)))
#define FADDS_POTRF(n_) ((n_) * (((1. / 6.) * (n_)      ) * (n_) - (1. / 6.)))
#define FLOPS_DPOTRF(n_) (     FMULS_POTRF((double)(n_)) +       FADDS_POTRF((double)(n_)) )


using namespace std;



int main( int argc, char** argv)
{


    
    double *h_A, *h_R, *d_A;
    
    int N, n2, lda, ldda, info;
   
    
    
    int Nsize[] = {5120, 7680, 10240, 12800, 15360, 17920, 20480, 23040, 25600, 28160, 30720, 33280, 16};
    for( int i = 0; i < 12; ++i ) {
            N  = Nsize[i];
            lda = N;
            n2  = lda*N;
            ldda = ((N+31)/32)*32;
            double gflops = FLOPS_DPOTRF( N ) / 1e9;
            
            cudaMalloc((void**)&d_A,N*ldda*sizeof(double));
            
//            h_A = new double[n2];
//            magma_dmalloc_pinned(&h_R, n2 * sizeof(double));
            magma_dmalloc(&d_A, ldda*N*sizeof(double));
//            
            
            /* Initialize the matrix */
//            lapackf77_dlarnv( &ione, ISEED, &n2, h_A );
//            magma_dmake_hpd( N, h_A, lda );
//            lapackf77_dlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );
   //         magma_dsetmatrix( N, N, h_A, lda, d_A, ldda );
            
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
            
            
     //       delete[] h_A;
     //       magma_free(h_R);
     //       magma_free(d_A);
            
            fflush( stdout );
    }

    return 0;
}
