/*Blocked Cholesky Factorization v1.4.
*potf on CPU and dtrsm on GPU, dgemm on GPU. Compute either upper or lower. Initial data is on GPU, so transfer the data to GPU is not taken care of.
*Jieyang Chen, University of California, Riverside
**/

//Initial Data on GPU
//Hybird GPU (DTRSM & DGEMM)and CPU (DPOTRF) version MAGMA way
//Column Major
//Either upper and lower triangle
//testing function are made to facilitate testing
//CPU and GPU are asynchronized
//CUBLAS are used in DTRSM & DGEMM
//Leading Dimension is used
//Add CUDA Event timing

#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath> 
#include<ctime>
#include"cublas_v2.h"
#include<cuda_runtime.h>
#include<curand.h>
#include"acml.h"
#include"papi.h"
#include"cblas.h"
//#include"lapacke.h"
//#include"blas.h"
//#include<cuda.h>
//#include<cuda_runtime.h>

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )

using namespace std;

void printMatrix_host(double * matrix_host, int N);
void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int N);
void POTF2_CPU(char uplo, double * matrix, int ld, int B);
__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo, double alpha, double beta);
void matrixGenerator_gpu(char uplo, double * matrix, int matrix_ld, double * result, int result_ld, int N,  int B);
__global__ void resultVerify_gpu_help(double * realResult,int real_ld, double * testResult,int test_ld,double * diff, int N);
bool resultVerify_gpu(double * realResult,int real_ld, double * testResult, int test_ld, int N, int B);
void my_dpotrf(char uplo, double * matrix, int ld, int N, int B, double k, float * real_time, float * proc_time, long long * flpins, float * mflops);


long long int SYRK_Flops(long long N, long long B, long long I){
    long long int flop_num = 0;
    //int b_size;
    //for(int i=B;i<N;i+=B){
        //b_size = min(B,N-i);
        //flop_num += 2 * b_size * b_size * i + b_size*b_size;
    
    //}
    //return flop_num;
    if (N>0&&(!(I<B))&&I<N) {
        
        flop_num = 2 * B * B * I + B*B;
        if (flop_num<0) {
            cout<<"ERROR:syrk"<<flop_num<<"  "<<N<<" "<<B<<" "<<I<<endl;
        }
        return flop_num;
    }
    return 0;
}

long long GEMM_Flops(long long N, long long B, long long I){
    long long flop_num = 0;
    /*int b_size;
    for(int i=B;i<N-B;i+=B){
        b_size = min(B,N-i);
        flop_num += 2 * b_size * (N-i-b_size) * i + b_size*(N-i-b_size);
    }
    return flop_num;
    */
    if (N>0&&(!(I<B))&&N-I-B>0) {
        flop_num = 2 * B * (N-I-B) * I + B*(N-I-B);
        if (flop_num<0) {
            cout<<"ERROR:gemm"<<flop_num<<"  "<<N<<" "<<B<<" "<<I<<endl;
        }
        return flop_num;
    }
    return 0;
}
        
long long TRSM_Flops(long long N, long long B, long long I){
    long long flop_num = 0;
    /*int b_size;
    for(int i=0;i<N-B;i+=B){
        b_size = min(B,N-i);
        flop_num += b_size * b_size * (N-i-b_size);
    }
    return flop_num;
     */
    if (N>0&&(!(I<0))&&N-I-B>0) {
        flop_num = B * B * (N-I-B);
        if (flop_num<0) {
            cout<<"ERROR:trsm"<<flop_num<<"  "<<N<<" "<<B<<" "<<I<<endl;
        }
        return flop_num;
    }
    return 0;
}

long POTRF_Flops(int B){
    long long flop_num = 0;
    int b_size = B;
    if(B>64){
        for (int i=0; i<B; i+=64) {
            b_size = min(64, B-i);
            //SYRK
            if(i>0){
                flop_num += SYRK_Flops(B, b_size, i);
            }
            //GEMM
            if(i>0&&i<B-b_size){
                flop_num += GEMM_Flops(B, b_size, i);
            }
            //POTRF2
            for(int j=0;j<b_size;j++){
                flop_num += 2*j + 1 + (b_size-j-1)*i + b_size-j-1;
            }
            //TRSM
            flop_num += TRSM_Flops(B, b_size, i);
        }
    }else{
      for(int i=0;i<B;i++){
          flop_num += 2*i + 1 + (B-i-1)*i + B-i-1;
      }
    }
    if (flop_num<0) {
        cout<<"ERROR:potrf  "<<flop_num<<"  "<<B<<" "<<endl;
    }
    return flop_num;
}

double COPY_time(int B){
    
    float real_time = 0.0;
    float proc_time = 0.0;
    long long flpins = 0.0;
    float mflops = 0.0;
    
    
    float total_real_time = 0.0;
    float total_proc_time = 0.0;
    long long total_flpins = 0.0;
    float total_mflops = 0.0;
    
    int TEST_NUM = 10;
    
    char uplo = 'u';
    double * matrix;
    double * result;
    double * temp;
    
    size_t matrix_pitch;
    size_t result_pitch;
    //Memory allocation on RAM and DRAM
    cudaMallocPitch((void**)&matrix,&matrix_pitch,B*sizeof(double),B);
    cudaMallocPitch((void**)&result,&result_pitch,B*sizeof(double),B);
    cudaHostAlloc((void**)&temp,B*B*sizeof(double),cudaHostAllocDefault);
    
    int matrix_ld= matrix_pitch/sizeof(double);
    int result_ld= result_pitch/sizeof(double);
    
    matrixGenerator_gpu(uplo,matrix,matrix_ld,result,result_ld,B,2);
    
    
    for(int i=0;i<TEST_NUM;i++){
    
      if(PAPI_flops( &real_time, &proc_time, &flpins, &mflops)<PAPI_OK){
        cout<<"PAPI ERROR"<<endl;
        return -1;
      }
      cudaMemcpy2D(temp,B*sizeof(double),matrix,matrix_pitch,B*sizeof(double),B,cudaMemcpyDeviceToHost);
      cudaMemcpy2D(matrix,matrix_pitch,temp,B*sizeof(double),B*sizeof(double),B,cudaMemcpyHostToDevice);
      if(PAPI_flops( &real_time, &proc_time, &flpins, &mflops)<PAPI_OK){
        cout<<"PAPI ERROR"<<endl;
        return -1;
      }
      PAPI_shutdown();
      total_real_time += real_time;
      total_proc_time += proc_time;
    }
    

    
    
    
    cudaFreeHost(temp);
    cudaFree(matrix);
    cudaFree(result);
    
    double time =(total_real_time/(double)TEST_NUM);
    if (time<0) {
        cout<<"ERROR:copy"<<B<<" "<<endl;
    }
    return time;

}

int OptimalB(int N, int G){
    long long GPUflops = 300000000000;
    long long CPUflops = 9600000000;
    int bestB = 2;
    double bestT = 10000000;
    long long numc=0;
    long long numg=0;
    long long bnumc=0;
    long long bnumg=0;
    /*
    for(int b=2;b<N;b++){ cout<<"b="<<b;
        double syrk_time = SYRK_Flops(N,b)/GPUflops; cout<<"---syrk_time="<<syrk_time;
        double gemm_time = GEMM_Flops(N,b)/GPUflops;cout<<"---gemm_time="<<gemm_time;
        double trsm_time = TRSM_Flops(N,b)/GPUflops;cout<<"---trsm_time="<<trsm_time;
        double potrf_time = POTRF_Flops(N,b)/CPUflops; cout<<"---potrf_time="<<potrf_time;
        double copy_time = COPY_time(N,b); cout<<"---copy_time="<<copy_time;
        
        double total_time = syrk_time + trsm_time + max(gemm_time,potrf_time+copy_time); cout<<"---total_time="<<total_time<<endl;
        
        if(total_time<bestT){
            bestT = total_time;
            bestB = b;
        }
        
    }
     */
    //    long long num256 =0;
    for (int B=2; B<N+1; B+=G) {
      numc=0;
      numg=0;
        double current_time = 0;
        double copy_t = (double)COPY_time(B);
        double potrf_t = (double)POTRF_Flops(B)/CPUflops;
	numc+= POTRF_Flops(B)*N/B;
        for (int i=0; i<N; i+=B) {
	  if (i>0) {
	    numg+=SYRK_Flops(N,B,i);
	    current_time += (double)SYRK_Flops(N,B,i)/GPUflops;
	  }
	  if(i>0 && i<N-B){
	    numg+=GEMM_Flops(N,B,i);
	    current_time += (double)GEMM_Flops(N,B,i)/GPUflops;
	  }
            double t1 = 0;
            double t2 = 0;
            t1 += potrf_t;
            t1 += copy_t;
            if(i>0 && i<N-B){
	      numg+=GEMM_Flops(N,B,i);
	      t2 += (double)GEMM_Flops(N,B,i)/GPUflops;
            }
            current_time += max(t1,t2);
            if(i<N-B){
	      numg+=TRSM_Flops(N,B,i);
	      current_time += (double)TRSM_Flops(N,B,i)/GPUflops;
            }
        }
	if(B==256){
	  cout<<"num256:"<<numc<<"  "<<numg<<"   "<<current_time<<endl;
	}
        if(current_time < bestT){
	  bestT = current_time;
            bestB = B;
	    bnumc = numc;
	    bnumg = numg;
           
        }
    }
    cout<<"N"<<N<<"--numc:"<<bnumc<<"--numg"<<bnumg<<"   b:"<<bestB<<"time:"<<bestT<<endl;
    
    return bestB;
    
    
    
}


void printMatrix_host(double * matrix_host, int N){
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            cout.width(5);
            cout.setf(ios::left);
            cout<<matrix_host[j*N+i];
        }
        cout<<endl;
    }
    cout<<endl;
}

void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int N){
    double * matrix_host = new double[N*N]();
    cudaMemcpy2D(matrix_host,N*sizeof(double),matrix_device,matrix_pitch,N*sizeof(double),N,cudaMemcpyDeviceToHost);
    printMatrix_host(matrix_host,N);
    delete[] matrix_host;
}

//cpu version
void POTF2_CPU(char uplo, double * matrix, int ld, int B){
  if(uplo == 'u'){
    for(int i = 0; i<B;i++){
      matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
      for(int j=i+1;j<B;j++){
        matrix[j*ld+i] /=matrix[i*ld+i];
      }
      for(int j=i+1;j<B;j++){
        for(int k=i+1;k<j+1;k++){
          matrix[j*ld+k]-=matrix[j*ld+i]*matrix[k*ld+i];
        }
      }
    }
  } 
  if(uplo == 'l'){
    for(int i = 0; i<B;i++){
      matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
      for(int j=i+1;j<B;j++){
	matrix[i*ld+j] /=matrix[i*ld+i];
      }
      for(int j=i+1;j<B;j++){
	for(int k=i+1;k<j+1;k++){
	  matrix[k*ld+j]-=matrix[i*ld+j]*matrix[i*ld+k];
	}
      }
    }
  }
}



__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo, double alpha, double beta){
    int col = threadIdx.x+blockIdx.x*blockDim.x;
    int row = threadIdx.y+blockIdx.y*blockDim.y;
    if(uplo == 'u'){
        if(row<col+1){
            matrix[col*ld+row] = int(matrix[col*ld+row]*alpha+beta);
        }
        else{
            matrix[col*ld+row] = int(0.0);
        }
    }
    else{
        if(col<row+1){
            matrix[col*ld+row] = int(matrix[col*ld+row]*alpha+beta);
        }
        else{
            matrix[col*ld+row] = int(0.0);
        }
    } 
}


void matrixGenerator_gpu(char uplo, double * matrix, int matrix_ld, double * result, int result_ld, int N,  int B){
    double a = 10.0;
    //initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    //initialize curand
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,10ULL);
    //generate random number in range (0,1] on result using curand
    curandGenerateUniformDouble(gen,result,result_ld*N);
    cudaDeviceSynchronize();
    //printMatrix_gpu(result,result_ld*sizeof(double),N);
    matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(result, result_ld, uplo, a,1);
    cudaDeviceSynchronize();
    //printMatrix_gpu(result,N);
    //do matrix-matrix multiplcation using cublas
    cudaMemset(matrix,0,matrix_ld*N*sizeof(double));
    
    
    double alpha = 1.0;
    double beta = 1.0;
    if(uplo == 'u'){
        cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,&alpha,result,result_ld,result,result_ld,&beta,matrix,matrix_ld);
    }
    else if(uplo == 'l'){
        cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,N,N,N,&alpha,result,result_ld,result,result_ld,&beta,matrix,matrix_ld);
    }
    cudaDeviceSynchronize();
    //printMatrix_gpu(matrix,N);
    matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, matrix_ld, uplo, 1.0,0);
    cudaDeviceSynchronize();
    //printMatrix_gpu(matrix,matrix_ld*sizeof(double), N);
}
 

__global__ void resultVerify_gpu_help(double * realResult,int real_ld, double * testResult,int test_ld,double * diff, int N){
    int col = threadIdx.x+blockIdx.x*blockDim.x;
    int row = threadIdx.y+blockIdx.y*blockDim.y;
    diff[col*N+row] = testResult[col*test_ld+row] - realResult[col*real_ld+row];
}

bool resultVerify_gpu(double * realResult,int real_ld, double * testResult, int test_ld, int N, int B){
    double * diff;
    cudaMalloc((void**)&diff,N*N*sizeof(double));
    resultVerify_gpu_help<<<dim3(N/B,N/B),dim3(B,B)>>>(realResult,real_ld,testResult,test_ld,diff,N);
    
    //printMatrix_gpu(realResult,real_ld*sizeof(double),N);
    //printMatrix_gpu(testResult,test_ld*sizeof(double),N);
    
    double * diff_host = new double[N*N]();
    cudaMemcpy(diff_host,diff,N*N*sizeof(double),cudaMemcpyDeviceToHost);
    //  printMatrix(diff_host,N);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(abs(diff_host[i*N+j])>1e-3){
                //  cout<<"diff:"<<abs(diff_host[i*N+j])<<endl;
                delete[] diff_host;
                cudaFree(diff);
                return false;
            }
        }
    }
    delete[] diff_host;
    cudaFree(diff);
    return true;
    
    
    
}






//determin the next block size
int getDynamicBlockSize(int i, int N, int B){
  float root1 = (i/448+sqrt((4*N*i-3*i*i)/448))/2;
  float root2 = ((i/448)-sqrt((4*N*i)/448-(3*i*i)/448))/2;
    
  cout<<"root1:"<<(int)root1<<"  root2:"<<(int)root2<<endl;
  if((int)root1==0){
    root1=64;
   }
  return root1+10;


}


void my_dpotrf(char uplo, double * matrix, int ld, int N, int B,double k,float * real_time, float * proc_time, long long * flpins, float * mflops){
  //cout<<"start my_dpotrf"<<endl;
  //initial data
  int b_size = B;
  double * temp;
  float gemm_time =0;
  float cpu_time =0;
  //cudaHostAlloc((void**)&temp,B*B*sizeof(double),cudaHostAllocDefault);
  //cout<<"pinned memory initialized"<<endl;
  //intial streams----------------------------
  cudaStream_t stream0;//for main loop
  cudaStream_t stream1;//for dgemm part
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
 
  //cout<<"Streams initialized"<<endl;
  //intial cublas
  cublasStatus_t cublasStatus;
  cublasHandle_t handle0;
  cublasStatus = cublasCreate(&handle0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS NOT INITIALIZED(handle0)"<<endl;
  cublasStatus = cublasSetStream(handle0,stream0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS SET STREAM NOT INITIALIZED(handle0)"<<endl;


  cublasHandle_t handle1;
  cublasStatus = cublasCreate(&handle1);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS NOT INITIALIZED(handle1)"<<endl;

  cublasStatus = cublasSetStream(handle1,stream1);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS SET STREAM NOT INITIALIZED(handle1)"<<endl;

  //cout<<"cublas initialized"<<endl;
  //for timing
  cudaEvent_t start0, stop0, start1, stop1;
  cudaEventCreate(&start0);
  cudaEventCreate(&stop0);
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  //cout<<"cuda event"<<endl;
    
  
    
  //float t=0;
  //cout<<"Inital complete"<<endl;
  
  //cout<<"entering loop"<<endl;
  //start the loop of calculation----------------------------
    
  if(PAPI_flops(real_time, proc_time, flpins, mflops)<PAPI_OK){
    cout<<"PAPI ERROR"<<endl;
    return;
  }
    
   // clock_t t = clock();
    
  for(int i=0;i<N;i+=B){
    /*  if(i!=0){
      if(cpu_time<gemm_time){
          B+=150;
      }
      else{
          B-=150;
      }
      }
    */
        b_size = min(B,N-i);
    //cout<<"block size:"<<b_size<<"  ";
	
	if(i>0){
	  double alpha = -1;
	  double beta = 1;
	  //cudaEventRecord(start0,stream0);
	  cublasDsyrk(handle0,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,b_size,i,&alpha,matrix+i*ld,ld,&beta,matrix+i*ld+i,ld);
	  //cudaEventRecord(stop0,stream0);
	}
	
    if(i!=0&&i+b_size<N){
	  double alpha = -1;
	  double beta = 1;
	  //int l = (N-i-b_size)*k;
	  //cudaEventRecord(start1,stream1);
	  cublasDgemm(handle1,CUBLAS_OP_T,CUBLAS_OP_N,b_size,N-i-b_size,i,&alpha,matrix+i*ld,ld,matrix+(i+b_size)*ld,ld,&beta,matrix+(i+b_size)*ld+i,ld);
	  //cudaEventRecord(stop1    ,stream1);
	  /*if(l>0){
            double * temp2;
            cudaHostAlloc((void**)&temp2,i*b_size*sizeof(double),cudaHostAllocDefault);
	    double * temp3;
            cudaHostAlloc((void**)&temp3,i*l*sizeof(double),cudaHostAllocDefault);

            cudaMemcpy2DAsync(temp2,i*sizeof(double),matrix+i*ld,ld*sizeof(double),i*sizeof(double),b_size,cudaMemcpyDeviceToHost,stream0);
            cudaMemcpy2DAsync(temp3,i*sizeof(double),matrix+(i+b_size+(N-i-b_size-l))*ld,ld*sizeof(double),i*sizeof(double),l,cudaMemcpyDeviceToHost,stream0);
	    cudaStreamSynchronize(stream0);
            dgemm('t','n',b_size,l,i,-1,temp2,i,temp3,i,1,temp3,i);
            cudaMemcpy2DAsync(matrix+(i+b_size+(N-i-b_size-l))*ld,ld*sizeof(double),temp3,i*sizeof(double),i*sizeof(double),l,cudaMemcpyHostToDevice,stream0);
            cudaStreamSynchronize(stream0);
            cudaFreeHost(temp2);
	    cudaFreeHost(temp3);
          }
	  */
    }
      
	/*if(i>0){
	  cudaEventSynchronize(stop0);
	  cudaEventElapsedTime(&t,start0,stop0);
	  cout<<"SYRK: "<<t<<"ms  ";
	}*/

	
	cudaStreamSynchronize(stream0);	
	//cudaEventRecord(start0,stream0);
	cudaHostAlloc((void**)&temp,b_size*b_size*sizeof(double),cudaHostAllocDefault);
	cudaMemcpy2DAsync(temp,b_size*sizeof(double),matrix+i*ld+i,ld*sizeof(double),b_size*sizeof(double),b_size,cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
    //POTF2_CPU(uplo,temp,b_size,b_size);
	int info;
    dpotrf('U',b_size,temp,b_size,&info);
	
	cudaMemcpy2DAsync(matrix+i*ld+i,ld*sizeof(double),temp,b_size*sizeof(double),b_size*sizeof(double),b_size,cudaMemcpyHostToDevice,stream0);
    //cudaEventRecord(stop0,stream0);
	
	/*if(i!=0&&i+b_size<ld){
	  cudaEventSynchronize(stop1);
	  cudaEventElapsedTime(&gemm_time,start1,stop1);
	  cout<<"GEMM: "<<gemm_time<<"ms  ";
	}


    cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&cpu_time,start0,stop0);
      cout<<"CPU: "<<cpu_time<<"ms  "<<endl;
     */
	//update B                                                                      
    if(i+b_size<N){
	  cudaStreamSynchronize(stream1);
	  cudaStreamSynchronize(stream0);
	  double alpha = 1;
          int l = (N-i-b_size)*k; 
	  //cudaEventRecord(start0,stream0);
	  cublasDtrsm(handle0,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,b_size,N-i-b_size-l,&alpha,matrix+i*ld+i,ld,matrix+(i+b_size)*ld+i,ld);
	 
	  if(l>0){
	    double * temp2;
	    cudaHostAlloc((void**)&temp2,b_size*l*sizeof(double),cudaHostAllocDefault);  
	    cudaMemcpy2DAsync(temp2,b_size*sizeof(double),matrix+(i+b_size+(N-i-b_size-l))*ld+i,ld*sizeof(double),b_size*sizeof(double),l,cudaMemcpyDeviceToHost,stream1);
	    cudaStreamSynchronize(stream1);
	    dtrsm('l','u','t','n',B,l,1,temp,b_size,temp2,B);
	    cudaMemcpy2DAsync(matrix+(i+b_size+(N-i-b_size-l))*ld+i,ld*sizeof(double),temp2,b_size*sizeof(double),b_size*sizeof(double),l,cudaMemcpyHostToDevice,stream1);
	    cudaStreamSynchronize(stream1);
	    cudaFreeHost(temp2);
	  }
	  /*cudaEventRecord(stop0,stream0);
	  cudaEventSynchronize(stop0);
	  cudaEventElapsedTime(&t,start0,stop0);
	  cout<<"TRSM: "<<t<<"ms  "<<endl;*/
    }
    cudaFreeHost(temp);
  }
  //  t=clock()-t;
  //  float time =((float)t/CLOCKS_PER_SEC);
  //  cout<<"time[N="<<N<<"B="<<B<<"]:"<<time<<"s."<<endl;
    
    
  if(PAPI_flops( real_time, proc_time, flpins, mflops)<PAPI_OK){
    cout<<"PAPI ERROR"<<endl;
    return;
  }
  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);
  cublasDestroy(handle0);
  cublasDestroy(handle1);
 // cudaFreeHost(temp);
  PAPI_shutdown();

}





void test_mydpotrf(int N, int B,double k,float * real_time, float * proc_time, long long * flpins, float * mflops){

  char uplo = 'u';
  double * matrix;
  double * result;
  size_t matrix_pitch;
  size_t result_pitch;
  //Memory allocation on RAM and DRAM
  cudaMallocPitch((void**)&matrix,&matrix_pitch,N*sizeof(double),N);
  cudaMallocPitch((void**)&result,&result_pitch,N*sizeof(double),N);
 
  int matrix_ld= matrix_pitch/sizeof(double);
  int result_ld= result_pitch/sizeof(double);

  matrixGenerator_gpu(uplo,matrix,matrix_ld,result,result_ld,N,2);
    
  my_dpotrf(uplo,matrix,matrix_ld,N,B,k,real_time, proc_time, flpins, mflops);
  
  
 
  //Verify result
  if(resultVerify_gpu(result,result_ld,matrix,matrix_ld,N,2)){
    cout<<"Result passed!"<<endl;
  }else{
    cout<<"Result failed!"<<endl;
   }
  
    cudaFree(matrix);
    cudaFree(result);

}

int main(int argc, char**argv){    
    
    int TEST_NUM = 1;
    int n[16]={256,384,512,640,768,896,1024,2048,3072,4096,5120,6144,7168,8192,9216,10240};
    //int b[16]={88,96,104,80,154,112,86,256,512,512,732,1024,1792,2048,3072,3414};
    //int b=256;
    for(int k=0;k<16;k++){
        //for(int b=2;b<n;b*=2){
      for(double K=0.1;K<0.2;K+=0.1){
	int b=OptimalB(n[k],1)
          float total_real_time = 0.0;
          float total_proc_time = 0.0;
          long long total_flpins = 0.0;
          float total_mflops = 0.0;

	  float real_time = 0.0;
	  float proc_time = 0.0;
	  long long flpins = 0.0;
	  float mflops = 0.0;
	  double flops = FLOPS_DPOTRF(n[k])/1e9;
	  cout<<"flops:"<<flops<<"  ";

          for(int i=0;i<TEST_NUM;i++){
            test_mydpotrf(n[k],b,K,&real_time, &proc_time, &flpins, &mflops);
            total_real_time += real_time;
            total_proc_time += proc_time;
            total_flpins += flpins;
            total_mflops += mflops;
          }
      
          cout<<"Size:"<<n[k]<<"("<<b[k]<<")---Real_time:"<<total_real_time/(double)TEST_NUM<<"---"<<"Proc_time:"<<total_proc_time/(double)TEST_NUM<<"---"<<"Total GFlops:"<<flops/(total_proc_time/(double)TEST_NUM)<<endl;
      }
        //}
    }


     
  //cout<<OptimalB(16,1)<<endl;
    //cout<<OptimalB(32,1)<<endl;
    // cout<<OptimalB(64,1)<<endl;
    //    cout<<OptimalB(128,1)<<endl;

  //    cout<<OptimalB(256,1)<<endl;
    //cout<<OptimalB(384,1)<<endl;
    //cout<<OptimalB(512,1)<<endl;
    /*
    cout<<OptimalB(640,1)<<endl;
    cout<<OptimalB(768,1)<<endl;
    cout<<OptimalB(896,1)<<endl;
    cout<<OptimalB(1024,1)<<endl;
    cout<<OptimalB(2048,1)<<endl;
    cout<<OptimalB(3072,1)<<endl;
    cout<<OptimalB(4096,1)<<endl;
    cout<<OptimalB(5120,1)<<endl;
    cout<<OptimalB(6144,1)<<endl;
    cout<<OptimalB(7168,1)<<endl;
    cout<<OptimalB(8192,1)<<endl;
    cout<<OptimalB(9216,1)<<endl;
    cout<<OptimalB(10240,1)<<endl;
    */

}
