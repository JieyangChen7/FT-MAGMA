#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath>
#include<ctime>
//#include"cblas.h"
#include"cublas_v2.h"

#include<curand.h>

//#include "lapacke.h"
#include "acml.h"
//#include "blas.h"
#include "papi.h"

#include <pthread.h>


#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )


#define NUM_THREADS 15

using namespace std;

struct dtrsm_data{
  char SIDE;
  char UPLO;
  char TRANSA;
  char DIAG;
  int M;
  int N;
  double ALPHA;
  double * A;
  int LDA;
  double * B;
  int LDB;
};

struct dgemm_data{
  char TRANSA;
  char TRANSB;
  int M;
  int N;
  int K;
  double ALPHA;
  double * A;
  int LDA;
  double * B;
  int LDB;
  double BETA;
  double * C;
  int LDC;
};

struct dsyrk_data{
    char UPLO;
    char TRANSA;
    int N;
    int K;
    double ALPHA;
    double * A;
    int LDA;
    double BETA;
    double * C;
    int LDC;
};


void printMatrix_host(double * matrix_host, int N);
void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int N);
void matrixGenerator_host(double * matrix, double * result, int N);
void matrixGenerator_gpu(char uplo, double * matrix, int matrix_ld, double * result, int result_ld, int N, int B);
bool resultVerify_host(double * realResult_gpu, double * testResult_gpu, int N);
__global__ void resultVerify_gpu_help(double * realResult,int real_ld, double * testResult,int test_ld,double * diff, int N);
bool resultVerify_gpu(double * realResult,int real_ld, double * testResult, int test_ld, int N, int B);
__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo, double alpha, double beta);
void * threadDtrsm(void * data);
void * threadDgemm(void * data);
void * threadDsyrk(void * data); 
void DPOTRF(char uplo,double * matrix_host,int matrix_host_ld, double * matrix_device, int matrix_device_ld, int N, int B, double k, float * real_time, float * proc_time, long long * flpins, float * mflops){
  
      
  int info = 0;
      
  int update_size = N-B;
  int update_block_host = 0;
  int update_size_host = 0;
  int update_size_device = 0;
  int update_index_host = 0;
  int update_index_device = 0;
  int new_update_index_device = 0;
    
  pthread_t thread[NUM_THREADS];
  int thread_index[NUM_THREADS];
  int thread_length[NUM_THREADS];
  int num_threads_used = 0;
  
    
  //cuda part*******************************
  cudaStream_t stream0;//for computing
  cudaStream_t stream1;//for transfering
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  cublasHandle_t handle0;
  cublasStatus_t cublasStatus;
  cublasStatus = cublasCreate(&handle0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS NOT INITIALIZED(handle0)"<<endl;

  cublasStatus = cublasSetStream(handle0,stream0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS SET STREAM ERROR(handle0)"<<endl;

  //threads data******************************
  
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  
  struct dtrsm_data * tdata = new struct dtrsm_data[NUM_THREADS];
  for(int j=0;j<NUM_THREADS;j++){
    tdata[j].SIDE = 'l';
    tdata[j].UPLO = 'u';
    tdata[j].TRANSA = 't';
    tdata[j].DIAG = 'n';
    tdata[j].ALPHA = 1;
    tdata[j].LDA = matrix_host_ld;
    tdata[j].LDB = matrix_host_ld;
  }

  struct dgemm_data * gdata = new struct dgemm_data[NUM_THREADS];
  for(int j=0;j<NUM_THREADS;j++){
    gdata[j].TRANSA = 't';
    gdata[j].TRANSB = 'n';
    gdata[j].ALPHA = -1;
    gdata[j].LDA = matrix_host_ld;
    gdata[j].LDB = matrix_host_ld;
    gdata[j].BETA = 1;
    gdata[j].LDC = matrix_host_ld;
  }






    if(PAPI_flops(real_time, proc_time, flpins, mflops)<PAPI_OK){
        cout<<"PAPI ERROR"<<endl;
        return;
    }

for(int i=0;i<N;i+=B){
  //int i=0;
  //Determine update index data for CPU and GPU                                                           
  update_size = N - i - B - B;

  update_block_host = int(update_size/B * k);
  update_size_host = update_block_host * B;
  update_size_device = max(update_size - update_size_host,0);
  update_index_host = min(i + B + B,N);
  new_update_index_device = min(update_index_host + update_size_host,N);

  //transfer data to CPU for load balancing                                                             
  if (i>0&&new_update_index_device>update_index_device) {
    //cout<<"Transfering data from GPU to CPU"<<endl;                                                     
    cudaMemcpy2DAsync(matrix_host + update_index_device * matrix_host_ld,
		      matrix_host_ld * sizeof(double),
		      matrix_device + update_index_device * matrix_device_ld,
		      matrix_device_ld * sizeof(double),
		      (i + B) * sizeof(double),
		      B,
		      cudaMemcpyDeviceToHost,
		      stream1);
  }
  //Update index                                                                                          
  update_index_device = new_update_index_device;

  //Do DPOTRF on CPU
  dpotrf(uplo,
         B,
         matrix_host+i*matrix_host_ld+i,
         matrix_host_ld,
         &info);
  
  //transfer dpotrf result to gpu
  cudaMemcpy2DAsync(matrix_device+i*matrix_device_ld+i,
		    matrix_device_ld*sizeof(double),
		    matrix_host+i*matrix_host_ld+i,
		    matrix_host_ld*sizeof(double),
		    B*sizeof(double),
		    B,
		    cudaMemcpyHostToDevice,
		    stream0);
    
  //preparing index data for each thread
  if(update_block_host<=NUM_THREADS){
    num_threads_used = update_block_host;
    for(int j=0;j<num_threads_used;j++){
      thread_index[j] = update_index_host+j*B;
      thread_length[j] = B;
    }
  }else{
    num_threads_used = NUM_THREADS;
    for(int j=0;j<num_threads_used;j++){
      thread_index[j] = update_index_host+floor(j*update_block_host/num_threads_used)*B;
      thread_length[j] = (floor((j+1)*update_block_host/num_threads_used)-floor(j*update_block_host/num_threads_used))*B;
    }
    
  }
 
  //CPU dtrsm**************** 
  //sync balance data
  cudaStreamSynchronize(stream1);
  for(int j=0;j<num_threads_used;j++){
    tdata[j].M = B;
    tdata[j].N = thread_length[j];
    tdata[j].A = matrix_host+i*matrix_host_ld+i;
    tdata[j].B = matrix_host+thread_index[j]*matrix_host_ld+i;
    //cout<<"start dtrsm thread:"<<j<<"["<<thread_index[j]<<","<<thread_length[j]<<"]"<<endl;
    pthread_create(&thread[j], &attr, threadDtrsm, (void *)&tdata[j]);
  }


  
    
  //GPU dtrsm****************
  double talpha = 1;
  cublasDtrsm(handle0,
	      CUBLAS_SIDE_LEFT,
	      CUBLAS_FILL_MODE_UPPER,
	      CUBLAS_OP_T,
	      CUBLAS_DIAG_NON_UNIT,
	      B,
	      update_size_device,
	      &talpha,
	      matrix_device+i*matrix_device_ld+i,
	      matrix_device_ld,
	      matrix_device+update_index_device*matrix_device_ld+i,
	      matrix_device_ld);
  //cout<<"Critical path dtrsm"<<endl;  
  //Critical path
  if(i+B<N){
  dtrsm('l',
        'u',
        't',
        'n',
        B,
        B,
        1,
        matrix_host+i*matrix_host_ld+i,
        matrix_host_ld,
        matrix_host+(i+B)*matrix_host_ld+i,
        matrix_host_ld
        ); 
  }
  if(i+B<N){

    //cout<<"Transfer a block to GPU for DGEMM later"<<endl; 
    cudaMemcpy2DAsync(matrix_device+(i+B)*matrix_device_ld,
		      matrix_device_ld*sizeof(double),
		      matrix_host+(i+B)*matrix_host_ld,
		      matrix_host_ld*sizeof(double),
		      (i+B)*sizeof(double),
		      B,
		      cudaMemcpyHostToDevice,
		      stream0);
  }
  
  //CPU synchronize****
  pthread_attr_destroy(&attr);
  for(int j=0;j<num_threads_used;j++){
    //cout<<"waiting dtrsm thread"<<j<<endl;
    pthread_join(thread[j], NULL);
    //cout<<"completed dtrsm thread:"<<j<<endl;
  }
 
  //CPU DGEMM
  for(int j=0;j<num_threads_used;j++){
    
    gdata[j].M = B;
    gdata[j].N = thread_length[j];
    gdata[j].K = i+B;
    gdata[j].A = matrix_host+(i+B)*matrix_host_ld;
    gdata[j].B = matrix_host+thread_index[j]*matrix_host_ld;
    gdata[j].C = matrix_host+thread_index[j]*matrix_host_ld+B+i;
    //cout<<"start dgemm thread:"<<j<<endl;
    pthread_create(&thread[j], &attr, threadDgemm, (void *)&gdata[j]);
  }
  


  //cout<<"CUBLAS DGEMM"<<endl;
  //GPU DGEMM
  double galpha = -1;
  double gbeta = 1;
  cublasDgemm(handle0,
	      CUBLAS_OP_T,
	      CUBLAS_OP_N,
	      B,
	      update_size_device,
	      i+B,
	      &galpha,
	      matrix_device+(i+B)*matrix_device_ld,
	      matrix_device_ld,
	      matrix_device+update_index_device*matrix_device_ld,
	      matrix_device_ld,
	      &gbeta,
	      matrix_device+update_index_device*matrix_device_ld+B+i,
	      matrix_device_ld
	      );
  
  //cout<<"Critical Path dsyrk"<<endl;
  if(i+B<N){
  dsyrk('u',
	't',
	B,
	i+B,
	-1,
        matrix_host+(i+B)*matrix_host_ld,
	matrix_host_ld,
	1,
	matrix_host+(i+B)*matrix_host_ld+i+B,
        matrix_host_ld
	);
  }




  //CPU Synchronize
  pthread_attr_destroy(&attr);
  for(int j=0;j<num_threads_used;j++){
    //cout<<"waiting dgemm thread"<<j<<endl;
    pthread_join(thread[j], NULL);
    //cout<<"completed dgemm thread:"<<j<<endl;
  }

  

}



    if(PAPI_flops(real_time, proc_time, flpins, mflops)<PAPI_OK){
        cout<<"PAPI ERROR"<<endl;
        return;
    }
    //cout<<"ENDING dpotrf"<<endl;
    
  cublasDestroy(handle0);
  
  PAPI_shutdown();

}





void test(int N, int B, double k, float * real_time, float * proc_time, long long * flpins, float * mflops){
  double * matrix_host = new double[N*N]();
  double * matrix_device;
  double * result_device;

  size_t matrix_pitch;
  size_t result_pitch;

  cudaMallocPitch((void**)&matrix_device,&matrix_pitch,N*sizeof(double),N);
  cudaMallocPitch((void**)&result_device,&result_pitch,N*sizeof(double),N);

  int matrix_ld= matrix_pitch/sizeof(double);
  int result_ld= result_pitch/sizeof(double);
  
  //cout<< matrix_pitch<<"  "<<result_pitch<<"  "<<matrix_ld<<"  "<<result_ld<<endl;
  matrixGenerator_gpu('u',matrix_device,matrix_ld,result_device,result_ld, N, 2);
  
  cudaMemcpy2D(matrix_host,N*sizeof(double),matrix_device,matrix_pitch,N*sizeof(double),N,cudaMemcpyDeviceToHost);

  // printMatrix_gpu(matrix_device,matrix_pitch,N);
  //cout<<"RESULT:"<<endl;
  //printMatrix_gpu(result_device, result_pitch,N);

  //printMatrix_host(matrix_host,N);

  DPOTRF('u',matrix_host, N, matrix_device, matrix_ld, N, B, k, real_time, proc_time, flpins, mflops);
  
  
   //cout<<"CPU:"<<endl;
   //printMatrix_host(matrix_host,N); 

   //cout<<"GPU:"<<endl;
   //printMatrix_gpu(matrix_device, matrix_pitch,N);
  

  //Verify result
  //  if(resultVerify_gpu(result_device,result_ld,matrix_device,matrix_ld,N,2)){
    //cout<<"Result passed!"<<endl;
  // }else{
  //   cout<<"Result failed!"<<endl;
  // }

  
  cudaFree(matrix_device);
  cudaFree(result_device);
  delete[] matrix_host;
}


int main(int argc, char *argv[]){
  
  for(int n=1024;n<10241;n+=1024){
    //float best_time = 10000;
    //int best_b = 0;
    //double best_k = 0;
    for(int b=16;b<1024;b*=2){
	for(double k=0.1;k<0.2;k+=0.1){
	  float real_time=0.0;
	  float proc_time=0.0;
	  long long flpins=0; 
	  float  mflops=0.0;
	  test(n,b,k, &real_time, &proc_time, &flpins, &mflops);
	  /*if(proc_time<best_time){
	    best_time = proc_time;
	    best_b = b;
	    best_k = k;
	  }
	  */
	  cout<<"Size:"<<n<<"("<<b<<")k="<<k<<"---gflops:"<<((double)FLOPS_DPOTRF(n)/1e9)/proc_time<<endl;
	}
    }
    // cout<<"Size:"<<n<<"("<<best_b<<")k="<<best_k<<"---Proc_time:"<<best_time<<endl;
  }
  
  // pthread_exit(NULL); Never put here when using CUDA
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
  //printMatrix_gpu(result,result_ld*sizeof(double), N);
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
  //printMatrix_gpu(matrix,matrix_ld*sizeof(double),N);
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
	cout<<"["<<i<<","<<j<<"]"<<"diff:"<<abs(diff_host[i*N+j])<<endl;
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


void * threadDtrsm(void * data){
  struct dtrsm_data * current_data = (struct dtrsm_data * ) data;
  dtrsm(current_data->SIDE,
	current_data->UPLO,
	current_data->TRANSA,
	current_data->DIAG,
	current_data->M,
	current_data->N,
	current_data->ALPHA,
	current_data->A,
	current_data->LDA,
	current_data->B,
	current_data->LDB
	);
  pthread_exit(NULL);

}


void * threadDgemm(void * data){
  struct dgemm_data * current_data = (struct dgemm_data * ) data;
  dgemm(current_data->TRANSA,
        current_data->TRANSB,
        current_data->M,
        current_data->N,
        current_data->K,
        current_data->ALPHA,
        current_data->A,
        current_data->LDA,
        current_data->B,
        current_data->LDB,
        current_data->BETA,
	current_data->C,
	current_data->LDC
        );
  pthread_exit(NULL);

}

void * threadDsyrk(void * data){
    struct dsyrk_data * current_data = (struct dsyrk_data * ) data;
    dsyrk(current_data->UPLO,
          current_data->TRANSA,
          current_data->N,
          current_data->K,
          current_data->ALPHA,
          current_data->A,
          current_data->LDA,
          current_data->BETA,
          current_data->C,
          current_data->LDC
          );
    pthread_exit(NULL);
    
}

