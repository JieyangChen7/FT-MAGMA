#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath>
#include<ctime>
#include"cblas.h"
#include"cublas_v2.h"

#include<curand.h>

#include "lapacke.h"
#include "blas.h"
#include "papi.h" 

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )

using namespace std;

void printMatrix_host(double * matrix_host, int N);
void printMatrix_gpu(double * matrix_gpu, int N);
void matrixGenerator_host(double * matrix, double * result, int N);
void matrixGenerator_gpu(char uplo, double * matrix, double * result, int N, int B);
bool resultVerify_host(double * realResult_gpu, double * testResult_gpu, int N);
__global__ void resultVerify_gpu_help(double * realResult, double * testResult,double * diff, int N);
bool resultVerify_gpu(double * realResult, double * testResult, int N, int B);
__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo, double alpha, double beta);
void POTF2_CPU(char uplo, double * matrix, int ld, int B);
__global__ void POTF2_cublas_helper(double * matrix, int ld);
void POTF2_cublas_full(double * matrix, int ld, int b_size);
void POTF2_cublas_magma(double * matrix, int ld, int b_size);
__global__ void POTF2_register(double * matrix, int ld);
__global__ void POTF2_register_magma(double * matrix, int ld);
__global__ void POTF2_shared(double * matrix, int ld, int b_size);
__global__ void POTF2_shared_magma(double * matrix, int ld, int b_size);
__global__ void POTF2(double * matrix, int ld, int B);
__global__ void POTF2_magma(double * matrix, int ld, int B);
__global__ void TRSM(char uplo, double * a, int lda, double * b, int ldb, int B, int l);
__global__ void RKSY(bool trans,bool triangular, double * b1, int ldb1, double * b2, int ldb2, double * c, int ldc, int i, int j,int k);
void my_dpotrf145(char uplo, double * matrix, int ld, int B, int b, bool debug, bool useCublas);
void my_dpotrf146(char uplo, double * matrix, int ld, int B);



double test(int c, int N, int B, char uplo, float * real_time, float * proc_time, long long * flpins, float * mflops){
    double * matrix;
    double * result;
    double * temp = new double[N*N]();
    cudaMalloc((void**)&matrix,N*N*sizeof(double));
    cudaMalloc((void**)&result,N*N*sizeof(double));
    cudaFuncSetCacheConfig(POTF2_shared,cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(POTF2_shared_magma,cudaFuncCachePreferShared);
    int b = 2;
    int info=0;
    //float real_time, proc_time, mflops;
    //long long flpins;
    matrixGenerator_gpu(uplo, matrix, result, N, B);
    
    if(PAPI_flops(real_time, proc_time, flpins, mflops)<PAPI_OK){
      cout<<"PAPI ERROR"<<endl;
      return -1;
    }

    //clock_t t = clock();
    
    

    // if(c == 0){
       
    //cout<<"LAPACK_dpotrf:";
    
    cudaMemcpy2D(temp,N*sizeof(double),matrix,N*sizeof(double),N*sizeof(double),N,cudaMemcpyDeviceToHost);
      LAPACK_dpotrf(&uplo,&N,temp,&N,&info);
      cudaMemcpy2D(matrix,N*sizeof(double),temp,N*sizeof(double),N*sizeof(double),N,cudaMemcpyHostToDevice);
      
      /* }else if(c == 1){
    
      cout<<"CPU:";
      
    cudaMemcpy2D(temp,N*sizeof(double),matrix,N*sizeof(double),N*sizeof(double),N,cudaMemcpyDeviceToHost);
    POTF2_CPU(uplo, temp, N, N);
    cudaMemcpy2D(matrix,N*sizeof(double),temp,N*sizeof(double),N*sizeof(double),N,cudaMemcpyHostToDevice);
       
    }else if(c == 2){
	 
     cout<<"GPU:";
    
  POTF2<<<dim3(1),dim3(N)>>>(matrix, N, N);
      
    }else if(c == 3){
    
    cout<<"GPU[magma]:";
      POTF2_magma<<<dim3(1),dim3(N)>>>(matrix, N, N);
      
    }else if(c == 4){
      
    cout<<"Shared:";
    POTF2_shared<<<dim3(1),dim3(N),N*N*sizeof(double)>>>(matrix, N, N);  
      
    }else if(c == 5){
      
    cout<<"Shared[magma]:";
    POTF2_shared_magma<<<dim3(1),dim3(N),N*N*sizeof(double)>>>(matrix, N, N);
      
    }else if(c == 6){
    
    cout<<"CUBLAS:";
      POTF2_cublas_full(matrix, N, N);
      
    }else if(c == 7){
      
    cout<<"CUBLAS[magma]:";
      POTF2_cublas_magma(matrix, N, N);
      
    }else 
    
    if(c == 8){
      
    cout<<"Register:";
      POTF2_register<<<dim3(1),dim3(1)>>>(matrix, N);
     
    }else if(c == 9){
     
    cout<<"Register[magma]:";
      POTF2_register_magma<<<dim3(1),dim3(1)>>>(matrix, N);
    }

      
    cudaDeviceSynchronize();
      */  
      /*  t=clock()-t;
    
    double time_in_sec =((double)t/(double)CLOCKS_PER_SEC);
    double gflop = (double)FLOPS_DPOTRF(N)/(double)1000000000.0;
    double gflops = (double)gflop/(double)time_in_sec;
    cout<<"N:"<<N<<";B:"<<B<<"----";
    cout<<"Time:"<<time_in_sec<<"s, "<<gflops<<"GFlops/s."<<endl;
      */
    if(PAPI_flops( real_time, proc_time, flpins, mflops)<PAPI_OK){
      cout<<"PAPI ERROR"<<endl;
      return -1;
    }
    
    //cout<<"Real_time:"<<real_time<<endl<<"Proc_time:"<<proc_time<<endl<<"Total flpins:"<<flpins<<endl<<"MFLOPS:"<<mflops<<endl;



    //    if(resultVerify_host(result,matrix,N)){
      //cout<<"Result passed!"<<endl;
    // }else{
    //  cout<<"Result failed!"<<endl;
    // }
    delete[] temp;
    cudaFree(matrix);
    cudaFree(result);
    PAPI_shutdown();
    return *mflops;

}






int main(){
  
  float real_time = 0.0;
  float proc_time = 0.0;
  long long flpins = 0.0;
  float mflops = 0.0;

  float total_real_time = 0.0;
  float total_proc_time = 0.0;
  long long total_flpins = 0.0;
  float total_mflops = 0.0;
  

  for(int n=64;n<1025;n*=2){
    for(int i=0;i<100000;i++){
      test(0,n,4,'u',&real_time,&proc_time,&flpins,&mflops);
      total_real_time += real_time;
      total_proc_time += proc_time;
      total_flpins += flpins;
      total_mflops += mflops;
    }
    cout<<"Size:"<<n<<"---Real_time:"<<total_real_time/(double)100000<<"---"<<"Proc_time:"<<total_proc_time/(double)100000<<"---"<<"Total flpins:"<<total_flpins/(double)100000<<"---"<<"MFLOPS:"<<total_mflops/(double)100000<<endl;
  }
  
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

void printMatrix_gpu(double * matrix_gpu, int N){
    double * matrix_host = new double[N*N]();
    cudaMemcpy(matrix_host,matrix_gpu,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
    printMatrix_host(matrix_host,N);
}

//Upper only
void matrixGenerator_host(double * matrix, double * result, int N){
    double * A = new double[N*N]();
    double * At = new double[N*N]();
    double * matrix_host_upper = new double[N*N]();
    double * matrix_host_lower = new double[N*N]();

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(j<i+1)
	      A[i*N+j] = (rand()%100000)/(double)10000+(double)1;
            else
                A[i*N+j] = 0;
            At[j*N+i] = A[i*N+j];
        }
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<i+1;j++){
            for(int k=0;k<N;k++){
                if(j<i+1)
                    matrix_host_upper[i*N+j]+=A[i*N+k]*At[k*N+j];
                if(j>i)
                    matrix_host_lower[i*N+j]+=A[i*N+k]*At[k*N+j];
            }
            if(matrix_host_upper[i*N+j]<0||matrix_host_lower[i*N+j]<0)
                cout<<"Matrix generate Error!"<<endl;
        }
    }
    
    cudaMemset(matrix,0,N*N*sizeof(double));
    cudaMemset(result,0,N*N*sizeof(double));
    cudaMemcpy(matrix,matrix_host_upper,sizeof(double)*N*N,cudaMemcpyHostToDevice);
    cudaMemcpy(result,A,sizeof(double)*N*N,cudaMemcpyHostToDevice);    
    /*
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            result[i*N+j]=A[i*N+j];
        }
    }
     */
    free(A);
    free(At);
}

void matrixGenerator_gpu(char uplo, double * matrix, double * result, int N, int B){
    double a = 10.0;
    //initialize cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    //initialize curand
    curandGenerator_t gen;
    curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,10ULL);
    //generate random number in range (0,1] on result using curand
    curandGenerateUniformDouble(gen,result,N*N);
    cudaDeviceSynchronize();
    //  printMatrix_gpu(result,N);
    matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(result, N, uplo, a,1);
    cudaDeviceSynchronize();
    //printMatrix_gpu(result,N);
    //do matrix-matrix multiplcation using cublas
    cudaMemset(matrix,0,N*N*sizeof(double));
   

    double alpha = 1.0;
    double beta = 1.0;
    if(uplo == 'u'){
        cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,&alpha,result,N,result,N,&beta,matrix,N);
    }
    else if(uplo == 'l'){
        cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,N,N,N,&alpha,result,N,result,N,&beta,matrix,N);
    }
    cudaDeviceSynchronize();
    //    printMatrix_gpu(matrix,N);
    matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, N, uplo, 1.0,0);
    cudaDeviceSynchronize();
    // printMatrix_gpu(matrix,N);
}

bool resultVerify_host(double * realResult_gpu, double * testResult_gpu, int N){
    double * realResult = new double[N*N]();
    double * testResult = new double[N*N]();
    cudaMemcpy(realResult,realResult_gpu,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
    cudaMemcpy(testResult,testResult_gpu,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
    //     printMatrix_host(testResult,N);
    // printMatrix_host(realResult,N);
    bool pass = true;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
	  if(abs(realResult[i*N+j]-testResult[i*N+j])>0){
	    // cout<<"["<<i<<","<<j<<"]:"<<realResult[i*N+j]<<":"<<testResult[i*N+j]<<"="<<realResult[i*N+j]-testResult[i*N+j]<<endl;  
	      pass = false;
            }
        }
    }
    return pass;
}

__global__ void resultVerify_gpu_help(double * realResult, double * testResult,double * diff, int N){
    int col = threadIdx.x+blockIdx.x*blockDim.x;
    int row = threadIdx.y+blockIdx.y*blockDim.y;
    diff[col*N+row] = testResult[col*N+row] - realResult[col*N+row];
}

bool resultVerify_gpu(double * realResult, double * testResult, int N, int B){
    double * diff;
    cudaMalloc((void**)&diff,N*N*sizeof(double));
    resultVerify_gpu_help<<<dim3(N/B,N/B),dim3(B,B)>>>(realResult,testResult,diff,N);
    
    double * diff_host = new double[N*N]();
    cudaMemcpy(diff_host,diff,N*N*sizeof(double),cudaMemcpyDeviceToHost);
    //  printMatrix(diff_host,N);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if(abs(diff_host[i*N+j])>1e-3){
                //  cout<<"diff:"<<abs(diff_host[i*N+j])<<endl;
                return false;
            }
        }
    }
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
__global__ void POTF2_cublas_helper(double * matrix, int ld){
  int id = threadIdx.x;
  if(id == 0){
    matrix[0] = sqrt(matrix[0]);
  }
  __syncthreads();
  if(id != 0){
    matrix[id*ld] /=matrix[0];
  }
}


void POTF2_cublas_full(double * matrix, int ld, int b_size){
  cudaStream_t stream0;                                  
  cudaStreamCreate(&stream0);
  cublasStatus_t cublasStatus;
  cublasHandle_t handle0;
  cublasStatus = cublasCreate(&handle0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS NOT INITIALIZED(handle0)"<<endl;
  cublasStatus = cublasSetStream(handle0,stream0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS SET STREAM NOT INITIALIZED(handle0)"<<endl;
  double alpha = -1;
  double beta = 1;
  
  for(int i=0;i<b_size;i++){
  //  int i=0;
    int current_size = b_size - i;
    POTF2_cublas_helper<<<dim3(1),dim3(current_size),0,stream0>>>(matrix+i*ld+i,ld);
     if(i<b_size-1)
    cublasDgemm(handle0,CUBLAS_OP_T,CUBLAS_OP_N,current_size-1,current_size-1,1,&alpha,matrix+(i+1)*ld+i,ld,matrix+(i+1)*ld+i,ld,&beta,matrix+(i+1)*ld+(i+1),ld);
     }
  
  cudaStreamSynchronize(stream0);
  cublasDestroy(handle0);
  cudaStreamDestroy(stream0);

}

void POTF2_cublas_magma(double * matrix, int ld, int b_size){
  cudaStream_t stream0;
  cudaStreamCreate(&stream0);
  cublasStatus_t cublasStatus;
  cublasHandle_t handle0;
  cublasStatus = cublasCreate(&handle0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS NOT INITIALIZED(handle0)"<<endl;
  cublasStatus = cublasSetStream(handle0,stream0);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS)
    cout<<"CUBLAS SET STREAM NOT INITIALIZED(handle0)"<<endl;
  double alpha = -1;
  double beta = 1;

  for(int i=0;i<b_size;i++){
    //  int i=0;                                                                                                    
    int current_size = b_size - i;
    POTF2_cublas_helper<<<dim3(1),dim3(current_size),0,stream0>>>(matrix+i*ld+i,ld);
    if(i<b_size-1)
      cublasDgemm(handle0,CUBLAS_OP_T,CUBLAS_OP_N,1,current_size-1,i+1,&alpha,matrix+(i+1)*ld,ld,matrix+(i+1)*ld,ld,&beta,matrix+(i+1)*ld+(i+1),ld);
  }

  cudaStreamSynchronize(stream0);
  cublasDestroy(handle0);
  cudaStreamDestroy(stream0);

}
//gpu version
__global__ void POTF2(double * matrix, int ld, int B){  
  int id = threadIdx.x;
    for(int i = 0; i<B;i++){
        if(id==i){
            matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
        }
        __syncthreads();
        if(id>i&&id<B){
            matrix[id*ld+i] /= matrix[i*ld+i];
            __syncthreads();
            for(int j=i+1;j<id+1;j++){
                matrix[id*ld+j]-=matrix[id*ld+i]*matrix[j*ld+i];
            }
        }
        __syncthreads();
    }
}

//gpu version(MAGMA way)
__global__ void POTF2_magma(double * matrix, int ld, int B){
  int id = threadIdx.x;
  for(int i = 0; i<B;i++){
    if(id==i){
      matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
    }
    __syncthreads();
    if(id>i&&id<B){
      matrix[id*ld+i] /= matrix[i*ld+i];
      __syncthreads();
      for(int j=0;j<i+1;j++){
	matrix[id*ld+i+1]-=matrix[(i+1)*ld+j]*matrix[id*ld+j];
      }
    }
    __syncthreads();
  }
}
//gpu register single thread
__global__ void POTF2_register(double * matrix, int ld){
  register  double localMatrix[256*256];
  int b_size = 256;
  for(int i=0;i<b_size;i++){
    for(int j=0;j<b_size;j++){
    localMatrix[i*b_size+j] = matrix[i*ld+j];
    }
  }
  
  for(int i = 0; i<b_size;i++){
    localMatrix[i*b_size+i] = sqrt(localMatrix[i*b_size+i]);
    for(int j = i+1;j<b_size;j++){
      localMatrix[j*b_size+i] /= localMatrix[i*b_size+i];
    }
    for(int j = i+1;j<b_size;j++){
      for(int k=i+1;k<j+1;k++){	
	localMatrix[j*b_size+k]-=localMatrix[k*b_size+i]*localMatrix[j*b_size+i];
      }
    }
    
  }
  for(int i=0;i<b_size;i++){
    for(int j=0;j<b_size;j++){
      matrix[i*ld+j] = localMatrix[i*b_size+j];
    }
  }
 
}
//GPU register single thread(MAGMA way) 
__global__ void POTF2_register_magma(double * matrix, int ld){
  register  double localMatrix[256*256];
  int b_size = 256;
  for(int i=0;i<b_size;i++){
    for(int j=0;j<b_size;j++){
      localMatrix[i*b_size+j] = matrix[i*ld+j];
    }
  }

  for(int i = 0; i<b_size;i++){
    localMatrix[i*b_size+i] = sqrt(localMatrix[i*b_size+i]);
    for(int j = i+1;j<b_size;j++){
      localMatrix[j*b_size+i] /= localMatrix[i*b_size+i];
    }
    for(int j=i+1;j<b_size;j++){
      for(int k=0;k<i+1;k++){
	localMatrix[j*b_size+i+1]-=localMatrix[(i+1)*b_size+k]*localMatrix[j*b_size+k];
      }
    }
  }
  for(int i=0;i<b_size;i++){
    for(int j=0;j<b_size;j++){
      matrix[i*ld+j] = localMatrix[i*b_size+j];
    }
  }

}

__global__ void POTF2_shared(double * matrix, int ld, int b_size){
  extern __shared__ double localMatrix[];
  int id = threadIdx.x;
  for(int i=0;i<b_size;i++){
    localMatrix[id*b_size+i] = matrix[id*ld+i];
  }
  __syncthreads();
  for(int i = 0; i<b_size;i++){
    if(id==i){
      localMatrix[i*b_size+i] = sqrt(localMatrix[i*b_size+i]);
    }
    __syncthreads();
    if(id>i&&id<b_size){
      localMatrix[id*b_size+i] /= localMatrix[i*b_size+i];
      //matrix[id*ld+i] = matrix[i*ld+id];                                                               
      __syncthreads();
      for(int j=i+1;j<id+1;j++){
	localMatrix[id*b_size+j]-=localMatrix[id*b_size+i]*localMatrix[j*b_size+i];
      }
    }
    __syncthreads();
  }
  for(int i=0;i<b_size;i++){
    matrix[id*ld+i] = localMatrix[id*b_size+i];
  }
}


__global__ void POTF2_shared_magma(double * matrix, int ld, int b_size){
  extern __shared__ double localMatrix[];
  int id = threadIdx.x;
  for(int i=0;i<b_size;i++){
    localMatrix[id*b_size+i] = matrix[id*ld+i];
  }
  __syncthreads();
  for(int i = 0; i<b_size;i++){
    if(id==i){
      localMatrix[i*b_size+i] = sqrt(localMatrix[i*b_size+i]);
    }
    __syncthreads();
    if(id>i&&id<b_size){
      localMatrix[id*b_size+i] /= localMatrix[i*b_size+i];
      __syncthreads();
      for(int j=0;j<i+1;j++){
        localMatrix[id*b_size+i+1]-=localMatrix[(i+1)*b_size+j]*localMatrix[id*b_size+j];
      }
    }
    __syncthreads();
  }
  for(int i=0;i<b_size;i++){
    matrix[id*ld+i] = localMatrix[id*b_size+i];
  }
}

//a->A; b->B; B->block size, column of B&A; l->row of B
__global__ void TRSM(char uplo, double * a, int lda, double * b, int ldb, int B, int l){
    
    register int id = threadIdx.x+blockIdx.x*blockDim.x;
    if(uplo == 'l'){
        register int row = id;
        for(int j=0;j<B;j++){
            register double sum = 0;
            for(int k=0;k<j;k++){
                sum+=b[k*ldb+row]*a[k*lda+j];
            }
            b[j*ldb+row]-=sum;
            b[j*ldb+row]/=a[j*lda+j];
        }
    }
    else if(uplo == 'u'){
        register int col=id;
        for(int j=0;j<B;j++){
            register double sum = 0;
            for(int k=0;k<j;k++){
                sum+=b[col*ldb+k]*a[j*lda+k];
            }
            b[col*ldb+j]-=sum;
            b[col*ldb+j]/=a[j*lda+j];
        }
    }
}





__global__ void RKSY(bool trans,bool triangular, double * b1, int ldb1, double * b2, int ldb2, double * c, int ldc, int i, int j,int k){
    register int col = threadIdx.x+blockIdx.x*blockDim.x;
    register int row = threadIdx.y+blockIdx.y*blockDim.y;
    register int local_col = threadIdx.x;
    register int local_row = threadIdx.y;
    register int b = blockDim.x;
    register int local_col_b = local_col*b;
    register int local_row_b = local_row*b;
    register int local_col_b_local_row = local_col_b+local_row;
    register int local_row_b_local_col = local_row_b+local_col;
    extern __shared__ double bs[];
    register double * b1s = bs;
    register double * b2s = &bs[b*b];
    //C = C - (B^T)*B for upper
    if(trans){
        register int row_ldb1 = row*ldb1;
        register int col_ldb2 = col*ldb2;
        register double sum = 0;
        for(int m=0;m<k;m+=b){
            b1s[local_row_b_local_col] = b1[row_ldb1+m+local_col];
            b2s[local_col_b_local_row] = b2[col_ldb2+m+local_row];
            __syncthreads();
            for(int s=0;s<b;s+=2){
                //sum+=b1s[local_row*b+s]*b2s[local_col*b+s];
                register int local_row_b_s = local_row_b+s;
                register int local_col_b_s = local_col_b+s;
                register double b11 = b1s[local_row_b_s];
                register double b12 = b1s[local_row_b_s+1];
                register double b21 = b2s[local_col_b_s];
                register double b22 = b2s[local_col_b_s+1];
                sum+=b11*b21;
                sum+=b12*b22;
            }
            __syncthreads();
        }
        if(triangular&&row<col+1)
            c[col*ldc+row]-=sum;
        else if(!triangular)
            c[col*ldc+row]-=sum;
    }
    //C = C - B*(B^T) for lower
    else{
        
        register double sum = 0;
        for(int m=0;m<k;m+=b){
            b1s[local_col_b_local_row] = b1[(m+local_col)*ldb1+row];
            b2s[local_row_b_local_col] = b2[(m+local_row)*ldb2+col];
            __syncthreads();
            for(int s=0;s<b;s++)
                sum+=b1s[s*b+local_row]*b2s[s*b+local_col];
            __syncthreads();
        }
        if(triangular&&col<row+1)
            c[col*ldc+row]-=sum;
        else if(!triangular)
            c[col*ldc+row]-=sum;
    }
    
}



void my_dpotrf145(char uplo, double * matrix, int ld, int B, int b, bool debug, bool useCublas){
    //for debug use
    /*
     double rksy1_time=0;
     double rksy2_time=0;
     double cpu_time=0;
     double trsm_time=0;
     */
    //initial data
    double * temp;
    cudaHostAlloc((void**)&temp,B*B*sizeof(double),cudaHostAllocDefault);
    //intial streams----------------------------
    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    //intial cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle,stream0);
    
    
    if(!useCublas){
        if(uplo == 'u'){
            //start the loop of calculation----------------------------
            for(int i=0;i<ld;i+=B){
                //update A
                if(i!=0){
                    RKSY<<<dim3(B/b,B/b),dim3(b,b),b*b*2*sizeof(double),stream0>>>(true,true,matrix+i*ld,ld,matrix+i*ld,ld,matrix+i*ld+i,ld,B,B,i);
                }
                cudaStreamSynchronize(stream0);
                cudaMemcpy2DAsync(temp,B*sizeof(double),matrix+i*ld+i,ld*sizeof(double),B*sizeof(double),B,cudaMemcpyDeviceToHost,stream1);
                if(i!=0&&i+B<ld){
                    RKSY<<<dim3((ld-i-B)/b,B/b),dim3(b,b),b*b*2*sizeof(double),stream0>>>(true,false,matrix+i*ld,ld,matrix+(i+B)*ld,ld,matrix+(i+B)*ld+i,ld,B,ld-i-B,i);
                }
                //factorize A on CPU
                //printMatrix(temp, B);
                cudaStreamSynchronize(stream1);
                POTF2_CPU(uplo,temp,B,B);
                //printMatrix(temp,B);
                cudaMemcpy2DAsync(matrix+i*ld+i,ld*sizeof(double),temp,B*sizeof(double),B*sizeof(double),B,cudaMemcpyHostToDevice,stream1);
                
                //update B
                if(i+B<ld){
                    cudaStreamSynchronize(stream1);
                    TRSM<<<dim3(int(ld-i-B)/b),dim3(b),0,stream0>>>(uplo,matrix+i*ld+i,ld,matrix+(i+B)*ld+i,ld,B,ld-i-B);
                }
            }
            
            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);
            
        }
        if(uplo == 'l'){
            //start the loop of calculation----------------------------
            for(int i=0;i<ld;i+=B){
                //update A
                if(i!=0){
                    RKSY<<<dim3(B/b,B/b),dim3(b,b),b*b*2*sizeof(double),stream0>>>(false,true,matrix+i,ld,matrix+i,ld,matrix+i*ld+i,ld,B,B,i);
                }
                cudaStreamSynchronize(stream0);
                cudaMemcpy2DAsync(temp,B*sizeof(double),matrix+i*ld+i,ld*sizeof(double),B*sizeof(double),B,cudaMemcpyDeviceToHost,stream1);
                if(i!=0&&i+B<ld){
                    RKSY<<<dim3(B/b,(ld-i-B)/b),dim3(b,b),b*b*2*sizeof(double),stream0>>>(false,false,matrix+i+B,ld,matrix+(i),ld,matrix+(i)*ld+i+B,ld,ld-i-B,B,i);
                }
                //factorize A on CPU
                //printMatrix(temp, B);
                cudaStreamSynchronize(stream1);
                POTF2_CPU(uplo,temp,B,B);
                //printMatrix(temp,B);
                cudaMemcpy2DAsync(matrix+i*ld+i,ld*sizeof(double),temp,B*sizeof(double),B*sizeof(double),B,cudaMemcpyHostToDevice,stream1);
                //cudaStreamSynchronize(stream1);
                //update B
                if(i+B<ld){
                    cudaStreamSynchronize(stream1);
                    //cudaFuncSetCacheConfig(TRSM,cudaFuncCachePreferShared);
                    TRSM<<<dim3(int(ld-i-B)/b),dim3(b),0,stream0>>>(uplo,matrix+i*ld+i,ld,matrix+(i)*ld+i+B,ld,B,ld-i-B);
                }
            }
            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);
            
        }
    }
    if(useCublas){
        
        if(uplo == 'u'){
            //start the loop of calculation----------------------------
            for(int i=0;i<ld;i+=B){
                //update A
                if(i>0){
                    
                    double alpha = -1;
                    double beta = 1;
                    cublasDsyrk(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,B,i,&alpha,matrix+i*ld,ld,&beta,matrix+i*ld+i,ld);
                }
                cudaStreamSynchronize(stream0);
                cudaMemcpy2DAsync(temp,B*sizeof(double),matrix+i*ld+i,ld*sizeof(double),B*sizeof(double),B,cudaMemcpyDeviceToHost,stream1);
                if(i!=0&&i+B<ld){
                    double alpha = -1;
                    double beta = 1;
                    cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,B,ld-i-B,i,&alpha,matrix+i*ld,ld,matrix+(i+B)*ld,ld,&beta,matrix+(i+B)*ld+i,ld);
                }
                //factorize A on CPU
                //printMatrix(temp, B);                                                             
                cudaStreamSynchronize(stream1);
                POTF2_CPU(uplo,temp,B,B);
                //printMatrix(temp,B);                                                              
                cudaMemcpy2DAsync(matrix+i*ld+i,ld*sizeof(double),temp,B*sizeof(double),B*sizeof(double),B,cudaMemcpyHostToDevice,stream1);
                //update B                                                                      
                if(i+B<ld){
                    //cudaStreamSynchronize(stream1);
                    //TRSM<<<dim3(int(ld-i-B)/b),dim3(b),0,stream0>>>(uplo,matrix+i*ld+i,ld,matrix+(i+B)*ld+i,ld,B,ld-i-B);
                    double alpha = 1;
                    cublasDtrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,B,ld-i-B,&alpha,matrix+i*ld+i,ld,matrix+(i+B)*ld+i,ld);
                }
            }
            
            cudaStreamSynchronize(stream0);
            cudaStreamSynchronize(stream1);
            cublasDestroy(handle);
        }
        
        
    }
    
    
}


void my_dpotrf146(char uplo, double * matrix, int ld, int B){
  clock_t t1 = clock();
    //initial data
    int b_size = B;
    double * temp;
    //double * work;
    //    float gemm_time = 0;
    //float cpu_time = 0;
    //float gemm_time_total = 0;
    //float cpu_time_total = 0;
    //float syrk_time_total = 0;
    //float trsm_time_total = 0;
    cudaHostAlloc((void**)&temp,B*B*sizeof(double),cudaHostAllocDefault);
    //cout<<"pinned memory initialized"<<endl;
    //intial streams----------------------------
    cudaStream_t stream0;//for main loop
    cudaStream_t stream1;//for dgemm part
    //  cudaStream_t stream2;//for cpu part
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
        float t=0;
    //cout<<"Inital complete"<<endl;
    t1 = clock() - t1;
    cout<<"initialize time:"<<((double)t1/CLOCKS_PER_SEC)*1000<<endl;
    t1 = clock();
    if(uplo == 'u'){
      //cout<<"entering loop"<<endl;
      //start the loop of calculation----------------------------
      for(int i=0;i<ld;i+=B){
	//printMatrix_gpu(matrix,ld);
	//cout<<"loop"<<i<<endl;
	//if(i!=0){
	//  if(gemm_time>cpu_time)
	//    B+=20;
	//  else
	//    B-=20;
	//}
	//if(B<=0)
	//  B=1;
	
	//B = getDynamicBlockSize(i,ld, B);
	
	//b_size = min(B,ld-i);
	
	//cout<<"block size:"<<b_size<<"  ";
	
	//update A
	
	if(i>0){
	  double alpha = -1;
	  double beta = 1;
	  //t = clock();
	  //cudaEventRecord(start0,stream0);
	  cublasDsyrk(handle0,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,b_size,i,&alpha,matrix+i*ld,ld,&beta,matrix+i*ld+i,ld);
	  //cudaEventRecord(stop0,stream0);
	  
	  //******************
	  //cudaStreamSynchronize(stream0); 
	  //cudaEventElapsedTime(&t,start0,stop0);
	  //t = clock() - t;
          //syrk_time_total += t;                                                                                                                                           
          //cout<<"SYRK: "<<t<<"ms  ";
	  //*********************

	  //cudaHostAlloc((void**)&work,b_size*(i+b_size)*sizeof(double),cudaHostAllocDefault);
	  //cudaMemcpy2DAsync(work,(i+b_size)*sizeof(double),matrix+i*ld,ld*sizeof(double),(i+b_size)*sizeof(double),b_size,cudaMemcpyDeviceToHost,stream0);
	  //cudaStreamSynchronize(stream0);
	  
	  /*cblas_dgemm(CblasColMajor, CblasTrans,
			   CblasNoTrans, b_size, b_size,
			   i, -1, work,
			   i+b_size, work, i+b_size,
			   1, work+i, i+b_size);
	  
	  cudaMemcpy2DAsync(matrix+i*ld,ld*sizeof(double),work,(i+b_size)*sizeof(double),(i+b_size)*sizeof(double),b_size,cudaMemcpyHostToDevice,stream0);
	  cudaStreamSynchronize(stream0);
	  cudaFreeHost(work);
	  */
	  }
	
	if(i!=0&&i+b_size<ld){
	  double alpha = -1;
	  double beta = 1;
	  //cudaEventRecord(start0,stream0);
	  //t = clock();
	  cublasDgemm(handle1,CUBLAS_OP_T,CUBLAS_OP_N,b_size,ld-i-b_size,i,&alpha,matrix+i*ld,ld,matrix+(i+b_size)*ld,ld,&beta,matrix+(i+b_size)*ld+i,ld);
	  //cudaEventRecord(stop0,stream0);

	  //*********************
	  //cudaStreamSynchronize(stream0);
          //t = clock() - t;
	  //cudaEventElapsedTime(&t,start0,stop0);
          //gemm_time_total += t;
          //cout<<"DGEMM: "<<((double)t/CLOCKS_PER_SECOND)*1000<<"ms  ";

	  //********************
	}
	//factorize A on CPU
	/*
	if(i>0){
	  cudaEventSynchronize(stop0);
	  cudaEventElapsedTime(&t,start0,stop0);
	  syrk_time_total += t;
	  cout<<"SYRK: "<<t<<"ms  ";
	}
	*/
	
	//cudaStreamSynchronize(stream0);
	//cudaEventRecord(start0,stream0);
	//cudaHostAlloc((void**)&temp,b_size*b_size*sizeof(double),cudaHostAllocDefault);
	


	cudaMemcpy2DAsync(temp,b_size*sizeof(double),matrix+i*ld+i,ld*sizeof(double),b_size*sizeof(double),b_size,cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
	POTF2_CPU(uplo,temp,b_size,b_size);
	//t = clock();
	//POTF2_register<<<dim3(1),dim3(1),0,stream0>>>(matrix+i*ld+i,ld);
	//POTF2_shared<<<dim3(1),dim3(b_size),b_size*b_size*sizeof(double),stream0>>>(matrix+i*ld+i,ld,b_size);
	//int info;
	
	//LAPACK_dpotrf(&uplo,&b_size,temp,&b_size,&info);
	//cudaStreamSynchronize(stream0);
	//printMatrix_gpu(matrix,ld);	
	cudaMemcpy2DAsync(matrix+i*ld+i,ld*sizeof(double),temp,b_size*sizeof(double),b_size*sizeof(double),b_size,cudaMemcpyHostToDevice,stream0);
	




	//cudaEventRecord(stop0,stream0);
	//********************
	//cudaStreamSynchronize(stream0);
	//t = clock() - t;
	//cudaEventElapsedTime(&t,start0,stop0);			\
	//syrk_time_total += t;						\
	//cout<<"POTF2: "<<((double)t/CLOCKS_PER_SECOND)*1000<<"ms  ";
	//***********************


	/*
	if(i!=0&&i+b_size<ld){
	  cudaEventSynchronize(stop1);
	  cudaEventElapsedTime(&t,start1,stop1);
	  gemm_time = t;
	  gemm_time_total += t;
	  cout<<"GEMM: "<<t<<"ms  ";
	}
	*/
	
	/*
	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&t,start0,stop0);
	cpu_time = t;
	cpu_time_total += t;
	cout<<"CPU: "<<t<<"ms  ";
	*/
	//cudaFreeHost(temp);
	//update B                                            
	if(i+b_size<ld){
	  
	  //cudaStreamSynchronize(stream0);
	  //cudaStreamSynchronize(stream1);
	  double alpha = 1;
          
	  //cudaEventRecord(start0,stream0);
	  //t = clock();
	  cublasDtrsm(handle0,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,b_size,ld-i-b_size,&alpha,matrix+i*ld+i,ld,matrix+(i+b_size)*ld+i,ld);
	  
	  //cudaEventRecord(stop0,stream0);
	  //**********************
	  //cudaStreamSynchronize(stream0);
          //t = clock() - t;
	  //cudaEventElapsedTime(&t,start0,stop0);
          //trsm_time_total += t;
          //cout<<"TRSM: "<<((double)t/CLOCKS_PER_SECOND)*1000<<"ms  ";


	  //**********************
	  /*
	  cudaEventSynchronize(stop0);
	  cudaEventElapsedTime(&t,start0,stop0);
	  trsm_time_total += t;
	  cout<<"TRSM: "<<t<<"ms  "<<endl;
          */
	}
	
      }
      
     
      
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cublasDestroy(handle0);
    cublasDestroy(handle1);
    //t1 = clock() - t1;
    //cout<<"loop time:"<<((double)t1/CLOCKS_PER_SEC)*1000<<endl;
    //cout<<endl<<"Total: SYRK="<<syrk_time_total<<"ms, GEMM="<<gemm_time_total<<"ms, CPU="<<cpu_time_total<<"ms, TRSM="<<trsm_time_total<<"ms."<<endl;
    
}



