 /*Blocked Cholesky Factorization v1.4.
 *potf on cpu, dtrsm and dgemm on cublas, compute both upper and lower, double the computation time. Initial data is on RAM, so transfer to GPU is taken care of.
 *Jieyang Chen, University of California, Riverside
 **/



#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath> 
#include<ctime>
#include"cublas_v2.h"
#include<cuda_runtime.h>
//#define N 1024
//#define B 64
using namespace std;

void POTF2(double * matrix, int ld, int N, int B){
  for(int i = 0; i<B;i++){
    matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
    for(int j=i+1;j<B;j++){
      matrix[i*ld+j] /=matrix[i*ld+i];
      matrix[j*ld+i] = matrix[i*ld+j];
    }
    for(int j=i+1;j<B;j++){
      for(int k=i+1;k<B;k++){
	matrix[j*ld+k]-=matrix[i*ld+k]*matrix[j*ld+i];
      }
    }
  }
}



__global__ void TRSM(double * matrix, int I,int N, int B){
  
  int id = threadIdx.x+blockIdx.x*blockDim.x;
  double * localMatrix = matrix+I*N+I;
  extern __shared__ double sharedLocalMatrix[];
  for(int i=0;i<B;i++){
      sharedLocalMatrix[i*B+threadIdx.x] = localMatrix[i*N+threadIdx.x];
      __syncthreads();
  }
  //__syncthreads();
  
  for(int row=id;row<N-I-B;row+=blockDim.x*gridDim.x){
    for(int j=0;j<B;j++){
      double sum = 0;
      for(int k=0;k<j;k++){
	sum+=localMatrix[(B+row)*N+k]*sharedLocalMatrix[k*B+j];
      }
      localMatrix[(B+row)*N+j]-=sum;
      localMatrix[(B+row)*N+j]/=sharedLocalMatrix[j*B+j];
      localMatrix[j*N+B+row] = localMatrix[(B+row)*N+j];
    }
  }
}




__global__ void RKSY(double * matrix, int I,int N, int B){
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  double * localMatrix = matrix+I*N+I;
  for(int row =y;row<N-I-B;row+=blockDim.y*gridDim.y){
   for(int col = x;col<N-I-B;col+=blockDim.x*gridDim.x){
      double sum = 0;
      for(int k=0;k<B;k++){         
	sum+=localMatrix[(B+row)*N+k]*localMatrix[k*N+B+col];   
      }     
      localMatrix[(B+row)*N+B+col]-=sum;
      }
    }  
}

void CPU_CholeskyFactorization(double * matrix, int N){
    for(int i = 0; i<N;i++){ 
      matrix[i*N+i] = sqrt(matrix[i*N+i]);
      for(int j=i+1;j<N;j++){
	matrix[j*N+i] = matrix[j*N+i]/matrix[i*N+i];
	matrix[i*N+j] = matrix[j*N+i];
      }
      for(int j=i+1;j<N;j++){
	for(int k=i+1;k<N;k++){
	  matrix[j*N+k]-=matrix[j*N+i]*matrix[i*N+k];
	}
      }
  }
}



void matrixGenerator(double * matrix, double * result, int N){
  double * A = new double[N*N]();
  double * At = new double[N*N]();
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if(j<i+1)
	A[i*N+j] = rand()%10+1;
      else
	A[i*N+j] = 0;
      At[j*N+i] = A[i*N+j];
    }
  }
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      for(int k=0;k<N;k++){
	matrix[i*N+j]+=A[i*N+k]*At[k*N+j];
      }
      if(matrix[i*N+j]<0)
	cout<<"Matrix generate Error!"<<endl;
    }
  }
  for(int i=0;i<N;i++){
    for(int j=i+1;j<N;j++){
      A[i*N+j] = At[i*N+j];
    }
  }
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      result[i*N+j]=A[i*N+j];
    }
  }
  free(A);
  free(At);
}

bool resultVerify(double * realResult, double * testResult, int N){
  bool pass = true;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if(realResult[i*N+j]!=testResult[i*N+j]){
	pass = false;
	break;
      }
    }
  }
  return pass;
}

void printMatrix(double * matrix, int N){
  // cout.width(5);
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      cout.width(2);
      cout.setf(ios::left);   
      cout<<matrix[j*N+i];
    }
    cout<<endl;
  }
  cout<<endl;
}


void test(int N, int B){  
  double * input = new double[N*N]();
  double * result = new double[N*N]();
  
  matrixGenerator(input,result,N);
  //intial cublas
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  double * d_a;
  double * d_b1;
  double * d_b2;
  double * d_c;
  double alpha = 1.0;
  double neg_alpha = -1.0;
  stat = cublasCreate(&handle);
  if(stat!=CUBLAS_STATUS_SUCCESS){
      cout<<"ERROR"<<endl;
    }
  

clock_t t = clock();

   for(int i=0;i<N;i+=B){
   
    double * a = input+i*N+i;
    double * b1 = a+N*B;
    double * b2 = a+B;
    double * c = a+B*N+B;
    POTF2(a,N,N,B);
    int l =N-i-B;
    if(i+B<N){
      //cudaMalloc--------------------------------------------
      cudaStat = cudaMalloc((void**)&d_a,B*B*sizeof(double));
      cudaStat = cudaMalloc((void**)&d_b1,B*l*sizeof(double));
      cudaStat = cudaMalloc((void**)&d_b2,B*l*sizeof(double));
      cudaStat = cudaMalloc((void**)&d_c,l*l*sizeof(double));
      //cudaSetMatrix-----------------------------------------
      stat = cublasSetMatrix(B,B,sizeof(double),a,N,d_a,B);
      stat = cublasSetMatrix(B,l,sizeof(double),b1,N,d_b1,B);
      stat = cublasSetMatrix(l,B,sizeof(double),b2,N,d_b2,l);
      stat = cublasSetMatrixAsync(l,l,sizeof(double),c,N,d_c,l,0);
      //cublas compute---------------------------------------- 
      cudaFuncSetCacheConfig(cublasDtrsm,cudaFuncCachePreferShared);
      stat = cublasDtrsm(handle,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,B,l,&alpha,d_a,B,d_b1,B);
      cudaFuncSetCacheConfig(cublasDtrsm,cudaFuncCachePreferShared);
      stat = cublasDtrsm(handle,CUBLAS_SIDE_RIGHT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_N,CUBLAS_DIAG_NON_UNIT,l,B,&alpha,d_a,B,d_b2,l);
      cudaFuncSetCacheConfig(cublasDgemm,cudaFuncCachePreferShared);
      stat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,l,B,&neg_alpha,d_b2,l,d_b1,B,&alpha,d_c,l);
      //cublas GetMatrix--------------------------------------
      stat = cublasGetMatrixAsync(B,l,sizeof(double),d_b1,B,b1,N,0);
      stat = cublasGetMatrixAsync(l,B,sizeof(double),d_b2,l,b2,N,0);
      stat = cublasGetMatrix(l,l,sizeof(double),d_c,l,c,N);
    }

   }
 t=clock()-t;
  float time =((float)t/CLOCKS_PER_SEC)*1000.0;
  cout<<"CUBLAS time[N="<<N<<"B="<<B<<"]:"<<time<<"ms."<<endl;
   
   
   //PrintMatrix for debug---------------------------------
   //printMatrix(result,N);
   //printMatrix(input,N);
   //Verify result-----------------------------------------
   if(resultVerify(input,result,N)){
     cout<<"Result passed!"<<endl;
   }else{
     cout<<"Result failed!"<<endl;
   }
   //clean-------------------------------------------------
   delete input;
   delete result;
   cudaFree(d_a);
   cudaFree(d_b1);
   cudaFree(d_b2);
   cudaFree(d_c);
   cublasDestroy(handle);
}




int main(int argc, char**argv){
   test(1024,64);
 
}

