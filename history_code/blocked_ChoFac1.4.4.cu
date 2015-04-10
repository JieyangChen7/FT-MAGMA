/*Blocked Cholesky Factorization v1.4.
 *potf and dtrsm on GPU, dgemm on cublas. Compute both upper and lower, double the computation time. Initial data is on RAM, so transfer the data to GPU is taken care of.
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
//cpu version

void POTF2_CPU(double * matrix, int ld, int N, int B){
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

//gpu version
__global__ void POTF2(double * matrix, int ld, int B){
  int id = threadIdx.x;
  for(int i = 0; i<B;i++){
    if(id==i){
      matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
    }
    __syncthreads();
    if(id>i&&id<B){
      matrix[i*ld+id] /= matrix[i*ld+i];
      matrix[id*ld+i] = matrix[i*ld+id];
      __syncthreads();
      for(int j=i+1;j<B;j++){
        matrix[j*ld+id]-=matrix[i*ld+id]*matrix[j*ld+i];
      }
    }
    __syncthreads();
  }
}


//temporary use only upper triagular matrix
//a->A; b->B; B->block size, column of B&A; l->row of B
__global__ void TRSM(char uplo, double * a, int lda, double * b, int ldb, int B, int l){
  
  int id = threadIdx.x+blockIdx.x*blockDim.x;
  int stride = blockDim.x*gridDim.x;
  //double * localMatrix = matrix+I*N+I;
  //extern __shared__ double sharedLocalMatrix[];
  //for(int i=0;i<B;i++){
  //    sharedLocalMatrix[i*B+threadIdx.x] = localMatrix[i*N+threadIdx.x];
  //    __syncthreads();
  //}
  //__syncthreads();
  
  if(uplo == 'u'){
    for(int row=id;row<l;row+=stride){
      for(int j=0;j<B;j++){
	double sum = 0;
	for(int k=0;k<j;k++){
	  sum+=b[k*ldb+row]*a[j*lda+k];
	}
	b[j*ldb+row]-=sum;
	b[j*ldb+row]/=a[j*lda+j];
	//localMatrix[j*N+B+row] = localMatrix[(B+row)*N+j];
      }
    }
  }
  else if(uplo == 'l'){
     for(int col=id;col<l;col+=stride){
      for(int j=0;j<B;j++){
	double sum = 0;
	for(int k=0;k<j;k++){
	  sum+=b[col*ldb+k]*a[k*lda+j];
	}
	b[col*ldb+j]-=sum;
	b[col*ldb+j]/=a[j*lda+j];
	//localMatrix[j*N+B+row] = localMatrix[(B+row)*N+j];
      }
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
  //initial data
  double * input;
  cudaHostAlloc((void**)&input,N*N*sizeof(double),cudaHostAllocDefault);
  double * result = new double[N*N]();
  matrixGenerator(input,result,N);
  double ** dev = new double* [(N/B)*(N/B)]();
  
  //intial streams----------------------------
  cudaStream_t stream0;
  cudaStream_t stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
  //initial cublas
  //  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  double * d_a;
  double * d_b1;
  double * d_b2;
  double * d_c;
  double alpha = 1.0;
  double neg_alpha = -1.0;
  stat = cublasCreate(&handle);
  cublasSetStream(handle,stream0);
  if(stat!=CUBLAS_STATUS_SUCCESS){
      cout<<"ERROR"<<endl;
  }
  

  clock_t t = clock();
  //start the loop of calculation----------------------------
     for(int i=0;i<N;i+=B){
       double * a = input+i*N+i;
       double * b1 = a+N*B;
       double * b2 = a+B;
       double * c = a+B*N+B;
       int l = N-i-B;
       
       //cudaMalloc--------------------------------------------
       cudaMalloc((void**)&d_a,B*B*sizeof(double));
       if(i+B<N){
	 cudaMalloc((void**)&d_b2,B*l*sizeof(double));
	 cudaMalloc((void**)&d_b1,B*l*sizeof(double));
	 cudaMalloc((void**)&d_c,l*l*sizeof(double));
       }
       //cudaSetMatrix-----------------------------------------
       stat = cublasSetMatrixAsync(B,B,sizeof(double),a,N,d_a,B,stream0);
       if(i+B<N){
	 stat = cublasSetMatrixAsync(B,l,sizeof(double),b1,N,d_b1,B,stream0);
	 stat = cublasSetMatrixAsync(l,B,sizeof(double),b2,N,d_b2,l,stream1);
	 stat = cublasSetMatrixAsync(l,l,sizeof(double),c,N,d_c,l,stream0);
       }
       //cublas compute----------------------------------------
       POTF2<<<1,dim3(B),0,stream0>>>(d_a,B,B);
       if(i+B<N){
	 TRSM<<<dim3(B),dim3(B),0,stream0>>>('l',d_a,B,d_b1,B,B,l);
	 TRSM<<<dim3(B),dim3(B),0,stream1>>>('u',d_a,B,d_b2,l,B,l);
	 cudaFuncSetCacheConfig(cublasDgemm,cudaFuncCachePreferShared);
	 stat = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,l,l,B,&neg_alpha,d_b2,l,d_b1,B,&alpha,d_c,l);
       }
       //cublas GetMatrix--------------------------------------
       stat = cublasGetMatrixAsync(B,B,sizeof(double),d_a,B,a,N,stream0);
       if(i+B<N){ 
	 stat = cublasGetMatrixAsync(B,l,sizeof(double),d_b1,B,b1,N,stream0);
	 stat = cublasGetMatrixAsync(l,B,sizeof(double),d_b2,l,b2,N,stream1);
	 stat = cublasGetMatrix(l,l,sizeof(double),d_c,l,c,N);
       }
     }
cudaStreamSynchronize(stream0);
cudaStreamSynchronize(stream1);
t=clock()-t;
float time =((float)t/CLOCKS_PER_SEC)*1000.0;
// float ctime =((float)c/CLOCKS_PER_SEC)*1000.0;
//float mtime =((float)m/CLOCKS_PER_SEC)*1000.0;
cout<<"CUBLAS time[N="<<N<<"B="<<B<<"]:"<<time<<"ms."<<endl;
  //cout<<"Calculation time:"<<c<<"ms."; 
  // cout<<"Memory copy time:"<<mtime<<"ms."<<endl;
  
  
  //PrintMatrix for debug---------------------------------
  // printMatrix(result,N);
  //printMatrix(input,N);
  //Verify result-----------------------------------------
  if(resultVerify(input,result,N)){
    cout<<"Result passed!"<<endl;
  }else{
    cout<<"Result failed!"<<endl;
  }
  //clean-------------------------------------------------
  //delete input;
  //delete result;
  cudaFreeHost(input);
  delete result;
  cudaFree(d_a);
  cudaFree(d_b1);
  cudaFree(d_b2);
  cudaFree(d_c);
  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
  cublasDestroy(handle);
}




int main(int argc, char**argv){
   test(1024,64);
 
}

