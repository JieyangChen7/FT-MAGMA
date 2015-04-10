#include<iostream>
#include<cstdlib>
#include <iomanip> 
#define N 1000
#define B 10
using namespace std;


__global__ void POTF2(double * matrix, int I){
  int id = threadIdx.x;
  double * localMatrix = matrix+I*N+I;
    for(int i = 0; i<B;i++){
    if(id==i){
      localMatrix[i*N+i] = sqrt(localMatrix[i*N+i]);
    }
    __syncthreads();
    if(id>i&&id<B){
      localMatrix[id*N+i] = localMatrix[id*N+i]/localMatrix[i*N+i];
      localMatrix[i*N+id] = localMatrix[id*N+i];
      __syncthreads();
      for(int j=i+1;j<B;j++){
       	localMatrix[id*N+j]-=localMatrix[id*N+i]*localMatrix[j*N+i];
      }
    }
    __syncthreads();
  }
}


__global__ void TRSM(double * matrix, int I){
  int id = threadIdx.x+blockIdx.x*blockDim.x;
  double * localMatrix = matrix+I*N+I;
  for(int row=id;row<N-I-B;row+=blockDim.x*gridDim.x){
    for(int j=0;j<B;j++){
      double sum = 0;
      for(int k=0;k<j;k++){
	sum+=localMatrix[(B+row)*N+k]*localMatrix[k*N+j];
      }
       localMatrix[(B+row)*N+j]-=sum;                                                                      
      localMatrix[(B+row)*N+j]/=localMatrix[j*N+j];
      localMatrix[j*N+B+row] = localMatrix[(B+row)*N+j];
    }
  }
}


__global__ void RKSY(double * matrix, int I){
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  double * localMatrix = matrix+I*N+I;
  for(int row =y;row<N-I-B;row+=blockDim.y*gridDim.y){                                                                  
   for(int col = x;col<N-I-B;col+=blockDim.x*gridDim.x){                                                               
     // int row =y;
     // int col =x;
      double sum = 0;                                                                                         
      for(int k=0;k<B;k++){                                                                                  
	sum+=localMatrix[(B+row)*N+k]*localMatrix[k*N+B+col];                                                
      }                                                                                                      
      localMatrix[(B+row)*N+B+col]-=sum;                                                                     
      }                                                                                                         
    }  
}
/*
__global__ void blockCholeskyFactorization(double * matrix){
  int x = threadIdx.x+blockIdx.x*blockDim.x;
  int y = threadIdx.y+blockIdx.y*blockDim.y;
  int id = x+y*blockDim.x*gridDim.x;
  //   for(int I=0;I<N;I+=B){
  int I=0; 
  double * localMatrix = matrix+I*N+I;
    //unblocked Cholesky Factorization performed within block(0,0)
    if(x<blockDim.x&&y<blockDim.y){
      int id0 = x+y*blockDim.x;
      for(int i = 0; i<B;i++){
	if(id0==i){
	  localMatrix[i*N+i] = sqrt(localMatrix[i*N+i]);
	}
	__syncthreads();
	if(id0>i&&id0<B){
	  localMatrix[id0*N+i] = localMatrix[id0*N+i]/localMatrix[i*N+i];
	  localMatrix[i*N+id0] = localMatrix[id0*N+i];
	  for(int j=i+1;j<B;j++)
	    localMatrix[id0*N+j]-=localMatrix[id0*N+i]*localMatrix[j*N+i];
	}
	__syncthreads();
      }
    }
    __syncthreads();
    if(id<N-I-B){
      // int row = id;
      for(int row=id;row<N-I-B;row+=blockDim.x*blockDim.y*gridDim.x*gridDim.y){
	for(int j=0;j<B;j++){
	  float sum = 0;
	  for(int k=0;k<j;k++){
	    sum+=localMatrix[(B+row)*N+k]*localMatrix[k*N+j];
	  }
	  //localMatrix[(B+row)*N+j]-=sum;
	  localMatrix[(B+row)*N+j]=localMatrix[j*N+j];
	  localMatrix[j*N+B+row] = localMatrix[(B+row)*N+j];
	}
      }
    // }
    __syncthreads();
    /*
    for(int row =x;row<N-I-B;row+=blockDim.y){
      for(int col = y;col<N-I-B;col+=blockDim.x){
	 float sum = 0;
	 for(int k=0;k<B;k++){
	   sum+=localMatrix[(B+row)*N+k]*localMatrix[k*N+B+col];
	 }
	 localMatrix[(B+row)*N+B+col]-=sum;
      }
    }
    __syncthreads();
  } 
}
*/
void matrixGenerator(double * matrix, double * result){
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

bool resultVerify(double * realResult, double * testResult){
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




int main(){  


  // float input[N*N] = {4.0,12.0,-16.0,12.0,37.0,-43.0,-16.0,-43.0,98.0};
  double * input = new double[N*N]();
  double * result = new double[N*N]();
  matrixGenerator(input,result);
  double * output = new double[N*N]();
  double * dev_input;
  //  float * dev_output;
  /*  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      cout.width(15);
      cout.setf(ios::left);
      cout<<setprecision(10)<<result[i*N+j];

    }
    cout<<endl;
  }
  */
  
  cudaMalloc((void**)&dev_input, N*N*sizeof(double));
  cudaMemset((void**)&dev_input,0,N*N*sizeof(double));
  cudaMemcpy(dev_input,input,N*N*sizeof(double),cudaMemcpyHostToDevice);
  

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);


   for(int I=0;I<N;I+=B){
     POTF2<<<dim3(1),dim3(B)>>>(dev_input,I);
     TRSM<<<dim3(B),dim3(B)>>>(dev_input, I);
     RKSY<<<dim3(B,B),dim3(10,10)>>>(dev_input, I);
   }

   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime,start,stop);
   cout<<"Performing blocked Cholesky Factorization on GPU.Time:"<<elapsedTime<<"."<<endl;
  cout<<"Matrix Size:"<<N<<"*"<<N<<", Block Size:"<<B<<"*"<<B<<"."<<endl;
  cudaMemcpy(output,dev_input,N*N*sizeof(double),cudaMemcpyDeviceToHost);
     for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      // cout.width(2);
      // cout.setf(ios::left);
      // cout<<output[i*N+j]<<"  ";
      if(output[i*N+j]<0){
	cout<<i<<"--"<<j<<endl;
      }
    }
    //   cout<<endl<<endl<<endl;
  }
  
  
  cout<<"Verify result on CPU...";
  if(resultVerify(result,output)){
    cout<<"Result passed"<<endl;
  }
  else{
    cout<<"Result failed"<<endl;

  }
  cudaFree(dev_input);
  free(input);
  free(output);
  return 0;
}

