#include<iostream>
//#include<cmath>
#include<cstdlib>
#include <iomanip> 
#define N 9
#define B 3
using namespace std;

__global__ void blockCholeskyFactorization(double * matrix){
 
 
  int id = threadIdx.x;
   for(int I=0;I<N;I+=B){
     //    int I=0;
    double * localMatrix = matrix+I*N+I;
    for(int i = 0; i<B;i++){
      if(id==i){
	//	double x=36;
	// if(i==2)
	//   if(sqrtf(localMatrix[i*N+i])==sqrtf(x))
	//     localMatrix[i*N+i]=100000*(x-localMatrix[i*N+i]);
	//   else
	//     localMatrix[i*N+i]=888;
	//	 else
	  localMatrix[i*N+i] = sqrt(localMatrix[i*N+i]);
      }
      __syncthreads();
      if(id>i&&id<B){
	localMatrix[id*N+i] = localMatrix[id*N+i]/localMatrix[i*N+i];
	localMatrix[i*N+id] = localMatrix[id*N+i];
	for(int j=i+1;j<B;j++)
	  localMatrix[id*N+j]-=localMatrix[id*N+i]*localMatrix[j*N+i];
      }
      __syncthreads();
    }
   
    __syncthreads();
    if(id<N-I-B){      
      for(int j=0;j<B;j++){
	float sum = 0;
	for(int k=0;k<j;k++){
	  sum+=localMatrix[(B+id)*N+k]*localMatrix[k*N+j];
	}
	localMatrix[(B+id)*N+j]-=sum;
	localMatrix[(B+id)*N+j]/=localMatrix[j*N+j];
	localMatrix[j*N+B+id] = localMatrix[(B+id)*N+j];
      }
    }
    __syncthreads();
    
    if(id<N-I-B){
      for(int j=0;j<N-I-B;j++){
	float sum = 0;
	for(int k=0;k<B;k++){
	  sum+=localMatrix[(B+id)*N+k]*localMatrix[k*N+B+j];
	}
		float a=162;
		float b=158;
		localMatrix[(B+id)*N+B+j]-=sum;
	//localMatrix[(B+id)*N+B+j]=a-b;
      }
    }
    __syncthreads();
    
      }
    
}

void matrixGenerator(double * matrix){
  double A[N*N];
  double At[N*N];
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      if(j<i+1)
	A[i*N+j] = rand()%10+1;
      else
	A[i*N+j] = 0;
      At[j*N+i] = A[i*N+j];
      cout.width(15);
      cout.setf(ios::left);
      cout<<setprecision(10)<<A[i*N+j];
    }
    cout<<endl;
  }
  cout<<endl;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      for(int k=0;k<N;k++){
	matrix[i*N+j]+=A[i*N+k]*At[k*N+j];
      }
      cout.width(20);
      cout.setf(ios::left);
      cout<<setprecision(18)<<matrix[i*N+j];
      if(matrix[i*N+j]<0)
	cout<<"Matrix generate Error!"<<endl;

    }
    cout<<endl;

  }
  cout<<endl;
  


}

int main(){  


  // float input[N*N] = {4.0,12.0,-16.0,12.0,37.0,-43.0,-16.0,-43.0,98.0};
  double input[N*N]={};
  matrixGenerator(input);
  double output[N*N] ={0};
  double * dev_input;
  //  float * dev_output;
  
  cudaMalloc((void**)&dev_input, N*N*sizeof(double));
  cudaMemset((void**)&dev_input,0,N*N*sizeof(double));
  cudaMemcpy(dev_input,input,N*N*sizeof(double),cudaMemcpyHostToDevice);
   // cudaMalloc((void**)&dev_output, N*N*sizeof(float));
   // cudaMemset((void**)&dev_output,0,N*N*sizeof(float));
  blockCholeskyFactorization<<<1,N>>>(dev_input);
  cudaMemcpy(output,dev_input,N*N*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      cout.width(15);
      cout.setf(ios::left);
      cout<<setprecision(10)<<output[i*N+j];
    }
    cout<<endl;
  }
  cudaFree(dev_input);
  //cudaFree(dev_output);
  //free(input);
  //  free(output);
  return 0;
}

