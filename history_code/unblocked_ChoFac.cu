#include<iostream>
#include<cmath>
#define N 3
using namespace std;
__global__ void CholeskyFactorization(float * input, float * output){
   //  cout<<"GPU hello world!"<<endl;
   int id = threadIdx.x;
  // output[id]=input[id];
   for(int i = 0; i<N;i++){
     if(id==i){
       output[i*N+i] = sqrt(input[i*N+i]);
       
     }
     __syncthreads();
     if(id>i){
       output[id*N+i] = input[id*N+i]/output[i*N+i];
       output[i*N+id] = output[id*N+i];
       for(int j=i+1;j<N;j++)
	 input[id*N+j]-=output[id*N+i]*output[j*N+i];
       // __syncthreads();
      }
     __syncthreads();
   }
}


int main(){
  float input[N*N] = {4.0,12.0,-16.0,12.0,37.0,-43.0,-16.0,-43.0,98.0};
  float output[N*N] ={0};
  float * dev_input;
  float * dev_output;
  
  cudaMalloc((void**)&dev_input, N*N*sizeof(float));
  cudaMemset((void**)&dev_input,0,N*N*sizeof(float));
   cudaMemcpy(dev_input,input,N*N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_output, N*N*sizeof(float));
  cudaMemset((void**)&dev_output,0,N*N*sizeof(float));
   CholeskyFactorization<<<1,N>>>(dev_input,dev_output);
   cudaMemcpy(output,dev_output,N*N*sizeof(float),cudaMemcpyDeviceToHost);
    for(int i=0;i<N*N;i++)
     cout<<output[i]<<"  ";
  cudaFree(dev_input);
  cudaFree(dev_output);
  //free(input);
  //  free(output);
  return 0;
}

