 /*Blocked Cholesky Factorization v1.4.
 *This version mainly optimized on accessing global memory.
 *Shared memory bank conflicts are not taken care of.
 *Jieyang Chen, University of California, Riverside
 **/



#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath> 
#include<ctime>
//#define N 1024
//#define B 64
using namespace std;

__global__ void POTF2(double * matrix, int I, int N, int B){
  int id = threadIdx.x;
  
  double * localMatrix = matrix+I*N+I;
  extern __shared__ double sharedLocalMatrix[];
  register int idB =id*B;
  //copy sub-matrix to shared memory
  for(int i=0;i<B;i++){
    sharedLocalMatrix[i*B+id] = localMatrix[i*N+id]; 
    __syncthreads();
  }

    for(int i = 0; i<B;i++){
    if(id==i){
      sharedLocalMatrix[i*B+i] = sqrt(sharedLocalMatrix[i*B+i]);
    }
    __syncthreads();
    if(id>i&&id<B){
      sharedLocalMatrix[idB+i] = sharedLocalMatrix[idB+i]/sharedLocalMatrix[i*B+i];
     sharedLocalMatrix[i*B+id] = sharedLocalMatrix[idB+i];
      __syncthreads();
      for(int j=i+1;j<B;j++){
       	sharedLocalMatrix[idB+j]-=sharedLocalMatrix[idB+i]*sharedLocalMatrix[j*B+i];
      }
    }
    __syncthreads();
  }
    __syncthreads();
    for(int i=0;i<B;i++){
      localMatrix[i*N+id]=sharedLocalMatrix[i*B+id];
      __syncthreads();
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
      cout<<matrix[i*N+j];
    }
    cout<<endl;
  }
  cout<<endl;
}


void test(int N, int B){  
  double * input = new double[N*N]();
  double * result = new double[N*N]();
  matrixGenerator(input,result,N);
  double * output = new double[N*N]();
  double * dev_input;
 
  double * a =new double[N*N]();
  memcpy(a,input,N*N*sizeof(double));

  clock_t t = clock();
  CPU_CholeskyFactorization(a,N);
  t=clock()-t;
  float time =((float)t/CLOCKS_PER_SEC)*1000.0;
  cout<<"CPU time:"<<time<<"ms."<<endl;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  cudaMalloc((void**)&dev_input, N*N*sizeof(double));
  cudaMemset((void**)&dev_input,0,N*N*sizeof(double));
  cudaMemcpy(dev_input,input,N*N*sizeof(double),cudaMemcpyHostToDevice);
  

  //cudaEvent_t start, stop;
  //cudaEventCreate(&start);
  //cudaEventCreate(&stop);
  //cudaEventRecord(start,0);

  cudaFuncSetCacheConfig(POTF2,cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TRSM,cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(RKSY,cudaFuncCachePreferShared);
   for(int I=0;I<N;I+=B){
     POTF2<<<dim3(1),dim3(B),B*B*sizeof(double)>>>(dev_input,I,N,B);
     TRSM<<<dim3(B),dim3(B),B*B*sizeof(double)>>>(dev_input,I,N,B);
     RKSY<<<dim3(32,32),dim3(8,8)>>>(dev_input,I,N,B);
   }
   cudaMemcpy(output,dev_input,N*N*sizeof(double),cudaMemcpyDeviceToHost);
   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime,start,stop);
   cout<<"Performing blocked Cholesky Factorization on GPU.Time:"<<elapsedTime<<"ms."<<endl;
  cout<<"Matrix Size:"<<N<<"*"<<N<<", Block Size:"<<B<<"*"<<B<<"."<<endl;
  //cudaMemcpy(output,dev_input,N*N*sizeof(double),cudaMemcpyDeviceToHost);
  cout<<"Verify result on CPU...";
  if(resultVerify(result,output,N)){
    cout<<"Result passed"<<endl;
  }
  else{
    cout<<"Result failed"<<endl;
  }
  cudaFree(dev_input);
  free(input);
  free(output);
  // return 0;
}

int main(){
  // test(16,2);
  //test(16,4);
  //test(16,8);
  //test(32,4);
  //test(32,8);
  //test(32,16);
  //test(64,8);
  //test(64,16);
  //test(64,32);
  test(128,16);
  test(128,32);
  test(128,64);

  test(256,16);
  test(256,32);
  test(256,64);
 
  test(512,16);
  test(512,32);
  test(512,64);
  
  test(1024,16);
  test(1024,32);
  test(1024,64);

  test(2048,16);
  test(2048,32);
  test(2048,64);

  /*
  test(4096,512);
  test(4096,1024);
  test(4096,2048); 
 
  test(2048,64);
  */
}

