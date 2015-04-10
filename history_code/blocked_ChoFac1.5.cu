 /*Blocked Cholesky Factorization v1.5.
 *This version mainly added online fault tolerance feature.
 *Shared memory bank conflicts are not taken care of. 100x faster than global memory is fair enough for now.
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

__global__ void POTF2(double * matrix, int I, int N, int B, double * checkSum1, double * checkSum2){
  int id = threadIdx.x;
  
  double * localMatrix = matrix+I*N+I;
  extern __shared__ double sharedAll[];
  double * sharedLocalMatrix = sharedAll;
  double * sharedCheckSum1 = &sharedAll[B*B];
  double * sharedCheckSum2 = &sharedAll[B*B+B];
  register int idB =id*B;
  //copy sub-matrix to shared memory
  for(int i=0;i<B;i++){
    sharedLocalMatrix[i*B+id] = localMatrix[i*N+id]; 
    __syncthreads();
  }
  //Do unblocked Cholesky Factorization 
  for(int i = 0; i<B;i++){
    if(id==i){
      sharedLocalMatrix[i*B+i] = sqrt(sharedLocalMatrix[i*B+i]);
    }
    __syncthreads();
    if(id>i&&id<B){
      sharedLocalMatrix[idB+i] = sharedLocalMatrix[idB+i]/sharedLocalMatrix[i*B+i];
      sharedLocalMatrix[i*B+id] = 0;//sharedLocalMatrix[idB+i];
      __syncthreads();
      for(int j=i+1;j<id+1;j++){
       	sharedLocalMatrix[idB+j]-=sharedLocalMatrix[idB+i]*sharedLocalMatrix[j*B+i];
      }
    }
    __syncthreads();
  }
  __syncthreads();
  //do online fault check and correction
  //copy two checksums to shared memory for speed up
  sharedCheckSum1[id] = checkSum1[I*N+I*B+id];
  sharedCheckSum2[id] = checkSum2[I*N+I*B+id];
  __syncthreads();
  //update CheckSum1
  if(id==0){
    for(int i=0;i<B;i++){
      sharedCheckSum1[i]/=sharedLocalMatrix[i*B+i];
      for(int j=i+1;j<B;j++){
	sharedCheckSum1[j]-=sharedCheckSum1[i]*sharedLocalMatrix[j*B+i];
      }
    }
    //update CheckSum2
    for(int i=0;i<B;i++){
      sharedCheckSum2[i]/=sharedLocalMatrix[i*B+i];
      for(int j=i+1;j<B;j++){
	sharedCheckSum2[j]-=sharedCheckSum2[i]*sharedLocalMatrix[j*B+i];
      }
    }
    
  }
  __syncthreads();
  //get checksums of result
  double sum1=0;
  double sum2=0;
  for(int i=0;i<B;i++){
    sum1+=sharedLocalMatrix[i*B+id];
    sum2+=sharedLocalMatrix[i*B+id]*(i+1);
  }
  __syncthreads();
  //comparision
  if(sum1!=sharedCheckSum1[id]){
    //error occured
    double r1 = sum1-sharedCheckSum1[id];
    double r2 = sum2-sharedCheckSum2[id];
    int pos = (int)(r2/r1)-1;
    // sharedLocalMatrix[pos*B+id]-=r1;
  }
  


  //copy the result back to global memory
  for(int i=0;i<B;i++){
    localMatrix[i*N+id]=sharedLocalMatrix[i*B+id];
    __syncthreads();
  }
  //copy updated checksums back to global memory
  checkSum1[I*N+I*B+id] = sharedCheckSum1[id];
  checkSum2[I*N+I*B+id] = sharedCheckSum2[id];
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
	sum+=localMatrix[(B+row)*N+k]*sharedLocalMatrix[j*B+k];
      }
      localMatrix[(B+row)*N+j]-=sum;
      localMatrix[(B+row)*N+j]/=sharedLocalMatrix[j*B+j];
      localMatrix[j*N+B+row] = 0;//localMatrix[(B+row)*N+j];
    }
    __syncthreads();

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
	sum+=localMatrix[(B+row)*N+k]*localMatrix[(B+col)*N+k];   
      }     
      if(row>=col)
	localMatrix[(B+row)*N+B+col]-=sum;
      else
	localMatrix[(B+row)*N+B+col]=0;
      __syncthreads();
      }
    }  

}


__global__ void InitCheckSum(double * matrix,double * vector, double * checkSum, int N, int B){
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  //int cacheIndex = ty;
  double * localMatrix = matrix+by*B*N+bx*B;
  extern __shared__ double sharedAll[];
  double * sharedVector = sharedAll;
  double * sharedBlock = &sharedAll[B];
  if(ty==0){
    sharedVector[tx] = vector[tx];
  }
  sharedBlock[ty*B+tx] = localMatrix[ty*N+tx]*sharedVector[ty];
  __syncthreads();
  
  int i = B/2;
  while(i!=0){
    if(ty<i){
      sharedBlock[ty*B+tx]+=sharedBlock[(ty+i)*B+tx];
    }
    __syncthreads();
    i/=2;
  }
  
  if(ty==0){
    checkSum[by*N+bx*B+tx] = sharedBlock[tx];
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
  //  for(int i=0;i<N;i++){
  //  for(int j=i+1;j<N;j++){
  //    A[i*N+j] = At[i*N+j];
  //  }
  // }
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
      cout.width(5);
      cout.setf(ios::left);   
      cout<<matrix[i*N+j];
    }
    cout<<endl;
  }
  cout<<endl;
}

void CPUtest(double * matrix, int N){
  double * cpuMatrix =new double[N*N]();
  memcpy(cpuMatrix,matrix,N*N*sizeof(double));

  clock_t t = clock();
  CPU_CholeskyFactorization(cpuMatrix,N);
  t=clock()-t;
  float time =((float)t/CLOCKS_PER_SEC)*1000.0;
   cout<<"CPU time:"<<time<<"ms."<<endl;
}



void test(int N, int B){  
  double * input = new double[N*N]();
  double * result = new double[N*N]();
  matrixGenerator(input,result,N);
  double * output = new double[N*N]();
  double * dev_input;

  double * dev_checkSum1;
  double * dev_checkSum2;
  double * dev_checkVector1;
  double * dev_checkVector2;
  double * checkVector1 = new double[B]();
  double * checkVector2 = new double[B]();
  for(int i=0;i<B;i++){
    checkVector1[i]=1;
    checkVector2[i]=i+1;
  }

  cudaMalloc((void**)&dev_checkSum1,(N*N/B)*sizeof(double));
  cudaMalloc((void**)&dev_checkSum2,(N*N/B)*sizeof(double));
  cudaMalloc((void**)&dev_checkVector1,B*sizeof(double));
  cudaMalloc((void**)&dev_checkVector2,B*sizeof(double));
  
  cudaMemcpy(dev_checkVector1,checkVector1,B*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_checkVector2,checkVector2,B*sizeof(double),cudaMemcpyHostToDevice);

 



 
  cudaMalloc((void**)&dev_input, N*N*sizeof(double));
  cudaMemset((void**)&dev_input,0,N*N*sizeof(double));
  cudaMemcpy(dev_input,input,N*N*sizeof(double),cudaMemcpyHostToDevice);
  

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  cudaFuncSetCacheConfig(POTF2,cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(TRSM,cudaFuncCachePreferShared);
  cudaFuncSetCacheConfig(RKSY,cudaFuncCachePreferShared);
cudaFuncSetCacheConfig(InitCheckSum,cudaFuncCachePreferShared);
  InitCheckSum<<<dim3(N/B,N/B),dim3(B,B),(B+1)*B*sizeof(double)>>>(dev_input,dev_checkVector1,dev_checkSum1, N, B);
  InitCheckSum<<<dim3(N/B,N/B),dim3(B,B),(B+1)*B*sizeof(double)>>>(dev_input,dev_checkVector2,dev_checkSum2, N, B);
  
  double * checkSum1 = new double[N*N/B];
  double * checkSum2 = new double[N*N/B];
  cudaMemcpy(checkSum1, dev_checkSum1,(N*N/B)*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(checkSum2, dev_checkSum2,(N*N/B)*sizeof(double),cudaMemcpyDeviceToHost);
  cout<<"Original"<<endl;
  printMatrix(input,N);
  cout<<"checkSum"<<endl;
  for(int i=0;i<N/B;i++){
    for(int j=0;j<N;j++){
      cout.width(5);
      cout.setf(ios::left); 
      cout<<checkSum1[i*N+j];
    }
    cout<<endl;
  }
  cout<<"result"<<endl;

  printMatrix(result,N);

   for(int I=0;I<N;I+=B){
     POTF2<<<dim3(1),dim3(B),(B+2)*B*sizeof(double)>>>(dev_input,I,N,B,dev_checkSum1, dev_checkSum2);
     TRSM<<<dim3(B),dim3(B),B*B*sizeof(double)>>>(dev_input,I,N,B);
     RKSY<<<dim3(32,32),dim3(8,8)>>>(dev_input,I,N,B);
   }

   

   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime,start,stop);
   cout<<"Performing blocked Cholesky Factorization on GPU.Time:"<<elapsedTime<<"ms."<<endl;
  cout<<"Matrix Size:"<<N<<"*"<<N<<", Block Size:"<<B<<"*"<<B<<"."<<endl;
  cudaMemcpy(output,dev_input,N*N*sizeof(double),cudaMemcpyDeviceToHost);



cudaMemcpy(checkSum1, dev_checkSum1,(N*N/B)*sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(checkSum2, dev_checkSum2,(N*N/B)*sizeof(double),cudaMemcpyDeviceToHost);
 cout<<"Updated"<<endl;
  printMatrix(output,N);
    cout<<"checkSum"<<endl;
  for(int i=0;i<N/B;i++){
    for(int j=0;j<N;j++){
      cout.width(5);
      cout.setf(ios::left); 
      cout<<checkSum1[i*N+j];
    }
    cout<<endl;
  }




  cout<<"Verify result on CPU...";
  if(resultVerify(result,output,N)){
    cout<<"Result passed"<<endl;
  }
  else{
    cout<<"Result failed"<<endl;
  }
  cudaFree(dev_input);
  cudaFree(dev_checkSum1);
  cudaFree(dev_checkSum2);
  cudaFree(dev_checkVector1);
  cudaFree(dev_checkVector2);

  free(input);
  free(output);
  free(checkVector1);
  free(checkVector2);
  // return 0;
}

int main(){
  /* test(16,2);
  test(16,4);
  test(16,8);
  test(32,4);
  test(32,8);
  test(32,16);
  test(64,8);
  test(64,16);
  test(64,32);
  test(128,16);
  test(128,32);
  test(128,64);
  test(256,32);
  test(256,64);
  test(256,128);
  test(512,64);
  test(512,128);
  test(512,256);
  
  test(1024,128);
  test(1024,256);
  test(1024,512);
  test(2048,256);
  test(2048,512);
  test(2048,1024);

  
  test(4096,512);
  test(4096,1024);
  test(4096,2048); 
  */
  test(8,4);
  
}

