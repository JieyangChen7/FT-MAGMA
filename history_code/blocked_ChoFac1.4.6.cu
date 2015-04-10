/*Blocked Cholesky Factorization v1.4.
*potf on CPU and dtrsm on GPU, dgemm on GPU. Compute either upper or lower. Initial data is on GPU, so transfer the data to GPU is not taken care of.
*Jieyang Chen, University of California, Riverside
**/



#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath> 
#include<ctime>
#include"cublas_v2.h"
#include<cuda_runtime.h>
#include<curand.h>
//#include"lapacke.h"
//#include"blas.h"
//#include<cuda.h>
//#include<cuda_runtime.h>

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )

using namespace std;


void printMatrix(double * matrix, int N){
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      cout.width(5);
      cout.setf(ios::left);
      cout<<matrix[j*N+i];
    }
    cout<<endl;
  }
  cout<<endl;
}

void printMatrix_gpu(double * matrix, int N){
  double * matrix_host = new double[N*N]();
  cudaMemcpy(matrix_host,matrix,sizeof(double)*N*N,cudaMemcpyDeviceToHost);
  printMatrix(matrix_host,N);
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
    for(int j=0;j<i+1;j++){
      for(int k=0;k<N;k++){
        matrix[i*N+j]+=A[i*N+k]*At[k*N+j];
      }
      if(matrix[i*N+j]<0)
        cout<<"Matrix generate Error!"<<endl;
    }
  }
  /*
  for(int i=0;i<N;i++){
    for(int j=i+1;j<N;j++){
      A[i*N+j] = At[i*N+j];
    }
  }
  */
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      result[i*N+j]=A[i*N+j];
    }
  }
  free(A);
  free(At);
}



__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo, double alpha){
  int col = threadIdx.x+blockIdx.x*blockDim.x;
  int row = threadIdx.y+blockIdx.y*blockDim.y;
  if(uplo == 'u'){
    if(row<col+1){
      matrix[col*ld+row] *= alpha;      
    }
    else{
      matrix[col*ld+row] = 0.0;
    }
  }
  else{
    if(col<row+1){
      matrix[col*ld+row] *= alpha;
    }   
    else{
      matrix[col*ld+row] = 0.0;
    }
  } 
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
  matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(result, N, uplo, a);
  cudaDeviceSynchronize();
  
  //do matrix-matrix multiplcation using cublas 
  double alpha = 1.0;
  double beta = 1.0;
  if(uplo == 'u'){
    cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,&alpha,result,N,result,N,&beta,matrix,N);
  }
  else if(uplo == 'l'){
    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,N,N,N,&alpha,result,N,result,N,&beta,matrix,N);
  }

  matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, N, uplo, 1);
  cudaDeviceSynchronize();
}
 

__global__ void resultVerify_gpu(double * realResult, double * testResult,double * diff, int N){
  int col = threadIdx.x+blockIdx.x*blockDim.x;
  int row = threadIdx.y+blockIdx.y*blockDim.y;
  diff[col*N+row] = testResult[col*N+row] - realResult[col*N+row];
}

bool resultVerify(double * realResult, double * testResult, int N, int B){
  double * diff;
  cudaMalloc((void**)&diff,N*N*sizeof(double));
  resultVerify_gpu<<<dim3(N/B,N/B),dim3(B,B)>>>(realResult,testResult,diff,N);
  
  double * diff_host = new double[N*N]();
  cudaMemcpy(diff_host,diff,N*N*sizeof(double),cudaMemcpyDeviceToHost);  
  //  printMatrix(diff_host,N);
  for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
	if(abs(diff_host[i*N+j])>1e-5){
	  //  cout<<"diff:"<<abs(diff_host[i*N+j])<<endl;
	  return false;
	}
      }
    }
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


void my_dpotrf(char uplo, double * matrix, int ld, int B, int b, bool debug, bool useCublas){  
  cout<<"start my_dpotrf"<<endl;
  //initial data
  int b_size = B;
  double * temp;
  float gemm_time =0;
  float cpu_time =0;
  //cudaHostAlloc((void**)&temp,B*B*sizeof(double),cudaHostAllocDefault);
  cout<<"pinned memory initialized"<<endl;
  //intial streams----------------------------
  cudaStream_t stream0;//for main loop
  cudaStream_t stream1;//for dgemm part
  //  cudaStream_t stream2;//for cpu part
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);
 
  cout<<"Streams initialized"<<endl;
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

  cout<<"cublas initialized"<<endl;
  //for timing
  cudaEvent_t start0, stop0, start1, stop1;
  cudaEventCreate(&start0);
  cudaEventCreate(&stop0);
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  cout<<"cuda event"<<endl;
  float t=0;
  cout<<"Inital complete"<<endl;
  if(useCublas){

   if(uplo == 'u'){
     cout<<"entering loop"<<endl;
      //start the loop of calculation----------------------------
       for(int i=0;i<ld;i+=B){
	 //cout<<"loop"<<i<<endl;
	 /* if(i!=0){
	   if(gemm_time>cpu_time)
	     B+=10;
	   else
	     B-=10;
	 }
	 if(B<=0)
	   B=1;
	 //B = getDynamicBlockSize(i,ld, B);
	 */
	 //	b_size = min(B,ld-i);
	cout<<"block size:"<<b_size<<"  ";
        
//update A
	
	if(i>0){
	  double alpha = -1;
	  double beta = 1;
	  cudaEventRecord(start0,stream0);
	  cublasDsyrk(handle0,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,b_size,i,&alpha,matrix+i*ld,ld,&beta,matrix+i*ld+i,ld);
	  cudaEventRecord(stop0,stream0);
	}
	
        if(i!=0&&i+b_size<ld){
	  double alpha = -1;
	  double beta = 1;
	  cudaEventRecord(start1,stream1);
	  cublasDgemm(handle1,CUBLAS_OP_T,CUBLAS_OP_N,b_size,ld-i-b_size,i,&alpha,matrix+i*ld,ld,matrix+(i+b_size)*ld,ld,&beta,matrix+(i+b_size)*ld+i,ld);
	  cudaEventRecord(stop1,stream1);
	}
        //factorize A on CPU            
	if(i>0){
	  cudaEventSynchronize(stop0);
	  cudaEventElapsedTime(&t,start0,stop0);
	  cout<<"SYRK: "<<t<<"ms  ";
	}

	
	cudaStreamSynchronize(stream0);	
	cudaEventRecord(start0,stream0);
	cudaHostAlloc((void**)&temp,b_size*b_size*sizeof(double),cudaHostAllocDefault);
	cudaMemcpy2DAsync(temp,b_size*sizeof(double),matrix+i*ld+i,ld*sizeof(double),b_size*sizeof(double),b_size,cudaMemcpyDeviceToHost,stream0);
	cudaStreamSynchronize(stream0);
        POTF2_CPU(uplo,temp,b_size,b_size);
	//	int info;
        //LAPACKE_dpotrf(b_size,'U',b_size,temp,b_size);
	
	cudaMemcpy2DAsync(matrix+i*ld+i,ld*sizeof(double),temp,b_size*sizeof(double),b_size*sizeof(double),b_size,cudaMemcpyHostToDevice,stream0);
        cudaEventRecord(stop0,stream0);
	
	if(i!=0&&i+b_size<ld){
	  cudaEventSynchronize(stop1);
	  cudaEventElapsedTime(&t,start1,stop1);
          gemm_time = t;
	  cout<<"GEMM: "<<t<<"ms  ";
	}


	cudaEventSynchronize(stop0);
	cudaEventElapsedTime(&t,start0,stop0);
	cpu_time = t;
	cout<<"CPU: "<<t<<"ms  ";
	cudaFreeHost(temp);
	//update B                                                                      
        if(i+b_size<ld){
	  
	  cudaStreamSynchronize(stream0);
	  cudaStreamSynchronize(stream1);
	  double alpha = 1;
	  
	  cudaEventRecord(start0,stream0);
	  cublasDtrsm(handle0,CUBLAS_SIDE_LEFT,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,CUBLAS_DIAG_NON_UNIT,b_size,ld-i-b_size,&alpha,matrix+i*ld+i,ld,matrix+(i+b_size)*ld+i,ld);
	  cudaEventRecord(stop0,stream0);
	  cudaEventSynchronize(stop0);
	  cudaEventElapsedTime(&t,start0,stop0);
	  cout<<"TRSM: "<<t<<"ms  "<<endl;
	  
	}
	
       }
       
       cudaStreamSynchronize(stream0);
       cudaStreamSynchronize(stream1);
       cublasDestroy(handle0);
       cublasDestroy(handle1);
       
   }
   
   
  }
  

}





void test_mydpotrf(int N, int B, int b){
//int N=2048;
// int B=64;
//int b=8;
  char uplo = 'u';
  double * matrix_host;
  double * result_host;
  double * matrix;
  double * result;
  //Memory allocation on RAM and DRAM
  cudaMalloc((void**)&matrix,N*N*sizeof(double));
  cudaMalloc((void**)&result,N*N*sizeof(double));
  cudaMemset(matrix,0,N*N*sizeof(double));
  cudaMemset(result,0,N*N*sizeof(double));
  
  matrix_host =  new double[N*N]();
  result_host =  new double[N*N]();
   //Generate test data on RAM
  matrixGenerator(matrix_host,result_host,N);
  //matrixGenerator_gpu(uplo,matrix,result,N,B);
  cudaMemcpy(matrix,matrix_host,sizeof(double)*N*N,cudaMemcpyHostToDevice);
  cudaMemcpy(result,result_host,sizeof(double)*N*N,cudaMemcpyHostToDevice);
  //Do Cholesky Factorization
  cout<<"start timing"<<endl;
  clock_t t = clock();
  my_dpotrf(uplo,matrix,N,B,b,false,true);
  t=clock()-t;
  cout<<"end timing"<<endl;
  double time_in_sec =((double)t/CLOCKS_PER_SEC);
  double gflops = FLOPS_DPOTRF(N)/1000000000;
  cout<<"N:"<<N<<";B:"<<B<<";b:"<<b<<"----";
  cout<<"Time:"<<time_in_sec<<"s, "<<gflops/time_in_sec<<"GFlops/s."<<endl;
  //PrintMatrix for debug---------------------------------
  //double * host_matrix;
  //host_matrix = new double[N*N]();
  //cudaMemcpy(host_matrix,result,N*N*sizeof(double),cudaMemcpyDeviceToHost);
  //  printMatrix(host_matrix,N);

  //cudaMemcpy(host_matrix,matrix,N*N*sizeof(double),cudaMemcpyDeviceToHost); 
  //  printMatrix(host_matrix,N);
 
  //Verify result
  if(resultVerify(result,matrix,N,B)){
    cout<<"Result passed!"<<endl;
  }else{
    cout<<"Result failed!"<<endl;
  }
  //Clear
  //cudaFreeHost(matrix);
  //delete[] result;
  //cudaFree(dev_matrix);
}

int main(int argc, char**argv){

  test_mydpotrf(1024,64,8);

}
