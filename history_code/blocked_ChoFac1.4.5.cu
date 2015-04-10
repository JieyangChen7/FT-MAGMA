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

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )

using namespace std;

//cpu version
void POTF2_CPU(char uplo, double * matrix, int ld, int B){
  if(uplo == 'u'){
    for(int i = 0; i<B;i++){
      matrix[i*ld+i] = sqrt(matrix[i*ld+i]);
      for(int j=i+1;j<B;j++){
        matrix[j*ld+i] /=matrix[i*ld+i];
        // matrix[j*ld+i] = matrix[i*ld+j];                                             
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
	// matrix[j*ld+i] = matrix[i*ld+j];
      }
      for(int j=i+1;j<B;j++){
	for(int k=i+1;k<j+1;k++){
	  matrix[k*ld+j]-=matrix[i*ld+j]*matrix[i*ld+k];
	}
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
  //double * temp = new double[N*N]();
  //cudaMemcpy(temp,result,N*N*sizeof(double),cudaMemcpyDeviceToHost);
  //printMatrix(temp,N);
  
  //cudaMemcpy(temp,matrix,N*N*sizeof(double),cudaMemcpyDeviceToHost);
  //printMatrix(temp,N);

  
  //cudaFree(matrix);
  //cudaFree(result);
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
  //double * temp = new double[N*N]();
  cudaMemcpy(diff_host, diff, N*N*sizeof(double),cudaMemcpyDeviceToHost);
  //printMatrix(diff_host,N);
    
  //cudaMemcpy(temp, realResult, N*N*sizeof(double),cudaMemcpyDeviceToHost);
  //printMatrix(temp,N);                                                  

  //cudaMemcpy(temp, testResult, N*N*sizeof(double),cudaMemcpyDeviceToHost);
  //printMatrix(temp,N);                                                  

  bool pass = true;
    for(int i=0;i<N;i++){
      for(int j=0;j<N;j++){
	if(abs(diff_host[i*N+j])>1e-10){
	  pass = false;
	  break;
	}
      }
    }
    return pass;
  
  
}

void my_dpotrf(char uplo, double * matrix, int ld, int B, int b, bool debug, bool useCublas){  
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
	  // RKSY<<<dim3(B/b,B/b),dim3(b,b),b*b*2*sizeof(double),stream0>>>(true,true,matrix+i*ld,ld,matrix+i*ld,ld,matrix+i*ld+i,ld,B,B,i);
	  double alpha = -1;
	  double beta = 1; 
	  cublasDsyrk(handle,CUBLAS_FILL_MODE_UPPER,CUBLAS_OP_T,B,i,&alpha,matrix+i*ld,ld,&beta,matrix+i*ld+i,ld);
        }
        cudaStreamSynchronize(stream0);
        cudaMemcpy2DAsync(temp,B*sizeof(double),matrix+i*ld+i,ld*sizeof(double),B*sizeof(double),B,cudaMemcpyDeviceToHost,stream1);
        if(i!=0&&i+B<ld){
          //RKSY<<<dim3((ld-i-B)/b,B/b),dim3(b,b),b*b*2*sizeof(double),stream0>>>(true,false,matrix+i*ld,ld,matrix+(i+B)*ld,ld,matrix+(i+B)*ld+i,ld,B,ld-i-B,i);
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




void test_mydpotrf(int N, int B, int b){
//int N=2048;
// int B=64;
//int b=8;
  char uplo = 'u';
  int size = N*N*sizeof(double);
  double * matrix;
  double * result;
  //double * host_matrix;
  //Memory allocation on RAM and DRAM
  cudaMalloc((void**)&matrix,N*N*sizeof(double));
  cudaMalloc((void**)&result,N*N*sizeof(double));
  cudaMemset(matrix,0,N*N*sizeof(double));
  cudaMemset(result,0,N*N*sizeof(double));

//result = new double[N*N]();
//  host_matrix = new double[N*N]();
  //cudaMalloc(&dev_matrix,size);
  //Generate test data on RAM
  //matrixGenerator(uplo,matrix,result,N);
  matrixGenerator_gpu(uplo,matrix,result,N,64);
  //cudaMemcpy(host_matrix,matrix,size,cudaMemcpyDeviceToHost);
  //printMatrix(host_matrix,N);
  //Copy test data to DRAM
  //cudaMemcpy(dev_matrix,matrix,size,cudaMemcpyHostToDevice);
  //Do Cholesky Factorization
  clock_t t = clock();
  my_dpotrf(uplo,matrix,N,B,b,false,false);
  t=clock()-t;
  double time_in_sec =((double)t/CLOCKS_PER_SEC);
  double gflops = FLOPS_DPOTRF(N)/1000000000;
  cout<<"N:"<<N<<";B:"<<B<<";b:"<<b<<"----";
  cout<<"Time:"<<time_in_sec<<"s, "<<gflops/time_in_sec<<"GFlops/s."<<endl;
  //cudaMemcpy(host_matrix,matrix,size,cudaMemcpyDeviceToHost);
  //PrintMatrix for debug---------------------------------
  //printMatrix(host_matrix,N);
  //printMatrix(result,N);
  //Verify result
  if(resultVerify(result,matrix,N,64)){
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
  //double * matrix;
  //double * result;
  //matrixGenerator_gpu('l',matrix, result, 16);
  //test_mydpotrf(2048,64,8);
  //test_mydpotrf(2048,8,8);
  test_mydpotrf(2048,16,8);
  test_mydpotrf(2048,32,8);
  /*
  test_mydpotrf(2048,32,16);
  test_mydpotrf(2048,64,8);
  test_mydpotrf(2048,64,16);
  test_mydpotrf(2048,64,32);
  
  test_mydpotrf(2048,128,8);
  test_mydpotrf(2048,128,16);
  test_mydpotrf(2048,128,32);
  test_mydpotrf(2048,128,64);
  
  test_mydpotrf(2048,256,8);                                                                                                                                                     
  test_mydpotrf(2048,256,16);                                                                                                                                                    
  test_mydpotrf(2048,256,32);    

  test_mydpotrf(2048,256,64); 
  /*
  test_mydpotrf(2048,8,2);
  test_mydpotrf(2048,8,4);
  
  test_mydpotrf(2048,4,2);
  test_mydpotrf(2048,16,2);
  test_mydpotrf(2048,16,2);
  */
}
