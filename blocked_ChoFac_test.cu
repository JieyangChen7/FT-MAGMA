/*Blocked Cholesky Factorization with Fault tolerance.
dpotf on CPU and dtrsm on GPU, dgemm on GPU. Compute either upper or lower. Initial data is on GPU, so transfer the data to GPU is not taken care of.
*Jieyang Chen, University of California, Riverside
**/

//Initial Data on GPU
//Hybird GPU (DTRSM & DGEMM)and CPU (DPOTRF) version MAGMA way
//Column Major
//Either upper and lower triangle
//testing function are made to facilitate testing
//CPU and GPU are asynchronized
//CUBLAS are used in DTRSM & DGEMM
//Leading Dimension is used
//Add CUDA Event timing
#include<iostream>
#include<cstdlib>
#include<iomanip>
#include<cmath> 
#include<ctime>
#include"cublas_v2.h"
#include<cuda_runtime.h>
#include<curand.h>
#include"acml.h"
#include"papi.h"
#include"printHelper.h"
#include"matrixGenerator.h"
#include"dpotrfFT.h"
#include"dtrsmFT.h"
#include"dsyrkFT.h"
#include"dgemmFT.h"
#include"checksumGenerator.h"


#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )

using namespace std;


void my_dpotrf(char uplo, double * matrix, int ld, int N, int B,
		float * real_time, float * proc_time, long long * flpins,
		float * mflops) {
	
	
	double * temp;
	cudaHostAlloc((void**) &temp, B * B * sizeof(double), cudaHostAllocDefault);
	
	//intial streams----------------------------
	cudaStream_t stream0;  //for main loop
	cudaStream_t stream1;  //for dgemm part
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	//cout<<"Streams initialized"<<endl;
	//intial cublas
	cublasStatus_t cublasStatus;
	cublasHandle_t handle1;
	cublasStatus = cublasCreate(&handle1);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
		cout << "CUBLAS NOT INITIALIZED(handle1) in my_dpotrf " << endl;
	cublasStatus = cublasSetStream(handle1, stream1);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
		cout << "CUBLAS SET STREAM NOT INITIALIZED(handle1) in my_dpotrf"
				<< endl;   

	
	if (PAPI_flops(real_time, proc_time, flpins, mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	
	//intialize checksum1 and checksum2
	double * v1=new double[B];
	double * v2=new double[B];
	for(int i=0;i<B;i++){
		v1[i]=1;
		v2[i]=i+1;
	}
	
	double * v1d;
	size_t v1d_pitch;
	cudaMallocPitch((void**) &v1d, &v1d_pitch, N * sizeof(double), 1);
	cudaMemcpy2D(v1d, v1d_pitch, v1, N * sizeof(double), N * sizeof(double),
						1, cudaMemcpyHostToDevice);
	double * v2d;
	size_t v2d_pitch;
	cudaMallocPitch((void**) &v2d, &v2d_pitch, N * sizeof(double), 1);
	cudaMemcpy2D(v2d, v2d_pitch, v2, N * sizeof(double), N * sizeof(double),
								1, cudaMemcpyHostToDevice);
	
	//allocate space for recalculated checksum on CPU
	double * chk1 = new double[B];
	double * chk2 = new double[B];
	
	//allocate space for reclaculated checksum on CPU
	double * chk1d;
	double * chk2d;
	size_t chk1d_pitch;
	size_t chk2d_pitch;

	cudaMallocPitch((void**) &chk1d, &chk1d_pitch, (N / B) * sizeof(double), B);
	cudaMallocPitch((void**) &chk2d, &chk2d_pitch, (N / B) * sizeof(double), B);

	int chk1d_ld = chk1d_pitch / sizeof(double);
	int chk2d_ld = chk2d_pitch / sizeof(double);
	
	
	
			
			
	/*cout<<"checksum vector 1 on CPU:"<<endl;
	printVector_host(v1,B);
	cout<<"checksum vector 2 on CPU:"<<endl;
	printVector_host(v2,B);
	*/
	
	size_t checksum1_pitch;
	size_t checksum2_pitch;
	double * checksum1=initializeChecksum(handle1, matrix, ld, N, B, v1d, checksum1_pitch);
	double * checksum2=initializeChecksum(handle1, matrix, ld, N, B, v2d, checksum2_pitch);
	int checksum1_ld = checksum1_pitch/sizeof(double);
	int checksum2_ld = checksum2_pitch/sizeof(double);
	
	

	double one = 1;
	double negone = -1;
	double zero = 0;
	
	for (int i = 0; i < N; i += B) {
		
		
		/*cout<<"i="<<i<<endl;
		cout<<"matrix:"<<endl;
		printMatrix_gpu(matrix, ld*sizeof(double), N, N);
		cout<<"checksum1:"<<endl;
		printMatrix_gpu(checksum1, checksum1_pitch, N/B, N);
		cout<<"checksum2:"<<endl;
		printMatrix_gpu(checksum2, checksum2_pitch, N/B, N);
		*/
		if (i > 0) {
			dsyrkFT(handle1, B, i, matrix + i, ld, matrix + i * ld + i, ld,
					checksum1+i/B, checksum1_ld, checksum2+i/B, checksum2_ld,
					checksum1 + (i/B) + i*checksum1_ld,checksum1_ld, 
					checksum2 + (i/B) + i*checksum2_ld,checksum2_ld,
					v1d, v2d,
					chk1d, chk1d_ld, chk2d, chk2d_ld);
			
			/*cublasDsyrk(handle1, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, B, i,
					&negone, matrix + i, ld, &one, matrix + i * ld + i,
					ld);
			*/
		}
		
		cudaStreamSynchronize(stream1);
		
		cudaMemcpy2DAsync(temp, B * sizeof(double), matrix + i * ld + i,
				ld * sizeof(double), B * sizeof(double), B,
				cudaMemcpyDeviceToHost, stream0);
		
		
		cudaMemcpy2DAsync(chk1, 1 * sizeof(double), checksum1 + (i/B) + i*checksum1_ld,
				checksum1_pitch, 1 * sizeof(double), B,
				cudaMemcpyDeviceToHost, stream0);
		cudaMemcpy2DAsync(chk2, 1 * sizeof(double), checksum2 + (i/B) + i*checksum2_ld,
				checksum2_pitch, 1 * sizeof(double), B,
				cudaMemcpyDeviceToHost, stream0);
		
		if (i != 0 && i + B < N) {
			
			dgemmFT(handle1, N - i - B, B, i, matrix + (i + B), ld,
					matrix + i, ld, matrix + i * ld + (i + B), ld,
					checksum1+(i + B)/B, checksum1_ld, 
					checksum2+(i + B)/B, checksum2_ld, 
					checksum1 + i * checksum1_ld + (i + B)/B, checksum1_ld,
					checksum2 + i * checksum2_ld + (i + B)/B, checksum2_ld,
					v1d, v2d,
					chk1d, chk1d_ld, chk2d, chk2d_ld);
			
			/*cublasDgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_T, N - i - B, B, i,
					&negone, matrix + (i + B), ld, matrix + i, ld,
					&one, matrix + i * ld + (i + B), ld);
			 */                                               
		}
		cudaStreamSynchronize(stream0);
		
		//int info;
		//dpotrf('L', B, temp, B, &info);
		dpotrfFT(temp, B, B, chk1, 1, chk2, 1, v1, v2);
		
		
		
		cudaMemcpy2DAsync(matrix + i * ld + i, ld * sizeof(double), temp,
				B * sizeof(double), B * sizeof(double), B,
				cudaMemcpyHostToDevice, stream0);
		
		cudaMemcpy2DAsync(checksum1 + (i/B) + i*checksum1_ld,checksum1_pitch, chk1, 1 * sizeof(double), 
				1 * sizeof(double), B,
				cudaMemcpyHostToDevice, stream0);
		cudaMemcpy2DAsync(checksum2 + (i/B) + i*checksum2_ld,checksum2_pitch, chk2, 1 * sizeof(double), 
				1 * sizeof(double), B,
				cudaMemcpyHostToDevice, stream0);
		
	
		//update B                                                                      
		if (i + B < N) {
			cudaStreamSynchronize(stream0);
			dtrsmFT(handle1, N - i - B, B, matrix + i * ld + i, ld, matrix + i * ld + i + B, ld, 
				checksum1+(i+B)/B+i*checksum1_ld, checksum1_ld, checksum2+(i+B)/B+i*checksum2_ld, checksum2_ld,
				v1d, v2d,
				chk1d, chk1d_ld, chk2d, chk2d_ld);
			/*cublasDtrsm(handle1, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
					CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, N - i - B, B,  &one,
					matrix + i * ld + i, ld, matrix + i * ld + i + B, ld);
			*/
		}

	}
	
	if (PAPI_flops(real_time, proc_time, flpins, mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	
	cublasDestroy(handle1);
	cudaFreeHost(temp);
	PAPI_shutdown();

}

void test_mydpotrf(int N, int B, float * real_time, float * proc_time,
		long long * flpins, float * mflops) {

	char uplo = 'l';
	double * matrix;
	double * result;
	size_t matrix_pitch;
	size_t result_pitch;
	//Memory allocation on RAM and DRAM
	cudaMallocPitch((void**) &matrix, &matrix_pitch, N * sizeof(double), N);
	cudaMallocPitch((void**) &result, &result_pitch, N * sizeof(double), N);

	int matrix_ld = matrix_pitch / sizeof(double);
	int result_ld = result_pitch / sizeof(double);

	matrixGenerator_gpu(uplo, matrix, matrix_ld, result, result_ld, N, 2);

	my_dpotrf(uplo, matrix, matrix_ld, N, B, real_time, proc_time, flpins,
			mflops);

	//Verify result
	/*if(resultVerify_gpu(result,result_ld,matrix,matrix_ld,N,2)){
	cout<<"Result passed!"<<endl;
	}else{
	  cout<<"Result failed!"<<endl;
	}
	*/
	cudaFree(matrix);
	cudaFree(result);

}

int main(int argc, char**argv) {

	int TEST_NUM = 1;
	int n[10] = { 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240 };
	int b = 256; 
	for (int k = 9; k < 10; k++) {
		float total_real_time = 0.0;
		float total_proc_time = 0.0;
		long long total_flpins = 0.0;
		float total_mflops = 0.0;
		float real_time = 0.0;
		float proc_time = 0.0;
		long long flpins = 0.0;
		float mflops = 0.0;
		double flops = FLOPS_DPOTRF(n[k]) / 1e9;
		//cout<<"flops:"<<flops<<"  ";

		for (int i = 0; i < TEST_NUM; i++) {
			test_mydpotrf(n[k], b, &real_time, &proc_time, &flpins, &mflops);
			total_real_time += real_time;
			total_proc_time += proc_time;
			total_flpins += flpins;
			total_mflops += mflops;
		}
		cout << "Size:" << n[k] << "(" << b << ")---Real_time:"
				<< total_real_time / (double) TEST_NUM << "---" << "Proc_time:"
				<< total_proc_time / (double) TEST_NUM << "---"
				<< "Total GFlops:"
				<< flops / (total_proc_time / (double) TEST_NUM) << endl;
	}
}
