/*Blocked Cholesky Factorization with Fault tolerance.
 *potf on CPU and dtrsm on GPU, dgemm on GPU. Compute either upper or lower. Initial data is on GPU, so transfer the data to GPU is not taken care of.
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

#define FMULS_POTRF(__n) ((__n) * (((1. / 6.) * (__n) + 0.5) * (__n) + (1. / 3.)))
#define FADDS_POTRF(__n) ((__n) * (((1. / 6.) * (__n)      ) * (__n) - (1. / 6.)))
#define FLOPS_DPOTRF(__n) (FMULS_POTRF((double)(__n))+FADDS_POTRF((double)(__n)) )

using namespace std;

void printMatrix_host(double * matrix_host, int N);
void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int N);
void POTF2_CPU(char uplo, double * matrix, int ld, int B);
__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo,
		double alpha, double beta);
void matrixGenerator_gpu(char uplo, double * matrix, int matrix_ld,
		double * result, int result_ld, int N, int B);
__global__ void resultVerify_gpu_help(double * realResult, int real_ld,
		double * testResult, int test_ld, double * diff, int N);
bool resultVerify_gpu(double * realResult, int real_ld, double * testResult,
		int test_ld, int N, int B);
void my_dpotrf(char uplo, double * matrix, int ld, int N, int B,
		float * real_time, float * proc_time, long long * flpins,
		float * mflops);

void printMatrix_host(double * matrix_host, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			cout.width(5);
			cout.setf(ios::left);
			cout << matrix_host[j * N + i];
		}
		cout << endl;
	}
	cout << endl;
}

void printMatrix_gpu(double * matrix_device, size_t matrix_pitch, int N) {
	double * matrix_host = new double[N * N]();
	cudaMemcpy2D(matrix_host, N * sizeof(double), matrix_device, matrix_pitch,
			N * sizeof(double), N, cudaMemcpyDeviceToHost);
	printMatrix_host(matrix_host, N);
	delete[] matrix_host;
}

void printVector_host(double * vector_host, int N) {
	for (int i = 0; i < N; i++) {
		cout.width(5);
		cout.setf(ios::left);
		cout << vector_host[i];
	}
	cout << endl;
}

void printVector_gpu(double * vector_device, int N) {
	double * vector_host = new double[N]();
	cudaMemcpy(vector_host, vector_device, N * sizeof(double),
			cudaMemcpyDeviceToHost);
	printVector_host(vector_host, N);
	delete[] vector_host;
}

__global__ void matrixDiagonalizeAndScale(double * matrix, int ld, char uplo,
		double alpha, double beta) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (uplo == 'u') {
		if (row < col + 1) {
			matrix[col * ld + row] = int(matrix[col * ld + row] * alpha + beta);
		} else {
			matrix[col * ld + row] = int(0.0);
		}
	} else {
		if (col < row + 1) {
			matrix[col * ld + row] = int(matrix[col * ld + row] * alpha + beta);
		} else {
			matrix[col * ld + row] = int(0.0);
		}
	}
}

void matrixGenerator_gpu(char uplo, double * matrix, int matrix_ld,
		double * result, int result_ld, int N, int B) {
	double a = 10.0;
	//initialize cublas
	cublasStatus_t cublasStatus;
	cublasHandle_t handle;
	cublasStatus = cublasCreate(&handle);
	if (cublasStatus != CUBLAS_STATUS_SUCCESS)
		cout << "CUBLAS NOT INITIALIZED(handle1) in matrixGenerator_gpu"
				<< endl;

	//initialize curand
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 10ULL);
	//generate random number in range (0,1] on result using curand
	curandGenerateUniformDouble(gen, result, result_ld * N);
	cudaDeviceSynchronize();

	//print result
	//printMatrix_gpu(result,result_ld*sizeof(double),N);
	matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(result, result_ld, uplo, a,1);
	cudaDeviceSynchronize();

	//do matrix-matrix multiplcation using cublas
	cudaMemset(matrix, 0, matrix_ld * N * sizeof(double));

	double alpha = 1.0;
	double beta = 1.0;
	if (uplo == 'u') {
		cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, result,
				result_ld, result, result_ld, &beta, matrix, matrix_ld);
	} else if (uplo == 'l') {
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, N, N, &alpha, result,
				result_ld, result, result_ld, &beta, matrix, matrix_ld);
	}
	cudaDeviceSynchronize();

	matrixDiagonalizeAndScale<<<dim3(N/B,N/B),dim3(B,B)>>>(matrix, matrix_ld, uplo, 1.0,0);
	cudaDeviceSynchronize();

	//print matrix
	printMatrix_gpu(matrix, matrix_ld * sizeof(double), N);
}

__global__ void resultVerify_gpu_help(double * realResult, int real_ld,
		double * testResult, int test_ld, double * diff, int N) {
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	diff[col * N + row] = testResult[col * test_ld + row]
			- realResult[col * real_ld + row];
}

bool resultVerify_gpu(double * realResult, int real_ld, double * testResult,
		int test_ld, int N, int B) {
	double * diff;
	cudaMalloc((void**) &diff, N * N * sizeof(double));
	resultVerify_gpu_help<<<dim3(N/B,N/B),dim3(B,B)>>>(realResult,real_ld,testResult,test_ld,diff,N);

	//printMatrix_gpu(realResult,real_ld*sizeof(double),N);
	//printMatrix_gpu(testResult,test_ld*sizeof(double),N);

	double * diff_host = new double[N * N]();
	cudaMemcpy(diff_host, diff, N * N * sizeof(double), cudaMemcpyDeviceToHost);
	//  printMatrix(diff_host,N);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (abs(diff_host[i * N + j]) > 1e-3) {
				//  cout<<"diff:"<<abs(diff_host[i*N+j])<<endl;
				delete[] diff_host;
				cudaFree(diff);
				return false;
			}
		}
	}
	delete[] diff_host;
	cudaFree(diff);
	return true;

}








void cublasDsyrkFT(cublasHandle_t handle, cublasFillMode_t uplo,
		cublasOperation_t trans, int n, int k, const double * alpha,
		const double * A, int lda, const double * beta, double * C, int ldc,
		double * checkA, double * checkC) {

	cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);

}

double * initializeChecksum(cublasHandle_t handle, double * matrix, int ld, int N, int B, double * v) {

	double * vd;
	size_t vd_pitch;
	cudaMallocPitch((void**) &vd, &vd_pitch, B * sizeof(double), 1);
	cudaMemcpy2DAsync(vd, vd_pitch, v, B * sizeof(double), B * sizeof(double),
			1, cudaMemcpyHostToDevice);

	double * chksum;
	size_t chksum_pitch;
	cudaMallocPitch((void**) &chksum, &chksum_pitch, (N / B) * sizeof(double), N);
	cudaMemset2D((void*) chksum, chksum_pitch, 0, (N / B) * sizeof(double), N);
	int chksum_ld = chksum_pitch / sizeof(double);

	double alpha = 1;
	double beta = 0;
	for (int i = 0; i < N; i += B) {
		cublasDgemv(handle, CUBLAS_OP_T, N, B, &alpha, matrix + i, ld, vd, 1,
				&beta, chksum + (i / B), chksum_ld);
	}
	return chksum;

}

void my_dpotrf(char uplo, double * matrix, int ld, int N, int B,
		float * real_time, float * proc_time, long long * flpins,
		float * mflops) {
	//cout<<"start my_dpotrf"<<endl;
	//initial data
	//int b_size = B;
	double * temp;
	//float gemm_time =0;
	//float cpu_time =0;
	cudaHostAlloc((void**) &temp, B * B * sizeof(double), cudaHostAllocDefault);
	//cout<<"pinned memory initialized"<<endl;
	//intial streams----------------------------
	cudaStream_t stream0;  //for main loop
	cudaStream_t stream1;  //for dgemm part
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	//cout<<"Streams initialized"<<endl;
	//intial cublas
	cublasStatus_t cublasStatus;
	//cublasHandle_t handle0;
	//cublasStatus = cublasCreate(&handle0);
	//if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	//  cout<<"CUBLAS NOT INITIALIZED(handle0)"<<endl;
	//cublasStatus = cublasSetStream(handle0,stream0);
	//if(cublasStatus != CUBLAS_STATUS_SUCCESS)
	//  cout<<"CUBLAS SET STREAM NOT INITIALIZED(handle0)"<<endl;

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
		v2[i]=i;
	}
	double * checksum1=initializeChecksum(handle1, matrix, ld, N, B, v1);
	double * checksum2=initializeChecksum(handle1, matrix, ld, N, B, v2);

	
	for (int i = 0; i < N; i += B) {
		//b_size = min(B,N-i);
		//cout<<"block size:"<<b_size<<"  ";
		

		if (i > 0) {

			//prepare for checkA

			double alpha = -1;
			double beta = 1;
			//cudaEventRecord(start0,stream0);
			cublasDsyrk(handle1, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, B, i,
					&alpha, matrix + i * ld, ld, &beta, matrix + i * ld + i,
					ld);
			//cudaEventRecord(stop0,stream0);
		}
		
		cudaStreamSynchronize(stream1);
		//cudaEventRecord(start0,stream0);
		//cudaHostAlloc((void**)&temp,b_size*b_size*sizeof(double),cudaHostAllocDefault);
		cudaMemcpy2DAsync(temp, B * sizeof(double), matrix + i * ld + i,
				ld * sizeof(double), B * sizeof(double), B,
				cudaMemcpyDeviceToHost, stream0);

		if (i != 0 && i + B < N) {
			double alpha = -1;
			double beta = 1;
			//cudaEventRecord(start1,stream1);                                                   
			cublasDgemm(handle1, CUBLAS_OP_T, CUBLAS_OP_N, B, N - i - B, i,
					&alpha, matrix + i * ld, ld, matrix + (i + B) * ld, ld,
					&beta, matrix + (i + B) * ld + i, ld);
			//cudaEventRecord(stop1,stream1);                                                    
		}
		cudaStreamSynchronize(stream0);
		int info;
		dpotrf('U', B, temp, B, &info);
		cudaMemcpy2DAsync(matrix + i * ld + i, ld * sizeof(double), temp,
				B * sizeof(double), B * sizeof(double), B,
				cudaMemcpyHostToDevice, stream0);
		//cudaEventRecord(stop0,stream0);

		/*if(i!=0&&i+b_size<ld){
		 cudaEventSynchronize(stop1);
		 cudaEventElapsedTime(&gemm_time,start1,stop1);
		 cout<<"GEMM: "<<gemm_time<<"ms  ";
		 }


		 cudaEventSynchronize(stop0);
		 cudaEventElapsedTime(&cpu_time,start0,stop0);
		 cout<<"CPU: "<<cpu_time<<"ms  "<<endl;
		 */
		//update B                                                                      
		if (i + B < N) {
			//cudaStreamSynchronize(stream1);
			cudaStreamSynchronize(stream0);
			double alpha = 1;
			//cudaEventRecord(start0,stream0);
			cublasDtrsm(handle1, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
					CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, B, N - i - B, &alpha,
					matrix + i * ld + i, ld, matrix + (i + B) * ld + i, ld);
			/*cudaEventRecord(stop0,stream0);
			 cudaEventSynchronize(stop0);
			 cudaEventElapsedTime(&t,start0,stop0);
			 cout<<"TRSM: "<<t<<"ms  "<<endl;*/
		}
		//cudaFreeHost(temp);
	}
	//  t=clock()-t;
	//  float time =((float)t/CLOCKS_PER_SEC);
	//  cout<<"time[N="<<N<<"B="<<B<<"]:"<<time<<"s."<<endl;

	if (PAPI_flops(real_time, proc_time, flpins, mflops) < PAPI_OK) {
		cout << "PAPI ERROR" << endl;
		return;
	}
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);
	//  cublasDestroy(handle0);
	cublasDestroy(handle1);
	cudaFreeHost(temp);
	PAPI_shutdown();

}

void test_mydpotrf(int N, int B, float * real_time, float * proc_time,
		long long * flpins, float * mflops) {

	char uplo = 'u';
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
	//if(resultVerify_gpu(result,result_ld,matrix,matrix_ld,N,2)){
	//cout<<"Result passed!"<<endl;
	//}else{
	//  cout<<"Result failed!"<<endl;
	// }

	cudaFree(matrix);
	cudaFree(result);

}

int main(int argc, char**argv) {

	int TEST_NUM = 1;
	int n[10] = { 16, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240 };
	int b = 2;
	for (int k = 0; k < 1; k++) {
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
