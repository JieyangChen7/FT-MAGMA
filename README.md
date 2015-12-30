## Enhanced Online-ABFT Cholesky Decompostion Benchmark
Author: Jieyang Chen, Xin Liang, Zizhong Chen    
Affiliate: Department of Computer Science & Engineering @ University of California, Riverside

This is the instruction for configuring, installing and benchmarking our Enhanced Online-ABFT Cholesky Decompostion routine. Our program is designed to run on heterogeneous systems with GPUs.

#### File structures:
Each of the following folder contains the complete source code of MAGMA(v1.6.2) with our modified ABFT Cholesky Decompostion routine. Different settings are decribed as follow:
* [magma-1.6.2-enhanced-kepler] - containes Enhanced Online-ABFT Cholesky Decompostion for Nvidia GPUs with Kepler microarchitecture
* [magma-1.6.2-enhanced-fermi] - containes Enhanced Online-ABFT Cholesky Decompostion for Nvidia GPUs with Fermi microarchitecture
* [magma-1.6.2-online-kepler] - containes Online-ABFT Cholesky Decompostion for Nvidia GPUs with Kepler microarchitecture
* [magma-1.6.2-online-fermi] - containes Online-ABFT Cholesky Decompostion for Nvidia GPUs with Fermi microarchitecture
* [magma-1.6.2-offline-kepler] - containes Offline-ABFT Cholesky Decompostion for Nvidia GPUs with Kepler microarchitecture
* [magma-1.6.2-offline-fermi] - containes Offline-ABFT Cholesky Decompostion for Nvidia GPUs with Fermi microarchitecture
* [magma-1.6.2-cula] - containes the benchmark code for original Nvidia CULA's Cholesky Decomposition, no ABFT added. This code has nothing to do with MAGMA, we are just using the MAGMA's testing framework to test CULA. 

#### Configure and Install

Each version of our modified MAGMA has the similar way of configuration and installation as the origianl MAGMA except our program uses PAPI for more accurate timing. 

1. Install CUDA (version 6.5 and above), please refer to: [CUDA](https://developer.nvidia.com/cuda-downloads)
2. Install CPU-side optimizaed BLAS library (ACML, MKL, ATLAS, etc.)
3. Install the current lastest CULA R18, please refer to: [CULA](http://www.culatools.com/)
4. Install the lastest PAPI library, please refer to: [PAPI](http://icl.cs.utk.edu/papi/)
5. clone our code from GitHub
6. Configre make.inc file inside each folder to link with CUDA and CPU-side optimizaed BLAS library. For details, please refer to [MAGMA](http://icl.cs.utk.edu/magma/)
7. After confiure make.inc file, append "-lpapi" to the LIB variable and append "-I$(MAGMA_DIR)/FT" to the INC variable
8. To test CULA, please also append "-lcula_lapack" to the LIB variable and append "-I$(CULA_INC_PATH)" to the INC variable
9. Save the make.inc file and type command "make".
10. Please make sure there is no error before proceeding to the next step

#### Benckmark
1. Please make sure both software and hardware are set properly
2. Please also make sure there no other applications running
3. cd into a MAGMA folder and run command: ./testing/testing_dpotrf_gpu.
4. The program will run benchmarking by decomposes position definite matrices with size from 5120*5120 to maximum GPU memeory allowable size. Execution time will show for each different matrix size.

#### Questions
For any question or suggestion regards to our program, please feel free to send emails to: jchen098@ucr.edu. 

