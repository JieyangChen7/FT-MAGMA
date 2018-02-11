// #include "magma.h"
// #include "magma_lapack.h"
// #include "magmablas.h"
// #include  <ctime>
#include "magma_internal.h"

void printMatrix_host(double * matrix_host, int ld,  int M, int N, int row_block, int col_block);
void printMatrix_host_int(int * matrix_host, int ld,  int M, int N, int row_block, int col_block);
void printMatrix_gpu(double * matrix_device, int matrix_ld, int M, int N, int row_block, int col_block, magma_queue_t stream);

void printVector_host(double * vector_host, int N);
void printVector_gpu(double * vector_device, int N) ;

