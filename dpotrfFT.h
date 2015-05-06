using namespace std;
//Cholesky Factorization with FT on CPU using ACML
double get(double * matrix, int ld, int n, int i, int j) {
	if (i > ld || j > n)
		cout << "matrix_get_error" << endl;
	return *(matrix + j * ld + i);
}
/**
 * Cholesky factorization with FT on CPU using ACML
 * A: input matrix
 * lda: leading dimension of A
 * n: size of A
 * chksum1: checksum 1
 * inc1: stride between elememts in chksum1
 * chksum2: checksum 2
 * inc2: stride between elememts in chksum2
 */
void dpotrfFT(double * A, int lda, int n, 
		double * chksum1, int inc1, double * chksum2, int inc2, 
		double * v1, double * v2, bool FT ) {
	double alpha = 1;
	double beta = 0;
	/*double * v1 = new double[n];
	double * v2 = new double[n];
	for (int i = 0; i < n; i++) {
			v1[i] = 1;
			v2[i] = i+1;
	}
	*/
	//cout<<"matrix A before dpotrf:"<<endl;
	//printMatrix_host(A,n,n);
	
	//do Choleksy factorization
	int info;
	//dpotrf('L', n, A, n, &info);
	
	if(FT){
	
		//cout<<"matrix A after dpotrf:"<<endl;
		//printMatrix_host(A,n,n);
		
		/*cout<<"checksum on CPU before factorization:"<<endl;
		printVector_host(chksum1, n);
		printVector_host(chksum2, n);
		*/
		//recalculate checksum1 and checksum2
		/*
		double * chk1 = new double[n];
		double * chk2 = new double[n];
		dgemv('T', n, n, alpha, A, lda, v1, 1, beta, chk1, 1);
		dgemv('T', n, n, alpha, A, lda, v2, 1, beta, chk2, 1);
		*/
		//cout<<"recalcuated checksum on CPU after factorization:"<<endl;
		//printVector_host(chk1, n);
		//printVector_host(chk2, n);
		
		double * negone = -1;
		//update checksum1 and checksum2
		for (int i = 0; i < n; i++) {
			chksum1[i] = chksum1[i] / get(A, n, n, i, i);
			daxpy(n-i-1, negone*checksum1[i], A + i*lda + i+1, 1, checksum1 + i+1, 1 );
			/*for (int j = i + 1; j < n; j++) {
				chksum1[j] = chksum1[j] - chksum1[i] * get(A, n, n, j, i);
			}
			*/
		}
	
		for (int i = 0; i < n; i++) {
			chksum2[i] = chksum2[i] / get(A, n, n, i, i);
			daxpy(n-i-1, negone*checksum2[i], A + i*lda + i+1, 1, checksum2 + i+1, 1 );
			/*for (int j = i + 1; j < n; j++) {
				chksum2[j] = chksum2[j] - chksum2[i] * get(A, n, n, j, i);
			}
			*/
		}
		
		//cout<<"updated checksum on CPU after factorization:"<<endl;
		//printVector_host(chksum1, n);
		//printVector_host(chksum2, n);
	
		//checking error to be finished
		/*for(int i=0;i<n;i++){
			double diff = abs(chk1[i]-chksum1[i]);
			if(diff>0.1){//error detected
				//determine position
				cout<<"Error detected in dpotrf"<<endl;
				double diff2 = abs(chk2[i]-chksum2[i]);
				int j=(int)round(diff2/diff)-1;
				//correct error
				*(A+i*lda+j) += chksum1[i] - chk1[i];
			}
		}
		*/
	}
}