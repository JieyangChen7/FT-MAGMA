using namespace std;
//Cholesky Factorization with FT on CPU using ACML
double get(double * matrix, int ld, int n, int i, int j) {
	if (i > ld || j > n)
		cout << "matrix_get_error" << endl;
	return matrix + j * ld + i;
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
void dpotrfFT(double * A, int lda, int n, double * chksum1, int inc1, double * chksum2, int inc2 ) {
	//do Choleksy factorization
	int info;
	dpotrf('L', n, A, n, &info);
	
	//recalculate checksum1 and checksum2
	double * v1 = new double[n];
	double * v2 = new double[n];
	double * chk1 = new double[n];
	double * chk2 = new double[n];
	for (int i = 0; i < B; i++) {
		v1[i] = 1;
		v2[i] = i+1;
	}
	double alpha = 1;
	double beta = 0;
	dgemv('T', n, n, alpha, A, lda, v1, 1, beta, chk1, 1);
	dgemv('T', n, n, alpha, A, lda, v2, 1, beta, chk2, 1);

	//update checksum1 and checksum2
	for (int i = 0; i < n; i++) {
		chksum1[i] = chksum1[i] / get(A, n, n, i, i);
		for (int j = i + 1; j < n; j++) {
			chksum1[j] = chksum1[j] - chksum1[i] * get(A, n, n, j, i);
		}
	}

	for (int i = 0; i < n; i++) {
		chksum2[i] = chksum2[i] / get(A, n, n, i, i);
		for (int j = i + 1; j < n; j++) {
			chksum2[j] = chksum2[j] - chksum2[i] * get(A, n, n, j, i);
		}
	}

	//checking error to be finished
	for(int i=0;i<n;i++){
		double diff = abs(chk1[i]-chksum1[i]);
		if(diff>0.1){//error detected
			//determine position
			double diff2 = abs(chk2[i]-chksum2[i]);
			int j=(int)round(diff2/diff)-1;
			//correct error
			*(A+i*n+j) += chksum1[i] - chk1[i];
		}
	}
	
}