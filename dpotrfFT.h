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
	dpotrf('L', B, temp, B, &info);
	
	//recalculate checksum1 and checksum2
	double * v1 = new double[B];
	double * v2 = new double[B];
	double * chk1 = new double[B];
	double * chk2 = new double[B];
	for (int i = 0; i < B; i++) {
		v1[i] = 1;
		v2[i] = i+1;
	}
	double alpha = 1;
	double beta = 0;
	dgemv('T', n, n, &alpha, temp, B, v1, 1, &beta, chk1, 1);
	dgemv('T', n, n, &alpha, temp, B, v2, 1, &beta, chk2, 1);

	//update checksum1 and checksum2
	for (int i = 0; i < B; i++) {
		chksum1[i] = chksum1[i] / get(temp, B, B, i, i);
		for (int j = i + 1; j < B; j++) {
			chksum1[j] = chksum1[j] - chksum1[i] * get(temp, B, B, j, i);
		}
	}

	for (int i = 0; i < B; i++) {
		chksum2[i] = chksum2[i] / get(temp, B, B, i, i);
		for (int j = i + 1; j < B; j++) {
			chksum2[j] = chksum2[j] - chksum2[i] * get(temp, B, B, j, i);
		}
	}

	//checking error to be finished
	for(int i=0;i<B;i++){
		double diff = abs(chk1[i]-chksum1[i]);
		if(diff>0.1){//error detected
			//determine position
			double diff2 = abs(chk2[i]-chksum2[i]);
			int j=(int)round(diff2/diff1)-1;
			//correct error
			*(matrix+i*B+j) += chksum1[i] - chk1[i];
		}
	}
	
}