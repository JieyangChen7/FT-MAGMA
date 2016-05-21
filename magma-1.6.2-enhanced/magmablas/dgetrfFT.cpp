#include"FT.h"
#include<iostream>

using namespace std;

//LU Factorization with FT on CPU using ACML


void dgetrfFT(int m, int n, double * A, int lda, int * ipiv, int * infom
              double * chksum, int chksum_ld,
              double * v, int v_ld,
              bool FT , bool DEBUG, bool VERIFY) {
    
    
    lapackf77_dgetrf( &m, &n, A, &lda, ipiv, info);
    
    if (FT) {
        
    }
}

