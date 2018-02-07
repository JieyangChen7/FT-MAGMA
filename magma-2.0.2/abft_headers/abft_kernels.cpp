void abft_dpotf2(const char uplo, int n, double * A, int lda, int * info, 
			  	 int nb, 
			     double * colchk, int ld_colchk, 
			     double * rowchk, int ld_rowchk, 
			     double * chk_v, int ld_chk_v, 
			     bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER);