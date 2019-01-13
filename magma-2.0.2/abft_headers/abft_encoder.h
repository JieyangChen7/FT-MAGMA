void col_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * dcolchk, int ld_dcolchk, 
                 magma_queue_t stream);

void row_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * drowchk, int ld_drowchk, 
                 magma_queue_t stream);

void col_chk_enc(int m, int n, int nb, 
                 float * A, int lda,
                 float * chk_v, int ld_chk_v,
                 float * dcolchk, int ld_dcolchk, 
                 magma_queue_t stream);

void row_chk_enc(int m, int n, int nb, 
                 float * A, int lda,
                 float * chk_v, int ld_chk_v,
                 float * drowchk, int ld_drowchk, 
                 magma_queue_t stream);