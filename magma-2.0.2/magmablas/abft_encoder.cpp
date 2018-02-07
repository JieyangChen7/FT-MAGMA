void col_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * dcolchk, int ld_dcolchk, 
                 magma_queue_t stream) {

    for (int i = 0; i < m; i += nb) {        
        magma_dgemm(MagmaTrans, MagmaNoTrans,
                    2, n, nb,
                    MAGMA_D_ONE, 
                    chk_v, ld_chk_v,
                    A + i, lda,
                    MAGMA_D_ZERO, dcolchk + (i / nb) * 2, ld_dcolchk);           
    }
}

void row_chk_enc(int m, int n, int nb, 
                 double * A, int lda,
                 double * chk_v, int ld_chk_v,
                 double * drowchk, int ld_drowchk, 
                 magma_queue_t stream) {

    for (int i = 0; i < n; i += nb) {        
        magma_dgemm(MagmaNoTrans, MagmaNoTrans,
                    m, 2, nb,
                    MAGMA_D_ONE, 
                    A + i * lda, lda,
                    chk_v, ld_chk_v,
                    MAGMA_D_ZERO, drowchk + ((i / nb) * 2) * ld_drowchk, ld_drowchk);           
    }
}