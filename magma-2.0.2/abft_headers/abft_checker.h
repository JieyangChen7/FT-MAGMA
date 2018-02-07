void abft_checker_colchk(double * dA, int ldda, int m, int n, int nb,
                         double * dA_colchk,    int ldda_colchk,
                         double * dA_colchk_r,  int ldda_colchk_r,
                         double * dev_chk_v,    int ld_dev_chk_v,
                         bool DEBUG,
                         magma_queue_t stream)

void abft_checker_rowchk(double * dA, int ldda, int m, int n, int nb,
                         double * dA_rowchk,    int ldda_rowchk,
                         double * dA_rowchk_r,  int ldda_rowchk_r,
                         double * dev_chk_v,    int ld_dev_chk_v,
                         bool DEBUG,
                         magma_queue_t stream);

void abft_checker_fullchk(double * dA, int ldda, int m, int n, int nb,
                          double * dA_colchk,    int ldda_colchk,
                          double * dA_colchk_r,  int ldda_colchk_r,
                          double * dA_rowchk,    int ldda_rowchk,
                          double * dA_rowchk_r,  int ldda_rowchk_r,
                          double * dev_chk_v,    int ld_dev_chk_v,
                          bool DEBUG,
                          magma_queue_t stream);