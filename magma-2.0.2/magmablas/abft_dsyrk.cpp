#include "magma_internal.h"
#include"abft_checker.h"

//dsyrk with FT

/**
 * n: number of row of A
 * m: number of col of A
 */
void abft_dsyrk(magma_uplo_t uplo, magma_trans_t trans,
                 int n, int k, 
                 double alpha,
                 double * dA, int ldda,
                 double beta,
                 double * dC, int lddc,
                 double * dA_colchk,    int ldda_colchk,
                 double * dA_rowchk,    int ldda_rowchk,
                 double * dA_colchk_r,  int ldda_colchk_r,
                 double * dA_rowchk_r,  int ldda_rowchk_r,
                 double * dC_colchk,    int lddc_colchk,
                 double * dC_rowchk,    int lddc_rowchk,
                 double * dC_colchk_r,  int lddc_colchk_r,
                 double * dC_rowchk_r,  int lddc_rowchk_r,
                 double * chk_v,        int ld_chk_v, 
                 bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
                 magma_queue_t stream1, magma_queue_t stream2){
    
    /*         k                n
     * ******************   *********
     * *        A       * =>*   C   * n
     * *                *   *       *
     * ******************   *********
     */
    
    if (FT && CHECK_BEFORE) { 
        abft_checker_colchk(dA, ldda, n, n, nb,
                            dA_colchk,   ldda_colchk,
                            dA_colchk_r, ldda_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);

        abft_checker_colchk(dC, ldda, n, n, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);   
    }

    if (FT) {
        magma_dgemm(
                MagmaNoTrans, MagmaTrans,
                n, n, k,
                MAGMA_D_ONE * (-1),
                dA, ldda, dA, ldda,
                MAGMA_D_ONE,
                dC, lddc,
                stream1);
    } else {
        magma_dsyrk(uplo, trans, n, k,
                    alpha, dA, ldda,
                    beta,     dC, lddc,
                    stream1);
    }
    
    if(FT){
        //update checksums on GPU
        magma_dgemm(
                    MagmaNoTrans, MagmaTrans,
                    2, n, k,
                    MAGMA_D_ONE * (-1),
                    dA_colchk,  ldda_colchk,
                    dA,         ldda,
                    MAGMA_D_ONE,
                    dC_colchk,   lddc_colchk,
                    stream1);
    }


    if (FT && CHECK_AFTER) {
        //verify C after update
        abft_checker_colchk(dC, ldda, n, n, nb,
                            dC_colchk,   lddc_colchk,
                            dC_colchk_r, lddc_colchk_r,
                            chk_v,       ld_chk_v,
                            DEBUG,
                            stream1);   


}