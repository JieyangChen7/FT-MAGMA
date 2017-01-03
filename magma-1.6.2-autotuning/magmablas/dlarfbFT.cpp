#include"FT.h"
#include<iostream>
using namespace std;

int dlarfbFT( magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    						int m, int n, int k,
						  	double * dV, int lddv,
						  	double * dT, int lddt,
						  	double * dC, int lddc,
						  	double * dwork, int ldwork,
						  	ABFTEnv * abftEnv,
						  	double * col_chkV, int col_chkV_ld,
							double * row_chkV, int row_chkV_ld, 	
							double * col_chkT, int col_chkT_ld,
							double * row_chkT, int row_chkT_ld, 
							double * col_chkC, int col_chkC_ld,  
							double * row_chkC, int row_chkC_ld, 
							double * col_chkW, int col_chkW_ld,  
							double * row_chkW, int row_chkW_ld,
							bool FT, bool DEBUG, bool CHECK_BEFORE, bool CHECK_AFTER,
							magma_queue_t * stream) {

	#define dV(i_,j_)  (dV    + (i_) + (j_)*lddv)
    #define dT(i_,j_)  (dT    + (i_) + (j_)*lddt)
    #define dC(i_,j_)  (dC    + (i_) + (j_)*lddc)
    #define dwork(i_)  (dwork + (i_))
    
    double c_zero    = MAGMA_D_ZERO;
    double c_one     = MAGMA_D_ONE;
    double c_neg_one = MAGMA_D_NEG_ONE;

    /* Check input arguments */
    magma_int_t info = 0;
    if (m < 0) {
        info = -5;
    } else if (n < 0) {
        info = -6;
    } else if (k < 0) {
        info = -7;
    } else if ( ((storev == MagmaColumnwise) && (side == MagmaLeft) && lddv < max(1,m)) ||
                ((storev == MagmaColumnwise) && (side == MagmaRight) && lddv < max(1,n)) ||
                ((storev == MagmaRowwise) && lddv < k) ) {
        info = -9;
    } else if (lddt < k) {
        info = -11;
    } else if (lddc < max(1,m)) {
        info = -13;
    } else if ( ((side == MagmaLeft) && ldwork < max(1,n)) ||
                ((side == MagmaRight) && ldwork < max(1,m)) ) {
        info = -15;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Function Body */
    if (m <= 0 || n <= 0) {
        return info;
    }

    // opposite of trans
    magma_trans_t transt;
    if (trans == MagmaNoTrans)
        transt = MagmaTrans;
    else
        transt = MagmaNoTrans;
    
    // whether T is upper or lower triangular
    magma_uplo_t uplo;
    if (direct == MagmaForward)
        uplo = MagmaUpper;
    else
        uplo = MagmaLower;
    
    // whether V is stored transposed or not
    magma_trans_t notransV, transV;
    if (storev == MagmaColumnwise) {
        notransV = MagmaNoTrans;
        transV   = MagmaTrans;
    }
    else {
        notransV = MagmaTrans;
        transV   = MagmaNoTrans;
    }

    if ( side == MagmaLeft ) {
        // Form H C or H^H C
        // Comments assume H C. When forming H^H C, T gets transposed via transt.
        //cout << "dlarfb" << endl;
        // W = C^H V
        dgemmFT( MagmaTrans, notransV,
                n, k, m,
                c_one,  dC(0,0),  lddc,
                dV(0,0),  lddv,
                c_zero, dwork(0), ldwork,
                abftEnv,
                col_chkC, col_chkC_ld,  
			    row_chkC, row_chkC_ld,
			    col_chkV, col_chkV_ld,  
			    row_chkV, row_chkV_ld,
			    col_chkW, col_chkW_ld,  
			    row_chkW, row_chkW_ld,
			    FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
			    stream);

        // W = W T^H = C^H V T^H
        dtrmmFT( MagmaRight, uplo, transt, MagmaNonUnit,
                n, k,
                c_one, dT(0,0),  lddt,
                dwork(0), ldwork,
                abftEnv,
                col_chkT, col_chkT_ld,  
			    row_chkT, row_chkT_ld,
			    col_chkW, col_chkW_ld,  
			    row_chkW, row_chkW_ld,
			    FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
			    stream);

        // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C
        dgemmFT( notransV, MagmaTrans,
                m, n, k,
                c_neg_one, dV(0,0),  lddv,
                dwork(0), ldwork,
                c_one,     dC(0,0),  lddc,
                abftEnv,
                col_chkV, col_chkV_ld,  
			    row_chkV, row_chkV_ld,
			    col_chkW, col_chkW_ld,  
			    row_chkW, row_chkW_ld,,
			    col_chkC, col_chkC_ld,  
			    row_chkC, row_chkC_ld,
			    FT, DEBUG, CHECK_BEFORE, CHECK_AFTER,
			    stream);
    }
    else {
        // Form C H or C H^H
        // Comments assume C H. When forming C H^H, T gets transposed via trans.
        
        // W = C V
        magma_dgemm( MagmaNoTrans, notransV,
                     m, k, n,
                     c_one,  dC(0,0),  lddc,
                             dV(0,0),  lddv,
                     c_zero, dwork(0), ldwork);

        // W = W T = C V T
        magma_dtrmm( MagmaRight, uplo, trans, MagmaNonUnit,
                     m, k,
                     c_one, dT(0,0),  lddt,
                            dwork(0), ldwork);

        // C = C - W V^H = C - C V T V^H = C (I - V T V^H) = C H
        magma_dgemm( MagmaNoTrans, transV,
                     m, n, k,
                     c_neg_one, dwork(0), ldwork,
                                dV(0,0),  lddv,
                     c_one,     dC(0,0),  lddc);
    }

    return info;

}