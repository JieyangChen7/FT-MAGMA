void colchk_detect_correct(double * dA, int ldda, int nb,
				           double * dA_colchk, 	int ldda_colchk,
				           double * dA_colchk_r, 	int ldda_colchk_r,
						   magma_queue_t stream);

void rowchk_detect_correct(double * dA, int ldda, int nb,
					 	   double * dA_rowchk, 	int ldda_rowchk,
						   double * dA_rowchk_r, 	int ldda_rowchk_r,
						   magma_queue_t stream);