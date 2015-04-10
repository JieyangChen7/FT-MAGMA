!**
!
! @file plasmaf.h
!
!  PLASMA FORTRAN header
!  PLASMA is a software package provided by Univ. of Tennessee,
!  Univ. of California Berkeley and Univ. of Colorado Denver
!
! @version 2.6.0
! @author Bilel Hadri
! @author Mathieu Faverge
! @date 2010-11-15
!
!**

!********************************************************************
!   PLASMA constants - precisions
!
      integer  PlasmaByte, PlasmaInteger, PlasmaRealFloat
      integer  PlasmaRealDouble, PlasmaComplexFloat, PlasmaComplexDouble
      parameter ( PlasmaByte          = 0 )
      parameter ( PlasmaInteger       = 1 )
      parameter ( PlasmaRealFloat     = 2 )
      parameter ( PlasmaRealDouble    = 3 )
      parameter ( PlasmaComplexFloat  = 4 )
      parameter ( PlasmaComplexDouble = 5 )

!********************************************************************
!   PLASMA constants - CBLAS & LAPACK
!
      integer PlasmaRM, PlasmaCM, PlasmaCCRB
      integer PlasmaCRRB, PlasmaRCRB, PlasmaRRRB
      parameter ( PlasmaRM         = 101 )
      parameter ( PlasmaCM         = 102 )
      parameter ( PlasmaCCRB       = 103 )
      parameter ( PlasmaCRRB       = 104 )
      parameter ( PlasmaRCRB       = 105 )
      parameter ( PlasmaRRRB       = 106 )

      integer  PlasmaNoTrans, PlasmaTrans, PlasmaConjTrans
      parameter ( PlasmaNoTrans    = 111 )
      parameter ( PlasmaTrans      = 112 )
      parameter ( PlasmaConjTrans  = 113 )

      integer PlasmaUpper, PlasmaLower
      integer PlasmaUpperLower
      parameter ( PlasmaUpper      = 121 )
      parameter ( PlasmaLower      = 122 )
      parameter ( PlasmaUpperLower = 123 )

      integer PlasmaNonUnit,PlasmaUnit
      parameter ( PlasmaNonUnit    = 131 )
      parameter ( PlasmaUnit       = 132 )

      integer PlasmaLeft,PlasmaRight
      parameter ( PlasmaLeft       = 141 )
      parameter ( PlasmaRight      = 142 )

      integer PlasmaOneNorm, PlasmaRealOneNorm
      integer PlasmaTwoNorm, PlasmaFrobeniusNorm
      integer PlasmaInfNorm, PlasmaRealInfNorm
      integer PlasmaMaxNorm, PlasmaRealMaxNorm
      parameter ( PlasmaOneNorm       = 171 )
      parameter ( PlasmaRealOneNorm   = 172 )
      parameter ( PlasmaTwoNorm       = 173 )
      parameter ( PlasmaFrobeniusNorm = 174 )
      parameter ( PlasmaInfNorm       = 175 )
      parameter ( PlasmaRealInfNorm   = 176 )
      parameter ( PlasmaMaxNorm       = 177 )
      parameter ( PlasmaRealMaxNorm   = 178 )

      integer PlasmaDistUniform
      integer PlasmaDistSymmetric
      integer PlasmaDistNormal
      parameter ( PlasmaDistUniform   = 201 )
      parameter ( PlasmaDistSymmetric = 202 )
      parameter ( PlasmaDistNormal    = 203 )

      integer PlasmaHermGeev
      integer PlasmaHermPoev
      integer PlasmaNonsymPosv
      integer PlasmaSymPosv
      parameter ( PlasmaHermGeev    = 241 )
      parameter ( PlasmaHermPoev    = 242 )
      parameter ( PlasmaNonsymPosv  = 243 )
      parameter ( PlasmaSymPosv     = 244 )

      integer PlasmaNoPacking     
      integer PlasmaPackSubdiag   
      integer PlasmaPackSupdiag   
      integer PlasmaPackColumn    
      integer PlasmaPackLowerBand 
      integer PlasmaPackRow       
      integer PlasmaPackUpeprBand 
      integer PlasmaPackAll       
      parameter ( PlasmaNoPacking     = 291 )
      parameter ( PlasmaPackSubdiag   = 292 )
      parameter ( PlasmaPackSupdiag   = 293 )
      parameter ( PlasmaPackColumn    = 294 )
      parameter ( PlasmaPackRow       = 295 )
      parameter ( PlasmaPackLowerBand = 296 )
      parameter ( PlasmaPackUpeprBand = 297 )
      parameter ( PlasmaPackAll       = 298 )

      integer PlasmaNoVec,PlasmaVec,PlasmaIvec, PlasmaAllVec
      parameter ( PlasmaNoVec  = 301 )
      parameter ( PlasmaVec    = 302 )
      parameter ( PlasmaIvec   = 303 )
      parameter ( PlasmaAllVec = 304 )

      integer PlasmaForward, PlasmaBackward
      parameter ( PlasmaForward    = 391 )
      parameter ( PlasmaBackward   = 392 )

      integer PlasmaColumnwise,PlasmaRowwise
      parameter ( PlasmaColumnwise = 401 )
      parameter ( PlasmaRowwise    = 402 )

!********************************************************************
!   PLASMA constants - boolean
!
      integer PLASMA_FALSE, PLASMA_TRUE
      parameter ( PLASMA_FALSE = 0 )
      parameter ( PLASMA_TRUE  = 1 )

!********************************************************************
!   State machine switches
!
      integer PLASMA_WARNINGS, PLASMA_ERRORS, PLASMA_AUTOTUNING
      integer PLASMA_DAG
      parameter ( PLASMA_WARNINGS   = 1 )
      parameter ( PLASMA_ERRORS     = 2 )
      parameter ( PLASMA_AUTOTUNING = 3 )
      parameter ( PLASMA_DAG        = 4 )

!********************************************************************
!   PLASMA constants - configuration  parameters
!
      integer PLASMA_CONCURRENCY, PLASMA_TILE_SIZE
      integer PLASMA_INNER_BLOCK_SIZE, PLASMA_SCHEDULING_MODE
      integer PLASMA_HOUSEHOLDER_MODE, PLASMA_HOUSEHOLDER_SIZE
      integer PLASMA_TRANSLATION_MODE
      parameter ( PLASMA_CONCURRENCY      = 1 )
      parameter ( PLASMA_TILE_SIZE        = 2 )
      parameter ( PLASMA_INNER_BLOCK_SIZE = 3 )
      parameter ( PLASMA_SCHEDULING_MODE  = 4 )
      parameter ( PLASMA_HOUSEHOLDER_MODE = 5 )
      parameter ( PLASMA_HOUSEHOLDER_SIZE = 6 )
      parameter ( PLASMA_TRANSLATION_MODE = 7 )

!********************************************************************
!   PLASMA constants - scheduling mode
!
      integer PLASMA_STATIC_SCHEDULING, PLASMA_DYNAMIC_SCHEDULING
      parameter ( PLASMA_STATIC_SCHEDULING   = 1 )
      parameter ( PLASMA_DYNAMIC_SCHEDULING  = 2 )

!********************************************************************
!   PLASMA constants - householder mode
!
      integer PLASMA_FLAT_HOUSEHOLDER, PLASMA_TREE_HOUSEHOLDER
      parameter ( PLASMA_FLAT_HOUSEHOLDER  = 1 )
      parameter ( PLASMA_TREE_HOUSEHOLDER  = 2 )

!*********************************************************************
!   PLASMA constants - translation mode
!
      integer PLASMA_INPLACE, PLASMA_OUTOFPLACE
      parameter ( PLASMA_INPLACE     = 1 )
      parameter ( PLASMA_OUTOFPLACE  = 2 )

!********************************************************************
!   PLASMA constants - success & error codes
!
      integer PLASMA_SUCCESS, PLASMA_ERR_NOT_INITIALIZED
      integer PLASMA_ERR_REINITIALIZED, PLASMA_ERR_NOT_SUPPORTED
      integer PLASMA_ERR_ILLEGAL_VALUE, PLASMA_ERR_NOT_FOUND
      integer PLASMA_ERR_OUT_OF_MEMORY, PLASMA_ERR_INTERNAL_LIMIT
      integer PLASMA_ERR_UNALLOCATED, PLASMA_ERR_FILESYSTEM
      integer PLASMA_ERR_UNEXPECTED, PLASMA_ERR_SEQUENCE_FLUSHED
      parameter ( PLASMA_SUCCESS             =    0 )
      parameter ( PLASMA_ERR_NOT_INITIALIZED = -101 )
      parameter ( PLASMA_ERR_REINITIALIZED   = -102 )
      parameter ( PLASMA_ERR_NOT_SUPPORTED   = -103 )
      parameter ( PLASMA_ERR_ILLEGAL_VALUE   = -104 )
      parameter ( PLASMA_ERR_NOT_FOUND       = -105 )
      parameter ( PLASMA_ERR_OUT_OF_MEMORY   = -106 )
      parameter ( PLASMA_ERR_INTERNAL_LIMIT  = -107 )
      parameter ( PLASMA_ERR_UNALLOCATED     = -108 )
      parameter ( PLASMA_ERR_FILESYSTEM      = -109 )
      parameter ( PLASMA_ERR_UNEXPECTED      = -110 )
      parameter ( PLASMA_ERR_SEQUENCE_FLUSHED= -111 )
