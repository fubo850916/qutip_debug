#cython: language_level=3
#cython: linetrace=True
#distutils: define_macros=CYTHON_TRACE_NOGIL=1
import numpy as np
cimport numpy as cnp
cimport cython
cimport libc.math
from libcpp cimport bool
#from qutip.cy.spmatfuncs import spmmpy_c
import line_profiler

include "complex_math.pxi"

cdef extern from "src/zspmv.hpp" nogil:
    void zspmvpy(double complex *data, int *ind, int *ptr, double complex *vec,
                double complex a, double complex *out, int nrows)

cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyDataMem_NEW(size_t size)

cdef extern from "<complex>" namespace "std" nogil:
    double abs(double complex x)
    double real(double complex x)
    double imag(double complex x)

ctypedef int MKL_INT
cdef extern from "mkl.h" nogil:
#    ctypedef double complex MKL_Complex16
#    ctypedef int MKL_INT
    ctypedef struct MKL_Complex16:
        double real
        double imag
    ctypedef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO = 0
        SPARSE_INDEX_BASE_ONE = 1
    
    ctypedef enum sparse_status_t:
        SPARSE_STATUS_SUCCESS = 0 # the operation was successful
        SPARSE_STATUS_NOT_INITIALIZED = 1 # empty handle or matrix arrays
        SPARSE_STATUS_ALLOC_FAILED = 2 # internal error: memory allocation failed
        SPARSE_STATUS_INVALID_VALUE = 3 # invalid input value
        SPARSE_STATUS_EXECUTION_FAILED = 4 # e.g. 0-diagonal element for triangular solver, etc.
        SPARSE_STATUS_INTERNAL_ERROR = 5 # internal error
        SPARSE_STATUS_NOT_SUPPORTED = 6 # e.g. operation for double precision doesn't support other types */
    
    ctypedef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE = 10
        SPARSE_OPERATION_TRANSPOSE = 11
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
    
    ctypedef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL = 20 # General case
        SPARSE_MATRIX_TYPE_SYMMETRIC = 21 # Triangular part of the matrix is to be processed
        SPARSE_MATRIX_TYPE_HERMITIAN = 22
        SPARSE_MATRIX_TYPE_TRIANGULAR = 23
        SPARSE_MATRIX_TYPE_DIAGONAL = 24 # diagonal matrix; only diagonal elements will be processed
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL = 26 # block-diagonal matrix; only diagonal blocks will be processed
    
    ctypedef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER = 40 # lower triangular part of the matrix is stored
        SPARSE_FILL_MODE_UPPER = 41 # upper triangular part of the matrix is stored
        SPARSE_FILL_MODE_FULL = 42 # upper triangular part of the matrix is stored
    
    ctypedef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT = 50 # triangular matrix with non-unit diagonal
        SPARSE_DIAG_UNIT = 51 # triangular matrix with unit diagonal
    
    ctypedef enum sparse_layout_t:
        SPARSE_LAYOUT_ROW_MAJOR = 101 # C-style
        SPARSE_LAYOUT_COLUMN_MAJOR = 102 # Fortran-style
    
    struct sparse_matrix:
        pass
    
    ctypedef sparse_matrix* sparse_matrix_t
    
    struct matrix_descr:
        sparse_matrix_type_t type # matrix type: general, diagonal or triangular / symmetric / hermitian
        sparse_fill_mode_t mode # upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case)
        sparse_diag_type_t diag # unit or non-unit diagonal ( for triangular / symmetric / hermitian case)
    sparse_status_t mkl_sparse_z_mm( sparse_operation_t    operation,
                                     MKL_Complex16         alpha,
#                                     double complex         alpha,
                                     const sparse_matrix_t A,
                                     matrix_descr          descr,
                                     sparse_layout_t       layout,
                                     const MKL_Complex16   *x,
#                                     const double complex   *x,
                                     MKL_INT               columns,
                                     MKL_INT               ldx,
                                     MKL_Complex16         beta,
                                     MKL_Complex16         *y,
#                                     double complex         beta,
#                                     double complex         *y,
                                     MKL_INT               ldy )
    sparse_status_t mkl_sparse_z_create_csr( sparse_matrix_t           *A,
                                             const sparse_index_base_t indexing,
                                             const MKL_INT             rows,
                                             const MKL_INT             cols,
                                                   MKL_INT             *rows_start,
                                                   MKL_INT             *rows_end,
                                                   MKL_INT             *col_indx,
                                                   MKL_Complex16       *values )
#                                                   double complex       *values )
    sparse_status_t mkl_sparse_z_spmmd( sparse_operation_t        operation,
                                        const sparse_matrix_t     A,
                                        const sparse_matrix_t     B,
                                        sparse_layout_t           layout,
                                        MKL_Complex16              *C,
                                        MKL_INT                   ldc)
    void cblas_zdotu_sub(const MKL_INT                            n,
                         const void                               *x,
                         const MKL_INT                            incx,
                         const void                               *y,
                         const MKL_INT                            incy,
                         void                                     *dotu)        
    void cblas_zdotui_sub(const MKL_INT                           nz,
                          const void                              *x,
                          const MKL_INT                           *indx,
                          const void                              *y,
                          void                                    *dotui)
    void mkl_zimatcopy(   const char                              ordering,
                          const char                              trans,                 
                          size_t                                  rows,
                          size_t                                  cols,
                          const MKL_Complex16                     alpha,
                          MKL_Complex16                           *AB,
                          size_t                                  lda,
                          size_t                                  ldb)
    void mkl_zomatcopy(   const char                              ordering,
                          const char                              trans,                 
                          size_t                                  rows,
                          size_t                                  cols,
                          const MKL_Complex16                     alpha,
                          MKL_Complex16                           *A,
                          size_t                                  lda,
                          MKL_Complex16                           *B,
                          size_t                                  ldb)
    void mkl_zomatadd(    char                                    ordering,
                          char                                    transa,
                          char                                    transb,
                          size_t                                  m,
                          size_t                                  n,
                          const MKL_Complex16                     alpha,
                          const MKL_Complex16                     *A,
                          size_t                                  lda,
                          const MKL_Complex16                     beta,
                          const MKL_Complex16                     *B,
                          size_t                                  ldb,
                          MKL_Complex16                           *C,
                          size_t                                  ldc)
    sparse_status_t mkl_sparse_destroy(sparse_matrix_t A)
 
#include "complex_math.pxi"

#from Cython.Compiler.Options import directive_defaults
#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

class MKLCallError(Exception):
   pass

#functions from qutip
@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _spmm_c_py(complex* data, int* ind, int* ptr,
            complex* mat, complex a, complex* out,
            unsigned int sp_rows, unsigned int nrows, unsigned int ncols):
    """
    sparse*dense "C" ordered.
    """
    cdef int row, col, ii, jj, row_start, row_end
    for row from 0 <= row < sp_rows :
        row_start = ptr[row]
        row_end = ptr[row+1]
        for jj from row_start <= jj < row_end:
            for col in range(ncols):
                out[row * ncols + col] += a*data[jj]*mat[ind[jj] * ncols + col]

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void spmmpy_c(complex[::1] data, int[::1] ind, int[::1] ptr,
             complex[:,::1] M, complex a, complex[:,::1] out):
    """
    Sparse matrix, c ordered dense matrix multiplication.
    The sparse matrix must be in CSR format and have complex entries.
    Parameters
    ----------
    data : array
        Data for sparse matrix.
    idx : array
        Indices for sparse matrix data.
    ptr : array
        Pointers for sparse matrix data.
    mat : array 2d
        Dense matrix for multiplication.  Must be in c mode.
    alpha : complex
        Numerical coefficient for sparse matrix.
    out: array
        Output array. Must be in c mode.
    """
    cdef unsigned int sp_rows = ptr.shape[0]-1
    cdef unsigned int nrows = M.shape[0]
    cdef unsigned int ncols = M.shape[1]
    _spmm_c_py(&data[0], &ind[0], &ptr[0], &M[0,0], 1.,
               &out[0,0], sp_rows, nrows, ncols)

#helper functions for mkl
@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef sparse_matrix_t to_mkl_matrix(
#                                   complex[::1] A_data,
                                   MKL_Complex16[::1] A_data,
                                   int[::1] A_ind,
                                   int[::1] A_indptr,
#                                   MKL_INT[::1] A_ind,
#                                   MKL_INT[::1] A_indptr,
                                   MKL_INT nrows,
                                   MKL_INT ncols):
    cdef sparse_matrix_t A
    cdef sparse_index_base_t base_index = SPARSE_INDEX_BASE_ZERO
    cdef MKL_INT * start = &A_indptr[0]
    cdef MKL_INT * end = &A_indptr[1]
    cdef MKL_INT * index = &A_ind[0]
    cdef MKL_Complex16 * values = &A_data[0]
#    cdef double complex  * values = &A_data[0]

    create_status = mkl_sparse_z_create_csr(&A, base_index,nrows,ncols,
                                         start, end, index, values)
    if create_status != SPARSE_STATUS_SUCCESS:
        raise MKLCallError("Creating an MKL sparse matrix failed.")
    return A

   
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void MKL_Complex16_to_double_complex_1d(
        MKL_Complex16 * a, 
        double complex * b, 
        unsigned int ndata):
    cdef size_t ii
    for ii in range(ndata):
        b[ii].real = a[ii].real
        b[ii].imag = a[ii].imag

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void double_complex_to_MKL_Complex16_1d(
        MKL_Complex16 * a, 
        double complex * b, 
        unsigned int ndata):
    cdef size_t ii
    for ii in range(ndata):
        a[ii].real = b[ii].real
        a[ii].imag = b[ii].imag

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void MKL_Complex16_to_double_complex_2d(
        MKL_Complex16 * a, 
        double complex * b, 
        unsigned int nrows, 
        unsigned int ncols):
    cdef size_t ii,jj
    for ii in range(nrows):
        for jj in range(ncols):
            b[ii * ncols + jj].real = a[ii * ncols + jj].real
            b[ii * ncols + jj].imag = a[ii * ncols + jj].imag

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void double_complex_to_MKL_Complex16_2d(
        MKL_Complex16 * a, 
        double complex * b, 
        unsigned int nrows,
        unsigned int ncols):
    cdef size_t ii,jj
    for ii in range(nrows):
        for jj in range(ncols):
            a[ii * ncols + jj].real = b[ii * ncols + jj].real
            a[ii * ncols + jj].imag = b[ii * ncols + jj].imag

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void spmm_c_mkl(
#                     MKL_Complex16[::1] A_data,
                     complex[::1] A_data,
                     int[::1] A_ind,
                     int[::1] A_indptr,
#                     MKL_INT[::1] A_ind,
#                     MKL_INT[::1] A_indptr,
                     MKL_INT nrows,
                     MKL_INT ncols,
#                     MKL_Complex16 * x,
#                     MKL_Complex16 * y):
                     double complex * x,
                     double complex * y):
    cdef sparse_matrix_t A_mkl 
    cdef sparse_operation_t operation
    cdef sparse_layout_t layout
    cdef matrix_descr descr
    cdef MKL_Complex16 alpha
    alpha.real = 1.0
    alpha.imag = 0.0
    cdef MKL_Complex16 beta
    beta.real = 1.0
    beta.imag = 0.0
#    cdef MKL_INT ldx = nrows
#    cdef MKL_INT ldy = nrows
#    cdef size_t nnz = A_data.shape[0]
#    cdef MKL_Complex16 * A_data_mkl_buffer = <MKL_Complex16 *>PyDataMem_NEW(nnz*sizeof(MKL_Complex16))
#    cdef MKL_Complex16[::1] A_data_mkl = <MKL_Complex16[:nnz]>A_data_mkl_buffer
#    cdef MKL_Complex16[::1] A_data_mkl = <MKL_Complex16 * >PyDataMem_NEW(nnz*sizeof(MKL_Complex16))
#    cdef MKL_Complex16[::1] A_data_mkl = np.zeros((nnz,), dtype=np.complex128,order='c') 
    cdef MKL_Complex16[::1] A_data_mkl = np.ascontiguousarray(A_data,dtype=complex)
#    cdef int ndata = A_data.shape[0]
#    double_complex_to_MKL_Complex16_1d(&A_data_mkl[0],&A_data[0],ndata)
    
    A_mkl = to_mkl_matrix(A_data_mkl,A_ind,A_indptr,nrows,ncols)
#    print("mkl matrix has been created.")
    operation = SPARSE_OPERATION_NON_TRANSPOSE
    layout = SPARSE_LAYOUT_ROW_MAJOR
    descr.type = SPARSE_MATRIX_TYPE_GENERAL
    descr.diag = SPARSE_DIAG_NON_UNIT
    
#    print("we are about to create the buffer.")
#    cdef MKL_Complex16 * x_mkl_buffer = \
#        <MKL_Complex16 *>PyDataMem_NEW_ZEROED(nrows*nrows,sizeof(MKL_Complex16))
#    cdef MKL_Complex16 * y_mkl_buffer = \
#        <MKL_Complex16 *>PyDataMem_NEW_ZEROED(nrows*nrows,sizeof(MKL_Complex16))
#    print("we are about to assign the buffer")
#    cdef MKL_Complex16[:,::1] x_mkl = <MKL_Complex16[:nrows,:nrows]>x_mkl_buffer
#    cdef MKL_Complex16[:,::1] y_mkl = <MKL_Complex16[:nrows,:nrows]>y_mkl_buffer
    cdef double complex[:,::1] x_view = <double complex [:nrows,:nrows]> x
    cdef MKL_Complex16[:,::1] x_mkl = np.ascontiguousarray(x_view,dtype=np.complex128)
    cdef double complex[:,::1] y_view = <double complex [:nrows,:nrows]> y
    cdef MKL_Complex16[:,::1] y_mkl = np.ascontiguousarray(y_view,dtype=np.complex128)

#    cdef MKL_Complex16[:,::1] x_mkl = np.zeros((nrows,nrows), dtype=np.complex128,order='c')
#    cdef MKL_Complex16[:,::1] y_mkl = np.zeros((nrows,nrows), dtype=np.complex128,order='c')

#    double_complex_to_MKL_Complex16_2d(&x_mkl[0,0],x,nrows,nrows)
#    double_complex_to_MKL_Complex16_2d(&y_mkl[0,0],y,nrows,nrows)
    mkl_sparse_z_mm(operation,alpha,A_mkl,descr,layout,&x_mkl[0,0],nrows,nrows,beta,&y_mkl[0,0],nrows)
#    cnp.ndarray[complex, ndim=2, mode="c"] y_arr = np.ascontiguousarray(y_mkl,dtype=complex) 
#    MKL_Complex16_to_double_complex_2d(&y_mkl[0,0],y,nrows,nrows) 
    cdef complex[:,::1] y_view2 = np.ascontiguousarray(y_mkl,dtype=complex)
    y = &y_view2[0,0]

def spmm_c_mkl_sentinel():
    pass

## different versions of cy_ode_rhs functions
@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs_single_aop_mkl(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr):

    #reshape the 1d rho into 2d rho
    cdef cnp.ndarray[complex, ndim=2, mode="c"] rho2d = \
        np.ascontiguousarray(rho,dtype=complex).reshape((nrows,nrows),order='c')
    #compute the product of -iH0-KKp with rho2d
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out1 = \
        np.zeros((nrows,nrows), dtype=complex,order='c')
    spmm_c_mkl(H0KKpsdata,H0KKpsind,H0KKpsindptr,nrows,nrows,&rho2d[0,0],&out1[0,0])
    #compute the product rho2d*K in its adjoint form,i.e.,
    #(rho2d*K).dag()=K*rho2d
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out2 = \
        np.zeros((nrows,nrows), dtype=complex,order='c')
    spmm_c_mkl(Kdata,Kind,Kindptr,nrows,nrows,&rho2d[0,0],&out2[0,0])
#    spmmpy_c(Kdata,Kind,Kindptr,rho2d,1.0,out2)
    ##Calculate Kp*rho2d*K
    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut2 = \
         np.ascontiguousarray(np.transpose(out2).conjugate(),dtype=complex)# out2.T.copy(order='C')
    spmm_c_mkl(Kpdata,Kpind,Kpindptr,nrows,nrows,&AdjointOut2[0,0],&out1[0,0])
#    spmmpy_c(Kpdata,Kpind,Kpindptr,AdjointOut2,1.0,out1)
    #compute the adjoint of out1 and add it to out1
    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut1 = \
         np.ascontiguousarray(np.transpose(out1).conjugate(),dtype=complex)
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out3 = out1 + AdjointOut1
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out4 = \
        out3.reshape(nrows*nrows,order='c')
    return out4

def cy_ode_rhs_single_aop_mkl_sentinel():
    pass

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs_single_aop(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr):

    cdef unsigned int nrows2 = nrows*nrows
    #reshape the 1d rho into 2d rho
    cdef cnp.ndarray[complex, ndim=1, mode="c"] rho_ndarray = \
        np.asarray(rho,dtype=complex,order='c')
    cdef cnp.ndarray[complex, ndim=2, mode="c"] rho2d = \
        rho_ndarray.reshape((nrows,nrows),order='c')
    #compute the product of -iH0-KKp with rho2d
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out1 = \
        np.zeros((nrows,nrows), dtype=complex,order='c')
    spmmpy_c(H0KKpsdata,H0KKpsind,H0KKpsindptr,rho2d,1.0,out1)
    #compute the product rho2d*K in its adjoint form,i.e.,
    #(rho2d*K).dag()=K*rho2d
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out2 = \
        np.zeros((nrows,nrows), dtype=complex,order='c')
    spmmpy_c(Kdata,Kind,Kindptr,rho2d,1.0,out2)

    ##Calculate Kp*rho2d*K
    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut2 = \
         np.ascontiguousarray(out2.T.conjugate(),dtype=complex)# out2.T.copy(order='C')
    spmmpy_c(Kpdata,Kpind,Kpindptr,AdjointOut2,1.0,out1)
    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut1 = \
         np.ascontiguousarray(out1.T.conjugate(),dtype=complex)
         #out1.T.conjugate().copy(order='C')
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out3 = out1 + AdjointOut1

    cdef cnp.ndarray[complex, ndim=1, mode="c"] out4 = \
        out3.reshape(nrows*nrows,order='c')
    return out4

def cy_ode_rhs_single_aop_sentinel():
    pass

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs_single_aop_mkl_v2(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr):
    cdef MKL_Complex16 alpha
    alpha.real = 1.0
    alpha.imag = 0.0
    cdef MKL_Complex16 beta
    beta.real = 1.0
    beta.imag = 0.0
    #reshape the 1d rho into 2d rho,the 1d rho here has been in c-order, 
    cdef cnp.ndarray[complex, ndim=2, mode="c"] rho2d = \
        np.ascontiguousarray(rho,dtype=complex).reshape((nrows,nrows),order='c')
#        np.asarray(rho,dtype=complex,order='c').reshape((nrows,nrows),order='c')
    #compute the product of -iH0-KKp with rho2d
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out1 = \
        np.zeros((nrows,nrows), dtype=complex,order='c')
    #spmm_c_mkl(H0KKpsdata,H0KKpsind,H0KKpsindptr,nrows,nrows,&rho2d[0,0],&out1[0,0])
    #Now Let's expand this function here.
    cdef sparse_matrix_t H0KKps_mkl
    cdef sparse_operation_t H0KKps_operation
    cdef sparse_layout_t H0KKps_layout
    cdef matrix_descr H0KKps_descr
#    cdef MKL_INT ldx = nrows
#    cdef MKL_INT ldy = nrows
#    cdef size_t nnz = H0KKpsdata.shape[0]
    cdef MKL_Complex16[::1] H0KKpsdata_mkl = np.ascontiguousarray(H0KKpsdata,dtype=complex)
#    cdef MKL_Complex16[::1] A_data_mkl = np.zeros((nnz,), dtype=np.complex128,order='c') 
#    cdef int ndata = H0KKpsdata.shape[0]
#    double_complex_to_MKL_Complex16_1d(&A_data_mkl[0],&H0KKpsdata[0],ndata)
    H0KKps_mkl = to_mkl_matrix(H0KKpsdata_mkl,H0KKpsind,H0KKpsindptr,nrows,nrows)
    H0KKps_operation = SPARSE_OPERATION_NON_TRANSPOSE
    H0KKps_layout = SPARSE_LAYOUT_ROW_MAJOR
    H0KKps_descr.type = SPARSE_MATRIX_TYPE_GENERAL
    H0KKps_descr.diag = SPARSE_DIAG_NON_UNIT

    cdef MKL_Complex16[:,::1] rho2d_mkl = rho2d #np.zeros((nrows,nrows), dtype=np.complex128,order='c')
    cdef MKL_Complex16[:,::1] out1_mkl = out1 #np.zeros((nrows,nrows), dtype=np.complex128,order='c')

#    double_complex_to_MKL_Complex16_2d(&x_mkl[0,0],&rho2d[0,0],nrows,nrows)
#    double_complex_to_MKL_Complex16_2d(&y_mkl[0,0],&out1[0,0],nrows,nrows)
    mkl_sparse_z_mm(H0KKps_operation,alpha,H0KKps_mkl,H0KKps_descr,H0KKps_layout,&rho2d_mkl[0,0],nrows,nrows,beta,&out1_mkl[0,0],nrows)
#    MKL_Complex16_to_double_complex_2d(&y_mkl[0,0],&out1[0,0],nrows,nrows) 
#    out1 =  np.ascontiguousarray(out1_mkl,dtype=complex)


    #compute the product rho2d*K in its adjoint form,i.e.,(rho2d*K).dag()=K*rho2d
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out2 = \
        np.zeros((nrows,nrows), dtype=complex,order='c')
#    spmm_c_mkl(Kdata,Kind,Kindptr,nrows,nrows,&rho2d[0,0],&out2[0,0])
    #Now Let's expand this function here.
    cdef sparse_matrix_t K_mkl
    cdef sparse_operation_t K_operation
    cdef sparse_layout_t K_layout
    cdef matrix_descr K_descr
    cdef MKL_Complex16[::1] Kdata_mkl = np.ascontiguousarray(Kdata,dtype=complex)
    K_mkl = to_mkl_matrix(Kdata_mkl,Kind,Kindptr,nrows,nrows)
    K_operation = SPARSE_OPERATION_NON_TRANSPOSE
    K_layout = SPARSE_LAYOUT_ROW_MAJOR
    K_descr.type = SPARSE_MATRIX_TYPE_GENERAL
    K_descr.diag = SPARSE_DIAG_NON_UNIT

#    cdef MKL_Complex16[:,::1] rho2d_mkl = rho2d #np.zeros((nrows,nrows), dtype=np.complex128,order='c')
    cdef MKL_Complex16[:,::1] out2_mkl = out2 #np.zeros((nrows,nrows), dtype=np.complex128,order='c')

    mkl_sparse_z_mm(K_operation,alpha,K_mkl,K_descr,K_layout,&rho2d_mkl[0,0],nrows,nrows,beta,&out2_mkl[0,0],nrows)
    out2 =  np.ascontiguousarray(out2_mkl,dtype=complex)

    ##Calculate Kp*rho2d*K
    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut2 = \
         np.ascontiguousarray(np.transpose(out2).conjugate(),dtype=complex)
#    spmm_c_mkl(Kpdata,Kpind,Kpindptr,nrows,nrows,&AdjointOut2[0,0],&out1[0,0])
    #Now Let's expand this function here.
    cdef sparse_matrix_t Kp_mkl
    cdef sparse_operation_t Kp_operation
    cdef sparse_layout_t Kp_layout
    cdef matrix_descr Kp_descr
    cdef MKL_Complex16[::1] Kpdata_mkl = np.ascontiguousarray(Kpdata,dtype=complex)
    Kp_mkl = to_mkl_matrix(Kpdata_mkl,Kpind,Kpindptr,nrows,nrows)
    Kp_operation = SPARSE_OPERATION_NON_TRANSPOSE
    Kp_layout = SPARSE_LAYOUT_ROW_MAJOR
    Kp_descr.type = SPARSE_MATRIX_TYPE_GENERAL
    Kp_descr.diag = SPARSE_DIAG_NON_UNIT

    cdef MKL_Complex16[:,::1] AdjointOut2_mkl = AdjointOut2 #np.zeros((nrows,nrows), dtype=np.complex128,order='c')
#    cdef MKL_Complex16[:,::1] out1_mkl = out1 #np.zeros((nrows,nrows), dtype=np.complex128,order='c')

    mkl_sparse_z_mm(Kp_operation,alpha,Kp_mkl,Kp_descr,Kp_layout,&AdjointOut2_mkl[0,0],nrows,nrows,beta,&out1_mkl[0,0],nrows)
    out1 =  np.ascontiguousarray(out1_mkl,dtype=complex)

    #compute the adjoint of out1 and add it to out1
    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut1 = \
         np.ascontiguousarray(np.transpose(out1).conjugate(),dtype=complex)
    cdef cnp.ndarray[complex, ndim=2, mode="c"] out3 = out1 + AdjointOut1
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out4 = \
        out3.reshape(nrows*nrows,order='c')
    return out4

def cy_ode_rhs_single_aop_mkl_v2_sentinel():
    pass

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void run_cy_ode_rhs_single_aop(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr,
        int N):
    cdef size_t ii=0
    cdef cnp.ndarray[complex, ndim=1, mode="c"] arr
    while ii < N:
        arr = cy_ode_rhs_single_aop(t,rho,nrows,H0KKpsdata,H0KKpsind,H0KKpsindptr,Kdata,Kind,Kindptr,Kpdata,Kpind,Kpindptr)
        ii += 1

def run_cy_ode_rhs_single_aop_sentinel():
    pass


@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void run_cy_ode_rhs_single_aop_mkl(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr,
        int N):
    cdef size_t ii=0
    cdef cnp.ndarray[complex, ndim=1, mode="c"] arr
    while ii < N:
        arr = cy_ode_rhs_single_aop_mkl(t,rho,nrows,H0KKpsdata,H0KKpsind,H0KKpsindptr,Kdata,Kind,Kindptr,Kpdata,Kpind,Kpindptr)
        ii += 1

def run_cy_ode_rhs_single_aop_mkl_sentinel():
    pass

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void run_cy_ode_rhs_single_aop_mkl_v2(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr,
        int N):
    cdef size_t ii=0
    cdef cnp.ndarray[complex, ndim=1, mode="c"] arr
    while ii < N:
        arr = cy_ode_rhs_single_aop_mkl_v2(t,rho,nrows,H0KKpsdata,H0KKpsind,H0KKpsindptr,Kdata,Kind,Kindptr,Kpdata,Kpind,Kpindptr)
        ii += 1

def run_cy_ode_rhs_single_aop_mkl_v2_sentinel():
    pass

@cython.binding(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[complex, ndim=1, mode="c"] cy_ode_rhs_single_aop_mkl_v3(
        double t,
        complex[::1] rho,
        int nrows,
        complex[::1] H0KKpsdata,
        int[::1] H0KKpsind,
        int[::1] H0KKpsindptr,
        complex[::1] Kdata,
        int[::1] Kind,
        int[::1] Kindptr,
        complex[::1] Kpdata,
        int[::1] Kpind,
        int[::1] Kpindptr):
    cdef MKL_Complex16 alpha
    alpha.real = 1.0
    alpha.imag = 0.0
    cdef MKL_Complex16 beta
    beta.real = 1.0
    beta.imag = 0.0

    cdef sparse_operation_t operation
    cdef sparse_layout_t layout
    cdef matrix_descr descr
    operation = SPARSE_OPERATION_NON_TRANSPOSE
    layout = SPARSE_LAYOUT_ROW_MAJOR
    descr.type = SPARSE_MATRIX_TYPE_GENERAL
    descr.diag = SPARSE_DIAG_NON_UNIT

#    #reshape the 1d rho into 2d rho,the 1d rho here has been in c-order, 
#    cdef cnp.ndarray[complex, ndim=2, mode="c"] rho2d = \
#        np.asarray(rho,dtype=complex,order='c').reshape((nrows,nrows),order='c')
#
#    #1st multiplication: compute the product of -iH0-KKp with rho2d
#    cdef cnp.ndarray[complex, ndim=2, mode="c"] out1 = \
#        np.zeros((nrows,nrows), dtype=complex,order='c')

    #1st multiplication: compute the product of -iH0-KKp with rho2d

    #spmm_c_mkl(H0KKpsdata,H0KKpsind,H0KKpsindptr,nrows,nrows,&rho2d[0,0],&out1[0,0])
    #Now Let's expand this function here.
    cdef sparse_matrix_t H0KKps_mkl
    cdef MKL_Complex16[::1] H0KKpsdata_mkl = np.ascontiguousarray(H0KKpsdata,dtype=complex)
    H0KKps_mkl = to_mkl_matrix(H0KKpsdata_mkl,H0KKpsind,H0KKpsindptr,nrows,nrows)
#    cdef MKL_Complex16[:,::1] rho2d_mkl = rho2d 
#    cdef MKL_Complex16[:,::1] out1_mkl = out1 
    #reshape the 1d rho into 2d rho,the 1d rho here has been in c-order, 
    cdef MKL_Complex16[:,::1] rho2d_mkl = \
        np.asarray(rho,dtype=complex,order='c').reshape((nrows,nrows),order='c')
    cdef MKL_Complex16[:,::1] out1_mkl = \
        np.zeros((nrows,nrows), dtype=complex,order='c')

    mkl_sparse_z_mm(operation,alpha,H0KKps_mkl,descr,layout,&rho2d_mkl[0,0],nrows,nrows,beta,&out1_mkl[0,0],nrows)
    #free up memory
#    PyDataMem_FREE(&H0KKpsdata_mkl[0])
#    mkl_sparse_destroy(H0KKps_mkl)
#    out1 =  np.ascontiguousarray(out1_mkl,dtype=complex)


    #compute the product rho2d*K in its adjoint form,i.e.,(rho2d*K).dag()=K*rho2d
#    cdef cnp.ndarray[complex, ndim=2, mode="c"] out2 = \
#        np.zeros((nrows,nrows), dtype=complex,order='c')
#    spmm_c_mkl(Kdata,Kind,Kindptr,nrows,nrows,&rho2d[0,0],&out2[0,0])
    #Now Let's expand this function here.
    cdef sparse_matrix_t K_mkl
    cdef MKL_Complex16[::1] Kdata_mkl = np.ascontiguousarray(Kdata,dtype=complex)
    K_mkl = to_mkl_matrix(Kdata_mkl,Kind,Kindptr,nrows,nrows)
#    cdef MKL_Complex16[:,::1] rho2d_mkl = rho2d #np.zeros((nrows,nrows), dtype=np.complex128,order='c')
    cdef MKL_Complex16[:,::1] out2_mkl = \
        np.zeros((nrows,nrows), dtype=complex,order='c')#out2 #np.zeros((nrows,nrows), dtype=np.complex128,order='c')

    mkl_sparse_z_mm(operation,alpha,K_mkl,descr,layout,&rho2d_mkl[0,0],nrows,nrows,beta,&out2_mkl[0,0],nrows)
#    out2 =  np.ascontiguousarray(out2_mkl,dtype=complex)
    #Free up memory
#    PyDataMem_FREE(&Kdata_mkl[0])
#    mkl_sparse_destroy(K_mkl)

    ##Calculate Kp*rho2d*K
#    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut2 = \
#         np.ascontiguousarray(np.transpose(out2).conjugate(),dtype=complex)
#    spmm_c_mkl(Kpdata,Kpind,Kpindptr,nrows,nrows,&AdjointOut2[0,0],&out1[0,0])
    #Now Let's expand this function here.
    cdef sparse_matrix_t Kp_mkl
    cdef MKL_Complex16[::1] Kpdata_mkl = np.ascontiguousarray(Kpdata,dtype=complex)
    Kp_mkl = to_mkl_matrix(Kpdata_mkl,Kpind,Kpindptr,nrows,nrows)

    cdef MKL_Complex16[:,::1] AdjointOut2_mkl = np.zeros((nrows,nrows), dtype=complex,order='c')
    cdef char ordering = b'R'
    cdef char trans = b'C'
    mkl_zomatcopy(ordering, trans, nrows, nrows, alpha,&out2_mkl[0,0],nrows,&AdjointOut2_mkl[0,0],nrows)

    #the out1_mkl = Kp*rho2d*K+out1_mkl=Kp*rho2d*K+H0KKps*rho2d
    mkl_sparse_z_mm(operation,alpha,Kp_mkl,descr,layout,&AdjointOut2_mkl[0,0],nrows,nrows,beta,&out1_mkl[0,0],nrows)
#    PyDataMem_FREE(&Kpdata_mkl[0])
#    PyDataMem_FREE(&out2_mkl[0,0])
#    PyDataMem_FREE(&AdjointOut2_mkl[0,0])
#    mkl_sparse_destroy(Kp_mkl)
##NOTE: AdjointOut2 maybe reused later.

#    out1 =  np.ascontiguousarray(out1_mkl,dtype=complex)

    #compute the adjoint of out1 and add it to out1
#    cdef cnp.ndarray[complex, ndim=2, mode="c"] AdjointOut1 = \
#         np.ascontiguousarray(np.transpose(out1).conjugate(),dtype=complex)
    cdef MKL_Complex16[:,::1] out3_mkl = np.zeros((nrows,nrows), dtype=complex,order='c')
    cdef MKL_Complex16[:,::1] AdjointOut1_mkl = np.zeros((nrows,nrows), dtype=complex,order='c')
    #Compute the adjoint of out1_mkl
    mkl_zomatcopy(ordering, trans, nrows, nrows, alpha,&out1_mkl[0,0],nrows,&AdjointOut1_mkl[0,0],nrows)
    
    #Add up out1_mkl and out1, store the result in out3_mkl
    cdef char transa = b'N'
    cdef char transb = b'N'
    mkl_zomatadd(ordering,transa,transb,nrows,nrows,alpha,&out1_mkl[0,0],nrows,\
                 beta,&AdjointOut1_mkl[0,0],nrows,&out3_mkl[0,0],nrows)

    #Convert out3_mkl back to compelx type, reshape it to 1d array and store it in out4
    cdef cnp.ndarray[complex, ndim=1, mode="c"] out4 = \
        np.asarray(out3_mkl,dtype=complex,order='c').reshape(nrows*nrows,order='c')
    
    #Free up memory
#    PyDataMem_FREE(&out3_mkl[0,0])
#    PyDataMem_FREE(&AdjointOut1_mkl[0,0])
#    PyDataMem_FREE(&out1_mkl[0,0])
    return out4

def cy_ode_rhs_single_aop_mkl_v3_sentinel():
    pass


