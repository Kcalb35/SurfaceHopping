//
// Created by grass on 4/15/21.
//

#include "gslExtra.h"
#include "gsl/gsl_blas.h"
#include <stdexcept>


void copy_complex_vector_to_real_vector(const gsl_vector_complex *src, gsl_vector *dst) {
    if (dst->size < src->size)
        throw std::runtime_error("dst vector size is smaller than dst size");
    for (int i = 0; i < src->size; ++i)
        gsl_vector_set(dst, i, gsl_vector_complex_get(src, i).dat[0]);
}

void gsl_extra_multiply(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* dst){
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,m1,m2,0,dst);
}

void gsl_extra_multiply(gsl_matrix_complex *m1,gsl_matrix_complex *m2, gsl_matrix_complex *dst){
    gsl_blas_zgemm(CblasNoTrans,CblasNoTrans,gsl_complex{1,0},m1,m2,gsl_complex{0,0},dst);
}
