//
// Created by grass on 4/15/21.
//

#include "gslExtra.h"
#include "gsl/gsl_blas.h"
#include "easylogging++.h"
#include <stdexcept>
#include <string>

void copy_complex_vector_to_real_vector(const gsl_vector_complex *src, gsl_vector *dst) {
    if (dst->size < src->size)
        throw std::runtime_error("dst vector size is smaller than dst size");
    for (int i = 0; i < src->size; ++i)
        gsl_vector_set(dst, i, gsl_vector_complex_get(src, i).dat[0]);
}

void gsl_extra_multiply(gsl_matrix *m1, gsl_matrix *m2, gsl_matrix *dst) {
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, m1, m2, 0, dst);
}

void gsl_extra_multiply(gsl_matrix_complex *m1, gsl_matrix_complex *m2, gsl_matrix_complex *dst) {
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex{1, 0}, m1, m2, gsl_complex{0, 0}, dst);
}

void log_matrix(gsl_matrix *m, int size_1, int size_2, const std::string &name) {
    LOG(INFO) << name;
    for (int i = 0; i < size_1; ++i) {
        for (int j = 0; j < size_2; ++j) {
            LOG(INFO) << i + 1 << j + 1 << ' ' << gsl_matrix_get(m, i, j);
        }
    }
}

void log_matrix(gsl_matrix_complex *m, int size_1, int size_2, const std::string &name) {
    LOG(INFO) << name;
    for (int i = 0; i < size_1; ++i) {
        for (int j = 0; j < size_2; ++j) {
            gsl_complex tmp = gsl_matrix_complex_get(m, i, j);
            LOG(INFO) << i + 1 << j + 1 << ' ' << tmp.dat[0] << ((tmp.dat[1] > 0) ? " +" : " ") << tmp.dat[1];
        }
    }
}

int sgn(double val) {
    return (0 < val) - (val < 0);
}
