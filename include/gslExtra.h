//
// Created by grass on 4/15/21.
//

#ifndef FSSH_GSLEXTRA_H
#define FSSH_GSLEXTRA_H

#include "gsl/gsl_matrix.h"
#include <string>

void copy_complex_vector_to_real_vector(const gsl_vector_complex *src, gsl_vector *dst);

void gsl_extra_multiply(gsl_matrix *m1, gsl_matrix *m2, gsl_matrix *dst);

void gsl_extra_multiply(gsl_matrix_complex *m1, gsl_matrix_complex *m2, gsl_matrix_complex *dst);

void log_matrix(gsl_matrix *m, int size_1, int size_2, const std::string &name);

void log_matrix(gsl_matrix_complex *m, int size_1, int size_2, const std::string &name);

int sgn(double val);

#endif //FSSH_GSLEXTRA_H
