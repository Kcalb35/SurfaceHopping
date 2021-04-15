//
// Created by grass on 4/15/21.
//

#ifndef FSSH_GSLEXTRA_H
#define FSSH_GSLEXTRA_H

#include "gsl/gsl_matrix.h"

void copy_complex_vector_to_real_vector(const gsl_vector_complex *src, gsl_vector *dst);
void gsl_extra_multiply(gsl_matrix* m1,gsl_matrix* m2,gsl_matrix* dst);

#endif //FSSH_GSLEXTRA_H
