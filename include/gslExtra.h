//
// Created by grass on 4/15/21.
//

#ifndef FSSH_GSLEXTRA_H
#define FSSH_GSLEXTRA_H

#include "gsl/gsl_matrix.h"
#include <string>

void log_matrix(gsl_matrix *m, int size_1, int size_2, const std::string &name);

void log_matrix(gsl_matrix_complex *m, int size_1, int size_2, const std::string &name);

int sgn(double val);

#endif //FSSH_GSLEXTRA_H
