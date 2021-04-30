//
// Created by grass on 4/15/21.
//

#include "gslExtra.h"
#include "gsl/gsl_blas.h"
#include "easylogging++.h"
#include <string>

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
