//
// Created by grass on 4/27/21.
//

#include "FSSHMath.h"
#include "gsl/gsl_matrix.h"
#include "easylogging++.h"
#include <ctime>
#include <iostream>


INITIALIZE_EASYLOGGINGPP

int main() {
    auto h = gsl_matrix_alloc(2, 2);
    auto v1 = gsl_vector_alloc(2);
    auto v2 = gsl_vector_alloc(2);
    auto wb = gsl_eigen_symmv_alloc(2);
    auto vals = gsl_vector_alloc(2);
    auto vecs = gsl_matrix_alloc(2, 2);
    double e1, e2;
    int cnt = 5e7;

    ECR ecr;
    std::clock_t start = clock();
    for (int i = 0; i < cnt; ++i) {
        ecr.hamitonian_cal(h, 5);
        diagonalize(h, e1, e2, v1, v2, wb, vals, vecs);
    }
    std::cout << "numerical time:" << (clock() - start) / (double) CLOCKS_PER_SEC << std::endl;

    start = clock();
    for (int i = 0; i < cnt; ++i) {
        ecr.diagonal_analytic(5, h, e1, e2, v1, v2);
    }
    std::cout << "analytic time:" << (clock() - start) / (double) CLOCKS_PER_SEC << std::endl;

}