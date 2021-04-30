//
// Created by grass on 4/29/21.
//

#include "ModelBase.h"

void ECR::hamitonian_cal(gsl_matrix *m, double x) {
    double h12 = x < 0 ? 0.1 * exp(0.9 * x) : 0.1 * (2 - exp(-0.9 * x));
    gsl_matrix_set(m, 0, 0, 6e-4);
    gsl_matrix_set(m, 1, 1, -6e-4);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

void ECR::d_hamitonian_cal(gsl_matrix *m, double x) {
    double d12 = 0.1 * 0.9 * exp((x > 0 ? -1 : 1) * 0.9 * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, 0);
    gsl_matrix_set(m, 1, 0, d12);
    gsl_matrix_set(m, 0, 1, d12);
}

void ECR::diagonal_analytic(double x, gsl_matrix *h, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2) {
    double h12 = x < 0 ? 0.1 * exp(0.9 * x) : 0.1 * (2 - exp(-0.9 * x));
    e1 = -sqrt(9 + 25e6 * h12 * h12) / 5000;
    e2 = -e1;
    gsl_matrix_set_zero(h);
    gsl_matrix_set(h, 0, 0, e1);
    gsl_matrix_set(h, 1, 1, e2);
    gsl_vector_set(s1, 0, (3 - sqrt(9 + 25e6 * h12 * h12)) / 5000 / h12);
    gsl_vector_set(s1, 1, 1);
    gsl_vector_set(s2, 0, (3 + sqrt(9 + 25e6 * h12 * h12)) / 5000 / h12);
    gsl_vector_set(s2, 1, 1);
    gsl_vector_scale(s1, gsl_blas_dnrm2(s1));
    gsl_vector_scale(s2, gsl_blas_dnrm2(s2));
}

double ECR::nac_analytic(double x) {
    if (x < 0) {
        double tmp = sqrt(9 + 25e4 * exp(1.8 * x));
        return 337500. / tmp / sqrt(25e4 - 3 * exp(-1.8 * x) * (-3 + tmp)) /
               sqrt(25e4 + 3 * exp(-1.8 * x) * (3 + tmp));
    } else {
        double tmp1 = pow(-2 + exp(-0.9 * x), 2);
        double tmp2 = 9 + 25e4 * tmp1;
        return 168750 / (-0.5 + exp(0.9 * x)) / sqrt(tmp2) / sqrt(tmp2 - 3 * sqrt(tmp2)) / sqrt(tmp2 + 3 * sqrt(tmp2)) *
               tmp1;
    }
}

double ECR::energy_grad_analytic(double x, int state) {
    double d;
    if (x < 0) {
        d = -45 * exp(1.8 * x) / sqrt(9 + 25e4 * exp(1.8 * x));
    } else {
        d = exp(-1.8 * x) * (45 - 90 * exp(0.9 * x)) / sqrt(9 + 25e4 * pow(-2 + exp(-0.9 * x), 2));
    }
    if (state == 1) {
        d = -d;
    }
    return d;
}


void SAC::hamitonian_cal(gsl_matrix *m, double x) {
    int flag = (x > 0 ? 1 : -1);
    double h11 = flag * 0.01 * (1 - exp(-flag * 1.6 * x));
    double h12 = 0.005 * exp(-x * x);
    gsl_matrix_set(m, 0, 0, h11);
    gsl_matrix_set(m, 1, 1, -h11);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

void SAC::d_hamitonian_cal(gsl_matrix *m, double x) {
    double h11 = 0.01 * 1.6 * exp((x < 0 ? 1 : -1) * 1.6 * x);
    double h12 = -2 * 0.005 * x * exp(-x * x);
    gsl_matrix_set(m, 0, 0, h11);
    gsl_matrix_set(m, 1, 1, -h11);
    gsl_matrix_set(m, 1, 0, h12);
    gsl_matrix_set(m, 0, 1, h12);
}


void DAC::hamitonian_cal(gsl_matrix *m, double x) {

    double h12 = 0.015 * exp(-0.06 * x * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, -0.1 * exp(-0.28 * x * x) + 0.05);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

void DAC::d_hamitonian_cal(gsl_matrix *m, double x) {
    double d12 = -2 * 0.015 * 0.06 * x * exp(-0.06 * x * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, 2 * 0.1 * 0.28 * x * exp(-0.28 * x * x));
    gsl_matrix_set(m, 1, 0, d12);
    gsl_matrix_set(m, 0, 1, d12);
}
