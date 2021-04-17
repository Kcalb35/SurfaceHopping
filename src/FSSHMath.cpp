//
// Created by grass on 4/15/21.
//
#include "FSSHMath.h"
#include "gslExtra.h"
#include "cmath"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_eigen.h"

void diagonalize(void (*f)(gsl_matrix*,double), double x, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_nonsymmv_workspace *wb) {
    auto hamitonian= gsl_matrix_alloc(2,2);
    f(hamitonian,x);
    auto *e_value = gsl_vector_complex_alloc(2);
    auto *e_vector = gsl_matrix_complex_alloc(2, 2);
    auto *tmp_complex = gsl_vector_complex_alloc(2);
    gsl_eigen_nonsymmv(hamitonian, e_value, e_vector, wb);

    e1 = gsl_vector_complex_get(e_value, 0).dat[0];
    e2 = gsl_vector_complex_get(e_value, 1).dat[0];
    gsl_matrix_complex_get_col(tmp_complex, e_vector, 0);
    copy_complex_vector_to_real_vector(tmp_complex, s1);
    gsl_matrix_complex_get_col(tmp_complex, e_vector, 1);
    copy_complex_vector_to_real_vector(tmp_complex, s2);

    gsl_matrix_free(hamitonian);
    gsl_vector_complex_free(e_value);
    gsl_matrix_complex_free(e_vector);
    gsl_vector_complex_free(tmp_complex);
}


void calculate_density_matrix(gsl_matrix_complex *density_matrix, gsl_vector_complex *expand) {

    auto v1 = gsl_matrix_complex_view_vector(expand, 2, 1);
    auto v2 = gsl_matrix_complex_view_vector(expand, 1, 2);
    gsl_extra_multiply(&v1.matrix, &v2.matrix, density_matrix);
}


double NAC(void (*f)(gsl_matrix*,double), double x, gsl_vector *s1, gsl_vector *s2, const double e1, const double e2) {
    auto dh = gsl_matrix_alloc(2,2);
    f(dh,x);
    auto nac = gsl_matrix_alloc(1, 1);
    gsl_matrix_view t1 = gsl_matrix_view_vector(s1, 1, 2);
    gsl_matrix_view t2 = gsl_matrix_view_vector(s2, 2, 1);
    auto tmp = gsl_matrix_alloc(2, 1);
    gsl_extra_multiply(dh, &t2.matrix, tmp);
    gsl_extra_multiply(&t1.matrix, tmp, nac);

    double result = gsl_matrix_get(nac, 0, 0);
    gsl_matrix_free(dh);
    gsl_matrix_free(nac);
    gsl_matrix_free(tmp);
    return result / (e2 - e1);
}


/// tully model 1
/// \param m hamitonian matrix to set
/// \param x position
void model_1(gsl_matrix *m, double x) {
    int flag = x < 0 ? -1 : 1;
    double h11 = flag * 0.01 * (1 - exp(-flag * 1.6 * x));
    double h12 = 0.005 * exp(-x * x);
    gsl_matrix_set(m, 0, 0, h11);
    gsl_matrix_set(m, 1, 1, -h11);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

/// tully model 1 derived hamitonian
/// \param m derived hamitonian matrix to set
/// \param x position
void model_1_derive(gsl_matrix *m, double x) {
    double h11 = 0.01 * 1.6 * exp((x < 0 ? 1 : -1) * 1.6 * x);
    double h12 = -2 * 0.005 * x * exp(-x * x);
    gsl_matrix_set(m, 0, 0, h11);
    gsl_matrix_set(m, 1, 1, -h11);
    gsl_matrix_set(m, 1, 0, h12);
    gsl_matrix_set(m, 0, 1, h12);
}

/// tully model 2
/// \param m hamitonian matrix to set
/// \param x position
void model_2(gsl_matrix *m, double x) {
    double h12 = 0.015 * exp(-0.06 * x * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, -0.1 * exp(-0.28 * x * x) + 0.05);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

/// tully model 1 derived hamitonian
/// \param m derived hamitonian matrix to set
/// \param x position
void model_2_derive(gsl_matrix *m, double x) {
    double d12 = -2 * 0.015 * 0.06 * exp(-0.06 * x * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, 2 * 0.1 * 0.28 * x * exp(-0.28 * x * x));
    gsl_matrix_set(m, 1, 0, d12);
    gsl_matrix_set(m, 0, 1, d12);
}


/// tully model 3
/// \param m hamitonian matrix to set
/// \param x position
void model_3(gsl_matrix *m, double x) {
    double d12 = x < 0 ? 0.1 * exp(0.9 * x) : 0.1 * (2 - exp(-0.9 * x));
    gsl_matrix_set(m, 0, 0, 6e-4);
    gsl_matrix_set(m, 1, 1, -6e-4);
    gsl_matrix_set(m, 0, 1, d12);
    gsl_matrix_set(m, 1, 0, d12);
}

/// tully model 1 derived hamitonian
/// \param m derived hamitonian matrix to set
/// \param x position
void model_3_derive(gsl_matrix *m, double x) {
    double d12 = 0.1 * 0.9 * exp((x > 0 ? -1 : 1) * 0.9 * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, 0);
    gsl_matrix_set(m, 1, 0, d12);
    gsl_matrix_set(m, 0, 1, d12);
}
