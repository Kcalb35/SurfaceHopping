//
// Created by grass on 4/15/21.
//
#include "FSSHMath.h"
#include "gslExtra.h"
#include "cmath"
#include <stdexcept>
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_eigen.h"

void diagonalize(gsl_matrix *(*f)(double), double x1, double x2, int steps, double e1[], double e2[], gsl_matrix *t1,
                 gsl_matrix *t2) {
    auto wb = gsl_eigen_nonsymmv_alloc(2);
    auto tmp_complex = gsl_vector_complex_alloc(2);
    auto tmp_real = gsl_vector_alloc(2);
    for (int i = 0; i < steps; ++i) {
        // diagonalize hamitonian
        double x = x1 + (x2 - x1) * i / (steps - 1);
        gsl_matrix *hamitonian = f(x);
        gsl_vector_complex *e_value = gsl_vector_complex_alloc(2);
        gsl_matrix_complex *e_vector = gsl_matrix_complex_alloc(2, 2);
        int error = gsl_eigen_nonsymmv(hamitonian, e_value, e_vector, wb);
        if (error > 1)
            throw std::runtime_error("diagonalize error");

        // copy eigenvalue and eigenvector
        e1[i] = gsl_vector_complex_get(e_value, 0).dat[0];
        e2[i] = gsl_vector_complex_get(e_value, 1).dat[0];
        gsl_matrix_complex_get_col(tmp_complex, e_vector, 0);
        copy_complex_vector_to_real_vector(tmp_complex, tmp_real);
        gsl_matrix_set_row(t1, i, tmp_real);
        gsl_matrix_complex_get_col(tmp_complex, e_vector, 1);
        copy_complex_vector_to_real_vector(tmp_complex, tmp_real);
        gsl_matrix_set_row(t2, i, tmp_real);
        gsl_matrix_free(hamitonian);
    }
    gsl_eigen_nonsymmv_free(wb);
}

double NAC(gsl_matrix *(*f)(double), double x, gsl_vector *s1, gsl_vector *s2, const double e1, const double e2) {
    auto dh = f(x);
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


/// tully model 1 hamitonian
/// \param x position
/// \return Hamitonian matrix ,remember to delete
gsl_matrix *model_1(double x) {
    auto h = gsl_matrix_alloc(2, 2);
    int flag = x < 0 ? -1 : 1;
    double h11 = flag * 0.01 * (1 - exp(-flag * 1.6 * x));
    double h12 = 0.005 * exp(-x * x);
    gsl_matrix_set(h, 0, 0, h11);
    gsl_matrix_set(h, 1, 1, -h11);
    gsl_matrix_set(h, 0, 1, h12);
    gsl_matrix_set(h, 1, 0, h12);
    return h;
}

/// tully model 1 derived hamitonian
/// \param x position
/// \return derived Hamiltonian matrix, remember to delete
gsl_matrix *model_1_derive(double x) {
    auto h = gsl_matrix_alloc(2, 2);
    double h11 = 0.01 * 1.6 * exp((x < 0 ? 1 : -1) * 1.6 * x);
    double h12 = -2 * 0.005 * x * exp(-x * x);
    gsl_matrix_set(h, 0, 0, h11);
    gsl_matrix_set(h, 1, 1, -h11);
    gsl_matrix_set(h, 1, 0, h12);
    gsl_matrix_set(h, 0, 1, h12);
    return h;
}

/// tully model 2 hamitonian
/// \param x position
/// \return Hamitonian matrix ,remember to delete
gsl_matrix *model_2(double x) {
    auto h = gsl_matrix_alloc(2, 2);
    double h12 = 0.015*exp(-0.06*x*x);
    gsl_matrix_set(h, 0, 0, 0);
    gsl_matrix_set(h, 1, 1, -0.1*exp(-0.28*x*x)+0.05);
    gsl_matrix_set(h, 0, 1, h12);
    gsl_matrix_set(h, 1, 0, h12);
    return h;
}

/// tully model 2 derived hamitonian
/// \param x position
/// \return derived Hamiltonian matrix, remember to delete
gsl_matrix *model_2_derive(double x) {
    auto h = gsl_matrix_alloc(2, 2);
    double d12 = -2*0.015*0.06*exp(-0.06*x*x);
    gsl_matrix_set(h, 0, 0, 0);
    gsl_matrix_set(h, 1, 1, 2*0.1*0.28*x*exp(-0.28*x*x));
    gsl_matrix_set(h, 1, 0, d12);
    gsl_matrix_set(h, 0, 1, d12);
    return h;
}

/// tully model 3 hamitonian
/// \param x position
/// \return Hamitonian matrix ,remember to delete
gsl_matrix *model_3(double x) {
    auto h = gsl_matrix_alloc(2, 2);
    double d12= x<0 ? 0.1*exp(0.9*x): 0.1*(2-exp(-0.9*x));
    gsl_matrix_set(h, 0, 0, 6e-4);
    gsl_matrix_set(h, 1, 1, -6e-4);
    gsl_matrix_set(h, 0, 1, d12);
    gsl_matrix_set(h, 1, 0, d12);
    return h;
}

/// tully model 3 derived hamitonian
/// \param x position
/// \return derived Hamiltonian matrix, remember to delete
gsl_matrix *model_3_derive(double x) {
    auto h = gsl_matrix_alloc(2, 2);
    double d12 = 0.1*0.9*exp((x>0?-1:1)*0.9*x);
    gsl_matrix_set(h, 0, 0, 0);
    gsl_matrix_set(h, 1, 1, 0);
    gsl_matrix_set(h, 1, 0, d12);
    gsl_matrix_set(h, 0, 1, d12);
    return h;
}
