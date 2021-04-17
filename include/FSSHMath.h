//
// Created by grass on 4/15/21.
//

#ifndef FSSH_FSSHMATH_H
#define FSSH_FSSHMATH_H

#include <gsl/gsl_eigen.h>
#include "gsl/gsl_matrix.h"

struct Atom {
    double x = -10;
    double mass = 2000;
    double kinetic_energy{};
    double velocity{};
    double potential_energy{};
    int state{};
    gsl_vector_complex *expand{};
    gsl_complex population{};
};


void diagonalize(void (*f)(gsl_matrix *, double), double x, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_nonsymmv_workspace *wb);


void calculate_density_matrix(gsl_matrix_complex *density_matrix, gsl_vector_complex *expand);

double NAC(void (*f)(gsl_matrix *, double), double x, gsl_vector *s1, gsl_vector *s2, double e1, double e2);

void model_1(gsl_matrix *m, double x);

void model_1_derive(gsl_matrix *m, double x);

void model_2(gsl_matrix *m, double x);

void model_2_derive(gsl_matrix *m, double x);

void model_3(gsl_matrix *m, double x);

void model_3_derive(gsl_matrix *m, double x);

#endif //FSSH_FSSHMATH_H
