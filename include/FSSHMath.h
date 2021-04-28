//
// Created by grass on 4/15/21.
//

#ifndef FSSH_FSSHMATH_H
#define FSSH_FSSHMATH_H

#include <gsl/gsl_eigen.h>
#include "gsl/gsl_matrix.h"
#include <string>


struct Atom {
    double x{};
    double mass = 2000;
    double kinetic_energy{};
    double velocity{};
    double potential_energy{};
    int state{};

    void log(const std::string &s) const;
};


enum FinalPosition {
    lower_transmission,
    upper_transmission,
    lower_reflection,
    upper_reflection
};

typedef void (*H_matrix_function)(gsl_matrix *, double);


void diagonalize(gsl_matrix *hamitonian, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_symmv_workspace *wb, gsl_vector *e_value, gsl_matrix *e_vector);

double integral(gsl_vector *left, gsl_matrix *m, gsl_vector *right, gsl_matrix *tmp_mid, gsl_matrix *result_wb);

double NAC(gsl_matrix *dh, gsl_vector *s1, gsl_vector *s2, double e1,
           double e2, gsl_matrix *result_wb, gsl_matrix *tmp_mid);

void model_1(gsl_matrix *m, double x);

void model_1_derive(gsl_matrix *m, double x);

void model_2(gsl_matrix *m, double x);

void model_2_derive(gsl_matrix *m, double x);

void model_3(gsl_matrix *m, double x);

void model_3_derive(gsl_matrix *m, double x);

FinalPosition
run_single_trajectory(H_matrix_function hamitonian_f,
                      H_matrix_function d_hamitonian_f,
                      int start_state, double start_momenta, double dt, bool debug);

#endif //FSSH_FSSHMATH_H
