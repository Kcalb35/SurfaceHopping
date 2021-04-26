//
// Created by grass on 4/15/21.
//

#ifndef FSSH_FSSHMATH_H
#define FSSH_FSSHMATH_H

#include <gsl/gsl_eigen.h>
#include "gsl/gsl_matrix.h"
#include <string>
#include <functional>


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


void diagonalize(gsl_matrix *hamitonian, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_symmv_workspace *wb);

double integral(gsl_vector *left, gsl_matrix *m, gsl_vector *right);

void calculate_density_matrix(gsl_matrix_complex *density_matrix, gsl_vector_complex *expand);

double
NAC(const std::function<void(gsl_matrix *, double)> &f, double x, gsl_vector *s1, gsl_vector *s2, double e1, double e2);

void model_1(gsl_matrix *m, double x);

void model_1_derive(gsl_matrix *m, double x);

void model_2(gsl_matrix *m, double x);

void model_2_derive(gsl_matrix *m, double x);

void model_3(gsl_matrix *m, double x);

void model_3_derive(gsl_matrix *m, double x);

FinalPosition
run_single_trajectory(const std::function<void(gsl_matrix *, double)> &hamitonian_f,
                      const std::function<void(gsl_matrix *, double)> &d_hamitonian_f,
                      int start_state, double start_momenta, double dt, bool debug);

#endif //FSSH_FSSHMATH_H
