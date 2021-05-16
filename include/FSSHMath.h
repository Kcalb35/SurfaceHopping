//
// Created by grass on 4/15/21.
//

#ifndef FSSH_FSSHMATH_H
#define FSSH_FSSHMATH_H

#include <gsl/gsl_eigen.h>
#include "gsl/gsl_matrix.h"
#include "ModelBase.h"
#include <string>


struct Atom {
    double x;
    double mass;
    double kinetic_energy;
    double velocity;
    double potential_energy;
    int state;

    void log(const std::string &s) const;

    Atom(double mass);
};


enum FinalPosition {
    lower_transmission,
    upper_transmission,
    lower_reflection,
    upper_reflection,
    timeout
};

static const char *const FinalPositionName[]{
        "lower_transmission",
        "upper_transmission",
        "lower_reflection",
        "upper_reflection"
};

enum model_type {
    analytic,
    numerical
};

static const char *const ModelTypeName[]{
        "analytic",
        "numerical"
};


enum SHMethod {
    FSSH,
    PCFSSH,
    PCBCSH
};

static const char *SHMethodName[]{
        "FSSH",
        "PC-FSSH",
        "PC-BCSH"
};

void phase_correction(gsl_vector *ref, gsl_vector *t);

double integral(gsl_vector *left, gsl_matrix *op, gsl_vector *right, gsl_vector *wb);


void
density_matrix_grad_cal(gsl_matrix_complex *density_grad, gsl_matrix_complex *density, gsl_matrix_complex *hamitonian,
                        double v, gsl_matrix *nac);

void diagonalize(gsl_matrix *hamitonian, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_symmv_workspace *wb, gsl_vector *e_value, gsl_matrix *e_vector);

double NAC(gsl_matrix *dh, gsl_vector *s1, gsl_vector *s2, double e1,
           double e2, gsl_vector *tmp_wb);

FinalPosition
run_single_trajectory(NumericalModel *num_model, AnalyticModel *ana_model, int start_state, double start_momenta,
                      double dt, bool debug, model_type type, SHMethod method, double left, double right);

#endif //FSSH_FSSHMATH_H
