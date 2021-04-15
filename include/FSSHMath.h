//
// Created by grass on 4/15/21.
//

#ifndef FSSH_FSSHMATH_H
#define FSSH_FSSHMATH_H

#include "gsl/gsl_matrix.h"


void diagonalize(gsl_matrix *(*f)(double), double x1, double x2, int steps, double *e1, double *e2, gsl_matrix *t1,
                 gsl_matrix *t2);


double NAC(gsl_matrix *(*f)(double), double x, gsl_vector *s1, gsl_vector *s2, double e1, double e2);

gsl_matrix *model_1(double x);

gsl_matrix *model_1_derive(double x);

gsl_matrix *model_2_derive(double x);

gsl_matrix *model_2(double x);

gsl_matrix *model_3_derive(double x);

gsl_matrix *model_3(double x);

#endif //FSSH_FSSHMATH_H
