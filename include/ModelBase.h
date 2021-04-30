#ifndef FSSH_MODELBASE_H
#define FSSH_MODELBASE_H

#include <cmath>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

class AnalyticModel {
public:
    virtual double nac_analytic(double x) = 0;

    virtual double energy_grad_analytic(double x, int state) = 0;

    virtual void diagonal_analytic(double x, gsl_matrix *h, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2) = 0;
};

class NumericalModel {
public:
    virtual void hamitonian_cal(gsl_matrix *m, double x) = 0;

    virtual void d_hamitonian_cal(gsl_matrix *m, double x) = 0;
};

class ECR : public NumericalModel, public AnalyticModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    void diagonal_analytic(double x, gsl_matrix *h, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2) override;

    double nac_analytic(double x) override;

    double energy_grad_analytic(double x, int state) override;

};

class SAC : public NumericalModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;
};

class DAC : public NumericalModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;
};

#endif //FSSH_MODELBASE_H
