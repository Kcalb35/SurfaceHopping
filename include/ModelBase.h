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

    virtual double sigma_x(double k) = 0;

    virtual double sigma_p(double k) = 0;

    double x0;
};

class ECR : public NumericalModel, public AnalyticModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    void diagonal_analytic(double x, gsl_matrix *h, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2) override;

    double nac_analytic(double x) override;

    double energy_grad_analytic(double x, int state) override;

    double sigma_x(double k) override;

    double sigma_p(double k) override;

    ECR() {
        x0 = -17.5;
    }

};

class SAC : public NumericalModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    double sigma_x(double k) override;

    double sigma_p(double k) override;

    SAC() {
        x0 = -17.5;
    }
};

class DAC : public NumericalModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    double sigma_x(double k) override;

    double sigma_p(double k) override;

    DAC() {
        x0 = -17.5;
    }
};

class DBG : public NumericalModel {
public:
    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    void hamitonian_cal(gsl_matrix *m, double x) override;

    double sigma_x(double k) override;

    double sigma_p(double k) override;

    DBG() {
        x0 = -22.5;
    };
};

class DAG : public NumericalModel {
public:
    void hamitonian_cal(gsl_matrix *m, double x) override;

    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    double sigma_x(double k) override;

    double sigma_p(double k) override;

    DAG() {
        x0 = -27.5;
    }
};

class DRN : public NumericalModel {
public:
    void d_hamitonian_cal(gsl_matrix *m, double x) override;

    void hamitonian_cal(gsl_matrix *m, double x) override;

    double sigma_x(double k) override;

    double sigma_p(double k) override;

    DRN() {
        x0 = -12.5;
    }
};

#endif //FSSH_MODELBASE_H
