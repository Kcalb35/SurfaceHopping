#ifndef SH_MFMATH_H
#define SH_MFMATH_H

#include "gsl/gsl_matrix.h"
#include "ModelBase.h"

enum MFMethod {
    EMF,
    BCMF_s,
    BCMF_w
};

struct weighted_trajectory {
    double x;
    double mass;
    double velocity;
    double kinetic_energy;
    double potential_energy;
    double weight;
    double lifetime;
    bool finish;
    gsl_matrix_complex *density;

    weighted_trajectory(double x) {
        mass = 2000;
        this->x = x;
        density = gsl_matrix_complex_calloc(2, 2);
        lifetime = 0;
        finish = false;
    }

    void calculate_population(double result[]) {
        double p1 = GSL_REAL(gsl_matrix_complex_get(density, 0, 0));
        double p2 = GSL_REAL(gsl_matrix_complex_get(density, 1, 1));
        if (velocity > 0) {
            result[0] = p1;
            result[1] = p2;
            result[2] = result[3] = 0;
        } else {
            result[2] = p1;
            result[3] = p2;
            result[0] = result[1] = 0;
        }
    }

    ~weighted_trajectory() {
        gsl_matrix_complex_free(density);
    }
};

int run_single_MF(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
                  const MFMethod method, double result[], const double start_x, const double left, const double right,
                  bool debug);


#endif //SH_MFMATH_H
