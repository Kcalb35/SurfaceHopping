#ifndef SH_MFMATH_H
#define SH_MFMATH_H

#include "gsl/gsl_matrix.h"
#include "ModelBase.h"
#include "easylogging++.h"

enum MFMethod {
    EMF,
    BCMF_s,
    BCMF_w
};

struct weighted_trajectory {
    double mass;
    double x;
    double velocity;
    double kinetic_energy;
    double potential_energy;
    double weight;
    bool finish;
    double e_total;
    double e[2]{};
    double p_prev[2]{};
    gsl_vector *t[2];
    gsl_matrix_complex *density;

    weighted_trajectory(double mass, double x, double weight) {
        this->mass = mass;
        this->x = x;
        this->weight = weight;
        density = gsl_matrix_complex_calloc(2, 2);
        for (int i = 0; i < 2; ++i)
            t[i] = gsl_vector_alloc(2);
        finish = false;
    }

    void calculate_population(double result[]) {
        double p1 = GSL_REAL(gsl_matrix_complex_get(density, 0, 0));
        double p2 = GSL_REAL(gsl_matrix_complex_get(density, 1, 1));
        if (velocity > 0) {
            result[0] = weight * p1;
            result[1] = weight * p2;
            result[2] = result[3] = 0;
        } else {
            result[2] = weight * p1;
            result[3] = weight * p2;
            result[0] = result[1] = 0;
        }
    }

    bool is_finish(const double left, const double right) {
        if (velocity > 0 && x > right || velocity < 0 && x < left) {
            finish = true;
        } else finish = false;
        return finish;
    }

    ~weighted_trajectory() {
        gsl_matrix_complex_free(density);
        for (int i = 0; i < 2; ++i)
            gsl_vector_free(t[i]);
    }

    void log() {
        LOG(INFO) << "move Ep:" << potential_energy << " Ek:" << kinetic_energy << " E:"
                  << potential_energy + kinetic_energy << " x:" << x << " v:" << velocity << " finish:"
                  << (finish ? "yes" : "no") << " pop.:"
                  << GSL_REAL(gsl_matrix_complex_get(density, 0, 0)) + GSL_REAL(gsl_matrix_complex_get(density, 1, 1))
                  << " w:" << weight;
    }
};

int run_single_MF(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
                  const MFMethod method, double result[], const double start_x, const double left, const double right,
                  bool debug);

int run_BCMF_w(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
               double result[], const double start_x, const double left, const double right,
               const int max_num_trajectories, bool debug, double timeout_w, double timeout_t);


#endif //SH_MFMATH_H
