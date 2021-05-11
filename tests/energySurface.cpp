//
// Created by grass on 4/20/21.
//
#include <fstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <FSSHMath.h>
#include "easylogging++.h"
#include "ModelBase.h"

INITIALIZE_EASYLOGGINGPP

using namespace std;


double test(AnalyticModel *model, double x) {
    return model->nac_analytic(x);
}

int main(int argc, char **argv) {
    int model = 1;
    bool ana_flag = false;
    ofstream f;
    f.open("data/model-1");


    auto wb = gsl_eigen_symmv_alloc(2);
    auto h = gsl_matrix_alloc(2, 2);
    gsl_vector *tmp_t[] = {
            gsl_vector_calloc(2),
            gsl_vector_calloc(2)
    };
    gsl_vector *t[] = {
            gsl_vector_calloc(2),
            gsl_vector_calloc(2)
    };
    auto tmp_e_vals = gsl_vector_alloc(2);
    auto tmp_e_vecs = gsl_matrix_alloc(2, 2);
    double e1, e2;
    int max = 100;
    SAC sac;
    DAC dac;
    ECR ecr;
    NumericalModel *num_ms[] = {&sac, &dac, &ecr};
    AnalyticModel *ana_ms[] = {nullptr, nullptr, &ecr};
    if (ana_flag && ana_ms[model - 1] == nullptr || !ana_flag && num_ms[model - 1] == nullptr)
        throw runtime_error("not supported");
    for (int i = 0; i < max; i++) {
        double x = -10.0 + 20.0 * i / (max - 1);
        double nac;
        if (ana_flag) {
            nac = ana_ms[model - 1]->nac_analytic(x);
            ana_ms[model - 1]->diagonal_analytic(x, h, e1, e2, t[0], t[1]);
        } else {
            num_ms[model - 1]->hamitonian_cal(h, x);
            diagonalize(h, e1, e2, tmp_t[0], tmp_t[1], wb, tmp_e_vals, tmp_e_vecs);
            for (int j = 0; j < 2; ++j) {
                if (gsl_vector_get(tmp_t[j], 0) * gsl_vector_get(t[j], 0) < 0 ||
                    gsl_vector_get(tmp_t[j], 1) * gsl_vector_get(t[j], 1) < 0)
                    gsl_vector_scale(tmp_t[j], -1);
                gsl_vector_memcpy(t[j], tmp_t[j]);
            }
            num_ms[model - 1]->d_hamitonian_cal(h, x);
            nac = NAC(h, t[0], t[1], e1, e2, tmp_e_vals);
        }
        f << x << '\t' << e1 << '\t' << e2 << '\t' << nac << '\t' << gsl_vector_get(t[0], 0) << '\t'
          << gsl_vector_get(t[0], 1)
          << '\t' << gsl_vector_get(t[1], 0) << '\t' << gsl_vector_get(t[1], 1) << endl;
    }
    f.close();
}