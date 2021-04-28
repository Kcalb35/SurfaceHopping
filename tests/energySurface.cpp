//
// Created by grass on 4/20/21.
//
#include <fstream>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <FSSHMath.h>
#include "easylogging++.h"
#include "CLI11.hpp"
#include <functional>

INITIALIZE_EASYLOGGINGPP

using namespace std;

int main(int argc, char **argv) {
    CLI::App app("Energy surface and nac calculation");
    int model = 1;
    string path("1.dat");

    app.add_option("--model", model, "Tully model index");
    app.add_option("--path", path, "save file path");
    CLI11_PARSE(app, argc, argv);

    ofstream f;
    f.open(path);

    H_matrix_function h_f[3] = {model_1, model_2, model_3};
    H_matrix_function d_h_f[3] = {model_1_derive, model_2_derive, model_3_derive};

    auto wb = gsl_eigen_symmv_alloc(2);
    auto h = gsl_matrix_alloc(2, 2);
    auto t1 = gsl_vector_alloc(2);
    auto t2 = gsl_vector_alloc(2);
    auto tmp1 = gsl_vector_alloc(2);
    auto tmp2 = gsl_vector_alloc(2);
    auto tmp_mid = gsl_matrix_alloc(2, 1);
    auto tmp_matrix_wb = gsl_matrix_alloc(1, 1);
    double e1, e2;
    int max = 10000;
    for (int i = 0; i < max; i++) {
        double x = -10.0 + 20.0 * i / (max - 1);
        h_f[model - 1](h, x);
        diagonalize(h, e1, e2, tmp1, tmp2, wb);
        if (i > 1) {
            if (gsl_vector_get(tmp1, 0) * gsl_vector_get(t1, 0) < 0 ||
                gsl_vector_get(tmp1, 1) * gsl_vector_get(t1, 1) < 0) {
                gsl_vector_scale(tmp1, -1);
            }
            if (gsl_vector_get(tmp2, 0) * gsl_vector_get(t2, 0) < 0 ||
                gsl_vector_get(tmp2, 1) * gsl_vector_get(t2, 1) < 0) {
                gsl_vector_scale(tmp2, -1);
            }
        }
        gsl_vector_memcpy(t1, tmp1);
        gsl_vector_memcpy(t2, tmp2);
        d_h_f[model - 1](h, x);
        double nac = NAC(h, t1, t2, e1, e2, tmp_matrix_wb, tmp_mid);
        f << x << '\t' << e1 << '\t' << e2 << '\t' << nac << '\t' << gsl_vector_get(t1, 0) << '\t'
          << gsl_vector_get(t1, 1)
          << '\t' << gsl_vector_get(t2, 0) << '\t' << gsl_vector_get(t2, 1) << endl;
    }
    f.close();
}