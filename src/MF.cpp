#include "MF.h"
#include <gsl/gsl_matrix.h>
#include <FSSHMath.h>
#include "gsl/gsl_eigen.h"
#include "ModelBase.h"
#include "stdexcept"
#include "gslExtra.h"
#include "easylogging++.h"

double avg_energy(const gsl_matrix_complex *density, const double e[]) {
    double result = 0;
    for (int i = 0; i < density->size1; ++i) {
        result += e[i] * GSL_REAL(gsl_matrix_complex_get(density, i, i));
    }
    return result;
}

double avg_acceleration(gsl_matrix_complex *density_grad, gsl_matrix_complex *density, gsl_matrix *d_hamitonian,
                        gsl_vector *t[], gsl_vector *wb, double mass, double v, double e[]) {
    double result = 0;
    for (int i = 0; i < density->size1; ++i) {
        result -= GSL_REAL(gsl_matrix_complex_get(density_grad, i, i)) / v * e[i];
        result -= GSL_REAL(gsl_matrix_complex_get(density, i, i)) * integral(t[i], d_hamitonian, t[i], wb);
    }
    return result / mass;
}

void run_single_MF(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
                   const MFMethod method, double result[], const double right, const double left, bool debug) {
    if (model == nullptr) throw std::runtime_error("null model");

    // pre-allocate
    gsl_matrix_complex *density = gsl_matrix_complex_calloc(2, 2);
    gsl_matrix_complex *tmp_density = gsl_matrix_complex_alloc(2, 2);
    gsl_matrix *hamitonian = gsl_matrix_alloc(2, 2);
    gsl_matrix_complex *hamitonian_z = gsl_matrix_complex_calloc(2, 2);
    gsl_vector *t[]{gsl_vector_calloc(2), gsl_vector_calloc(2)};
    gsl_vector *tmp_t[]{gsl_vector_calloc(2), gsl_vector_calloc(2)};
    gsl_matrix *nac = gsl_matrix_calloc(2, 2);
    gsl_matrix_complex *tmp_density_grad[]{
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2)
    };
    double e[2]{0, 0};

    // workspace
    auto wb = gsl_eigen_symmv_alloc(2);
    auto wb_vecs = gsl_matrix_alloc(2, 2);
    auto wb_vals = gsl_vector_alloc(2);
    auto wb_dot_vec = gsl_vector_alloc(2);

    // init an atom
    Atom atom(2000);
    atom.velocity = start_momenta / atom.mass;
    atom.kinetic_energy = start_momenta * start_momenta / 2 / atom.mass;
    atom.x = -17.5 - 30 / start_momenta;
    gsl_matrix_complex_set(density, start_state, start_state, gsl_complex{1, 0});

    model->hamitonian_cal(hamitonian, atom.x);
    diagonalize(hamitonian, e[0], e[1], t[0], t[1], wb, wb_vals, wb_vecs);
    atom.potential_energy = avg_energy(density, e);

    double nac_x, v[4], a[4], tmp_x, tmp_v, tmp_e, tmp_v_all[2]{0, 0}, rk4dt[]{dt / 2, dt / 2, dt};
    while ((atom.velocity > 0 && atom.x <= right) || (atom.velocity < 0 && atom.x >= left)) {
        tmp_x = atom.x;
        tmp_v = atom.velocity;
        gsl_matrix_complex_memcpy(tmp_density, density);

        // RK4
        for (int i = 0; i < 4; ++i) {
            v[i] = tmp_v;
            model->hamitonian_cal(hamitonian, tmp_x);
            diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);
            for (int j = 0; j < 2; ++j) {
                if (gsl_vector_get(tmp_t[j], 0) * gsl_vector_get(t[j], 0) < 0 ||
                    gsl_vector_get(tmp_t[j], 1) * gsl_vector_get(t[j], 1) < 0)
                    gsl_vector_scale(tmp_t[j], -1);
            }
            tmp_e = avg_energy(tmp_density, e);
            model->d_hamitonian_cal(hamitonian, tmp_x);
            nac_x = NAC(hamitonian, tmp_t[0], tmp_t[1], e[0], e[1], wb_dot_vec);
            gsl_matrix_set(nac, 0, 1, nac_x);
            gsl_matrix_set(nac, 1, 0, -nac_x);

            for (int j = 0; j < 2; ++j) {
                double v_square = tmp_v * tmp_v + 2 * (tmp_e - e[j]) / atom.mass;
                if (v_square < 0) tmp_v_all[j] = 0;
                else tmp_v_all[j] = sgn(tmp_v) * sqrt(v_square);
//                gsl_matrix_complex_set(hamitonian_z, j, j, gsl_complex{-tmp_v * tmp_v_all[j] * atom.mass, 0});
            }
            gsl_matrix_complex_set(hamitonian_z, 0, 0, gsl_complex{e[0], 0});
            gsl_matrix_complex_set(hamitonian_z, 1, 1, gsl_complex{e[1], 0});
            density_matrix_grad_cal(tmp_density_grad[i], tmp_density, hamitonian_z, tmp_v, nac);
            a[i] = avg_acceleration(tmp_density_grad[i], tmp_density, hamitonian, tmp_t, wb_dot_vec, atom.mass,
                                    tmp_v, e);
            if (i == 3) break;
            tmp_x = atom.x + rk4dt[i] * v[i];
            tmp_v = atom.velocity + rk4dt[i] * a[i];
            gsl_matrix_complex_memcpy(tmp_density_grad[4], tmp_density_grad[i]);
            gsl_matrix_complex_scale(tmp_density_grad[4], gsl_complex{rk4dt[i], 0});
            gsl_matrix_complex_memcpy(tmp_density, density);
            gsl_matrix_complex_add(tmp_density, tmp_density_grad[4]);
        }

        // finish RK4 update all properties
        atom.x += dt / 6 * (v[0] + 2 * v[1] + 2 * v[2] + v[3]);
        atom.velocity += dt / 6 * (a[0] + 2 * a[1] + 2 * a[2] + a[3]);
        gsl_matrix_complex_scale(tmp_density_grad[0], gsl_complex{dt / 6, 0});
        gsl_matrix_complex_scale(tmp_density_grad[1], gsl_complex{dt / 3, 0});
        gsl_matrix_complex_scale(tmp_density_grad[2], gsl_complex{dt / 3, 0});
        gsl_matrix_complex_scale(tmp_density_grad[3], gsl_complex{dt / 6, 0});
        for (int i = 0; i < 4; ++i) {
            gsl_matrix_complex_add(density, tmp_density_grad[i]);
        }

        // update energy
        model->hamitonian_cal(hamitonian, atom.x);
        diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);

        atom.kinetic_energy = 0.5 * atom.velocity * atom.velocity * atom.mass;
        atom.potential_energy = avg_energy(density, e);

        if (debug) {
            atom.log("move");

            LOG(INFO) << GSL_REAL(gsl_matrix_complex_get(density, 0, 0)) +
                         GSL_REAL(gsl_matrix_complex_get(density, 1, 1)) << " acc:" << a[3];
            log_matrix(density, 2, 2, "density");
        }
    }

    if (atom.velocity > 0) {
        //trans
        result[0] = GSL_REAL(gsl_matrix_complex_get(density, 0, 0));
        result[1] = GSL_REAL(gsl_matrix_complex_get(density, 1, 1));
        result[2] = result[3] = 0;
    } else {
        result[2] = GSL_REAL(gsl_matrix_complex_get(density, 0, 0));
        result[3] = GSL_REAL(gsl_matrix_complex_get(density, 1, 1));
        result[0] = result[1] = 0;
    }

    gsl_matrix_complex_free(density);
    gsl_matrix_free(nac);
    gsl_matrix_free(wb_vecs);
    gsl_matrix_free(hamitonian);
    gsl_matrix_complex_free(hamitonian_z);
    gsl_vector_free(wb_vals);
    gsl_vector_free(wb_dot_vec);
    for (auto &i : t) gsl_vector_free(i);
    for (auto &i : tmp_t) gsl_vector_free(i);
    for (auto &i : tmp_density_grad) gsl_matrix_complex_free(i);

    gsl_eigen_symmv_free(wb);
}