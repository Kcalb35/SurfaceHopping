//
// Created by grass on 4/15/21.
//
#include "FSSHMath.h"
#include "gslExtra.h"
#include "easylogging++.h"
#include <cmath>
#include <random>
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_complex_math.h"
#include "ModelBase.h"
#include <stdexcept>
#include <iomanip>


void phase_correction(gsl_vector *ref, gsl_vector *t) {
    if (gsl_vector_get(t, 0) * gsl_vector_get(ref, 0) < 0 ||
        gsl_vector_get(t, 1) * gsl_vector_get(ref, 1) < 0) {
        gsl_vector_scale(t, -1);
    }
}

void cal_momenta(const double e[], const double mass, const int state, const double v_active, double p[]) {
    double p_active = v_active * mass;
    p[state] = p_active;
    if (p_active * p_active / 2 / mass + e[state] < e[1 - state]) {
        p[1 - state] = 0;
    } else {
        p[1 - state] = sgn(p_active) * sqrt(p_active * p_active + 2 * mass * (e[state] - e[1 - state]));
    }
}

void set_hamitonian_z_by_pc(gsl_matrix_complex *hamitonian_z, const double p[], const double mass, const int state) {
    gsl_matrix_complex_set_zero(hamitonian_z);
    gsl_matrix_complex_set(hamitonian_z, 0, 0, gsl_complex{-p[state] * p[0] / mass, 0});
    gsl_matrix_complex_set(hamitonian_z, 1, 1, gsl_complex{-p[state] * p[1] / mass, 0});
}

double integral(gsl_vector *left, gsl_matrix *op, gsl_vector *right, gsl_vector *wb) {
    gsl_vector_set_zero(wb);
    gsl_blas_dgemv(CblasNoTrans, 1, op, right, 0, wb);
    double result;
    gsl_blas_ddot(left, wb, &result);
    return result;
}

void diagonalize(gsl_matrix *hamitonian, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_symmv_workspace *wb, gsl_vector *e_value, gsl_matrix *e_vector) {
    gsl_eigen_symmv(hamitonian, e_value, e_vector, wb);
    gsl_eigen_symmv_sort(e_value, e_vector, GSL_EIGEN_SORT_VAL_ASC);

    e1 = gsl_vector_get(e_value, 0);
    e2 = gsl_vector_get(e_value, 1);
    gsl_matrix_get_col(s1, e_vector, 0);
    gsl_matrix_get_col(s2, e_vector, 1);

}

double NAC(gsl_matrix *dh, gsl_vector *s1, gsl_vector *s2, double e1,
           double e2, gsl_vector *tmp_wb) {
    double result;
    gsl_vector_set_zero(tmp_wb);
    gsl_blas_dgemv(CblasNoTrans, 1, dh, s2, 0, tmp_wb);
    gsl_blas_ddot(s1, tmp_wb, &result);
    return result / (e2 - e1);
}

void set_hamitonian_z_by_e(gsl_matrix_complex *h, const double e[]) {
    gsl_matrix_complex_set_zero(h);
    for (int i = 0; i < h->size1; ++i) {
        gsl_matrix_complex_set(h, i, i, gsl_complex{e[i], 0});
    }
}

void
density_matrix_grad_cal(gsl_matrix_complex *density_grad, gsl_matrix_complex *density, gsl_matrix_complex *hamitonian,
                        double v, gsl_matrix *nac) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            gsl_complex grad_complex = gsl_complex{0, 0};
            for (int k = 0; k < 2; ++k) {
                grad_complex = gsl_complex_add(grad_complex,
                                               gsl_complex_mul(gsl_matrix_complex_get(density, k, j),
                                                               gsl_complex_sub_imag(
                                                                       gsl_matrix_complex_get(hamitonian, i, k),
                                                                       v * gsl_matrix_get(nac, i, k))));
                grad_complex = gsl_complex_sub(grad_complex,
                                               gsl_complex_mul(gsl_matrix_complex_get(density, i, k),
                                                               gsl_complex_sub_imag(
                                                                       gsl_matrix_complex_get(hamitonian, k, j),
                                                                       v * gsl_matrix_get(nac, k, j))));
            }
            grad_complex = gsl_complex_div_imag(grad_complex, 1);
            gsl_matrix_complex_set(density_grad, i, j, grad_complex);
        }
    }
}

FinalPosition
run_single_trajectory(NumericalModel *num_model, AnalyticModel *ana_model, int start_state, double start_momenta,
                      double dt, bool debug, model_type type, SHMethod method, const double left, const double right) {

    if (type == model_type::analytic && ana_model == nullptr || type == model_type::numerical && num_model == nullptr)
        throw std::runtime_error("not supported");

    auto nac = gsl_matrix_calloc(2, 2);
    auto hamitonian = gsl_matrix_alloc(2, 2);
    auto hamitonian_z = gsl_matrix_complex_alloc(2, 2);
    gsl_vector *t[] = {gsl_vector_alloc(2), gsl_vector_alloc(2)};
    gsl_vector *tmp_t[] = {
            gsl_vector_alloc(2),
            gsl_vector_alloc(2)
    };
    auto density_matrix = gsl_matrix_complex_calloc(2, 2);
    auto tmp_density_matrix = gsl_matrix_complex_alloc(2, 2);
    gsl_matrix_complex *tmp_density_matrix_grad[]{
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2)
    };

    // prepare work space
    auto wb = gsl_eigen_symmv_alloc(2);
    auto wb_vecs = gsl_matrix_alloc(2, 2);
    auto wb_vals = gsl_vector_alloc(2);
    auto wb_dot_vec = gsl_vector_calloc(2);

    double e[] = {0, 0};
    double p_prev[] = {0, 0};
    double p_now[]{0, 0};
    int log_cnt = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0, 1.0);

    // initial an atom
    Atom atom(2000);
    atom.velocity = start_momenta / atom.mass;
    atom.kinetic_energy = start_momenta * start_momenta / 2 / atom.mass;
    atom.x = -17.5 - 30 / start_momenta;
    atom.state = start_state;
    if (type == model_type::analytic) {
        ana_model->diagonal_analytic(atom.x, hamitonian, e[0], e[1], t[0], t[1]);
    } else {
        num_model->hamitonian_cal(hamitonian, atom.x);
        diagonalize(hamitonian, e[0], e[1], t[0], t[1], wb, wb_vals, wb_vecs);
    }
    atom.potential_energy = e[atom.state];
    gsl_matrix_complex_set(density_matrix, atom.state, atom.state, gsl_complex{1, 0});
    if (debug) atom.log("start");

    // start dynamic evolve
    double rk4dt[]{dt / 2, dt / 2, dt, 0};
    double v[4], a[4], tmp_x, tmp_v;
    double nac_x;
    while ((atom.velocity > 0 && atom.x <= right) || (atom.velocity < 0 && atom.x > left)) {
        // using RK4
        tmp_x = atom.x;
        tmp_v = atom.velocity;
        gsl_matrix_complex_memcpy(tmp_density_matrix, density_matrix);

        for (int i = 0; i < 4; ++i) {
            v[i] = tmp_v;
            if (type == model_type::analytic) {
                a[i] = -ana_model->energy_grad_analytic(tmp_x, atom.state) / atom.mass;
                ana_model->diagonal_analytic(tmp_x, hamitonian, e[0], e[1], tmp_t[0], tmp_t[1]);
                nac_x = ana_model->nac_analytic(tmp_x);
            } else {
                num_model->hamitonian_cal(hamitonian, tmp_x);
                diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);

                for (int j = 0; j < 2; ++j) {
                    if (gsl_vector_get(tmp_t[j], 0) * gsl_vector_get(t[j], 0) < 0 ||
                        gsl_vector_get(tmp_t[j], 1) * gsl_vector_get(t[j], 1) < 0)
                        gsl_vector_scale(tmp_t[j], -1);
                }

                // set d_hamitonian
                num_model->d_hamitonian_cal(hamitonian, tmp_x);
                nac_x = NAC(hamitonian, tmp_t[0], tmp_t[1], e[0], e[1], wb_dot_vec);
                a[i] = -integral(tmp_t[atom.state], hamitonian, tmp_t[atom.state], wb_dot_vec) / atom.mass;
            }
            switch (method) {
                case FSSH:
                    set_hamitonian_z_by_e(hamitonian_z, e);
                    break;
                case PCFSSH:
                case PCBCSH:
                    cal_momenta(e, atom.mass, atom.state, tmp_v, p_now);
                    set_hamitonian_z_by_pc(hamitonian_z, p_now, atom.mass, atom.state);
                    break;
            }
            gsl_matrix_set(nac, 0, 1, nac_x);
            gsl_matrix_set(nac, 1, 0, -nac_x);
            density_matrix_grad_cal(tmp_density_matrix_grad[i], tmp_density_matrix, hamitonian_z, tmp_v, nac);
            if (i == 3) break;
            tmp_x = atom.x + rk4dt[i] * v[i];
            tmp_v = atom.velocity + rk4dt[i] * a[i];
            gsl_matrix_complex_memcpy(tmp_density_matrix_grad[4], tmp_density_matrix_grad[i]);
            gsl_matrix_complex_scale(tmp_density_matrix_grad[4], gsl_complex{rk4dt[i], 0});
            gsl_matrix_complex_memcpy(tmp_density_matrix, density_matrix);
            gsl_matrix_complex_add(tmp_density_matrix, tmp_density_matrix_grad[4]);
        }

        // finish RK4 update all properties
        atom.x += dt / 6.0 * (v[0] + 2 * v[1] + 2 * v[2] + v[3]);
        atom.velocity += dt / 6.0 * (a[0] + 2 * a[1] + 2 * a[2] + a[3]);
        gsl_matrix_complex_scale(tmp_density_matrix_grad[0], gsl_complex{dt / 6, 0});
        gsl_matrix_complex_scale(tmp_density_matrix_grad[1], gsl_complex{dt / 3, 0});
        gsl_matrix_complex_scale(tmp_density_matrix_grad[2], gsl_complex{dt / 3, 0});
        gsl_matrix_complex_scale(tmp_density_matrix_grad[3], gsl_complex{dt / 6, 0});
        for (int i = 0; i < 4; ++i) {
            gsl_matrix_complex_add(density_matrix, tmp_density_matrix_grad[i]);
        }

        // update energies
        if (type == model_type::analytic) {
            ana_model->diagonal_analytic(atom.x, hamitonian, e[0], e[1], t[0], t[1]);
            nac_x = ana_model->nac_analytic(atom.x);
        } else {
            num_model->hamitonian_cal(hamitonian, atom.x);
            diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);

            for (int j = 0; j < 2; ++j) {
                if (gsl_vector_get(tmp_t[j], 0) * gsl_vector_get(t[j], 0) < 0 ||
                    gsl_vector_get(tmp_t[j], 1) * gsl_vector_get(t[j], 1) < 0)
                    gsl_vector_scale(tmp_t[j], -1);
                gsl_vector_memcpy(t[j], tmp_t[j]);
            }
            num_model->d_hamitonian_cal(hamitonian, atom.x);
            nac_x = NAC(hamitonian, t[0], t[1], e[0], e[1], wb_dot_vec);
        }
        atom.potential_energy = e[atom.state];
        atom.kinetic_energy = 0.5 * atom.mass * atom.velocity * atom.velocity;
        // update nac
        gsl_matrix_set(nac, 0, 1, nac_x);
        gsl_matrix_set(nac, 1, 0, -nac_x);
        switch (method) {
            case FSSH:
                set_hamitonian_z_by_e(hamitonian_z, e);
                break;
            case PCFSSH:
            case PCBCSH:
                cal_momenta(e, atom.mass, atom.state, atom.velocity, p_now);
                set_hamitonian_z_by_pc(hamitonian_z, p_now, atom.mass, atom.state);
                break;
        }

        // calculate switching probability
        int k = atom.state;
        gsl_complex tmp_complex;
        tmp_complex = gsl_matrix_complex_get(density_matrix, 1 - k, k);
        tmp_complex = gsl_complex_conjugate(tmp_complex);
        double b = 2 * GSL_IMAG(gsl_complex_mul(tmp_complex, gsl_matrix_complex_get(hamitonian_z, 1 - k, k))) -
                   2 * GSL_REAL(gsl_complex_mul_real(tmp_complex, atom.velocity * gsl_matrix_get(nac, 1 - k, k)));

        double prob = dt * b / GSL_REAL(gsl_matrix_complex_get(density_matrix, k, k));
        if (prob < 0) prob = 0;

        // generate random number
        double zeta = distrib(gen);
        // decide whether to switch
        if (zeta < prob) {
            // switch from atom.state to 3-atom.state or k -> 1-k
            double de = e[1 - k] - e[k];
            if (de <= atom.kinetic_energy) {
                // allowed
                if (debug) atom.log("jump_before");
                atom.kinetic_energy -= de;
                atom.potential_energy = e[1 - k];
                atom.velocity = (sgn(atom.velocity)) * sqrt(2 * atom.kinetic_energy / atom.mass);
                atom.state = 1 - k;
                if (debug) {
                    LOG(INFO) << "jump " << zeta << '/' << prob;
                    atom.log("jump_after");
                }
            }
        }

        // BC
        if (method == SHMethod::PCBCSH) {
            // perform BC
            if (p_prev[atom.state] * p_now[atom.state] < 0 || p_prev[1 - atom.state] * p_now[1 - atom.state] < 0) {
                gsl_complex ca = gsl_matrix_complex_get(density_matrix, atom.state, atom.state);
                gsl_matrix_complex_set_zero(density_matrix);
                gsl_matrix_complex_set(density_matrix, atom.state, atom.state,
                                       gsl_complex_mul_real(ca, 1 / gsl_complex_abs(ca)));
            }
            p_prev[0] = p_now[0];
            p_prev[1] = p_now[1];
        }

        log_cnt++;
        if (debug)
            if (log_cnt % int(10 / dt) == 0) {
                atom.log("move");
//                log_matrix(density_matrix, 2, 2, "density_matrix");
//                LOG(INFO) << zeta << '/' << prob;
                LOG(INFO) << GSL_REAL(gsl_matrix_complex_get(density_matrix, 0, 0)) +
                             GSL_REAL(gsl_matrix_complex_get(density_matrix, 1, 1));
            }
        if (log_cnt * dt > 1e7) {
            LOG(INFO) << "timeout k:" << start_momenta;
            return FinalPosition::timeout;
        }
    }

    // judge final state
    if (debug) atom.log("end");
    FinalPosition final;
    if (atom.x > 0) {
        if (atom.state == 1)
            final = FinalPosition::upper_transmission;
        else
            final = FinalPosition::lower_transmission;
    } else {
        if (atom.state == 1)
            final = FinalPosition::upper_reflection;
        else
            final = FinalPosition::lower_reflection;
    }
    if (debug) {
        LOG(INFO) << "final " << FinalPositionName[final];
//        log_matrix(density_matrix, 2, 2, "density matrix");
    }

    gsl_vector_free(t[0]);
    gsl_vector_free(t[1]);
    gsl_matrix_free(hamitonian);
    gsl_matrix_complex_free(hamitonian_z);
    gsl_matrix_complex_free(density_matrix);
    gsl_matrix_complex_free(tmp_density_matrix);

    gsl_eigen_symmv_free(wb);
    gsl_matrix_free(wb_vecs);
    gsl_vector_free(wb_vals);
    gsl_vector_free(wb_dot_vec);
    for (auto &i : tmp_density_matrix_grad) {
        gsl_matrix_complex_free(i);
    }
    return final;
}

void Atom::log(const std::string &s) const {
    LOG(INFO) << s << std::setprecision(10) << " Ep:" << potential_energy << " Ek:" << kinetic_energy << " E:"
              << potential_energy + kinetic_energy << " x:" << x << " state:" << state << " v:" << velocity;
}

Atom::Atom(double mass) : mass(mass),
                          x(),
                          kinetic_energy(),
                          potential_energy(),
                          velocity(),
                          state() {}
