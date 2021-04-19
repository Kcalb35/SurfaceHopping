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
#include <mutex>


const char *FinalPositionName[] = {
        "lower_transmission",
        "upper_transmission",
        "lower_reflection",
        "upper_reflection"
};

std::mutex m;

void diagonalize(gsl_matrix *hamitonian, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_nonsymmv_workspace *wb) {
    auto *e_value = gsl_vector_complex_alloc(2);
    auto *e_vector = gsl_matrix_complex_alloc(2, 2);
    auto *tmp_complex = gsl_vector_complex_alloc(2);
    gsl_eigen_nonsymmv(hamitonian, e_value, e_vector, wb);

    e1 = gsl_vector_complex_get(e_value, 0).dat[0];
    e2 = gsl_vector_complex_get(e_value, 1).dat[0];
    gsl_matrix_complex_get_col(tmp_complex, e_vector, 0);
    copy_complex_vector_to_real_vector(tmp_complex, s1);
    gsl_matrix_complex_get_col(tmp_complex, e_vector, 1);
    copy_complex_vector_to_real_vector(tmp_complex, s2);

    gsl_vector_complex_free(e_value);
    gsl_matrix_complex_free(e_vector);
    gsl_vector_complex_free(tmp_complex);
}


void calculate_density_matrix(gsl_matrix_complex *density_matrix, gsl_vector_complex *expand) {

    auto v1 = gsl_matrix_complex_view_vector(expand, 2, 1);
    auto v2 = gsl_matrix_complex_view_vector(expand, 1, 2);
    gsl_extra_multiply(&v1.matrix, &v2.matrix, density_matrix);
}


double NAC(H_matrix_function f, double x, gsl_vector *s1, gsl_vector *s2, const double e1, const double e2) {
    auto dh = gsl_matrix_alloc(2, 2);
    f(dh, x);
    auto nac = gsl_matrix_alloc(1, 1);
    gsl_matrix_view t1 = gsl_matrix_view_vector(s1, 1, 2);
    gsl_matrix_view t2 = gsl_matrix_view_vector(s2, 2, 1);
    auto tmp = gsl_matrix_alloc(2, 1);
    gsl_extra_multiply(dh, &t2.matrix, tmp);
    gsl_extra_multiply(&t1.matrix, tmp, nac);

    double result = gsl_matrix_get(nac, 0, 0);
    gsl_matrix_free(dh);
    gsl_matrix_free(nac);
    gsl_matrix_free(tmp);
    return result / (e2 - e1);
}


/// tully model 1
/// \param m hamitonian matrix to set
/// \param x position
void model_1(gsl_matrix *m, double x) {
    int flag = x < 0 ? -1 : 1;
    double h11 = flag * 0.01 * (1 - exp(-flag * 1.6 * x));
    double h12 = 0.005 * exp(-x * x);
    gsl_matrix_set(m, 0, 0, h11);
    gsl_matrix_set(m, 1, 1, -h11);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

/// tully model 1 derived hamitonian
/// \param m derived hamitonian matrix to set
/// \param x position
void model_1_derive(gsl_matrix *m, double x) {
    double h11 = 0.01 * 1.6 * exp((x < 0 ? 1 : -1) * 1.6 * x);
    double h12 = -2 * 0.005 * x * exp(-x * x);
    gsl_matrix_set(m, 0, 0, h11);
    gsl_matrix_set(m, 1, 1, -h11);
    gsl_matrix_set(m, 1, 0, h12);
    gsl_matrix_set(m, 0, 1, h12);
}

/// tully model 2
/// \param m hamitonian matrix to set
/// \param x position
void model_2(gsl_matrix *m, double x) {
    double h12 = 0.015 * exp(-0.06 * x * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, -0.1 * exp(-0.28 * x * x) + 0.05);
    gsl_matrix_set(m, 0, 1, h12);
    gsl_matrix_set(m, 1, 0, h12);
}

/// tully model 1 derived hamitonian
/// \param m derived hamitonian matrix to set
/// \param x position
void model_2_derive(gsl_matrix *m, double x) {
    double d12 = -2 * 0.015 * 0.06 * exp(-0.06 * x * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, 2 * 0.1 * 0.28 * x * exp(-0.28 * x * x));
    gsl_matrix_set(m, 1, 0, d12);
    gsl_matrix_set(m, 0, 1, d12);
}


/// tully model 3
/// \param m hamitonian matrix to set
/// \param x position
void model_3(gsl_matrix *m, double x) {
    double d12 = x < 0 ? 0.1 * exp(0.9 * x) : 0.1 * (2 - exp(-0.9 * x));
    gsl_matrix_set(m, 0, 0, 6e-4);
    gsl_matrix_set(m, 1, 1, -6e-4);
    gsl_matrix_set(m, 0, 1, d12);
    gsl_matrix_set(m, 1, 0, d12);
}

/// tully model 1 derived hamitonian
/// \param m derived hamitonian matrix to set
/// \param x position
void model_3_derive(gsl_matrix *m, double x) {
    double d12 = 0.1 * 0.9 * exp((x > 0 ? -1 : 1) * 0.9 * x);
    gsl_matrix_set(m, 0, 0, 0);
    gsl_matrix_set(m, 1, 1, 0);
    gsl_matrix_set(m, 1, 0, d12);
    gsl_matrix_set(m, 0, 1, d12);
}

FinalPosition
run_single_trajectory(H_matrix_function h_f, H_matrix_function d_h_f, int start_state, double start_momenta,
                      double dt, bool debug) {
    // prepare a lock
    std::unique_lock<std::mutex> l(m, std::defer_lock);

    // pre-allocate space
    auto wb = gsl_eigen_nonsymmv_alloc(2);
    auto nac = gsl_matrix_alloc(2, 2);
    gsl_matrix_set_all(nac, 0);
    auto hamitonian = gsl_matrix_alloc(2, 2);
    auto t1 = gsl_vector_alloc(2);
    auto t2 = gsl_vector_alloc(2);
    auto density_matrix = gsl_matrix_complex_alloc(2, 2);
    auto density_matrix_grad = gsl_matrix_complex_alloc(2, 2);

    double e[] = {0, 0};
    int log_cnt = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0, 1.0);
    FinalPosition final;

    // initial an atom
    Atom atom;
    atom.state = start_state;
    h_f(hamitonian, atom.x);
    diagonalize(hamitonian, e[0], e[1], t1, t2, wb);
    atom.potential_energy = e[start_state - 1];
    atom.velocity = start_momenta / atom.mass;
    atom.kinetic_energy = atom.mass * atom.velocity * atom.velocity / 2;

    auto expand = gsl_vector_complex_alloc(2);
    gsl_vector_complex_set(expand, start_state - 1, gsl_complex{1, 0});
    gsl_vector_complex_set(expand, 2 - start_state, gsl_complex{0, 0});
    calculate_density_matrix(density_matrix, expand);
    l.lock();
    atom.log("start");
    l.unlock();

    // start dynamic evolve
    while (atom.x <= 10 && atom.x >= -10) {
        // move the atom
        double x = atom.x + atom.velocity * dt;
        // calculate hamitonian and diagonalize it
        h_f(hamitonian, atom.x);
        diagonalize(hamitonian, e[0], e[1], t1, t2, wb);

        // if potential energy surface change
        int target = atom.state;
        if (fabs(e[2 - target] - atom.potential_energy) < fabs(e[target - 1] - atom.potential_energy)) {
            target = 3 - target;
        }

        // update potential and kinetic energy, update velocity
        double d_energy = e[target - 1] - atom.potential_energy;
        if (d_energy < atom.kinetic_energy) {
            atom.x = x;
            atom.potential_energy = e[target - 1];
            atom.kinetic_energy -= d_energy;
            atom.velocity = sgn(atom.velocity) * sqrt(2 * atom.kinetic_energy / atom.mass);
        } else {
            atom.velocity = -atom.velocity;
        }
        atom.state = target;


        // calculate nac
        double nac_x = NAC(d_h_f, atom.x, t1, t2, e[0], e[1]);
        gsl_matrix_set(nac, 0, 1, nac_x);
        gsl_matrix_set(nac, 1, 0, -nac_x);

        // update density matrix
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                gsl_complex grad_complex = gsl_complex{0, 0};
                for (int k = 0; k < 2; ++k) {
                    grad_complex = gsl_complex_add(grad_complex,
                                                   gsl_complex_mul(gsl_matrix_complex_get(density_matrix, k, j),
                                                                   gsl_complex{gsl_matrix_get(hamitonian, i, k),
                                                                               -atom.velocity *
                                                                               gsl_matrix_get(nac, i, k)}));
                    grad_complex = gsl_complex_sub(grad_complex,
                                                   gsl_complex_mul(gsl_matrix_complex_get(density_matrix, i, k),
                                                                   gsl_complex{gsl_matrix_get(hamitonian, k, j),
                                                                               -atom.velocity *
                                                                               gsl_matrix_get(nac, k, j)}));
                }
                grad_complex = gsl_complex_div_imag(grad_complex, 1);
                gsl_matrix_complex_set(density_matrix_grad, i, j, grad_complex);
            }
        }
        gsl_matrix_complex_scale(density_matrix_grad, gsl_complex{dt, 0});
        gsl_matrix_complex_add(density_matrix, density_matrix_grad);

        // calculate switching probability
        int k = atom.state - 1;
        gsl_complex tmp_complex;
        tmp_complex = gsl_matrix_complex_get(density_matrix, 1 - k, k);
        tmp_complex = gsl_complex_conjugate(tmp_complex);
        double b = 2 * GSL_IMAG(gsl_complex_mul_real(tmp_complex, gsl_matrix_get(hamitonian, 1 - k, k))) -
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
                if (debug) {
                    l.lock();
                    atom.log("jump_before");
                    l.unlock();
                }
                atom.kinetic_energy -= de;
                atom.potential_energy = e[1 - k];
                atom.velocity = (sgn(atom.velocity)) * sqrt(2 * atom.kinetic_energy / atom.mass);
                atom.state = 2 - k;
                if (debug) {
                    l.lock();
                    LOG(INFO) << "jump " << zeta << '/' << prob;
                    atom.log("jump_after");
                    l.unlock();
                }
            }
        }

        if (debug)
            if (++log_cnt % LOG_INTERVAL == 0) {
                log_cnt = 0;
                l.lock();
                atom.log("move");
                log_matrix(density_matrix,2,2,"density_matrix");
                LOG(INFO) << zeta << '/' << prob;
                l.unlock();
            }
    }

    // judge final state
    l.lock();
    atom.log("end");
    l.unlock();
    if (atom.x > 0) {
        if (atom.potential_energy > 0)
            final = FinalPosition::upper_transmission;
        else
            final = FinalPosition::lower_transmission;
    } else {
        if (atom.potential_energy < 0)
            final = FinalPosition::lower_reflection;
        else
            final = FinalPosition::upper_reflection;
    }
    l.lock();
    LOG(INFO) << "final " << FinalPositionName[final];
    log_matrix(density_matrix, 2, 2, "density matrix");
    l.unlock();

    gsl_vector_free(t1);
    gsl_vector_free(t2);
    gsl_matrix_free(hamitonian);
    gsl_matrix_complex_free(density_matrix);
    gsl_matrix_complex_free(density_matrix_grad);
    gsl_eigen_nonsymmv_free(wb);

    return final;
}

void Atom::log(const std::string &s) const {
    LOG(INFO) << s << " Ep:" << potential_energy << " Ek:" << kinetic_energy << " x:" << x << " state:" << state
              << " v:" << velocity;
}

