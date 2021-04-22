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


void diagonalize(gsl_matrix *hamitonian, double &e1, double &e2, gsl_vector *s1, gsl_vector *s2,
                 gsl_eigen_symmv_workspace *wb) {
    auto *e_value = gsl_vector_alloc(2);
    auto *e_vector = gsl_matrix_alloc(2, 2);
    gsl_eigen_symmv(hamitonian, e_value, e_vector, wb);
    gsl_eigen_symmv_sort(e_value, e_vector, GSL_EIGEN_SORT_VAL_ASC);

    e1 = gsl_vector_get(e_value, 0);
    e2 = gsl_vector_get(e_value, 1);
    gsl_matrix_get_col(s1, e_vector, 0);
    gsl_matrix_get_col(s2, e_vector, 1);

    gsl_vector_free(e_value);
    gsl_matrix_free(e_vector);
}

double integral(gsl_vector *left, gsl_matrix *m, gsl_vector *right) {
    auto tmp = gsl_matrix_alloc(right->size, 1);
    auto result_matrix = gsl_matrix_alloc(1, 1);
    auto lm = gsl_matrix_view_vector(left, 1, left->size);
    auto rm = gsl_matrix_view_vector(right, right->size, 1);

    gsl_extra_multiply(m, &rm.matrix, tmp);
    gsl_extra_multiply(&lm.matrix, tmp, result_matrix);

    double result = gsl_matrix_get(result_matrix, 0, 0);

    gsl_matrix_free(tmp);
    gsl_matrix_free(result_matrix);
    return result;
}


void calculate_density_matrix(gsl_matrix_complex *density_matrix, gsl_vector_complex *expand) {

    auto v1 = gsl_matrix_complex_view_vector(expand, 2, 1);
    auto v2 = gsl_matrix_complex_view_vector(expand, 1, 2);
    gsl_extra_multiply(&v1.matrix, &v2.matrix, density_matrix);
}


double
NAC(const std::function<void(gsl_matrix *, double)> &f, double x, gsl_vector *s1, gsl_vector *s2, const double e1,
    const double e2) {
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
    int flag = (x > 0 ? 1 : -1);
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
run_single_trajectory(const std::function<void(gsl_matrix *, double)> &h_f,
                      const std::function<void(gsl_matrix *, double)> &d_h_f, int start_state, double start_momenta,
                      double dt, bool debug) {
    // pre-allocate space
    auto wb = gsl_eigen_symmv_alloc(2);
    auto nac = gsl_matrix_alloc(2, 2);
    gsl_matrix_set_all(nac, 0);
    auto hamitonian = gsl_matrix_alloc(2, 2);
    gsl_vector *t[] = {gsl_vector_alloc(2), gsl_vector_alloc(2)};
    auto density_matrix = gsl_matrix_complex_alloc(2, 2);
    auto density_matrix_grad = gsl_matrix_complex_alloc(2, 2);
    // space for calculate force
    auto tmp_hamitonian = gsl_matrix_alloc(2, 2);
    gsl_vector *tmp_t[] = {gsl_vector_alloc(2), gsl_vector_alloc(2)};

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
    diagonalize(hamitonian, e[0], e[1], t[0], t[1], wb);

    atom.potential_energy = e[atom.state];
    atom.velocity = start_momenta / atom.mass;
    atom.kinetic_energy = atom.mass * atom.velocity * atom.velocity / 2;

    d_h_f(tmp_hamitonian, atom.x);
    double acceleration = -integral(t[atom.state], tmp_hamitonian, t[atom.state]) / atom.mass;

    auto expand = gsl_vector_complex_alloc(2);
    gsl_vector_complex_set(expand, start_state, gsl_complex{1, 0});
    gsl_vector_complex_set(expand, 1 - start_state, gsl_complex{0, 0});
    calculate_density_matrix(density_matrix, expand);
    if (debug) atom.log("start");

    // start dynamic evolve
    while (atom.x <= 10 && atom.x >= -10) {
        // move the atom using verlet method
        atom.velocity += 0.5 * dt * acceleration;
        atom.x += dt * atom.velocity;
        // update potential energy
        h_f(hamitonian, atom.x);
        diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb);

        atom.potential_energy = e[atom.state];
        // calculate force
        d_h_f(tmp_hamitonian, atom.x);
        acceleration = -integral(tmp_t[atom.state], tmp_hamitonian, tmp_t[atom.state]) / atom.mass;
        atom.velocity += 0.5 * dt * acceleration;
        atom.kinetic_energy = 0.5 * atom.mass * atom.velocity * atom.velocity;

        // wave function phase correction
        for (int i = 0; i < 2; ++i) {
            if (gsl_vector_get(tmp_t[i], 0) * gsl_vector_get(t[i], 0) < 0 ||
                gsl_vector_get(tmp_t[i], 1) * gsl_vector_get(t[i], 1) < 0)
                gsl_vector_scale(tmp_t[i], -1);
            gsl_vector_memcpy(t[i], tmp_t[i]);
        }


        // calculate nac
        double nac_x = NAC(d_h_f, atom.x, t[0], t[1], e[0], e[1]);
        gsl_matrix_set(nac, 0, 1, nac_x);
        gsl_matrix_set(nac, 1, 0, -nac_x);

        // calculate diagonalize hamitonian
        gsl_matrix_set_all(tmp_hamitonian, 0);
        gsl_matrix_set(tmp_hamitonian, 0, 0, e[0]);
        gsl_matrix_set(tmp_hamitonian, 1, 1, e[1]);

        // update density matrix
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                gsl_complex grad_complex = gsl_complex{0, 0};
                for (int k = 0; k < 2; ++k) {
                    grad_complex = gsl_complex_add(grad_complex,
                                                   gsl_complex_mul(gsl_matrix_complex_get(density_matrix, k, j),
                                                                   gsl_complex{gsl_matrix_get(tmp_hamitonian, i, k),
                                                                               -atom.velocity *
                                                                               gsl_matrix_get(nac, i, k)}));
                    grad_complex = gsl_complex_sub(grad_complex,
                                                   gsl_complex_mul(gsl_matrix_complex_get(density_matrix, i, k),
                                                                   gsl_complex{gsl_matrix_get(tmp_hamitonian, k, j),
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
        int k = atom.state;
        gsl_complex tmp_complex;
        tmp_complex = gsl_matrix_complex_get(density_matrix, 1 - k, k);
        tmp_complex = gsl_complex_conjugate(tmp_complex);
        double b = -2 * GSL_REAL(gsl_complex_mul_real(tmp_complex, atom.velocity * gsl_matrix_get(nac, 1 - k, k)));
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

        if (debug)
            if (++log_cnt % int(10 / dt) == 0) {
                log_cnt = 0;
                atom.log("move");
                log_matrix(density_matrix, 2, 2, "density_matrix");
//                LOG(INFO) << zeta << '/' << prob;
            }
    }

    // judge final state
    if (debug) atom.log("end");
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
    if (debug) {
        LOG(INFO) << "final " << FinalPositionName[final];
//        log_matrix(density_matrix, 2, 2, "density matrix");
    }

    gsl_vector_free(t[0]);
    gsl_vector_free(t[1]);
    gsl_vector_free(tmp_t[0]);
    gsl_vector_free(tmp_t[1]);
    gsl_matrix_free(hamitonian);
    gsl_matrix_free(tmp_hamitonian);
    gsl_matrix_complex_free(density_matrix);
    gsl_matrix_complex_free(density_matrix_grad);
    gsl_eigen_symmv_free(wb);
    return final;
}

void Atom::log(const std::string &s) const {
    LOG(INFO) << s << " Ep:" << potential_energy << " Ek:" << kinetic_energy << " E:"
              << potential_energy + kinetic_energy << " x:" << x << " state:" << state << " v:" << velocity;
}

