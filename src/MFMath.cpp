#include "MFMath.h"
#include <gsl/gsl_matrix.h>
#include "FSSHMath.h"
#include <random>
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_complex_math.h"
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

double avg_acceleration(gsl_matrix_complex *density, gsl_matrix *d_hamitonian,
                        gsl_vector *t[], gsl_vector *wb, double mass) {
    double result = 0;
    for (int i = 0; i < density->size1; ++i) {
        for (int j = 0; j < density->size1; ++j) {
            result -= GSL_REAL(gsl_matrix_complex_get(density, i, j)) * integral(t[i], d_hamitonian, t[j], wb);
        }
    }
    return result / mass;
}

void set_effective_hamitonian(gsl_matrix_complex *hamitonian, const double p_avg, const double p[], const double mass) {
    gsl_matrix_complex_set_zero(hamitonian);
    for (int i = 0; i < hamitonian->size1; ++i) {
        gsl_matrix_complex_set(hamitonian, i, i, gsl_complex{-p_avg * p[i] / mass, 0});
    }
}

void set_hamitonian_e(gsl_matrix_complex *hamitonian, const double e[]) {
    gsl_matrix_complex_set_zero(hamitonian);
    for (int i = 0; i < hamitonian->size1; ++i) {
        gsl_matrix_complex_set(hamitonian, i, i, gsl_complex{e[i], 0});
    }

}

void calculate_momenta(const double v, const double Ep, const double e[], double p[], const double mass) {
    for (int j = 0; j < 2; ++j) {
        double v_square = v * v + 2 * (Ep - e[j]) / mass;
        if (v_square < 0) p[j] = 0;
        else p[j] = sgn(v) * sqrt(v_square) * mass;
    }
}

void reset_wave_functions(const std::vector<int> &choose, const double population, gsl_matrix_complex *density) {
    for (int i = 0; i < density->size1; ++i) {
        for (int j = 0; j < density->size2; ++j) {
            if (std::find(choose.begin(), choose.end(), i) != choose.end() &&
                std::find(choose.begin(), choose.end(), j) != choose.end()) {
                gsl_matrix_complex_set(density, i, j,
                                       gsl_complex_mul_real(gsl_matrix_complex_get(density, i, j),
                                                            1 / population));
            } else {
                gsl_matrix_complex_set(density, i, j, gsl_complex{0, 0});
            }
        }
    }
}

void wave_packet_correction(gsl_matrix_complex *density, gsl_matrix_complex *tmp, const std::vector<int> &target,
                            const std::vector<int> &other, const double e[], const double e_total,
                            const double target_pop, const double other_pop, const double p_prev[]) {
    gsl_matrix_complex_memcpy(tmp, density);
    reset_wave_functions(target, target_pop, tmp);
    double tmp_e = avg_energy(tmp, e);
    // energy violate
    if (tmp_e > e_total) {
        bool energy_conserve = true;
        for (const auto &index : target) if (p_prev[index] == 0) energy_conserve = false;
        // energy conserve in the last step
        if (energy_conserve) {
            gsl_matrix_complex_memcpy(density, tmp);
        } else {
            reset_wave_functions(other, other_pop, density);
        }
    } else {
        gsl_matrix_complex_memcpy(density, tmp);
    }
}

int run_single_MF(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
                  const MFMethod method, double result[], const double start_x, const double left, const double right,
                  bool debug) {
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
    double p[2]{0, 0}, p_next[2]{0, 0}, p_prev[2]{0, 0};
    double p_avg_next;
    std::vector<int> RG{}, NRG{}, energy_allowed{};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0, 1.0);
    int log_cnt = 0;

    // workspace
    auto wb = gsl_eigen_symmv_alloc(2);
    auto wb_vecs = gsl_matrix_alloc(2, 2);
    auto wb_vals = gsl_vector_alloc(2);
    auto wb_dot_vec = gsl_vector_alloc(2);

    // init an atom
    Atom atom(2000);
    atom.velocity = start_momenta / atom.mass;
    atom.kinetic_energy = start_momenta * start_momenta / 2 / atom.mass;
    atom.x = start_x;
    gsl_matrix_complex_set(density, start_state, start_state, gsl_complex{1, 0});

    model->hamitonian_cal(hamitonian, atom.x);
    diagonalize(hamitonian, e[0], e[1], t[0], t[1], wb, wb_vals, wb_vecs);
    atom.potential_energy = avg_energy(density, e);
    double e_total = atom.kinetic_energy + atom.potential_energy;

    double nac_x, v[4], a[4], tmp_x, tmp_v, tmp_e, tmp_p_all[2]{0, 0}, rk4dt[]{dt / 2, dt / 2, dt};
    while ((atom.velocity >= 0 && atom.x <= right) || (atom.velocity <= 0 && atom.x >= left)) {
        tmp_x = atom.x;
        tmp_v = atom.velocity;
        gsl_matrix_complex_memcpy(tmp_density, density);

        // RK4
        for (int i = 0; i < 4; ++i) {
            v[i] = tmp_v;
            model->hamitonian_cal(hamitonian, tmp_x);
            diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);
            for (int j = 0; j < 2; ++j) {
                phase_correction(t[j], tmp_t[j]);
            }
            tmp_e = avg_energy(tmp_density, e);
            model->d_hamitonian_cal(hamitonian, tmp_x);
            nac_x = NAC(hamitonian, tmp_t[0], tmp_t[1], e[0], e[1], wb_dot_vec);
            gsl_matrix_set(nac, 0, 1, nac_x);
            gsl_matrix_set(nac, 1, 0, -nac_x);

            // set hamitonian
            switch (method) {
                case EMF:
                    set_hamitonian_e(hamitonian_z, e);
                    break;
                case BCMF_s:
                case BCMF_w:
                    calculate_momenta(tmp_v, tmp_e, e, tmp_p_all, atom.mass);
                    set_effective_hamitonian(hamitonian_z, atom.mass * tmp_v, tmp_p_all, atom.mass);
                    break;
            }
            density_matrix_grad_cal(tmp_density_grad[i], tmp_density, hamitonian_z, tmp_v, nac);
            a[i] = avg_acceleration(tmp_density, hamitonian, tmp_t, wb_dot_vec, atom.mass);
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
        for (int j = 0; j < 2; ++j) {
            phase_correction(t[j], tmp_t[j]);
            gsl_vector_memcpy(t[j], tmp_t[j]);
        }


        atom.kinetic_energy = 0.5 * atom.velocity * atom.velocity * atom.mass;
        atom.potential_energy = avg_energy(density, e);

        if (method == BCMF_s) {
            // calculate momenta on other PES
            calculate_momenta(atom.velocity, atom.potential_energy, e, p, atom.mass);
            // calculate momenta of next dt
            RG.clear();
            NRG.clear();
            energy_allowed.clear();
            model->d_hamitonian_cal(hamitonian, atom.x);
            for (int i = 0; i < 2; ++i) { ;
                p_next[i] = p[i] - integral(t[i], hamitonian, t[i], wb_dot_vec) * dt;
                if (p_next[i] * p[i] < 0 || p_next[i] * atom.velocity < 0) {
                    RG.push_back(i);
                } else {
                    NRG.push_back(i);
                }
            }
            p_avg_next =
                    atom.mass * (atom.velocity + avg_acceleration(density, hamitonian, t, wb_dot_vec, atom.mass) * dt);
            // todo 当能量不守衡，真实波包的动量设置为0，怎么判断反射和透射
            if (!RG.empty()) {
                // actual wave packet reflection
                gsl_matrix_complex_memcpy(tmp_density, density);
                double p_RG = 0, p_NRG = 0;
                for (int &i:RG)
                    p_RG += GSL_REAL(gsl_matrix_complex_get(density, i, i));
                for (int &i:NRG)
                    p_NRG += GSL_REAL(gsl_matrix_complex_get(density, i, i));
                if (distrib(gen) < p_RG)
                    // choose reflection group
                    wave_packet_correction(density, tmp_density, RG, NRG, e, e_total, p_RG, p_NRG, p_prev);
                else
                    // choose transmission group
                    wave_packet_correction(density, tmp_density, NRG, RG, e, e_total, p_NRG, p_RG, p_prev);
            } else if (RG.empty() && p_avg_next * atom.velocity < 0) {
                // effective wave packet reflection
                double population = 0;
                for (int i = 0; i < 2; ++i) {
                    if (e[i] <= e_total) {
                        energy_allowed.push_back(i);
                        population += GSL_REAL(gsl_matrix_complex_get(density, i, i));
                    }
                }
                reset_wave_functions(energy_allowed, population, density);
            }
            tmp_e = avg_energy(density, e);
            double tmp = 2 * (atom.potential_energy - tmp_e) / atom.mass / atom.velocity / atom.velocity;
            if (tmp < -1)
                atom.velocity = 0;
            else atom.velocity = atom.velocity * sqrt(tmp + 1);
            atom.potential_energy = tmp_e;
            atom.kinetic_energy = 0.5 * atom.mass * atom.velocity * atom.velocity;
            //record prev momenta
            p_prev[0] = p[0];
            p_prev[1] = p[1];
        }

        log_cnt++;
        if (debug) {
            if (log_cnt % int(10 / dt) == 0) {
                atom.log("move");
                LOG(INFO) << GSL_REAL(gsl_matrix_complex_get(density, 0, 0)) +
                             GSL_REAL(gsl_matrix_complex_get(density, 1, 1)) << " acc:" << a[3];
                LOG(INFO) << p[0] << ' ' << p[1];
                log_matrix(density, 2, 2, "density");
            }
        }
        if (log_cnt * dt > 1e7) {
            // timeout
            LOG(INFO) << "timeout k:" << start_momenta;
            return 1;
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
    return 0;
}

void run_BCMF_w(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
                double result[], const double start_x, const double left, const double right, const int count) {
    std::vector<weighted_trajectory> trajectories(count);
    LOG(INFO) << trajectories.size();
}
