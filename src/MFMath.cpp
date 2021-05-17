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

double calculate_group_population(gsl_matrix_complex *density, std::vector<int> &group) {
    double group_population = 0;
    for (const auto &index : group) {
        group_population += GSL_REAL(gsl_matrix_complex_get(density, index, index));
    }
    return group_population;
}


void update_split_trajectory(weighted_trajectory &tra, weighted_trajectory *split_traj, std::vector<int> group,
                             double population) {
    split_traj->e_total = tra.e_total;
    split_traj->turn = tra.turn;
    split_traj->weight = tra.weight * population;
    gsl_matrix_complex_memcpy(split_traj->density, tra.density);
    for (int i = 0; i < 2; ++i) {
        split_traj->e[i] = tra.e[i];
        gsl_vector_memcpy(split_traj->t[i], tra.t[i]);
    }
    reset_wave_functions(group, population, split_traj->density);
    split_traj->potential_energy = avg_energy(split_traj->density, split_traj->e);
    double tmp = 2 * (tra.potential_energy - split_traj->potential_energy) / tra.mass /
                 tra.velocity / tra.velocity;
    if (tmp < -1)
        split_traj->velocity = 0;
    else split_traj->velocity = tra.velocity * sqrt(tmp + 1);
    split_traj->kinetic_energy =
            0.5 * split_traj->velocity * split_traj->velocity * split_traj->mass;
    calculate_momenta(split_traj->velocity, split_traj->potential_energy, split_traj->e,
                      split_traj->p_prev, split_traj->mass);
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

int run_BCMF_w(NumericalModel *model, const double start_momenta, const int start_state, const double dt,
               double result[], const double start_x, const double left, const double right,
               const int max_num_trajectories,
               bool debug) {
    std::vector<weighted_trajectory *> trajectories, splited, deleted;
    trajectories.reserve(max_num_trajectories);

    // pre allocate
    gsl_matrix_complex *tmp_density = gsl_matrix_complex_alloc(2, 2);
    gsl_matrix *hamitonian = gsl_matrix_alloc(2, 2);
    gsl_matrix_complex *hamitonian_z = gsl_matrix_complex_calloc(2, 2);
    gsl_vector *tmp_t[]{gsl_vector_calloc(2), gsl_vector_calloc(2)};
    gsl_matrix *nac = gsl_matrix_calloc(2, 2);
    gsl_matrix_complex *tmp_density_grad[]{
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2),
            gsl_matrix_complex_alloc(2, 2)
    };
    double p[2]{}, p_next[2]{}, e[2]{};
    double p_avg_next;
    std::vector<int> RG, NRG, EAG, EFG;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0, 1.0);

    // work space
    auto wb = gsl_eigen_symmv_alloc(2);
    auto wb_vecs = gsl_matrix_alloc(2, 2);
    auto wb_vals = gsl_vector_alloc(2);
    auto wb_dot_vec = gsl_vector_alloc(2);
    // rk4
    double nac_x, v[4], a[4], tmp_x, tmp_v, tmp_e, tmp_p_all[2]{0, 0}, rk4dt[]{dt / 2, dt / 2, dt};

    // init the initial trajectory
    auto traj = new weighted_trajectory(2000, start_x, 1);
    traj->velocity = start_momenta / traj->mass;
    traj->kinetic_energy = start_momenta * start_momenta / 2 / traj->mass;
    model->hamitonian_cal(hamitonian, traj->x);
    diagonalize(hamitonian, traj->e[0], traj->e[1], traj->t[0], traj->t[1], wb, wb_vals, wb_vecs);
    gsl_matrix_complex_set_zero(traj->density);
    gsl_matrix_complex_set(traj->density, start_state, start_state, gsl_complex{1, 0});
    traj->potential_energy = avg_energy(traj->density, traj->e);
    traj->e_total = traj->potential_energy + traj->kinetic_energy;
    trajectories.push_back(std::move(traj));
    bool finish_all = false;
    while (!finish_all) {
        finish_all = true;
        for (auto &ptr : trajectories) {
            weighted_trajectory &tra = *ptr;
            if (!tra.finish) {
                tmp_x = tra.x;
                tmp_v = tra.velocity;
                gsl_matrix_complex_memcpy(tmp_density, tra.density);
                // RK4
                for (int i = 0; i < 4; ++i) {
                    v[i] = tmp_v;
                    model->hamitonian_cal(hamitonian, tmp_x);
                    diagonalize(hamitonian, e[0], e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);
                    for (int j = 0; j < 2; ++j)
                        phase_correction(tra.t[j], tmp_t[j]);
                    model->d_hamitonian_cal(hamitonian, tmp_x);
                    nac_x = NAC(hamitonian, tmp_t[0], tmp_t[1], e[0], e[1], wb_dot_vec);
                    gsl_matrix_set(nac, 0, 1, nac_x);
                    gsl_matrix_set(nac, 1, 0, -nac_x);
                    tmp_e = avg_energy(tmp_density, e);
                    // set hamitonian using PC
                    calculate_momenta(tmp_v, tmp_e, e, tmp_p_all, tra.mass);
                    set_effective_hamitonian(hamitonian_z, tra.mass * tmp_v, tmp_p_all, tra.mass);
                    density_matrix_grad_cal(tmp_density_grad[i], tmp_density, hamitonian_z, tmp_v, nac);
                    a[i] = avg_acceleration(tmp_density, hamitonian, tmp_t, wb_dot_vec, tra.mass);
                    if (i == 3) break;
                    tmp_x = tra.x + rk4dt[i] * v[i];
                    tmp_v = tra.velocity + rk4dt[i] * a[i];
                    gsl_matrix_complex_memcpy(tmp_density_grad[4], tmp_density_grad[i]);
                    gsl_matrix_complex_scale(tmp_density_grad[4], gsl_complex{rk4dt[i], 0});
                    gsl_matrix_complex_memcpy(tmp_density, tra.density);
                    gsl_matrix_complex_add(tmp_density, tmp_density_grad[4]);
                }
                // finish rk4 add up properties
                tra.x += dt / 6 * (v[0] + 2 * v[1] + 2 * v[2] + v[3]);
                tra.velocity += dt / 6 * (a[0] + 2 * a[1] + 2 * a[2] + a[3]);
                gsl_matrix_complex_scale(tmp_density_grad[0], gsl_complex{dt / 6, 0});
                gsl_matrix_complex_scale(tmp_density_grad[1], gsl_complex{dt / 3, 0});
                gsl_matrix_complex_scale(tmp_density_grad[2], gsl_complex{dt / 3, 0});
                gsl_matrix_complex_scale(tmp_density_grad[3], gsl_complex{dt / 6, 0});
                for (int i = 0; i < 4; ++i) {
                    gsl_matrix_complex_add(tra.density, tmp_density_grad[i]);
                }

                // update energy
                model->hamitonian_cal(hamitonian, tra.x);
                diagonalize(hamitonian, tra.e[0], tra.e[1], tmp_t[0], tmp_t[1], wb, wb_vals, wb_vecs);
                for (int i = 0; i < 2; ++i) {
                    phase_correction(tra.t[i], tmp_t[i]);
                    gsl_vector_memcpy(tra.t[i], tmp_t[i]);
                }
                tra.kinetic_energy = 0.5 * tra.mass * tra.velocity * tra.velocity;
                tra.potential_energy = avg_energy(tra.density, tra.e);
                if (debug && ++tra.turn % int(10 / dt) == 0) {
                    tra.log();
                    log_matrix(tra.density, 2, 2, "density");
                    LOG(INFO) << tra.weight;
                }

                // calculate momenta on other PES
                calculate_momenta(tra.velocity, tra.potential_energy, tra.e, p, tra.mass);
                RG.clear();
                NRG.clear();
                EAG.clear();
                EFG.clear();
                model->d_hamitonian_cal(hamitonian, tra.x);
                bool RG_energy_allowed_flag = true, NRG_energy_allowed_flag = true;
                bool RG_energy_allowed_prev = true, NRG_energy_allowed_prev = true;
                // category PES
                for (int i = 0; i < 2; ++i) {
                    p_next[i] = p[i] - integral(tra.t[i], hamitonian, tra.t[i], wb_dot_vec);
                    if (p_next[i] * p[i] < 0 || p_next[i] * tra.velocity < 0) {
                        RG.push_back(i);
                        if (tra.e[i] <= tra.e_total) {
                            EAG.push_back(i);
                        } else {
                            EFG.push_back(i);
                            RG_energy_allowed_flag = false;
                        }
                        RG_energy_allowed_prev = RG_energy_allowed_prev && (tra.p_prev[i] != 0);
                    } else {
                        NRG.push_back(i);
                        if (tra.e[i] <= tra.e_total) {
                            EAG.push_back(i);
                        } else {
                            EFG.push_back(i);
                            NRG_energy_allowed_flag = false;
                        }
                        NRG_energy_allowed_prev = NRG_energy_allowed_prev && (tra.p_prev[i] != 0);
                    }
                }
                p_avg_next = tra.mass * (tra.velocity +
                                         avg_acceleration(tra.density, hamitonian, tra.t, wb_dot_vec, tra.mass) * dt);
                // check reflection
                if (!RG.empty()) {
                    // split trajectories
                    double p_RG = 0, p_NRG = 0;
                    p_RG = calculate_group_population(tra.density, RG);
                    p_NRG = calculate_group_population(tra.density, NRG);
                    if (trajectories.size() >= max_num_trajectories) {
                        gsl_matrix_complex_memcpy(tmp_density, tra.density);
                        // no split, choose larger population group
                        if (p_NRG > p_RG)
                            // no reflection group
                            wave_packet_correction(tra.density, tmp_density, NRG, NRG, tra.e, tra.e_total, p_NRG, p_RG,
                                                   tra.p_prev);
                        else
                            // reflection group
                            wave_packet_correction(tra.density, tmp_density, RG, NRG, tra.e, tra.e_total, p_RG, p_NRG,
                                                   tra.p_prev);
                    } else {

                        // split
                        if ((RG_energy_allowed_prev || RG_energy_allowed_flag) &&
                            (NRG_energy_allowed_prev || NRG_energy_allowed_flag)) {
                            // split into 2
                            weighted_trajectory *tra_RG, *tra_NRG;
                            tra_RG = new weighted_trajectory(tra.mass, tra.x, tra.weight * p_RG);
                            tra_NRG = new weighted_trajectory(tra.mass, tra.x, tra.weight * p_NRG);
                            // update split trajectory and add them to a list
                            update_split_trajectory(tra, tra_RG, RG, p_RG);
                            update_split_trajectory(tra, tra_NRG, NRG, p_NRG);
                            splited.push_back(tra_RG);
                            splited.push_back(tra_NRG);

                            // collect traget trajectory to delete
                            deleted.push_back(ptr);
                            continue;
                        } else if (RG_energy_allowed_flag || RG_energy_allowed_prev) {
                            wave_packet_correction(tra.density, tmp_density, RG, NRG, tra.e, tra.e_total, p_RG, p_NRG,
                                                   tra.p_prev);
                        } else if (NRG_energy_allowed_flag || NRG_energy_allowed_prev) {
                            wave_packet_correction(tra.density, tmp_density, NRG, RG, tra.e, tra.e_total, p_NRG, p_RG,
                                                   tra.p_prev);
                        }
                    }
                } else if (RG.empty() && p_avg_next * tra.velocity < 0) {
                    // effective wave packet reflection
                    // reset wave function to EAG
                    double p_EAG = calculate_group_population(tra.density, EAG);
                    reset_wave_functions(EAG, p_EAG, tra.density);
                }
                // update energy
                tmp_e = avg_energy(tra.density, e);
                double tmp = 2 * (tra.potential_energy - tmp_e) / tra.mass / tra.velocity / tra.velocity;
                if (tmp < -1)
                    tra.velocity = 0;
                else tra.velocity = tra.velocity * sqrt(tmp + 1);
                tra.potential_energy = tmp_e;
                tra.kinetic_energy = 0.5 * tra.mass * tra.velocity * tra.velocity;
                // record prev momentas
                tra.p_prev[0] = p[0];
                tra.p_prev[1] = p[1];
            }
        }

        // delete trajectories
        for (auto tra:deleted) {
            std::remove(trajectories.begin(), trajectories.end(), tra);
            trajectories.pop_back();
            delete tra;
        }
        deleted.clear();
        // check all finish
        for (auto &tra:trajectories)
            finish_all = finish_all && tra->is_finish(left, right);
        // move splited trajectories
        for (auto &ptr : splited) {
            trajectories.push_back(ptr);
            finish_all = finish_all && ptr->is_finish(left, right);
        }
        splited.clear();
    }
    double population[4];
    for (int i = 0; i < 4; ++i) {
        result[i] = 0;
    }
    for (auto &tra:trajectories) {
        tra->calculate_population(population);
        for (int i = 0; i < 4; ++i) {
            result[i] += population[i];
        }
//        delete tra;
    }
    //fordebug
    double pop = 0;
    for (int i = 0; i < 4; ++i) {
        pop += result[i];
    }

    // free
    gsl_matrix_free(nac);
    gsl_matrix_free(wb_vecs);
    gsl_matrix_free(hamitonian);
    gsl_matrix_complex_free(hamitonian_z);
    gsl_vector_free(wb_vals);
    gsl_vector_free(wb_dot_vec);
    for (auto &i : tmp_t) gsl_vector_free(i);
    for (auto &i : tmp_density_grad) gsl_matrix_complex_free(i);
    gsl_eigen_symmv_free(wb);

    return 0;
}
