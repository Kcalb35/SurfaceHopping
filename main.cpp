#include "FSSHMath.h"
#include "easylogging++.h"
#include "CLI11.hpp"
#include <array>
#include <mutex>
#include <thread>

INITIALIZE_EASYLOGGINGPP

using namespace std;

void test() {
    cout << "test" << endl;
}

int main(int argc, char **argv) {
    // load log configure
    el::Configurations conf("log.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    // add CLI params parser
    CLI::App app{"Tully FSSH"};
    int model_index = 1;
    double momenta = 20;
    double dt = 1e-5;
    int start_state = 1;
    int cnt = 1;
    bool debug_flag = false;
    int cores = 1;
    app.add_option("-s", start_state, "start state (1,2)");
    app.add_option("-i", model_index, "model index (1,2,3)");
    app.add_option("-m", momenta, "atom momenta (0-30)");
    app.add_option("-t", dt, "simulate interval");
    app.add_option("-c", cnt, "run times");
    app.add_flag("--debug", debug_flag, "whether to log debug info");
    app.add_option("--cores", cores, "multi cores using(>=1)");
    CLI11_PARSE(app, argc, argv);
    // if multi cores than no debug info
    if (cores > 1) debug_flag = false;
    cnt = cnt / cores * cores;

    // log start settings
    LOG(INFO) << "simulate_setting start_state:" << start_state << " model:" << model_index << " momenta:" << momenta
              << " dt:" << dt << " times:" << cnt;
    LOG(INFO) << "runtime_setting debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;
    H_matrix_function h_f[3] = {model_1, model_2, model_3};
    H_matrix_function d_h_f[3] = {model_1_derive, model_2_derive, model_3_derive};


    // start here
    int result[] = {0, 0, 0, 0};
    if (cores == 1) {
        for (int i = 0; i < cnt; ++i) {
            auto final = run_single_trajectory(h_f[model_index - 1], d_h_f[model_index - 1], start_state,
                                               momenta, dt, debug_flag);
            result[final] += 1;
        }
    } else {
        mutex m;
        vector<thread> threads;
        for (int turn = 0; turn < cnt / cores; ++turn) {
            // add threads to list
            for (int j = 0; j < cores; ++j) {
                threads.push_back(move(
                        thread([&h_f, &d_h_f, model_index, start_state, momenta, dt, debug_flag, &result, &m]() {
                            auto final = run_single_trajectory(h_f[model_index - 1], d_h_f[model_index - 1],
                                                               start_state, momenta, dt, debug_flag);
                            m.lock();
                            result[final] += 1;
                            m.unlock();
                        })));
            }
            // wait a batch of threads to finish
            for (thread &t:threads) {
                t.join();
            }
            threads.clear();
        }

    }

    // log settings again and result
    LOG(INFO) << "simulate_setting start_state:" << start_state << " model:" << model_index << " momenta:" << momenta
              << " dt:" << dt << " times:" << cnt;
    LOG(INFO) << "runtime_setting debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;
    LOG(INFO) << "lower trans " << 100.0 * result[0] / cnt << "%";
    LOG(INFO) << "upper trans " << 100.0 * result[1] / cnt << "%";
    LOG(INFO) << "lower reflect " << 100.0 * result[2] / cnt << "%";
}
