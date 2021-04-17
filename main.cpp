#include "FSSHMath.h"
#include "easylogging++.h"
#include "CLI11.hpp"

INITIALIZE_EASYLOGGINGPP

using namespace std;

int main(int argc, char **argv) {
    el::Configurations conf("log.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    CLI::App app{"Tully FSSH"};

    int model_index = 1;
    double momenta = 20;
    double dt = 1e-5;
    int start_state = 1;
    int cnt = 1;
    bool debug_flag = false;
    app.add_option("-s", start_state, "start state (1,2)");
    app.add_option("-i", model_index, "model index (1,2,3)");
    app.add_option("-m", momenta, "atom momenta (0-30)");
    app.add_option("-t", dt, "simulate interval");
    app.add_option("-c", cnt, "run times");
    app.add_flag("--debug", debug_flag, "whether to log debug info");
    CLI11_PARSE(app, argc, argv);

    LOG(INFO) << "setting start_state:" << start_state << " model:" << model_index << " momenta:" << momenta << " dt:"
              << dt << " times:" << cnt << " debug:" << (debug_flag ? "yes" : "no");

    H_matrix_function h_f[3] = {model_1, model_2, model_3};
    H_matrix_function d_h_f[3] = {model_1_derive, model_2_derive, model_3_derive};

    int result[] = {0, 0, 0, 0};
    for (int i = 0; i < cnt; ++i) {
        auto final = run_single_trajectory(h_f[model_index - 1], d_h_f[model_index - 1], start_state, momenta, dt,
                                           debug_flag);
        result[final] += 1;
    }
    LOG(INFO) << "lower trans" << result[0] / cnt;
    LOG(INFO) << "upper trans" << result[1] / cnt;
    LOG(INFO) << "lower reflect" << result[2] / cnt;
}
