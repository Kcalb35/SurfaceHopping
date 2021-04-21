#include "FSSHMath.h"
#include "easylogging++.h"
#include "CLI11.hpp"
#include <mutex>
#include <thread>
#include "ThreadPool.h"

INITIALIZE_EASYLOGGINGPP

using namespace std;


int main(int argc, char **argv) {
    // load log configure
    el::Configurations conf("log.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    // add CLI params parser
    CLI::App app{"Tully FSSH"};
    int model_index = 1;
    double momenta = 20;
    double dt = 1;
    int start_state = 0;
    int cnt = 1;
    bool debug_flag = false;
    int cores = 1;
    double serial_interval = 0;
    string path("serial.dat");
    app.add_option("-s", start_state, "start state (0,1)");
    app.add_option("-i", model_index, "model index (1,2,3)");
    app.add_option("-m", momenta, "atom momenta (0-30)");
    app.add_option("-t", dt, "simulate interval");
    app.add_option("-c", cnt, "run times");
    app.add_flag("--debug", debug_flag, "whether to log debug info");
    app.add_option("--cores", cores, "how many threads to use(default=1)");
    app.add_option("--serial", serial_interval,
                   "enable it to carry out serial experiment by interval momenta, by default multi-threading");
    app.add_option("--path", path, "path to log serial experiment data");
    CLI11_PARSE(app, argc, argv);
    // if multi cores than no debug info
    if (cores > 1) debug_flag = false;
    ThreadPool pool(cores);
    queue<future<FinalPosition>> result_queue;
    mutex m;

    function<void(gsl_matrix *, double)> h_f[3] = {model_1, model_2, model_3};
    function<void(gsl_matrix *, double)> d_h_f[3] = {model_1_derive, model_2_derive, model_3_derive};

    if (serial_interval == 0) {
        // log start settings
        LOG(INFO) << "simulate_setting start_state:" << start_state << " model:" << model_index << " momenta:"
                  << momenta << " dt:" << dt << " times:" << cnt;
        LOG(INFO) << "runtime_setting debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;

        // start here
        int result[] = {0, 0, 0, 0};
        for (int i = 0; i < cnt; ++i) {
            auto res = pool.enqueue(run_single_trajectory, h_f[model_index - 1], d_h_f[model_index - 1], start_state,
                                    momenta, dt, debug_flag);
            result_queue.push(move(res));
        };
        while (!result_queue.empty()) {
            auto res = result_queue.front().get();
            result_queue.pop();
            result[res] += 1;
        }

        // log settings again and result
        LOG(INFO) << "simulate_setting start_state:" << start_state << " model:" << model_index << " momenta:"
                  << momenta << " dt:" << dt << " times:" << cnt;
        LOG(INFO) << "runtime_setting debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;
        LOG(INFO) << "lower trans " << 100.0 * result[0] / cnt << "%";
        LOG(INFO) << "upper trans " << 100.0 * result[1] / cnt << "%";
        LOG(INFO) << "lower reflect " << 100.0 * result[2] / cnt << "%";
    } else {
        ofstream fs;
        fs.open(path);
        cnt = 2000;
        for (double p = 1; p <= 30; p += serial_interval) {
            LOG(INFO) << "simulate_setting start_state:" << start_state << " model:" << model_index << " momenta:"
                      << p << " dt:" << dt << " times:" << cnt;
            int result[4] = {0, 0, 0, 0};
            for (int i = 0; i < cnt; ++i) {
                auto res = pool.enqueue(run_single_trajectory, h_f[model_index - 1], d_h_f[model_index - 1],
                                        start_state, p, dt, debug_flag);
                result_queue.push(move(res));
            }
            while (!result_queue.empty()) {
                auto res = result_queue.front().get();
                result_queue.pop();
                result[res] += 1;
            }
            fs << p << ' ' << 1.0 * result[0] / cnt << ' ' << 1.0 * result[1] / cnt << ' ' << 1.0 * result[2] / cnt
               << ' ' << 1.0 * result[3] / cnt << endl;
        }
        fs.close();
    }
}