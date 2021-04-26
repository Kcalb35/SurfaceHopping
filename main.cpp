#include "FSSHMath.h"
#include "easylogging++.h"
#include "CLI11.hpp"
#include <thread>
#include "ThreadPool.h"
#include <random>

INITIALIZE_EASYLOGGINGPP

using namespace std;


int main(int argc, char **argv) {
    // load log configure
    el::Configurations conf("log.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    // add CLI params parser
    CLI::App app{"Tully FSSH"};
    CLI::App *single = app.add_subcommand("single", "run single point FSSH");
    CLI::App *serial = app.add_subcommand("serial", "run serial FSSH");
    app.require_subcommand(1, 1);

    int model_index = 1;
    double momenta = 20;
    double dt = 1;
    int start_state = 0;
    int cnt = 1;
    bool debug_flag = false;
    int cores = 1;
    double serial_interval = 0;
    double start = 1, end = 30;
    bool norm_flag = false;
    string path("serial.dat");

    // shared options
    app.add_option("-s", start_state, "start state (0,1)");
    app.add_option("-m", model_index, "model index (1,2,3)");
    app.add_option("-t", dt, "simulate interval (default=1)");
    app.add_option("-c", cnt, "run times (default=2000)");
    // multi-threading option
    app.add_option("--cores", cores, "how many threads to use (default=1)");
    app.add_flag("--debug", debug_flag, "whether to log debug info");
    app.add_flag("--norm", norm_flag, "whether generate normal distribution momenta");

    // single options
    single->add_option("-k", momenta, "atom momenta (0-30)");

    // serial options
    serial->add_option("--serial", serial_interval, "serial momenta interval (default=1)");
    serial->add_option("--start", start, "start momenta (default=1)");
    serial->add_option("--end", end, "end momenta (default=30)");
    serial->add_option("--path", path, "path to log serial experiment data");

    CLI11_PARSE(app, argc, argv);
    // if multi cores than no debug info
    if (cores > 1) debug_flag = false;
    ThreadPool pool(cores);
    queue<future<FinalPosition>> result_queue;

    function<void(gsl_matrix *, double)> h_f[3] = {model_1, model_2, model_3};
    function<void(gsl_matrix *, double)> d_h_f[3] = {model_1_derive, model_2_derive, model_3_derive};

    if (app.got_subcommand(single)) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> distribution(momenta, momenta / 10);

        // log start settings
        LOG(INFO) << "simulate_setting start_state:" << start_state << " model:" << model_index << " momenta:"
                  << momenta << " dt:" << dt << " times:" << cnt << " norm:" << (norm_flag ? "yes" : "no");
        LOG(INFO) << "runtime_setting debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;

        // start here
        int result[] = {0, 0, 0, 0};
        for (int i = 0; i < cnt; ++i) {
            double m = momenta;
            if (norm_flag) {
                m = distribution(gen);
            }
            auto res = pool.enqueue(run_single_trajectory, h_f[model_index - 1], d_h_f[model_index - 1], start_state,
                                    m, dt, debug_flag);
            result_queue.push(move(res));
        };
        while (!result_queue.empty()) {
            auto res = result_queue.front().get();
            result_queue.pop();
            result[res] += 1;
        }

        // log settings again and result
        LOG(INFO) << "simulate_setting state:" << start_state << " model:" << model_index << " momenta:"
                  << momenta << " dt:" << dt << " times:" << cnt << " norm:" << (norm_flag ? "yes" : "no");
        LOG(INFO) << "runtime_setting debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;

        LOG(INFO) << "lower trans " << 100.0 * result[0] / cnt << "%";
        LOG(INFO) << "upper trans " << 100.0 * result[1] / cnt << "%";
        LOG(INFO) << "lower reflect " << 100.0 * result[2] / cnt << "%";
        LOG(INFO) << "upper reflect " << 100.0 * result[3] / cnt << "%";
    } else if (app.got_subcommand(serial)) {
        LOG(INFO) << "serial_setting start:" << start << " end:" << end << " interval:" << serial_interval
                  << " norm:" << (norm_flag ? "yes" : "no");
        ofstream fs;
        fs.open(path);
        if (cnt < 2000) cnt = 2000;
        double p = start;
        while (p <= end) {
            random_device rd;
            mt19937 gen(rd());
            normal_distribution<double> distribution(p, p / 20);
            LOG(INFO) << "simulate_setting state:" << start_state << " model:" << model_index << " momenta:"
                      << p << " dt:" << dt << " times:" << cnt;
            int result[4] = {0, 0, 0, 0};
            for (int i = 0; i < cnt; ++i) {
                double m = p;
                if (norm_flag) {
                    m = distribution(gen);
                }
                auto res = pool.enqueue(run_single_trajectory, h_f[model_index - 1], d_h_f[model_index - 1],
                                        start_state, m, dt, debug_flag);
                result_queue.push(move(res));
            }
            while (!result_queue.empty()) {
                auto res = result_queue.front().get();
                result_queue.pop();
                result[res] += 1;
            }
            fs << p << ' ' << 1.0 * result[0] / cnt << ' ' << 1.0 * result[1] / cnt << ' ' << 1.0 * result[2] / cnt
               << ' ' << 1.0 * result[3] / cnt << endl;
            p += serial_interval;
        }
        fs.close();
    }
}