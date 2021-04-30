#include "FSSHMath.h"
#include "easylogging++.h"
#include "CLI11.hpp"
#include "ThreadPool.h"
#include "ModelBase.h"
#include <thread>
#include <random>

INITIALIZE_EASYLOGGINGPP


using namespace std;

string format_time(long sec) {
    if (sec < 60) {
        return to_string(sec) + "s";
    } else if (sec < 3600) {
        return to_string(sec / 60.0) + "min";
    } else {
        return to_string(sec / 3600.0) + "h";
    }
}

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
    bool ana_flag = false;
    string path("serial.dat");

    // shared options
    app.add_option("-s", start_state, "start state (0,1)");
    app.add_option("-m", model_index, "model index (1,2,3)");
    app.add_option("-t", dt, "simulate interval (default=1)");
    app.add_option("-c", cnt, "run times (default=2000)");
    app.add_flag("--ana", ana_flag, "using analytic solution");
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
    model_type type = (ana_flag ? model_type::analytic : model_type::numerical);

    // prepare models
    SAC sac;
    DAC dac;
    ECR ecr;
    NumericalModel *num_models[3] = {&sac, &dac, &ecr};
    AnalyticModel *ana_models[3] = {nullptr, nullptr, &ecr};

    // single momenta experiment
    if (app.got_subcommand(single)) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> distribution(momenta, momenta / 20);

        // log start settings
        LOG(INFO) << "simulate state:" << start_state << " model:" << model_index << " momenta:"
                  << momenta << " dt:" << dt << " times:" << cnt << (norm_flag ? " norm" : "")
                  << (ana_flag ? " analytic" : " numerical");
        LOG(INFO) << "runtime debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;


        // start here
        const auto start_time = chrono::steady_clock::now();
        int result[] = {0, 0, 0, 0};
        for (int i = 0; i < cnt; ++i) {
            double m = momenta;
            if (norm_flag) {
                m = distribution(gen);
            }
            auto res = pool.enqueue(run_single_trajectory, num_models[model_index - 1], ana_models[model_index - 1],
                                    start_state, m, dt, debug_flag, type);
            result_queue.push(move(res));
        };
        while (!result_queue.empty()) {
            auto res = result_queue.front().get();
            result_queue.pop();
            result[res] += 1;
        }

        // if debug then log settings again and result
        if (debug_flag) {
            LOG(INFO) << "simulate state:" << start_state << " model:" << model_index << " momenta:"
                      << momenta << " dt:" << dt << " times:" << cnt << (norm_flag ? " norm" : "")
                      << (ana_flag ? " analytic" : " numerical");
            LOG(INFO) << "runtime debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;
        }

        LOG(INFO) << "lower trans " << 100.0 * result[0] / cnt << "%";
        LOG(INFO) << "upper trans " << 100.0 * result[1] / cnt << "%";
        LOG(INFO) << "lower reflect " << 100.0 * result[2] / cnt << "%";
        LOG(INFO) << "upper reflect " << 100.0 * result[3] / cnt << "%";
        long seconds =
                chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count() * cores;
        LOG(INFO) << "Total time:" << format_time(seconds);
    } else if (app.got_subcommand(serial)) {
        LOG(INFO) << "serial start:" << start << " end:" << end << " interval:" << serial_interval
                  << (norm_flag ? " norm" : "") << (ana_flag ? " analytic" : " numerical");
        LOG(INFO) << "runtime debug:" << (debug_flag ? "yes" : "no") << " cores:" << cores;
        ofstream fs;
        fs.open(path);
        if (cnt < 2000) cnt = 2000;
        const auto start_time = chrono::steady_clock::now();
        double p = start;
        while (p <= end) {
            random_device rd;
            mt19937 gen(rd());
            normal_distribution<double> distribution(p, p / 20);
            LOG(INFO) << "simulate state:" << start_state << " model:" << model_index << " momenta:"
                      << p << " dt:" << dt << " times:" << cnt;
            int result[4] = {0, 0, 0, 0};
            for (int i = 0; i < cnt; ++i) {
                double m = p;
                if (norm_flag) {
                    m = distribution(gen);
                }
                auto res = pool.enqueue(run_single_trajectory, num_models[model_index - 1], ana_models[model_index - 1],
                                        start_state, m, dt, debug_flag, type);
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
        long seconds =
                chrono::duration_cast<chrono::seconds>(chrono::steady_clock::now() - start_time).count() * cores;
        LOG(INFO) << "Total time:" << format_time(seconds);
        fs.close();
    }
}