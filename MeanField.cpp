#include "MFMath.h"
#include "easylogging++.h"
#include "ModelBase.h"
#include <random>
#include <fstream>
#include <omp.h>
#include "toml.hpp"

INITIALIZE_EASYLOGGINGPP

using namespace std;

struct Config {
    int model;
    int state;
    int cores;
    double dt;
    int count;
    string method;
    bool norm;
    bool debug;
    string save_path;
    int w_cnt;
    double timeout_w;
    double timeout_t;
};


void parse_toml(toml::basic_value<toml::discard_comments, unordered_map, vector> &data, Config &conf);

int main(int argc, char **argv) {
    // load log configure
    el::Configurations conf("log.conf");
    el::Loggers::reconfigureAllLoggers(conf);

    // prepare variables
    SAC sac;
    DAC dac;
    ECR ecr;
    DBG dbg;
    DAG dag;
    DRN drn;
    NumericalModel *models[6]{&sac, &dac, &ecr, &dbg, &dag, &drn};
    map<string, MFMethod> map{
            {"EMF",    EMF},
            {"BCMF_s", BCMF_s},
            {"BCMF_w", BCMF_w}
    };
    random_device rd;
    mt19937 gen(rd());
    vector<double> momenta_all;

    // parse configure file
    if (argc < 2)
        return 1;
    auto data = toml::parse(argv[1]);
    auto &shared = toml::find(data, "shared");
    Config runtime_conf;
    parse_toml(shared, runtime_conf);
    ofstream file(runtime_conf.save_path);
    omp_set_num_threads(runtime_conf.cores);

    if (!data.contains("single") && !data.contains("serial") && !data.contains("multi")) {
        LOG(ERROR) << "please input correct runtime settings";
        return 1;
    }
    if (data.contains("single")) {
        auto &single = toml::find(data, "single");
        double k = toml::find<double>(single, "momenta");
        momenta_all.push_back(k);
    }
    if (data.contains("serial")) {
        auto &serial = toml::find(data, "serial");
        double start = toml::find<double>(serial, "start");
        double end = toml::find<double>(serial, "end");
        double interval = toml::find<double>(serial, "interval");
        double k = start;
        while (k <= end) {
            momenta_all.push_back(k);
            k += interval;
        }
    }
    if (data.contains("multi")) {
        auto &multi = toml::find(data, "multi");
        auto m = toml::get<vector<double>>(toml::find(multi, "momenta"));
        momenta_all.insert(momenta_all.end(), m.begin(), m.end());
    }
    sort(momenta_all.begin(), momenta_all.end());

    LOG(INFO) << runtime_conf.method << " model:" << runtime_conf.model << (runtime_conf.norm ? " norm" : "")
              << " count:" << runtime_conf.count;
    LOG(INFO) << "save:" << runtime_conf.save_path << " cores:" << runtime_conf.cores;
    auto model = models[runtime_conf.model - 1];
    auto method = map[runtime_conf.method];
    for (auto &k: momenta_all) {
        LOG(INFO) << "start " << k;
        double result[4]{0, 0, 0, 0};
        auto tmp = new double[runtime_conf.count][5];
        double l = model->x0 - 3 * model->sigma_x(k);
        double r = -l;
        normal_distribution<double> distribution_p(k, model->sigma_p(k));
        normal_distribution<double> distribution_x(l, model->sigma_x(k));
#pragma omp parallel for
        for (int i = 0; i < runtime_conf.count; ++i) {
            double k1 = runtime_conf.norm ? distribution_p(gen) : k;
            double x1 = runtime_conf.norm ? distribution_x(gen) : l;
            if (method == EMF || method == BCMF_s)
                tmp[i][4] = run_single_MF(model, k1, runtime_conf.state, runtime_conf.dt, method, tmp[i], x1, l, r,
                                          runtime_conf.debug);
            else if (method == BCMF_w) {
                tmp[i][4] = run_BCMF_w(model, k1, runtime_conf.state, runtime_conf.dt, tmp[i], x1, l, r,
                                       runtime_conf.w_cnt, runtime_conf.debug, runtime_conf.timeout_w,
                                       runtime_conf.timeout_t);
            }
        }
        int count = 0;
        for (int i = 0; i < runtime_conf.count; ++i) {
            if (!tmp[i][4]) {
                for (int j = 0; j < 4; ++j) {
                    result[j] += tmp[i][j];
                }
                count++;
            }
        }
        delete[] tmp;
        for (int i = 0; i < 4; ++i)
            result[i] /= count;
        file << k;
        for (int i = 0; i < 4; ++i)
            file << ' ' << result[i];
        file << endl;
        if (count != runtime_conf.count)
            LOG(INFO) << "finish timeout:" << 1.0 - 1.0 * count / runtime_conf.count;
    }
    LOG(INFO) << "finish";
}

void parse_toml(toml::basic_value<toml::discard_comments, unordered_map, vector> &data, Config &conf) {
    conf.model = toml::find<int>(data, "model");
    conf.state = toml::find<int>(data, "state");
    conf.cores = toml::find<int>(data, "cores");
    conf.dt = toml::find<double>(data, "dt");
    conf.count = toml::find<int>(data, "count");
    conf.method = toml::find<string>(data, "method");
    conf.norm = toml::find<bool>(data, "norm");
    conf.debug = toml::find<bool>(data, "debug");
    conf.save_path = toml::find<string>(data, "save_path");
    conf.w_cnt = toml::find<int>(data, "w_count");
    conf.timeout_w = toml::find<double>(data, "timeout_w");
    conf.timeout_t = toml::find<double>(data, "timeout_t");
}