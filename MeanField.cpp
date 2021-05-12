#include "MF.h"
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
    NumericalModel *models[3]{&sac, &dac, &ecr};
    map<string, MFMethod> map{
            {"EMF",    EMF},
            {"BCMF_s", BCMF_s},
            {"BCMF_w", BCMF_w}
    };
    random_device rd;
    mt19937 gen(rd());
    double l = -10, r = 10;
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
    if (runtime_conf.model == 3) {
        l = -20;
    }

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

    for (auto &k: momenta_all) {
        LOG(INFO) << "start " << k;
        double result[4]{0, 0, 0, 0};
        auto tmp = new double[runtime_conf.count][4];
        normal_distribution<double> distribution(k, k / 20);
#pragma omp parallel for
        for (int i = 0; i < runtime_conf.count; ++i) {
            double k1 = runtime_conf.norm ? distribution(gen) : k;
            run_single_MF(models[runtime_conf.model - 1], k1, runtime_conf.state, runtime_conf.dt,
                          map[runtime_conf.method], tmp[i], r, l, runtime_conf.debug);
        }
        for (int i = 0; i < runtime_conf.count; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[j] += tmp[i][j];
            }
        }
        delete[] tmp;
        for (int i = 0; i < 4; ++i)
            result[i] /= runtime_conf.count;
        file << k;
        for (int i = 0; i < 4; ++i)
            file << ' ' << result[i];
        file << endl;
    }
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
}