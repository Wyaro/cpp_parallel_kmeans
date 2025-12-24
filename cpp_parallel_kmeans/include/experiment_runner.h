#pragma once

#include "dataset.h"
#include "kmeans_base.h"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

struct TimingResult {
    double T_fit_avg;
    double T_fit_std;
    double T_fit_min;
    double T_assign_total_avg;
    double T_update_total_avg;
    double T_iter_total_avg;
    double throughput_ops_avg;
    double T_transfer_avg;
    double T_transfer_ratio_avg;
    int n_iters_actual_avg;
    int repeats_done;
    int repeats_requested;
    std::vector<std::map<std::string, double>> runs;
};

class ExperimentRunner {
public:
    ExperimentRunner(Dataset& dataset, 
                    std::function<KMeansBase*(int, int, double)> model_factory);
    
    TimingResult run(int repeats = 100, int warmup = 3, double max_seconds = 0.0);

private:
    Dataset& dataset_;
    std::function<KMeansBase*(int, int, double)> model_factory_;
};

