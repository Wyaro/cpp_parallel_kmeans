#pragma once

#include "dataset.h"
#include "experiments_config.h"
#include "experiment_runner.h"
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

class ExperimentSuite {
public:
    ExperimentSuite(std::function<KMeansBase*(int, int, double)> model_factory,
                   bool gpu_only = false,
                   double max_seconds = 0.0);
    
    // Запуск всех экспериментов
    std::vector<std::map<std::string, std::string>> run_all();
    
    // Эксперимент 1: Baseline single
    std::vector<std::map<std::string, std::string>> run_exp1_baseline_single();
    
    // Эксперимент 2: Scaling N
    std::vector<std::map<std::string, std::string>> run_exp2_scaling_n();
    
    // Эксперимент 3: Scaling D
    std::vector<std::map<std::string, std::string>> run_exp3_scaling_d();
    
    // Эксперимент 4: Scaling K
    std::vector<std::map<std::string, std::string>> run_exp4_scaling_k();
    
    // Эксперимент 5: GPU Profile
    std::vector<std::map<std::string, std::string>> run_exp5_gpu_profile();

private:
    std::function<KMeansBase*(int, int, double)> model_factory_;
    bool gpu_only_;
    double max_seconds_;
    
    std::vector<std::string> get_gpu_implementations();
    std::vector<std::string> get_all_implementations();
    std::function<KMeansBase*(int, int, double)> get_implementation_factory(const std::string& impl_name);
    std::string find_dataset_path(const std::string& relative_path);
    std::vector<std::map<std::string, std::string>> format_result(
        ExperimentId exp_id,
        const std::string& impl_name,
        const Dataset& dataset,
        const TimingResult& timing);
};

