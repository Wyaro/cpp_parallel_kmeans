#include "../include/experiments_config.h"
#include <map>

const std::map<int, int> REPEATS_SCALING_N = {
    {1000, 50},
    {100000, 20},
    {1000000, 10},
    {5000000, 5}
};

const std::map<int, int> REPEATS_SCALING_D = {
    {2, 20},
    {10, 20},
    {50, 10},
    {100, 10},
    {200, 10}
};

const std::map<int, int> REPEATS_SCALING_K = {
    {4, 20},
    {8, 20},
    {16, 10},
    {32, 10}
};

ExperimentConfig get_experiment_config(ExperimentId id) {
    static std::map<ExperimentId, ExperimentConfig> configs = {
        {ExperimentId::BASELINE_SINGLE, {
            ExperimentId::BASELINE_SINGLE,
            "Эксперимент 1: baseline однопоточных реализаций N=100000,D=50,K=8",
            {"cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"},
            {{"N", 100000}, {"D", 50}, {"K", 8}, {"repeats", 30}, {"warmup", 3}}
        }},
        {ExperimentId::SCALING_N, {
            ExperimentId::SCALING_N,
            "Эксперимент 2: масштабирование по N",
            {"cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"},
            {}
        }},
        {ExperimentId::SCALING_D, {
            ExperimentId::SCALING_D,
            "Эксперимент 3: масштабирование по D",
            {"cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"},
            {}
        }},
        {ExperimentId::SCALING_K, {
            ExperimentId::SCALING_K,
            "Эксперимент 4: масштабирование по K",
            {"cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"},
            {}
        }},
        {ExperimentId::GPU_PROFILE, {
            ExperimentId::GPU_PROFILE,
            "Эксперимент 5: GPU профилирование",
            {"cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"},
            {{"N", 1000000}, {"D", 50}, {"K", 8}, {"repeats", 10}}
        }}
    };
    
    return configs[id];
}

std::string experiment_id_to_string(ExperimentId id) {
    switch (id) {
        case ExperimentId::BASELINE_SINGLE: return "exp1_baseline_single";
        case ExperimentId::SCALING_N: return "exp2_scaling_n";
        case ExperimentId::SCALING_D: return "exp3_scaling_d";
        case ExperimentId::SCALING_K: return "exp4_scaling_k";
        case ExperimentId::GPU_PROFILE: return "exp5_gpu_profile";
        default: return "unknown";
    }
}

