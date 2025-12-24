#pragma once

#include <string>
#include <map>
#include <vector>

enum class ExperimentId {
    BASELINE_SINGLE,
    SCALING_N,
    SCALING_D,
    SCALING_K,
    GPU_PROFILE
};

// Таблицы соответствия N/D/K → числа повторов
extern const std::map<int, int> REPEATS_SCALING_N;
extern const std::map<int, int> REPEATS_SCALING_D;
extern const std::map<int, int> REPEATS_SCALING_K;

struct ExperimentConfig {
    ExperimentId id;
    std::string description;
    std::vector<std::string> implementations;
    std::map<std::string, int> params;
};

// Получить конфигурацию эксперимента
ExperimentConfig get_experiment_config(ExperimentId id);

// Преобразование ExperimentId в строку
std::string experiment_id_to_string(ExperimentId id);

