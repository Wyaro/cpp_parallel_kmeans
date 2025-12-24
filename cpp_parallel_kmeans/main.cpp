#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "include/dataset.h"
#include "include/kmeans_base.h"
#include "include/kmeans_cuda.h"
#include "include/kmeans_cpu.h"
#include "include/timer.h"
#include "include/experiments_config.h"
#include "include/experiment_suite.h"
#include "include/json_writer.h"

void print_usage(const char* program_name) {
    std::cout << "Использование: " << program_name << " <mode> [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Режимы работы:" << std::endl;
    std::cout << "  single <dataset_path> [version]  - Запуск на одном датасете" << std::endl;
    std::cout << "  experiment <exp_id>              - Запуск эксперимента" << std::endl;
    std::cout << "  all                              - Запуск всех экспериментов" << std::endl;
    std::cout << std::endl;
    std::cout << "Эксперименты:" << std::endl;
    std::cout << "  exp1_baseline_single  - Baseline (N=100000, D=50, K=8)" << std::endl;
    std::cout << "  exp2_scaling_n        - Масштабирование по N" << std::endl;
    std::cout << "  exp3_scaling_d        - Масштабирование по D" << std::endl;
    std::cout << "  exp4_scaling_k        - Масштабирование по K" << std::endl;
    std::cout << "  exp5_gpu_profile     - GPU профилирование" << std::endl;
    std::cout << std::endl;
    std::cout << "Опции:" << std::endl;
    std::cout << "  --gpu-only           - Запускать только GPU реализации" << std::endl;
    std::cout << "  --max-seconds <sec>   - Лимит времени выполнения" << std::endl;
    std::cout << "  --output <file>       - Файл для сохранения результатов (JSON)" << std::endl;
    std::cout << std::endl;
    std::cout << "Версии реализации:" << std::endl;
    std::cout << "  0  - CPU (однопоточная)" << std::endl;
    std::cout << "  -1 - CPU OpenMP (многопоточная)" << std::endl;
    std::cout << "  1  - CUDA V1" << std::endl;
    std::cout << "  2  - CUDA V2" << std::endl;
    std::cout << "  3  - CUDA V3" << std::endl;
    std::cout << "  4  - CUDA V4" << std::endl;
    std::cout << std::endl;
    std::cout << "Примеры:" << std::endl;
    std::cout << "  " << program_name << " single ..\\datasets\\base\\N100000_D50_K8.txt 0" << std::endl;
    std::cout << "  " << program_name << " single ..\\datasets\\base\\N100000_D50_K8.txt 1" << std::endl;
    std::cout << "  " << program_name << " experiment exp1_baseline_single" << std::endl;
    std::cout << "  " << program_name << " all --output results.json" << std::endl;
}

ExperimentId parse_experiment_id(const std::string& str) {
    if (str == "exp1_baseline_single") return ExperimentId::BASELINE_SINGLE;
    if (str == "exp2_scaling_n") return ExperimentId::SCALING_N;
    if (str == "exp3_scaling_d") return ExperimentId::SCALING_D;
    if (str == "exp4_scaling_k") return ExperimentId::SCALING_K;
    if (str == "exp5_gpu_profile") return ExperimentId::GPU_PROFILE;
    throw std::invalid_argument("Неизвестный эксперимент: " + str);
}

int main_single(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Ошибка: требуется путь к датасету" << std::endl;
        return 1;
    }

    std::string dataset_path = argv[2];
    int version = (argc >= 4) ? std::stoi(argv[3]) : 1;

    try {
        std::cout << "Загрузка датасета: " << dataset_path << std::endl;
        Dataset dataset(dataset_path);

        std::cout << "Датасет загружен: N=" << dataset.N() 
                  << ", D=" << dataset.D() 
                  << ", K=" << dataset.K() << std::endl;

        std::unique_ptr<KMeansBase> model;
        if (version == 0) {
            std::cout << "Использование CPU (однопоточная)" << std::endl;
            model.reset(createKMeansCPU(dataset.K(), 100, 1e-6));
        } else if (version == -1) {
            std::cout << "Использование CPU OpenMP (многопоточная)" << std::endl;
            model.reset(createKMeansCPUOpenMP(dataset.K(), 100, 1e-6, 0));
        } else if (version == 1) {
            std::cout << "Использование CUDA V1" << std::endl;
            model.reset(createKMeansCUDAV1(dataset.K(), 100, 1e-6));
        } else if (version == 2) {
            std::cout << "Использование CUDA V2" << std::endl;
            model.reset(createKMeansCUDAV2(dataset.K(), 100, 1e-6));
        } else if (version == 3) {
            std::cout << "Использование CUDA V3" << std::endl;
            model.reset(createKMeansCUDAV3(dataset.K(), 100, 1e-6));
        } else if (version == 4) {
            std::cout << "Использование CUDA V4" << std::endl;
            model.reset(createKMeansCUDAV4(dataset.K(), 100, 1e-6));
        } else {
            std::cerr << "Неизвестная версия: " << version << std::endl;
            std::cerr << "Доступные версии: 0 (CPU), -1 (CPU OpenMP), 1-4 (CUDA V1-V4)" << std::endl;
            return 1;
        }

        std::cout << std::endl << "Запуск K-means..." << std::endl;
        Timer total_timer;
        total_timer.start();

        model->fit(dataset.X().data(), dataset.N(), dataset.D(), 
                   dataset.initial_centroids().data());

        total_timer.stop();

        std::cout << std::endl << "Результаты:" << std::endl;
        std::cout << "  Итераций выполнено: " << model->n_iters_actual() << std::endl;
        std::cout << "  Общее время: " << total_timer.elapsed() << " секунд" << std::endl;
        std::cout << "  T_assign_total: " << model->t_assign_total() << " секунд" << std::endl;
        std::cout << "  T_update_total: " << model->t_update_total() << " секунд" << std::endl;
        std::cout << "  T_iter_total: " << model->t_iter_total() << " секунд" << std::endl;
        std::cout << "  T_h2d: " << model->t_h2d() << " секунд" << std::endl;
        std::cout << "  T_d2h: " << model->t_d2h() << " секунд" << std::endl;
        std::cout << "  T_transfer: " << model->t_transfer() << " секунд" << std::endl;
        std::cout << "  T_assign_avg: " 
                  << model->t_assign_total() / model->n_iters_actual() << " секунд" << std::endl;
        std::cout << "  T_update_avg: " 
                  << model->t_update_total() / model->n_iters_actual() << " секунд" << std::endl;
        std::cout << "  T_iter_avg: " 
                  << model->t_iter_total() / model->n_iters_actual() << " секунд" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main_experiment(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Ошибка: требуется идентификатор эксперимента" << std::endl;
        return 1;
    }

    bool gpu_only = false;
    double max_seconds = 0.0;
    std::string output_file = "kmeans_timing_results.json";

    // Парсинг опций
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu-only") {
            gpu_only = true;
        } else if (arg == "--max-seconds" && i + 1 < argc) {
            max_seconds = std::stod(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    try {
        ExperimentId exp_id = parse_experiment_id(argv[2]);
        
        // Создаем фабрику (не используется для GPU-only режима)
        auto factory = [](int n_clusters, int n_iters, double tol) {
            return createKMeansCUDAV1(n_clusters, n_iters, tol);
        };
        
        ExperimentSuite suite(factory, gpu_only, max_seconds);
        
        std::vector<std::map<std::string, std::string>> results;
        
        switch (exp_id) {
            case ExperimentId::BASELINE_SINGLE:
                results = suite.run_exp1_baseline_single();
                break;
            case ExperimentId::SCALING_N:
                results = suite.run_exp2_scaling_n();
                break;
            case ExperimentId::SCALING_D:
                results = suite.run_exp3_scaling_d();
                break;
            case ExperimentId::SCALING_K:
                results = suite.run_exp4_scaling_k();
                break;
            case ExperimentId::GPU_PROFILE:
                results = suite.run_exp5_gpu_profile();
                break;
        }
        
        if (!results.empty()) {
            JsonWriter::write_results(results, output_file);
            std::cout << std::endl << "Результаты сохранены в: " << output_file << std::endl;
            std::cout << "Всего записей: " << results.size() << std::endl;
        } else {
            std::cout << "Нет результатов для сохранения" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main_all(int argc, char* argv[]) {
    bool gpu_only = false;
    double max_seconds = 0.0;
    std::string output_file = "kmeans_timing_results.json";

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gpu-only") {
            gpu_only = true;
        } else if (arg == "--max-seconds" && i + 1 < argc) {
            max_seconds = std::stod(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    try {
        auto factory = [](int n_clusters, int n_iters, double tol) {
            return createKMeansCUDAV1(n_clusters, n_iters, tol);
        };
        
        ExperimentSuite suite(factory, gpu_only, max_seconds);
        auto results = suite.run_all();
        
        if (!results.empty()) {
            JsonWriter::write_results(results, output_file);
            std::cout << std::endl << "Результаты сохранены в: " << output_file << std::endl;
            std::cout << "Всего записей: " << results.size() << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "single") {
        return main_single(argc, argv);
    } else if (mode == "experiment") {
        return main_experiment(argc, argv);
    } else if (mode == "all") {
        return main_all(argc, argv);
    } else {
        std::cerr << "Неизвестный режим: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}

