#include "../include/experiment_suite.h"
#include "../include/kmeans_cuda.h"
#include "../include/kmeans_cpu.h"
#include "../include/experiments_config.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <string>
#include <fstream>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

ExperimentSuite::ExperimentSuite(std::function<KMeansBase*(int, int, double)> model_factory,
                                 bool gpu_only,
                                 double max_seconds)
    : model_factory_(model_factory), gpu_only_(gpu_only), max_seconds_(max_seconds) {
}

std::vector<std::string> ExperimentSuite::get_gpu_implementations() {
    return {"cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"};
}

std::vector<std::string> ExperimentSuite::get_all_implementations() {
    if (gpu_only_) {
        return get_gpu_implementations();
    }
    // Возвращаем все реализации: CPU однопоточная, CPU OpenMP, затем GPU
    return {"cpp_cpu", "cpp_cpu_openmp", "cpp_cuda_v1", "cpp_cuda_v2", "cpp_cuda_v3", "cpp_cuda_v4"};
}

std::string ExperimentSuite::find_dataset_path(const std::string& relative_path) {
    // Список возможных базовых путей (относительно разных мест запуска)
    // Структура: корень_репозитория/x64/Debug/ (отсюда запускается)
    //            корень_репозитория/cpp_parallel_kmeans/datasets/ (здесь датасеты)
    std::vector<std::string> base_paths = {
        "../cpp_parallel_kmeans/datasets/",     // Из x64/Debug/ (основной путь)
        "../../cpp_parallel_kmeans/datasets/",   // Альтернативный путь
        "cpp_parallel_kmeans/datasets/",        // Из корня репозитория
        "../datasets/",                         // Из cpp_parallel_kmeans/
        "datasets/",                            // Если запускается из cpp_parallel_kmeans/
        "../../datasets/"                       // Старый путь (для совместимости)
    };
    
    for (const std::string& base : base_paths) {
        std::string full_path = base + relative_path;
        std::ifstream test(full_path);
        if (test.good()) {
            test.close();
            return full_path;
        }
    }
    
    // Если не найден, возвращаем исходный путь
    return relative_path;
}

std::function<KMeansBase*(int, int, double)> ExperimentSuite::get_implementation_factory(const std::string& impl_name) {
    if (impl_name == "cpp_cpu") {
        return [](int n_clusters, int n_iters, double tol) {
            return createKMeansCPU(n_clusters, n_iters, tol);
        };
    } else if (impl_name == "cpp_cpu_openmp") {
        return [](int n_clusters, int n_iters, double tol) {
            return createKMeansCPUOpenMP(n_clusters, n_iters, tol, 0);
        };
    } else if (impl_name == "cpp_cuda_v1") {
        return [](int n_clusters, int n_iters, double tol) {
            return createKMeansCUDAV1(n_clusters, n_iters, tol);
        };
    } else if (impl_name == "cpp_cuda_v2") {
        return [](int n_clusters, int n_iters, double tol) {
            return createKMeansCUDAV2(n_clusters, n_iters, tol);
        };
    } else if (impl_name == "cpp_cuda_v3") {
        return [](int n_clusters, int n_iters, double tol) {
            return createKMeansCUDAV3(n_clusters, n_iters, tol);
        };
    } else if (impl_name == "cpp_cuda_v4") {
        return [](int n_clusters, int n_iters, double tol) {
            return createKMeansCUDAV4(n_clusters, n_iters, tol);
        };
    }
    return model_factory_;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::format_result(
    ExperimentId exp_id,
    const std::string& impl_name,
    const Dataset& dataset,
    const TimingResult& timing) {
    
    std::vector<std::map<std::string, std::string>> results;
    
    // Основной результат
    std::map<std::string, std::string> result;
    result["experiment"] = experiment_id_to_string(exp_id);
    result["implementation"] = impl_name;
    result["dataset_N"] = std::to_string(dataset.N());
    result["dataset_D"] = std::to_string(dataset.D());
    result["dataset_K"] = std::to_string(dataset.K());
    
    // Форматируем числа: время в миллисекундах (ms), остальное как есть
    std::ostringstream ss;
    
    // Время в миллисекундах (умножаем на 1000)
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_fit_avg * 1000.0);
    result["T_fit_avg"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_fit_std * 1000.0);
    result["T_fit_std"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_fit_min * 1000.0);
    result["T_fit_min"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_assign_total_avg * 1000.0);
    result["T_assign_total_avg"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_update_total_avg * 1000.0);
    result["T_update_total_avg"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_iter_total_avg * 1000.0);
    result["T_iter_total_avg"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(2) << timing.throughput_ops_avg;
    result["throughput_ops_avg"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(3) << (timing.T_transfer_avg * 1000.0);
    result["T_transfer_avg"] = ss.str();
    
    ss.str("");
    ss << std::fixed << std::setprecision(2) << timing.T_transfer_ratio_avg;
    result["T_transfer_ratio_avg"] = ss.str();
    
    result["n_iters_actual_avg"] = std::to_string(timing.n_iters_actual_avg);
    result["repeats_done"] = std::to_string(timing.repeats_done);
    result["repeats_requested"] = std::to_string(timing.repeats_requested);
    
    results.push_back(result);
    
    return results;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::run_exp1_baseline_single() {
    ExperimentConfig cfg = get_experiment_config(ExperimentId::BASELINE_SINGLE);
    std::vector<std::map<std::string, std::string>> results;
    
    // Ищем датасет с N=100000, D=50, K=8
    std::string dataset_path = find_dataset_path("base/N100000_D50_K8.txt");
    
    try {
        Dataset dataset(dataset_path);
        
        if (dataset.N() == 100000 && dataset.D() == 50 && dataset.K() == 8) {
            int repeats = cfg.params.at("repeats");
            int warmup = cfg.params.at("warmup");
            
            for (const std::string& impl_name : get_all_implementations()) {
                std::cout << "Running " << impl_name << std::endl;
                auto factory = get_implementation_factory(impl_name);
                ExperimentRunner runner(dataset, factory);
                TimingResult timing = runner.run(repeats, warmup, max_seconds_);
                
                auto formatted = format_result(ExperimentId::BASELINE_SINGLE, impl_name, dataset, timing);
                results.insert(results.end(), formatted.begin(), formatted.end());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
    }
    
    return results;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::run_exp2_scaling_n() {
    ExperimentConfig cfg = get_experiment_config(ExperimentId::SCALING_N);
    std::vector<std::map<std::string, std::string>> results;
    
    // Список датасетов для масштабирования по N
    std::vector<std::string> dataset_paths = {
        find_dataset_path("scaling_N/N1000_D50_K8.txt"),
        find_dataset_path("scaling_N/N100000_D50_K8.txt"),
        find_dataset_path("scaling_N/N1000000_D50_K8.txt"),
        find_dataset_path("scaling_N/N5000000_D50_K8.txt")
    };
    
    for (const std::string& dataset_path : dataset_paths) {
        try {
            Dataset dataset(dataset_path);
            int N = dataset.N();
            int repeats = REPEATS_SCALING_N.count(N) > 0 ? REPEATS_SCALING_N.at(N) : 10;
            
            std::cout << "Dataset N=" << N << ", repeats=" << repeats << std::endl;
            
            for (const std::string& impl_name : get_all_implementations()) {
                std::cout << "  Running " << impl_name << std::endl;
                auto factory = get_implementation_factory(impl_name);
                ExperimentRunner runner(dataset, factory);
                TimingResult timing = runner.run(repeats, 3, max_seconds_);
                
                auto formatted = format_result(ExperimentId::SCALING_N, impl_name, dataset, timing);
                results.insert(results.end(), formatted.begin(), formatted.end());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading dataset " << dataset_path << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::run_exp3_scaling_d() {
    ExperimentConfig cfg = get_experiment_config(ExperimentId::SCALING_D);
    std::vector<std::map<std::string, std::string>> results;
    
    std::vector<std::string> dataset_paths = {
        find_dataset_path("scaling_D/N100000_D2_K8.txt"),
        find_dataset_path("scaling_D/N100000_D10_K8.txt"),
        find_dataset_path("scaling_D/N100000_D50_K8.txt"),
        find_dataset_path("scaling_D/N100000_D200_K8.txt")
    };
    
    for (const std::string& dataset_path : dataset_paths) {
        try {
            Dataset dataset(dataset_path);
            int D = dataset.D();
            int repeats = REPEATS_SCALING_D.count(D) > 0 ? REPEATS_SCALING_D.at(D) : 10;
            
            std::cout << "Dataset D=" << D << ", repeats=" << repeats << std::endl;
            
            for (const std::string& impl_name : get_all_implementations()) {
                std::cout << "  Running " << impl_name << std::endl;
                auto factory = get_implementation_factory(impl_name);
                ExperimentRunner runner(dataset, factory);
                TimingResult timing = runner.run(repeats, 3, max_seconds_);
                
                auto formatted = format_result(ExperimentId::SCALING_D, impl_name, dataset, timing);
                results.insert(results.end(), formatted.begin(), formatted.end());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading dataset " << dataset_path << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::run_exp4_scaling_k() {
    ExperimentConfig cfg = get_experiment_config(ExperimentId::SCALING_K);
    std::vector<std::map<std::string, std::string>> results;
    
    std::vector<std::string> dataset_paths = {
        find_dataset_path("scaling_K/N100000_D50_K4.txt"),
        find_dataset_path("scaling_K/N100000_D50_K8.txt"),
        find_dataset_path("scaling_K/N100000_D50_K16.txt"),
        find_dataset_path("scaling_K/N100000_D50_K32.txt")
    };
    
    for (const std::string& dataset_path : dataset_paths) {
        try {
            Dataset dataset(dataset_path);
            int K = dataset.K();
            int repeats = REPEATS_SCALING_K.count(K) > 0 ? REPEATS_SCALING_K.at(K) : 10;
            
            std::cout << "Dataset K=" << K << ", repeats=" << repeats << std::endl;
            
            for (const std::string& impl_name : get_all_implementations()) {
                std::cout << "  Running " << impl_name << std::endl;
                auto factory = get_implementation_factory(impl_name);
                ExperimentRunner runner(dataset, factory);
                TimingResult timing = runner.run(repeats, 3, max_seconds_);
                
                auto formatted = format_result(ExperimentId::SCALING_K, impl_name, dataset, timing);
                results.insert(results.end(), formatted.begin(), formatted.end());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error loading dataset " << dataset_path << ": " << e.what() << std::endl;
        }
    }
    
    return results;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::run_exp5_gpu_profile() {
    ExperimentConfig cfg = get_experiment_config(ExperimentId::GPU_PROFILE);
    std::vector<std::map<std::string, std::string>> results;
    
    std::string dataset_path = find_dataset_path("scaling_N/N1000000_D50_K8.txt");
    int repeats = cfg.params.at("repeats");
    
    try {
        Dataset dataset(dataset_path);
        
        if (dataset.N() == 1000000 && dataset.D() == 50 && dataset.K() == 8) {
            std::cout << "GPU Profile: N=" << dataset.N() << ", repeats=" << repeats << std::endl;
            
            for (const std::string& impl_name : get_all_implementations()) {
                std::cout << "  Running " << impl_name << std::endl;
                auto factory = get_implementation_factory(impl_name);
                ExperimentRunner runner(dataset, factory);
                TimingResult timing = runner.run(repeats, 3, max_seconds_);
                
                auto formatted = format_result(ExperimentId::GPU_PROFILE, impl_name, dataset, timing);
                results.insert(results.end(), formatted.begin(), formatted.end());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
    }
    
    return results;
}

std::vector<std::map<std::string, std::string>> ExperimentSuite::run_all() {
    std::vector<std::map<std::string, std::string>> results;
    
    auto exp1 = run_exp1_baseline_single();
    results.insert(results.end(), exp1.begin(), exp1.end());
    
    auto exp2 = run_exp2_scaling_n();
    results.insert(results.end(), exp2.begin(), exp2.end());
    
    auto exp3 = run_exp3_scaling_d();
    results.insert(results.end(), exp3.begin(), exp3.end());
    
    auto exp4 = run_exp4_scaling_k();
    results.insert(results.end(), exp4.begin(), exp4.end());
    
    auto exp5 = run_exp5_gpu_profile();
    results.insert(results.end(), exp5.begin(), exp5.end());
    
    return results;
}

