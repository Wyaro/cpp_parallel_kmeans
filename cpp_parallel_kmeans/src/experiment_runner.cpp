#include "../include/experiment_runner.h"
#include "../include/timer.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <map>
#include <string>

ExperimentRunner::ExperimentRunner(Dataset& dataset,
                                  std::function<KMeansBase*(int, int, double)> model_factory)
    : dataset_(dataset), model_factory_(model_factory) {
}

TimingResult ExperimentRunner::run(int repeats, int warmup, double max_seconds) {
    const double* X = dataset_.X().data();
    const double* initial_centroids = dataset_.initial_centroids().data();
    int N = dataset_.N();
    int D = dataset_.D();
    int K = dataset_.K();
    
    // Warmup прогоны
    if (warmup > 0) {
        std::cout << "Warmup x" << warmup << std::endl;
        for (int i = 0; i < warmup; ++i) {
            std::unique_ptr<KMeansBase> model(model_factory_(K, 100, 1e-6));
            model->fit(X, N, D, initial_centroids);
        }
    }
    
    std::vector<double> times;
    std::vector<double> assign_totals;
    std::vector<double> update_totals;
    std::vector<double> iter_totals;
    std::vector<double> throughput_ops;
    std::vector<double> transfer_totals;
    std::vector<double> transfer_ratios;
    std::vector<std::map<std::string, double>> runs;
    
    bool estimated = false;
    double estimated_total_seconds = 0.0;
    
    for (int run_idx = 1; run_idx <= repeats; ++run_idx) {
        std::cout << "Run " << run_idx << "/" << repeats << std::endl;
        
        std::unique_ptr<KMeansBase> model(model_factory_(K, 100, 1e-6));
        
        Timer t_fit;
        t_fit.start();
        model->fit(X, N, D, initial_centroids);
        t_fit.stop();
        
        double t_fit_val = t_fit.elapsed();
        times.push_back(t_fit_val);
        
        double t_assign = model->t_assign_total();
        double t_update = model->t_update_total();
        double t_iter = model->t_iter_total();
        int n_iters_actual = model->n_iters_actual();
        double t_transfer = model->t_transfer();
        
        assign_totals.push_back(t_assign);
        update_totals.push_back(t_update);
        iter_totals.push_back(t_iter);
        
        double throughput = (static_cast<double>(N) * K * D * n_iters_actual) / t_fit_val;
        throughput_ops.push_back(throughput);
        
        transfer_totals.push_back(t_transfer);
        double transfer_ratio = (t_fit_val > 0.0) ? (t_transfer / t_fit_val * 100.0) : 0.0;
        transfer_ratios.push_back(transfer_ratio);
        
        std::map<std::string, double> run_data;
        run_data["run_idx"] = static_cast<double>(run_idx);
        run_data["T_fit"] = t_fit_val;
        run_data["T_assign_total"] = t_assign;
        run_data["T_update_total"] = t_update;
        run_data["T_iter_total"] = t_iter;
        run_data["n_iters_actual"] = static_cast<double>(n_iters_actual);
        run_data["throughput_ops"] = throughput;
        run_data["T_transfer"] = t_transfer;
        run_data["T_transfer_ratio"] = transfer_ratio;
        runs.push_back(run_data);
        
        // Проверка лимита времени
        if (max_seconds > 0.0 && times.size() > 0) {
            double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
            double remaining = (repeats - run_idx) * avg_time;
            double spent = std::accumulate(times.begin(), times.end(), 0.0);
            
            if (spent + remaining > max_seconds) {
                estimated = true;
                estimated_total_seconds = spent + remaining;
                std::cout << "Ранний выход по лимиту времени: spent=" << spent 
                          << "s, remaining_est=" << remaining << "s, limit=" 
                          << max_seconds << "s" << std::endl;
                break;
            }
        }
    }
    
    // Вычисляем статистику
    TimingResult result;
    
    if (times.empty()) {
        return result;
    }
    
    // Среднее
    result.T_fit_avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.T_assign_total_avg = std::accumulate(assign_totals.begin(), assign_totals.end(), 0.0) / assign_totals.size();
    result.T_update_total_avg = std::accumulate(update_totals.begin(), update_totals.end(), 0.0) / update_totals.size();
    result.T_iter_total_avg = std::accumulate(iter_totals.begin(), iter_totals.end(), 0.0) / iter_totals.size();
    result.throughput_ops_avg = std::accumulate(throughput_ops.begin(), throughput_ops.end(), 0.0) / throughput_ops.size();
    result.T_transfer_avg = std::accumulate(transfer_totals.begin(), transfer_totals.end(), 0.0) / transfer_totals.size();
    result.T_transfer_ratio_avg = std::accumulate(transfer_ratios.begin(), transfer_ratios.end(), 0.0) / transfer_ratios.size();
    
    // Стандартное отклонение
    double variance = 0.0;
    for (double t : times) {
        variance += (t - result.T_fit_avg) * (t - result.T_fit_avg);
    }
    result.T_fit_std = std::sqrt(variance / times.size());
    
    // Минимум
    result.T_fit_min = *std::min_element(times.begin(), times.end());
    
    // Среднее количество итераций
    double n_iters_sum = 0.0;
    for (const auto& run : runs) {
        n_iters_sum += run.at("n_iters_actual");
    }
    result.n_iters_actual_avg = static_cast<int>(n_iters_sum / runs.size());
    
    result.repeats_done = static_cast<int>(times.size());
    result.repeats_requested = repeats;
    result.runs = runs;
    
    return result;
}

