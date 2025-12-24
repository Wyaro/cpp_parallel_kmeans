#include "../include/kmeans_base.h"
#include "../include/timer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

KMeansBase::KMeansBase(int n_clusters, int n_iters, double tol)
    : K_(n_clusters), n_iters_(n_iters), tol_(tol),
      t_assign_total_(0.0), t_update_total_(0.0), t_iter_total_(0.0),
      n_iters_actual_(0) {
}

void KMeansBase::fit(const double* X, int N, int D, const double* initial_centroids) {
    // Вызываем prepare_data для подготовки данных (например, перенос на GPU)
    prepare_data(X, N, D, initial_centroids);
    
    // Инициализация центроидов
    centroids_.resize(K_ * D);
    std::copy(initial_centroids, initial_centroids + K_ * D, centroids_.begin());

    labels_.resize(N);

    // Сбрасываем накопленные тайминги для нового запуска
    t_assign_total_ = 0.0;
    t_update_total_ = 0.0;
    t_iter_total_ = 0.0;
    n_iters_actual_ = 0;

    std::vector<double> old_centroids(K_ * D);
    std::vector<double> new_centroids(K_ * D);

    for (int i = 0; i < n_iters_; ++i) {
        // Сохраняем старые центроиды для проверки сходимости
        std::copy(centroids_.begin(), centroids_.end(), old_centroids.begin());

        // Шаг назначения
        Timer t_assign;
        t_assign.start();
        assign_clusters(X, N, D, centroids_.data(), labels_.data());
        t_assign.stop();
        double t_assign_elapsed = t_assign.elapsed();

        // Шаг обновления
        Timer t_update;
        t_update.start();
        update_centroids(X, N, D, labels_.data(), new_centroids.data());
        t_update.stop();
        double t_update_elapsed = t_update.elapsed();

        double t_iter_elapsed = t_assign_elapsed + t_update_elapsed;

        t_assign_total_ += t_assign_elapsed;
        t_update_total_ += t_update_elapsed;
        t_iter_total_ += t_iter_elapsed;
        n_iters_actual_ = i + 1;

        // Проверка сходимости: максимальное изменение центроидов
        double max_change = 0.0;
        for (int k = 0; k < K_ * D; ++k) {
            double change = std::abs(new_centroids[k] - old_centroids[k]);
            if (change > max_change) {
                max_change = change;
            }
        }

        bool converged = max_change < tol_;

        if (i == 0 || (i + 1) % 10 == 0 || converged) {
            std::cout << "  Iteration " << (i + 1) << "/" << n_iters_
                      << (converged ? " (converged)" : "")
                      << " (T_assign=" << t_assign_elapsed << "s, "
                      << "T_update=" << t_update_elapsed << "s, "
                      << "max_change=" << max_change << ")" << std::endl;
        }

        // Обновляем центроиды
        std::copy(new_centroids.begin(), new_centroids.end(), centroids_.begin());

        // Ранний выход при сходимости
        if (converged) {
            std::cout << "  Convergence reached after " << (i + 1) << " iterations "
                      << "(max_change=" << max_change << " < tol=" << tol_ << ")" << std::endl;
            break;
        }
    }
    
    // Вызываем finalize_data для финализации (например, перенос результатов обратно)
    finalize_data();
}

