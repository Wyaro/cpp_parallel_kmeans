#include "kmeans_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// Однопоточная CPU реализация
// ============================================================================

KMeansCPU::KMeansCPU(int n_clusters, int n_iters, double tol)
    : KMeansBase(n_clusters, n_iters, tol) {
}

void KMeansCPU::assign_clusters(const double* X, int N, int D,
                                const double* centroids, int* labels) {
    // Для каждой точки находим ближайший центроид
    for (int i = 0; i < N; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_k = 0;

        for (int k = 0; k < K_; ++k) {
            // Вычисляем квадрат расстояния: ||x - c||²
            double dist_sq = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = X[i * D + d] - centroids[k * D + d];
                dist_sq += diff * diff;
            }

            if (dist_sq < min_dist) {
                min_dist = dist_sq;
                best_k = k;
            }
        }
        labels[i] = best_k;
    }
}

void KMeansCPU::update_centroids(const double* X, int N, int D,
                                 const int* labels, double* new_centroids) {
    // Инициализируем суммы и счетчики
    std::vector<double> sums(K_ * D, 0.0);
    std::vector<int> counts(K_, 0);

    // Суммируем точки по кластерам
    for (int i = 0; i < N; ++i) {
        int k = labels[i];
        if (k >= 0 && k < K_) {
            counts[k]++;
            for (int d = 0; d < D; ++d) {
                sums[k * D + d] += X[i * D + d];
            }
        }
    }

    // Вычисляем новые центроиды (средние)
    // Используем старые центроиды для пустых кластеров
    const double* old_centroids = centroids_.data();
    for (int k = 0; k < K_; ++k) {
        if (counts[k] > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = sums[k * D + d] / counts[k];
            }
        } else {
            // Сохраняем старые центроиды для пустых кластеров
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = old_centroids[k * D + d];
            }
        }
    }
}

// ============================================================================
// Многопоточная CPU реализация с OpenMP
// ============================================================================

KMeansCPUOpenMP::KMeansCPUOpenMP(int n_clusters, int n_iters, double tol, int n_threads)
    : KMeansBase(n_clusters, n_iters, tol), n_threads_(n_threads) {
#ifdef _OPENMP
    if (n_threads_ > 0) {
        omp_set_num_threads(n_threads_);
    }
#endif
}

void KMeansCPUOpenMP::assign_clusters(const double* X, int N, int D,
                                      const double* centroids, int* labels) {
#ifdef _OPENMP
    // Параллелизуем по точкам
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_k = 0;

        for (int k = 0; k < K_; ++k) {
            // Вычисляем квадрат расстояния: ||x - c||²
            double dist_sq = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = X[i * D + d] - centroids[k * D + d];
                dist_sq += diff * diff;
            }

            if (dist_sq < min_dist) {
                min_dist = dist_sq;
                best_k = k;
            }
        }
        labels[i] = best_k;
    }
#else
    // Если OpenMP не доступен, используем однопоточную версию
    for (int i = 0; i < N; ++i) {
        double min_dist = std::numeric_limits<double>::max();
        int best_k = 0;

        for (int k = 0; k < K_; ++k) {
            double dist_sq = 0.0;
            for (int d = 0; d < D; ++d) {
                double diff = X[i * D + d] - centroids[k * D + d];
                dist_sq += diff * diff;
            }

            if (dist_sq < min_dist) {
                min_dist = dist_sq;
                best_k = k;
            }
        }
        labels[i] = best_k;
    }
#endif
}

void KMeansCPUOpenMP::update_centroids(const double* X, int N, int D,
                                      const int* labels, double* new_centroids) {
    // Инициализируем суммы и счетчики
    std::vector<double> sums(K_ * D, 0.0);
    std::vector<int> counts(K_, 0);

#ifdef _OPENMP
    // Параллелизуем суммирование по точкам
    // Используем reduction для сумм и счетчиков
    #pragma omp parallel
    {
        // Локальные суммы для каждого потока
        std::vector<double> local_sums(K_ * D, 0.0);
        std::vector<int> local_counts(K_, 0);

        #pragma omp for nowait
        for (int i = 0; i < N; ++i) {
            int k = labels[i];
            if (k >= 0 && k < K_) {
                local_counts[k]++;
                for (int d = 0; d < D; ++d) {
                    local_sums[k * D + d] += X[i * D + d];
                }
            }
        }

        // Критическая секция для объединения результатов
        #pragma omp critical
        {
            for (int k = 0; k < K_; ++k) {
                counts[k] += local_counts[k];
                for (int d = 0; d < D; ++d) {
                    sums[k * D + d] += local_sums[k * D + d];
                }
            }
        }
    }
#else
    // Если OpenMP не доступен, используем однопоточную версию
    for (int i = 0; i < N; ++i) {
        int k = labels[i];
        if (k >= 0 && k < K_) {
            counts[k]++;
            for (int d = 0; d < D; ++d) {
                sums[k * D + d] += X[i * D + d];
            }
        }
    }
#endif

    // Вычисляем новые центроиды (средние)
    // Используем старые центроиды для пустых кластеров
    const double* old_centroids = centroids_.data();
    for (int k = 0; k < K_; ++k) {
        if (counts[k] > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = sums[k * D + d] / counts[k];
            }
        } else {
            // Сохраняем старые центроиды для пустых кластеров
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = old_centroids[k * D + d];
            }
        }
    }
}

// ============================================================================
// Фабричные функции
// ============================================================================

KMeansBase* createKMeansCPU(int n_clusters, int n_iters, double tol) {
    return new KMeansCPU(n_clusters, n_iters, tol);
}

KMeansBase* createKMeansCPUOpenMP(int n_clusters, int n_iters, double tol, int n_threads) {
    return new KMeansCPUOpenMP(n_clusters, n_iters, tol, n_threads);
}

