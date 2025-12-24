#pragma once

#include "kmeans_base.h"

/**
 * Однопоточная CPU реализация K-means (baseline).
 * Простая реализация на CPU для сравнения производительности.
 */
class KMeansCPU : public KMeansBase {
public:
    KMeansCPU(int n_clusters, int n_iters = 100, double tol = 1e-6);
    ~KMeansCPU() = default;

protected:
    void assign_clusters(const double* X, int N, int D,
                         const double* centroids, int* labels) override;
    void update_centroids(const double* X, int N, int D,
                         const int* labels, double* new_centroids) override;
};

/**
 * Многопоточная CPU реализация K-means с OpenMP.
 * Параллелизует вычисления по потокам для ускорения на многоядерных CPU.
 */
class KMeansCPUOpenMP : public KMeansBase {
public:
    KMeansCPUOpenMP(int n_clusters, int n_iters = 100, double tol = 1e-6, int n_threads = 0);
    ~KMeansCPUOpenMP() = default;

protected:
    void assign_clusters(const double* X, int N, int D,
                         const double* centroids, int* labels) override;
    void update_centroids(const double* X, int N, int D,
                         const int* labels, double* new_centroids) override;

private:
    int n_threads_;  // Количество потоков (0 = автоматически)
};

// Фабричные функции для создания CPU реализаций
KMeansBase* createKMeansCPU(int n_clusters, int n_iters = 100, double tol = 1e-6);
KMeansBase* createKMeansCPUOpenMP(int n_clusters, int n_iters = 100, double tol = 1e-6, int n_threads = 0);

