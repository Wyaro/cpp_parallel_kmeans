// Этот файл должен компилироваться CUDA компилятором (nvcc)
// Убедитесь, что файл имеет расширение .cu и включен в проект как CudaCompile

#include "../include/kmeans_base.h"
#include "../include/kmeans_cuda.h"
#include "../include/timer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

// Убедимся, что функции экспортируются правильно
#ifdef __CUDACC__
// Компилируется CUDA компилятором
#else
// Если компилируется обычным компилятором, это ошибка
#error "This file must be compiled with CUDA compiler (nvcc)"
#endif

// Проверка ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::exit(1); \
        } \
    } while(0)

/**
 * CUDA реализация K-means (V1 - базовая версия).
 * Прямое вычисление расстояний через broadcasting.
 */
class KMeansCUDAV1 : public KMeansBase {
public:
    KMeansCUDAV1(int n_clusters, int n_iters = 100, double tol = 1e-6);
    ~KMeansCUDAV1();
    
    // Переопределяем методы для измерения времени передачи данных
    double t_h2d() const override { return t_h2d_; }
    double t_d2h() const override { return t_d2h_; }
    
    // Переопределяем prepare_data и finalize_data для измерения времени передачи
    void prepare_data(const double* X, int N, int D, const double* initial_centroids) override;
    void finalize_data() override;

protected:
    void assign_clusters(const double* X, int N, int D, 
                        const double* centroids, int* labels) override;
    void update_centroids(const double* X, int N, int D,
                         const int* labels, double* new_centroids) override;

private:
    double* d_X_;
    double* d_centroids_;
    int* d_labels_;
    double* d_distances_;
    double* d_sums_;
    int* d_counts_;
    int N_current_;
    int D_current_;
    double t_h2d_;  // Время передачи Host-to-Device
    double t_d2h_;  // Время передачи Device-to-Host
};

// CUDA kernel для вычисления расстояний (V1)
__global__ void compute_distances_v1(const double* X, const double* centroids,
                                     double* distances, int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double min_dist = 1e30;
    int best_k = 0;

    for (int k = 0; k < K; ++k) {
        double dist = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = X[i * D + d] - centroids[k * D + d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    distances[i] = best_k; // Сохраняем индекс кластера
}

// CUDA kernel для обновления центроидов (V1)
__global__ void update_centroids_v1(const double* X, const int* labels,
                                     double* sums, int* counts, int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int label = labels[idx];
    if (label < 0 || label >= K) return;

    atomicAdd(&counts[label], 1);
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[label * D + d], X[idx * D + d]);
    }
}

KMeansCUDAV1::KMeansCUDAV1(int n_clusters, int n_iters, double tol)
    : KMeansBase(n_clusters, n_iters, tol),
      d_X_(nullptr), d_centroids_(nullptr), d_labels_(nullptr),
      d_distances_(nullptr), d_sums_(nullptr), d_counts_(nullptr),
      N_current_(0), D_current_(0), t_h2d_(0.0), t_d2h_(0.0) {
}

KMeansCUDAV1::~KMeansCUDAV1() {
    if (d_X_) CUDA_CHECK(cudaFree(d_X_));
    if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
    if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
    if (d_distances_) CUDA_CHECK(cudaFree(d_distances_));
    if (d_sums_) CUDA_CHECK(cudaFree(d_sums_));
    if (d_counts_) CUDA_CHECK(cudaFree(d_counts_));
}

void KMeansCUDAV1::assign_clusters(const double* X, int N, int D,
                                   const double* centroids, int* labels) {
    // Данные уже на GPU (скопированы в prepare_data)
    // Обновляем только центроиды если они изменились
    std::vector<double> centroids_host(K_ * D);
    std::copy(centroids, centroids + K_ * D, centroids_host.begin());
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_host.data(), K_ * D * sizeof(double), cudaMemcpyHostToDevice));

    // Запускаем kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    compute_distances_v1<<<blocks, threads>>>(d_X_, d_centroids_, d_distances_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());

    // Копируем результаты обратно
    std::vector<double> distances_host(N);
    CUDA_CHECK(cudaMemcpy(distances_host.data(), d_distances_, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Преобразуем индексы в метки
    for (int i = 0; i < N; ++i) {
        labels[i] = static_cast<int>(distances_host[i]);
    }
}

void KMeansCUDAV1::update_centroids(const double* X, int N, int D,
                                    const int* labels, double* new_centroids) {
    // Выделяем память для сумм и счетчиков
    if (!d_sums_) {
        CUDA_CHECK(cudaMalloc(&d_sums_, K_ * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_counts_, K_ * sizeof(int)));
    }

    // Обнуляем массивы
    CUDA_CHECK(cudaMemset(d_sums_, 0, K_ * D * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_counts_, 0, K_ * sizeof(int)));

    // X уже на GPU, копируем только labels
    std::vector<int> labels_host(N);
    std::copy(labels, labels + N, labels_host.begin());
    CUDA_CHECK(cudaMemcpy(d_labels_, labels_host.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // Запускаем kernel
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_centroids_v1<<<blocks, threads>>>(d_X_, d_labels_, d_sums_, d_counts_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем результаты обратно
    std::vector<double> sums_host(K_ * D);
    std::vector<int> counts_host(K_);
    CUDA_CHECK(cudaMemcpy(sums_host.data(), d_sums_, K_ * D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts_host.data(), d_counts_, K_ * sizeof(int), cudaMemcpyDeviceToHost));

    // Вычисляем новые центроиды
    // Используем старые центроиды из базового класса
    const double* old_centroids = centroids_.data();
    for (int k = 0; k < K_; ++k) {
        if (counts_host[k] > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = sums_host[k * D + d] / counts_host[k];
            }
        } else {
            // Сохраняем старые центроиды для пустых кластеров
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = old_centroids[k * D + d];
            }
        }
    }
}

/**
 * CUDA реализация K-means (V2 - оптимизированная версия).
 * Использует матричную формулу ||x||² + ||c||² - 2x·c без материализации diff.
 */
class KMeansCUDAV2 : public KMeansBase {
public:
    KMeansCUDAV2(int n_clusters, int n_iters = 100, double tol = 1e-6);
    ~KMeansCUDAV2();
    
    // Переопределяем методы для измерения времени передачи данных
    double t_h2d() const override { return t_h2d_; }
    double t_d2h() const override { return t_d2h_; }
    
    void prepare_data(const double* X, int N, int D, const double* initial_centroids) override;
    void finalize_data() override;

protected:
    void assign_clusters(const double* X, int N, int D,
                        const double* centroids, int* labels) override;
    void update_centroids(const double* X, int N, int D,
                         const int* labels, double* new_centroids) override;

private:
    double* d_X_;
    double* d_centroids_;
    int* d_labels_;
    double* d_distances_;
    double* d_x_sq_;
    double* d_c_sq_;
    double* d_cross_;
    double* d_sums_;
    int* d_counts_;
    int N_current_;
    int D_current_;
    double t_h2d_;  // Время передачи Host-to-Device
    double t_d2h_;  // Время передачи Device-to-Host
};

// CUDA kernel для вычисления расстояний (V2) - оптимизированная формула
__global__ void compute_distances_v2(const double* X, const double* centroids,
                                     const double* x_sq, const double* c_sq,
                                     double* distances, int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double min_dist = 1e30;
    int best_k = 0;

    double x_norm_sq = x_sq[i];

    for (int k = 0; k < K; ++k) {
        double cross = 0.0;
        for (int d = 0; d < D; ++d) {
            cross += X[i * D + d] * centroids[k * D + d];
        }
        double dist = x_norm_sq + c_sq[k] - 2.0 * cross;
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    distances[i] = best_k;
}

KMeansCUDAV2::KMeansCUDAV2(int n_clusters, int n_iters, double tol)
    : KMeansBase(n_clusters, n_iters, tol),
      d_X_(nullptr), d_centroids_(nullptr), d_labels_(nullptr),
      d_distances_(nullptr), d_x_sq_(nullptr), d_c_sq_(nullptr),
      d_cross_(nullptr), d_sums_(nullptr), d_counts_(nullptr),
      N_current_(0), D_current_(0), t_h2d_(0.0), t_d2h_(0.0) {
}

KMeansCUDAV2::~KMeansCUDAV2() {
    if (d_X_) CUDA_CHECK(cudaFree(d_X_));
    if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
    if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
    if (d_distances_) CUDA_CHECK(cudaFree(d_distances_));
    if (d_x_sq_) CUDA_CHECK(cudaFree(d_x_sq_));
    if (d_c_sq_) CUDA_CHECK(cudaFree(d_c_sq_));
    if (d_cross_) CUDA_CHECK(cudaFree(d_cross_));
    if (d_sums_) CUDA_CHECK(cudaFree(d_sums_));
    if (d_counts_) CUDA_CHECK(cudaFree(d_counts_));
}

void KMeansCUDAV2::update_centroids(const double* X, int N, int D,
                                    const int* labels, double* new_centroids) {
    if (!d_sums_) {
        CUDA_CHECK(cudaMalloc(&d_sums_, K_ * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_counts_, K_ * sizeof(int)));
    }

    CUDA_CHECK(cudaMemset(d_sums_, 0, K_ * D * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_counts_, 0, K_ * sizeof(int)));

    // X уже на GPU, копируем только labels
    std::vector<int> labels_host(N);
    std::copy(labels, labels + N, labels_host.begin());
    CUDA_CHECK(cudaMemcpy(d_labels_, labels_host.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_centroids_v1<<<blocks, threads>>>(d_X_, d_labels_, d_sums_, d_counts_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> sums_host(K_ * D);
    std::vector<int> counts_host(K_);
    CUDA_CHECK(cudaMemcpy(sums_host.data(), d_sums_, K_ * D * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts_host.data(), d_counts_, K_ * sizeof(int), cudaMemcpyDeviceToHost));

    // Используем старые центроиды из базового класса
    const double* old_centroids = centroids_.data();
    for (int k = 0; k < K_; ++k) {
        if (counts_host[k] > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = sums_host[k * D + d] / counts_host[k];
            }
        } else {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = old_centroids[k * D + d];
            }
        }
    }
}

// Переопределяем prepare_data для измерения времени передачи H2D
void KMeansCUDAV1::prepare_data(const double* X, int N, int D, const double* initial_centroids) {
    Timer t_h2d_timer;
    t_h2d_timer.start();
    
    if (N != N_current_ || D != D_current_) {
        if (d_X_) CUDA_CHECK(cudaFree(d_X_));
        if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
        if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
        if (d_distances_) CUDA_CHECK(cudaFree(d_distances_));

        CUDA_CHECK(cudaMalloc(&d_X_, N * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_centroids_, K_ * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_labels_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_distances_, N * sizeof(double)));

        N_current_ = N;
        D_current_ = D;
    }
    
    CUDA_CHECK(cudaMemcpy(d_X_, X, N * D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_, initial_centroids, K_ * D * sizeof(double), cudaMemcpyHostToDevice));
    
    t_h2d_timer.stop();
    t_h2d_ = t_h2d_timer.elapsed();
}

// Переопределяем finalize_data для измерения времени передачи D2H
void KMeansCUDAV1::finalize_data() {
    Timer t_d2h_timer;
    t_d2h_timer.start();
    // Центроиды и labels уже на CPU, но можно скопировать если нужно
    t_d2h_timer.stop();
    t_d2h_ = t_d2h_timer.elapsed();
}

void KMeansCUDAV2::prepare_data(const double* X, int N, int D, const double* initial_centroids) {
    Timer t_h2d_timer;
    t_h2d_timer.start();
    
    if (N != N_current_ || D != D_current_) {
        if (d_X_) CUDA_CHECK(cudaFree(d_X_));
        if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
        if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
        if (d_distances_) CUDA_CHECK(cudaFree(d_distances_));
        if (d_x_sq_) CUDA_CHECK(cudaFree(d_x_sq_));
        if (d_c_sq_) CUDA_CHECK(cudaFree(d_c_sq_));

        CUDA_CHECK(cudaMalloc(&d_X_, N * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_centroids_, K_ * D * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_labels_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_distances_, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_x_sq_, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c_sq_, K_ * sizeof(double)));

        N_current_ = N;
        D_current_ = D;
    }
    
    CUDA_CHECK(cudaMemcpy(d_X_, X, N * D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_, initial_centroids, K_ * D * sizeof(double), cudaMemcpyHostToDevice));
    
    t_h2d_timer.stop();
    t_h2d_ = t_h2d_timer.elapsed();
}

void KMeansCUDAV2::finalize_data() {
    Timer t_d2h_timer;
    t_d2h_timer.start();
    t_d2h_timer.stop();
    t_d2h_ = t_d2h_timer.elapsed();
}

// CUDA kernels для вычисления x_sq и c_sq на GPU (оптимизация V2)
__global__ void compute_x_sq_kernel(const double* X, double* x_sq, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    double sum = 0.0;
    for (int d = 0; d < D; ++d) {
        sum += X[i * D + d] * X[i * D + d];
    }
    x_sq[i] = sum;
}

__global__ void compute_c_sq_kernel(const double* centroids, double* c_sq, int K, int D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    double sum = 0.0;
    for (int d = 0; d < D; ++d) {
        sum += centroids[k * D + d] * centroids[k * D + d];
    }
    c_sq[k] = sum;
}

// Обновляем V2 для использования GPU kernels для x_sq и c_sq
void KMeansCUDAV2::assign_clusters(const double* X, int N, int D,
                                   const double* centroids, int* labels) {
    // X уже на GPU, обновляем только центроиды
    std::vector<double> centroids_host(K_ * D);
    std::copy(centroids, centroids + K_ * D, centroids_host.begin());
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_host.data(), K_ * D * sizeof(double), cudaMemcpyHostToDevice));

    // Вычисляем ||x||² и ||c||² на GPU
    int threads = 256;
    int blocks_x = (N + threads - 1) / threads;
    int blocks_c = (K_ + threads - 1) / threads;
    
    compute_x_sq_kernel<<<blocks_x, threads>>>(d_X_, d_x_sq_, N, D);
    compute_c_sq_kernel<<<blocks_c, threads>>>(d_centroids_, d_c_sq_, K_, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    blocks_x = (N + threads - 1) / threads;
    compute_distances_v2<<<blocks_x, threads>>>(d_X_, d_centroids_, d_x_sq_, d_c_sq_, d_distances_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> distances_host(N);
    CUDA_CHECK(cudaMemcpy(distances_host.data(), d_distances_, N * sizeof(double), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; ++i) {
        labels[i] = static_cast<int>(distances_host[i]);
    }
}

/**
 * CUDA реализация K-means (V3 - быстрая версия с float32).
 * Использует float32 для меньшего трафика памяти и быстрых вычислений.
 */
class KMeansCUDAV3 : public KMeansBase {
public:
    KMeansCUDAV3(int n_clusters, int n_iters = 100, double tol = 1e-6, bool use_float32 = true);
    ~KMeansCUDAV3();
    
    double t_h2d() const override { return t_h2d_; }
    double t_d2h() const override { return t_d2h_; }
    void prepare_data(const double* X, int N, int D, const double* initial_centroids) override;
    void finalize_data() override;

protected:
    void assign_clusters(const double* X, int N, int D,
                        const double* centroids, int* labels) override;
    void update_centroids(const double* X, int N, int D,
                         const int* labels, double* new_centroids) override;

private:
    float* d_X_;
    float* d_centroids_;
    int* d_labels_;
    float* d_distances_;
    float* d_x_sq_;
    float* d_c_sq_;
    float* d_sums_;
    int* d_counts_;
    int N_current_;
    int D_current_;
    bool use_float32_;
    double t_h2d_;
    double t_d2h_;
};

// CUDA kernels для V3 (float32)
__global__ void compute_distances_v3(const float* X, const float* centroids,
                                     const float* x_sq, const float* c_sq,
                                     float* distances, int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float min_dist = 1e30f;
    int best_k = 0;

    float x_norm_sq = x_sq[i];

    for (int k = 0; k < K; ++k) {
        float cross = 0.0f;
        for (int d = 0; d < D; ++d) {
            cross += X[i * D + d] * centroids[k * D + d];
        }
        float dist = x_norm_sq + c_sq[k] - 2.0f * cross;
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    distances[i] = static_cast<float>(best_k);
}

__global__ void compute_x_sq_kernel_f32(const float* X, float* x_sq, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        sum += X[i * D + d] * X[i * D + d];
    }
    x_sq[i] = sum;
}

__global__ void compute_c_sq_kernel_f32(const float* centroids, float* c_sq, int K, int D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    float sum = 0.0f;
    for (int d = 0; d < D; ++d) {
        sum += centroids[k * D + d] * centroids[k * D + d];
    }
    c_sq[k] = sum;
}

__global__ void update_centroids_v3(const float* X, const int* labels,
                                     float* sums, int* counts, int N, int D, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int label = labels[idx];
    if (label < 0 || label >= K) return;

    atomicAdd(&counts[label], 1);
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[label * D + d], X[idx * D + d]);
    }
}

KMeansCUDAV3::KMeansCUDAV3(int n_clusters, int n_iters, double tol, bool use_float32)
    : KMeansBase(n_clusters, n_iters, tol),
      d_X_(nullptr), d_centroids_(nullptr), d_labels_(nullptr),
      d_distances_(nullptr), d_x_sq_(nullptr), d_c_sq_(nullptr),
      d_sums_(nullptr), d_counts_(nullptr),
      N_current_(0), D_current_(0), use_float32_(use_float32),
      t_h2d_(0.0), t_d2h_(0.0) {
}

KMeansCUDAV3::~KMeansCUDAV3() {
    if (d_X_) CUDA_CHECK(cudaFree(d_X_));
    if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
    if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
    if (d_distances_) CUDA_CHECK(cudaFree(d_distances_));
    if (d_x_sq_) CUDA_CHECK(cudaFree(d_x_sq_));
    if (d_c_sq_) CUDA_CHECK(cudaFree(d_c_sq_));
    if (d_sums_) CUDA_CHECK(cudaFree(d_sums_));
    if (d_counts_) CUDA_CHECK(cudaFree(d_counts_));
}

void KMeansCUDAV3::prepare_data(const double* X, int N, int D, const double* initial_centroids) {
    Timer t_h2d_timer;
    t_h2d_timer.start();
    
    if (N != N_current_ || D != D_current_) {
        if (d_X_) CUDA_CHECK(cudaFree(d_X_));
        if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
        if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
        if (d_distances_) CUDA_CHECK(cudaFree(d_distances_));
        if (d_x_sq_) CUDA_CHECK(cudaFree(d_x_sq_));
        if (d_c_sq_) CUDA_CHECK(cudaFree(d_c_sq_));

        CUDA_CHECK(cudaMalloc(&d_X_, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids_, K_ * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_distances_, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_sq_, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c_sq_, K_ * sizeof(float)));

        N_current_ = N;
        D_current_ = D;
    }
    
    // Конвертируем double в float и копируем на GPU
    std::vector<float> X_f32(N * D);
    std::vector<float> centroids_f32(K_ * D);
    for (int i = 0; i < N * D; ++i) {
        X_f32[i] = static_cast<float>(X[i]);
    }
    for (int i = 0; i < K_ * D; ++i) {
        centroids_f32[i] = static_cast<float>(initial_centroids[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_X_, X_f32.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_f32.data(), K_ * D * sizeof(float), cudaMemcpyHostToDevice));
    
    t_h2d_timer.stop();
    t_h2d_ = t_h2d_timer.elapsed();
}

void KMeansCUDAV3::finalize_data() {
    Timer t_d2h_timer;
    t_d2h_timer.start();
    t_d2h_timer.stop();
    t_d2h_ = t_d2h_timer.elapsed();
}

void KMeansCUDAV3::assign_clusters(const double* X, int N, int D,
                                   const double* centroids, int* labels) {
    // X уже на GPU в float32, обновляем только центроиды
    std::vector<float> centroids_f32(K_ * D);
    for (int k = 0; k < K_ * D; ++k) {
        centroids_f32[k] = static_cast<float>(centroids[k]);
    }
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_f32.data(), K_ * D * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks_x = (N + threads - 1) / threads;
    int blocks_c = (K_ + threads - 1) / threads;
    
    compute_x_sq_kernel_f32<<<blocks_x, threads>>>(d_X_, d_x_sq_, N, D);
    compute_c_sq_kernel_f32<<<blocks_c, threads>>>(d_centroids_, d_c_sq_, K_, D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    blocks_x = (N + threads - 1) / threads;
    compute_distances_v3<<<blocks_x, threads>>>(d_X_, d_centroids_, d_x_sq_, d_c_sq_, d_distances_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> distances_host(N);
    CUDA_CHECK(cudaMemcpy(distances_host.data(), d_distances_, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; ++i) {
        labels[i] = static_cast<int>(distances_host[i]);
    }
}

void KMeansCUDAV3::update_centroids(const double* X, int N, int D,
                                    const int* labels, double* new_centroids) {
    if (!d_sums_) {
        CUDA_CHECK(cudaMalloc(&d_sums_, K_ * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts_, K_ * sizeof(int)));
    }

    CUDA_CHECK(cudaMemset(d_sums_, 0, K_ * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_counts_, 0, K_ * sizeof(int)));

    // Конвертируем labels в int и копируем
    std::vector<int> labels_int(N);
    for (int i = 0; i < N; ++i) {
        labels_int[i] = labels[i];
    }
    CUDA_CHECK(cudaMemcpy(d_labels_, labels_int.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_centroids_v3<<<blocks, threads>>>(d_X_, d_labels_, d_sums_, d_counts_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> sums_host(K_ * D);
    std::vector<int> counts_host(K_);
    CUDA_CHECK(cudaMemcpy(sums_host.data(), d_sums_, K_ * D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts_host.data(), d_counts_, K_ * sizeof(int), cudaMemcpyDeviceToHost));

    const double* old_centroids = centroids_.data();
    for (int k = 0; k < K_; ++k) {
        if (counts_host[k] > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = static_cast<double>(sums_host[k * D + d]) / counts_host[k];
            }
        } else {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = old_centroids[k * D + d];
            }
        }
    }
}

/**
 * CUDA реализация K-means (V4 - raw CUDA kernels).
 * Самая быстрая версия с ручной оптимизацией kernels.
 */
class KMeansCUDAV4 : public KMeansBase {
public:
    KMeansCUDAV4(int n_clusters, int n_iters = 100, double tol = 1e-6, bool use_float32 = true);
    ~KMeansCUDAV4();
    
    double t_h2d() const override { return t_h2d_; }
    double t_d2h() const override { return t_d2h_; }
    void prepare_data(const double* X, int N, int D, const double* initial_centroids) override;
    void finalize_data() override;

protected:
    void assign_clusters(const double* X, int N, int D,
                        const double* centroids, int* labels) override;
    void update_centroids(const double* X, int N, int D,
                         const int* labels, double* new_centroids) override;

private:
    float* d_X_;
    float* d_centroids_;
    int* d_labels_;
    float* d_sums_;
    int* d_counts_;
    int N_current_;
    int D_current_;
    bool use_float32_;
    double t_h2d_;
    double t_d2h_;
};

// Raw CUDA kernels для V4
__global__ void assign_kernel_v4(const float* __restrict__ X,
                                  const float* __restrict__ C,
                                  int* __restrict__ labels,
                                  const int N, const int D, const int K) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;
    const float* x = X + i * D;
    float best = 1e30f;
    int best_k = 0;
    for (int k = 0; k < K; ++k) {
        const float* c = C + k * D;
        float dist = 0.0f;
        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float diff = x[d] - c[d];
            dist += diff * diff;
        }
        if (dist < best) { 
            best = dist; 
            best_k = k; 
        }
    }
    labels[i] = best_k;
}

__global__ void update_kernel_v4(const float* __restrict__ X,
                                  const int* __restrict__ labels,
                                  float* __restrict__ sums,
                                  int* __restrict__ counts,
                                  const int N, const int D, const int K) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    
    int label = labels[idx];
    if (label < 0 || label >= K) return;
    
    atomicAdd(&counts[label], 1);
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[label * D + d], X[idx * D + d]);
    }
}

KMeansCUDAV4::KMeansCUDAV4(int n_clusters, int n_iters, double tol, bool use_float32)
    : KMeansBase(n_clusters, n_iters, tol),
      d_X_(nullptr), d_centroids_(nullptr), d_labels_(nullptr),
      d_sums_(nullptr), d_counts_(nullptr),
      N_current_(0), D_current_(0), use_float32_(use_float32),
      t_h2d_(0.0), t_d2h_(0.0) {
}

KMeansCUDAV4::~KMeansCUDAV4() {
    if (d_X_) CUDA_CHECK(cudaFree(d_X_));
    if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
    if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
    if (d_sums_) CUDA_CHECK(cudaFree(d_sums_));
    if (d_counts_) CUDA_CHECK(cudaFree(d_counts_));
}

void KMeansCUDAV4::prepare_data(const double* X, int N, int D, const double* initial_centroids) {
    Timer t_h2d_timer;
    t_h2d_timer.start();
    
    if (N != N_current_ || D != D_current_) {
        if (d_X_) CUDA_CHECK(cudaFree(d_X_));
        if (d_centroids_) CUDA_CHECK(cudaFree(d_centroids_));
        if (d_labels_) CUDA_CHECK(cudaFree(d_labels_));
        if (d_sums_) CUDA_CHECK(cudaFree(d_sums_));
        if (d_counts_) CUDA_CHECK(cudaFree(d_counts_));

        CUDA_CHECK(cudaMalloc(&d_X_, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_centroids_, K_ * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_labels_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sums_, K_ * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_counts_, K_ * sizeof(int)));

        N_current_ = N;
        D_current_ = D;
    }
    
    std::vector<float> X_f32(N * D);
    std::vector<float> centroids_f32(K_ * D);
    for (int i = 0; i < N * D; ++i) {
        X_f32[i] = static_cast<float>(X[i]);
    }
    for (int i = 0; i < K_ * D; ++i) {
        centroids_f32[i] = static_cast<float>(initial_centroids[i]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_X_, X_f32.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_f32.data(), K_ * D * sizeof(float), cudaMemcpyHostToDevice));
    
    t_h2d_timer.stop();
    t_h2d_ = t_h2d_timer.elapsed();
}

void KMeansCUDAV4::finalize_data() {
    Timer t_d2h_timer;
    t_d2h_timer.start();
    t_d2h_timer.stop();
    t_d2h_ = t_d2h_timer.elapsed();
}

void KMeansCUDAV4::assign_clusters(const double* X, int N, int D,
                                   const double* centroids, int* labels) {
    // Обновляем центроиды на GPU
    std::vector<float> centroids_f32(K_ * D);
    for (int k = 0; k < K_ * D; ++k) {
        centroids_f32[k] = static_cast<float>(centroids[k]);
    }
    CUDA_CHECK(cudaMemcpy(d_centroids_, centroids_f32.data(), K_ * D * sizeof(float), cudaMemcpyHostToDevice));
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    assign_kernel_v4<<<blocks, threads>>>(d_X_, d_centroids_, d_labels_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> labels_host(N);
    CUDA_CHECK(cudaMemcpy(labels_host.data(), d_labels_, N * sizeof(int), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < N; ++i) {
        labels[i] = labels_host[i];
    }
}

void KMeansCUDAV4::update_centroids(const double* X, int N, int D,
                                    const int* labels, double* new_centroids) {
    CUDA_CHECK(cudaMemset(d_sums_, 0, K_ * D * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_counts_, 0, K_ * sizeof(int)));

    std::vector<int> labels_int(N);
    for (int i = 0; i < N; ++i) {
        labels_int[i] = labels[i];
    }
    CUDA_CHECK(cudaMemcpy(d_labels_, labels_int.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    update_kernel_v4<<<blocks, threads>>>(d_X_, d_labels_, d_sums_, d_counts_, N, D, K_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> sums_host(K_ * D);
    std::vector<int> counts_host(K_);
    CUDA_CHECK(cudaMemcpy(sums_host.data(), d_sums_, K_ * D * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(counts_host.data(), d_counts_, K_ * sizeof(int), cudaMemcpyDeviceToHost));

    const double* old_centroids = centroids_.data();
    for (int k = 0; k < K_; ++k) {
        if (counts_host[k] > 0) {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = static_cast<double>(sums_host[k * D + d]) / counts_host[k];
            }
        } else {
            for (int d = 0; d < D; ++d) {
                new_centroids[k * D + d] = old_centroids[k * D + d];
            }
        }
    }
}

// Фабричные функции для создания экземпляров
// Реализации должны соответствовать объявлениям в kmeans_cuda.h
// Используем extern "C++" для правильного именования
extern "C++" {
    KMeansBase* createKMeansCUDAV1(int n_clusters, int n_iters, double tol) {
        return new KMeansCUDAV1(n_clusters, n_iters, tol);
    }

    KMeansBase* createKMeansCUDAV2(int n_clusters, int n_iters, double tol) {
        return new KMeansCUDAV2(n_clusters, n_iters, tol);
    }

    KMeansBase* createKMeansCUDAV3(int n_clusters, int n_iters, double tol) {
        return new KMeansCUDAV3(n_clusters, n_iters, tol, true);
    }

    KMeansBase* createKMeansCUDAV4(int n_clusters, int n_iters, double tol) {
        return new KMeansCUDAV4(n_clusters, n_iters, tol, true);
    }
}

