#pragma once

#include "kmeans_base.h"

#ifdef _WIN32
// Для Windows используем явный экспорт
#define CUDA_EXPORT __declspec(dllexport)
#else
#define CUDA_EXPORT
#endif

// Объявления фабричных функций для создания CUDA реализаций
// Эти функции реализованы в kmeans_cuda.cu и должны быть скомпилированы CUDA компилятором
extern "C++" {
    CUDA_EXPORT KMeansBase* createKMeansCUDAV1(int n_clusters, int n_iters = 100, double tol = 1e-6);
    CUDA_EXPORT KMeansBase* createKMeansCUDAV2(int n_clusters, int n_iters = 100, double tol = 1e-6);
    CUDA_EXPORT KMeansBase* createKMeansCUDAV3(int n_clusters, int n_iters = 100, double tol = 1e-6);
    CUDA_EXPORT KMeansBase* createKMeansCUDAV4(int n_clusters, int n_iters = 100, double tol = 1e-6);
}

