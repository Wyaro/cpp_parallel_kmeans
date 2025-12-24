#pragma once

#include <vector>
#include <memory>

/**
 * Базовый класс для реализаций KMeans.
 * Отвечает за цикл итераций и сбор низкоуровневых таймингов.
 */
class KMeansBase {
public:
    KMeansBase(int n_clusters, int n_iters = 100, double tol = 1e-9);
    virtual ~KMeansBase() = default;

    /**
     * Основной цикл KMeans с адаптивной остановкой по сходимости.
     * Алгоритм останавливается, когда:
     * - Центроиды перестают меняться (изменение < tol), ИЛИ
     * - Достигнуто максимальное количество итераций (n_iters)
     */
    void fit(const double* X, int N, int D, const double* initial_centroids);
    
    // Виртуальные методы для подготовки и финализации (для GPU реализаций)
    virtual void prepare_data(const double* X, int N, int D, const double* initial_centroids) {}
    virtual void finalize_data() {}

    // Доступ к результатам
    const std::vector<double>& centroids() const { return centroids_; }
    const std::vector<int>& labels() const { return labels_; }
    int n_iters_actual() const { return n_iters_actual_; }

    // Тайминги
    double t_assign_total() const { return t_assign_total_; }
    double t_update_total() const { return t_update_total_; }
    double t_iter_total() const { return t_iter_total_; }
    
    // Время передачи данных (для GPU реализаций)
    virtual double t_h2d() const { return 0.0; }  // Host-to-Device
    virtual double t_d2h() const { return 0.0; }  // Device-to-Host
    virtual double t_transfer() const { return t_h2d() + t_d2h(); }

protected:
    /**
     * Шаг назначения точек кластерам.
     * @param X Массив данных (N * D)
     * @param centroids Текущие центроиды (K * D)
     * @param labels Выходной массив меток кластеров (N)
     */
    virtual void assign_clusters(const double* X, int N, int D, 
                                 const double* centroids, int* labels) = 0;

    /**
     * Шаг обновления центроидов по присвоенным меткам.
     * @param X Массив данных (N * D)
     * @param labels Метки кластеров (N)
     * @param new_centroids Выходной массив новых центроидов (K * D)
     */
    virtual void update_centroids(const double* X, int N, int D,
                                   const int* labels, double* new_centroids) = 0;

    int K_;              // Количество кластеров
    int n_iters_;        // Максимальное количество итераций
    double tol_;         // Порог сходимости

    std::vector<double> centroids_;  // Текущие центроиды (K * D)
    std::vector<int> labels_;        // Метки кластеров (N)

    // Агрегированные тайминги за один вызов fit(...)
    double t_assign_total_;
    double t_update_total_;
    double t_iter_total_;
    int n_iters_actual_;  // Реальное количество выполненных итераций
};

