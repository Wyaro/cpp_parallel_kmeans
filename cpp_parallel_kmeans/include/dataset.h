#pragma once

#include <vector>
#include <string>
#include <memory>

/**
 * Класс для загрузки и представления датасетов для экспериментов K-means.
 * Загружает данные из текстовых файлов в формате, созданном DatasetGenerator.
 */
class Dataset {
public:
    Dataset(const std::string& data_path);

    // Размеры данных
    int N() const { return N_; }  // Количество точек
    int D() const { return D_; }   // Размерность
    int K() const { return K_; }   // Количество кластеров

    // Доступ к данным
    const std::vector<double>& X() const { return X_; }  // Данные (N * D)
    const std::vector<int>& labels_true() const { return labels_true_; }  // Истинные метки
    const std::vector<double>& initial_centroids() const { return initial_centroids_; }  // Начальные центроиды (K * D)

    // Получить точку по индексу
    const double* get_point(int idx) const {
        return &X_[idx * D_];
    }

    // Получить центроид по индексу
    const double* get_centroid(int idx) const {
        return &initial_centroids_[idx * D_];
    }

private:
    void load_data(const std::string& data_path);

    int N_;
    int D_;
    int K_;
    std::vector<double> X_;                    // Данные: N точек по D измерений
    std::vector<int> labels_true_;             // Истинные метки кластеров
    std::vector<double> initial_centroids_;    // Начальные центроиды: K центроидов по D измерений
};

