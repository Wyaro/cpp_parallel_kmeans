#include "../include/dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>

Dataset::Dataset(const std::string& data_path) {
    load_data(data_path);
}

void Dataset::load_data(const std::string& data_path) {
    std::ifstream file(data_path);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + data_path);
    }

    std::string line;
    std::vector<std::vector<double>> centroids;
    std::vector<std::vector<double>> points;
    std::vector<int> labels;

    // Читаем метаданные из первой строки (начинается с #)
    if (std::getline(file, line) && line.find("#") == 0) {
        // Извлекаем K из JSON метаданных
        size_t k_pos = line.find("\"K\":");
        if (k_pos != std::string::npos) {
            size_t start = line.find_first_of("0123456789", k_pos);
            size_t end = line.find_first_not_of("0123456789", start);
            K_ = std::stoi(line.substr(start, end - start));
        }

        size_t d_pos = line.find("\"D\":");
        if (d_pos != std::string::npos) {
            size_t start = line.find_first_of("0123456789", d_pos);
            size_t end = line.find_first_not_of("0123456789", start);
            D_ = std::stoi(line.substr(start, end - start));
        }
    }

    // Читаем центроиды (первые K строк с метками 0..K-1)
    int centroids_read = 0;
    while (centroids_read < K_ && std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t")); // Убираем пробелы в начале
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int label;
        iss >> label;

        if (label < K_) {
            std::vector<double> coords;
            double val;
            while (iss >> val) {
                coords.push_back(val);
            }
            if (coords.size() == D_) {
                centroids.push_back(coords);
                centroids_read++;
            }
        }
    }

    // Пропускаем пустую строку
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        break;
    }

    // Читаем точки данных
    do {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        int label;
        iss >> label;

        std::vector<double> coords;
        double val;
        while (iss >> val) {
            coords.push_back(val);
        }

        if (coords.size() == D_) {
            points.push_back(coords);
            labels.push_back(label);
        }
    } while (std::getline(file, line));

    // Преобразуем в плоские массивы
    N_ = static_cast<int>(points.size());
    X_.resize(N_ * D_);
    labels_true_.resize(N_);
    initial_centroids_.resize(K_ * D_);

    for (int i = 0; i < N_; ++i) {
        for (int j = 0; j < D_; ++j) {
            X_[i * D_ + j] = points[i][j];
        }
        labels_true_[i] = labels[i];
    }

    for (int i = 0; i < K_; ++i) {
        for (int j = 0; j < D_; ++j) {
            initial_centroids_[i * D_ + j] = centroids[i][j];
        }
    }

    std::cout << "Dataset loaded: N=" << N_ << ", D=" << D_ << ", K=" << K_ << std::endl;
}

