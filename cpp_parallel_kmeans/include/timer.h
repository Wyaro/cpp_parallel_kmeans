#pragma once

#include <chrono>

/**
 * Высокоточный таймер для измерения производительности.
 * Использует std::chrono::steady_clock для измерений, не зависящих от системных часов.
 */
class Timer {
public:
    Timer() : start_(), end_(), elapsed_(0.0) {}

    void start() {
        start_ = std::chrono::steady_clock::now();
    }

    void stop() {
        end_ = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
        elapsed_ = duration.count() / 1000000.0; // Преобразуем в секунды
    }

    double elapsed() const {
        return elapsed_;
    }

    void reset() {
        elapsed_ = 0.0;
    }

private:
    std::chrono::steady_clock::time_point start_;
    std::chrono::steady_clock::time_point end_;
    double elapsed_;
};

