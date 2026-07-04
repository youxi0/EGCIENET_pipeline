// include/utils/Timer.h
#pragma once

#include <chrono>
#include <cstdint>

class ScopeTimer {
public:
    explicit ScopeTimer():start_(std::chrono::high_resolution_clock::now()) {}

    void reset(){
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto cost = std::chrono::duration<double, std::milli>(end - start_).count();
        // std::cout << "[TIME] " << name_ << ": " << cost << " ms" << std::endl;
        return cost;
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};