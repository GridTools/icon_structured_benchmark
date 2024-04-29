#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include "common.hpp"

#include <chrono>
using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <backend_impl I>
class timer {
#if not defined(__CUDACC__)
    static_assert(I == backend_impl::gpu, "GPU backend not supported");
#else
    cudaEvent_t start_event, stop_event;
#endif
    high_resolution_clock::time_point start_time, stop_time;

  public:
    timer() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu) {
            cudaEventCreate(&start_event);
            cudaEventCreate(&stop_event);
        }
#endif
    }
    inline void start() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu) {
            cudaEventRecord(start_event, 0);
        } else {
#endif
            start_time = high_resolution_clock::now();
#if defined(__CUDACC__)
        }
#endif
    }
    inline void stop() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu) {
            cudaEventRecord(stop_event, 0);
            cudaEventSynchronize(stop_event);
        } else {
#endif
            stop_time = high_resolution_clock::now();
#if defined(__CUDACC__)
        }
#endif
    }
    inline double elapsed() const {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu) {
            float elapsed_time{};
            cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
            return static_cast<double>(elapsed_time / 1000.0);
        } else {
#endif
            return duration<double>(stop_time - start_time).count();
#if defined(__CUDACC__)
        }
#endif
    }
    ~timer() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
        }
#endif
    }
};
