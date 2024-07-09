#pragma once

#if defined(__CUDACC__)
#include <cuda_runtime.h>

#include "gridtools/common/cuda_util.hpp"
#endif

#include "common.hpp"

#include <chrono>
using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <backend_impl I>
class timer {
#if not defined(__CUDACC__)
    static_assert(I != backend_impl::gpu_kloop || I != backend_impl::gpu_naive, "GPU backend not supported");
#else
    cudaEvent_t start_event, stop_event;
#endif
    high_resolution_clock::time_point start_time, stop_time;

  public:
    timer() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(cudaEventCreate(&start_event));
            cudaEventCreate(&stop_event);
        }
#endif
    }
    inline void start() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(cudaEventRecord(start_event, 0));
        } else {
#endif
            start_time = high_resolution_clock::now();
#if defined(__CUDACC__)
        }
#endif
    }
    inline void stop() {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(cudaEventRecord(stop_event, 0));
            GT_CUDA_CHECK(cudaEventSynchronize(stop_event));
        } else {
#endif
            stop_time = high_resolution_clock::now();
#if defined(__CUDACC__)
        }
#endif
    }
    inline double elapsed() const {
#if defined(__CUDACC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            float elapsed_time{};
            GT_CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event, stop_event));
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
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(cudaEventDestroy(start_event));
            GT_CUDA_CHECK(cudaEventDestroy(stop_event));
        }
#endif
    }
};
