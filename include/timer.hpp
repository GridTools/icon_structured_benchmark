#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

#include "gridtools/common/cuda_util.hpp"
#endif

#include "common.hpp"

#include <chrono>
using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <backend_impl I>
class timer {
#if not defined(__HIPCC__)
    static_assert(I != backend_impl::gpu_kloop || I != backend_impl::gpu_naive, "GPU backend not supported");
#else
    hipEvent_t start_event, stop_event;
#endif
    high_resolution_clock::time_point start_time, stop_time;

  public:
    timer() {
#if defined(__HIPCC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(hipEventCreate(&start_event));
            hipEventCreate(&stop_event);
        }
#endif
    }
    inline void start() {
#if defined(__HIPCC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(hipEventRecord(start_event, 0));
        } else {
#endif
            start_time = high_resolution_clock::now();
#if defined(__HIPCC__)
        }
#endif
    }
    inline void stop() {
#if defined(__HIPCC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(hipEventRecord(stop_event, 0));
            GT_CUDA_CHECK(hipEventSynchronize(stop_event));
        } else {
#endif
            stop_time = high_resolution_clock::now();
#if defined(__HIPCC__)
        }
#endif
    }
    inline double elapsed() const {
#if defined(__HIPCC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            float elapsed_time{};
            GT_CUDA_CHECK(hipEventElapsedTime(&elapsed_time, start_event, stop_event));
            return static_cast<double>(elapsed_time / 1000.0);
        } else {
#endif
            return duration<double>(stop_time - start_time).count();
#if defined(__HIPCC__)
        }
#endif
    }
    ~timer() {
#if defined(__HIPCC__)
        if constexpr (I == backend_impl::gpu_kloop || I == backend_impl::gpu_naive) {
            GT_CUDA_CHECK(hipEventDestroy(start_event));
            GT_CUDA_CHECK(hipEventDestroy(stop_event));
        }
#endif
    }
};
