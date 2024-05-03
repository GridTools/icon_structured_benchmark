#include <vector>

#include "common.hpp"

template <typename T, backend_impl I>
std::vector<std::vector<VP_TYPE>> run_validation(T &benchmark_object) {
#if defined(__CUDACC__)
    if constexpr (I == backend_impl::gpu) {
        cudaDeviceProp device_prop{};
        int current_device{0};
        GT_CUDA_CHECK(cudaGetDevice(&current_device));
        GT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, current_device));
        cudaStream_t stream_persistent_cache;
        GT_CUDA_CHECK(cudaStreamCreate(&stream_persistent_cache));
        timer<backend_impl::gpu> t;
        t.start(stream_persistent_cache);
        benchmark_object.template run<backend_impl::gpu>(device_prop, stream_persistent_cache);
        t.stop(stream_persistent_cache);
        GT_CUDA_CHECK(cudaStreamDestroy(stream_persistent_cache));
    } else {
#endif
        benchmark_object.template run<I>();
#if defined(__CUDACC__)
    }
#endif
    return benchmark_object.get_output();
}
