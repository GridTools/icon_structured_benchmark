#include <vector>

#include "nabla4_structured_simple.hpp"
#include "nabla4_structured_torus.hpp"
#include "nabla4_structured_torus_gridtools.hpp"
#include "nabla4_structured_torus_gridtools_halo.hpp"
#include "nabla4_unstructured.hpp"
#include "nabla4_unstructured_gridtools.hpp"
#include "timer.hpp"

template <typename T, backend_impl I>
std::vector<double> run_benchmark(T &benchmark_object, int repetitions, int dry_runs) {
    // for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
    //     benchmark_object.template run<I>();
    // }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
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
            runtimes.push_back(t.elapsed());
            GT_CUDA_CHECK(cudaStreamDestroy(stream_persistent_cache));
        } else {
#endif
            timer<I> t;
            t.start();
            benchmark_object.template run<I>();
            t.stop();
            runtimes.push_back(t.elapsed());
#if defined(__CUDACC__)
        }
#endif
    }
    return runtimes;
}

template <typename T, backend_impl I, typename... Args>
std::vector<double> run_benchmark(std::tuple<Args...> &&args, int repetitions, int dry_runs) {
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        T benchmark_object{std::apply([](auto &&...args) { return T{std::forward<decltype(args)>(args)...}; }, args)};
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        T benchmark_object{std::apply([](auto &&...args) { return T{std::forward<decltype(args)>(args)...}; }, args)};
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
            runtimes.push_back(t.elapsed());
            GT_CUDA_CHECK(cudaStreamDestroy(stream_persistent_cache));
        } else {
#endif
            timer<I> t;
            t.start();
            benchmark_object.template run<I>();
            t.stop();
            runtimes.push_back(t.elapsed());
#if defined(__CUDACC__)
        }
#endif
    }
    return runtimes;
}
