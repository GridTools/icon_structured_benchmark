#include <vector>

#include "interpolate_structured_gridtools.hpp"
#include "interpolate_unstructured_gridtools.hpp"
#include "nabla4_interpolate_roofline.hpp"
#include "nabla4_interpolate_structured_inlined.hpp"
#include "nabla4_interpolate_structured_inlined_cached.hpp"
#include "nabla4_interpolate_structured_separate.hpp"
#include "nabla4_interpolate_unstructured_inlined.hpp"
#include "nabla4_interpolate_unstructured_inlined_cached.hpp"
#include "nabla4_interpolate_unstructured_inlined_v2v.hpp"
#include "nabla4_interpolate_unstructured_separate.hpp"
#include "nabla4_interpolate_verts2cells_structured_separate.hpp"
#include "nabla4_interpolate_verts2cells_unstructured_separate.hpp"
#include "nabla4_structured_simple.hpp"
#include "nabla4_structured_torus.hpp"
#include "nabla4_structured_torus_gridtools.hpp"
#include "nabla4_structured_torus_gridtools_halo.hpp"
#include "nabla4_unstructured.hpp"
#include "nabla4_unstructured_gridtools.hpp"
#include "nabla4_vertical_interpolate_roofline.hpp"
#include "nabla4_vertical_interpolate_structured_inlined.hpp"
#include "nabla4_vertical_interpolate_structured_inlined_cached.hpp"
#include "nabla4_vertical_interpolate_structured_separate.hpp"
#include "nabla4_vertical_interpolate_unstructured_inlined.hpp"
#include "nabla4_vertical_interpolate_unstructured_inlined_v2v.hpp"
#include "nabla4_vertical_interpolate_unstructured_separate.hpp"
#include "nabla4_vertical_structured_torus_gridtools_halo.hpp"
#include "nabla4_vertical_unstructured_gridtools.hpp"
#include "timer.hpp"
#include "verts2cells_structured_gridtools.hpp"
#include "verts2cells_unstructured_gridtools.hpp"

template <typename T, backend_impl I>
std::vector<double> run_benchmark(T &benchmark_object, int repetitions, int dry_runs) {
#if defined(__CUDACC__)
    GT_CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        timer<I> t;
        t.start();
        benchmark_object.template run<I>();
        t.stop();
        runtimes.push_back(t.elapsed());
    }
    return runtimes;
}

template <typename T, backend_impl I, typename... Args>
std::vector<double> run_benchmark(std::tuple<Args...> &&args, int repetitions, int dry_runs) {
#if defined(__CUDACC__)
    GT_CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
#endif
    T benchmark_object{std::make_from_tuple<T>(args)};
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        timer<I> t;
        t.start();
        benchmark_object.template run<I>();
        t.stop();
        runtimes.push_back(t.elapsed());
    }
    return runtimes;
}
