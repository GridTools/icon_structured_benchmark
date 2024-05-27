#include <vector>

#include "copy_gridtools.hpp"
#include "copy_neighbor_gridtools.hpp"
#include "nabla4_structured_simple.hpp"
#include "nabla4_structured_torus.hpp"
#include "nabla4_structured_torus_gridtools.hpp"
#include "nabla4_structured_torus_gridtools_halo.hpp"
#include "nabla4_unstructured.hpp"
#include "nabla4_unstructured_gridtools.hpp"
#include "timer.hpp"

template <typename T, backend_impl I>
std::vector<double> run_benchmark(T &benchmark_object, int repetitions, int dry_runs) {
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
    T benchmark_object{std::apply([](auto &&...args) { return T{std::forward<decltype(args)>(args)...}; }, args)};
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
