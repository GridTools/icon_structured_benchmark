#include <chrono>
#include <vector>

#include "nabla4_unstructured.hpp"
#include "nabla4_structured.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

template<typename T, backend_impl I>
std::vector<double> run_benchmark(T& benchmark_object, int repetitions = 101, int dry_runs = 1) {
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        const auto start = high_resolution_clock::now();
        benchmark_object.template run<I>();
        const auto end = high_resolution_clock::now();
        runtimes.push_back(duration<double>(end - start).count());
    }
    return runtimes;
}
