#include <chrono>
#include <vector>

#include "nabla4_unstructured.hpp"

using std::chrono::high_resolution_clock;
using std::chrono::duration;

template<typename T, backend_impl I>
std::vector<double> run_benchmark(T& benchmark_object, int repetitions = 101) {
    /// TODO: Add dry runs
    std::vector<double> runtimes;
    for (auto rep = 0; rep < repetitions; ++rep) {
        const auto start = high_resolution_clock::now();
        benchmark_object.template run<I>();
        const auto end = high_resolution_clock::now();
        runtimes.push_back(duration<double>(end - start).count());
    }
    return runtimes;
}
