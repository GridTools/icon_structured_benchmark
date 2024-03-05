#include <chrono>
#include <vector>



using std::chrono::high_resolution_clock;
using std::chrono::nanoseconds;
using std::chrono::duration_cast;

template<typename T>
std::vector<long long> run_benchmark(T& benchmark_object, int repetitions = 101) {
    /// TODO: Add dry runs
    std::vector<long long> runtimes;
    for (auto rep = 0; rep < repetitions; ++rep) {
        const auto start = high_resolution_clock::now();
        benchmark_object.run();
        const auto end = high_resolution_clock::now();
        runtimes.push_back(duration_cast<nanoseconds>(end - start).count());
    }
    return runtimes;
}
