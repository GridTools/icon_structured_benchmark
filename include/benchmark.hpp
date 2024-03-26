#include <chrono>
#include <vector>

#include "nabla4_structured.hpp"
#include "nabla4_structured_torus.hpp"
#include "nabla4_unstructured.hpp"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

template <typename T, backend_impl I>
std::vector<double> run_benchmark(T &benchmark_object, int repetitions = 101, int dry_runs = 1) {
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

template <typename T, backend_impl I>
std::vector<double> run_benchmark(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions = 101,
    int dry_runs = 1) {
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        T benchmark_object{e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        T benchmark_object{e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
        const auto start = high_resolution_clock::now();
        benchmark_object.template run<I>();
        const auto end = high_resolution_clock::now();
        runtimes.push_back(duration<double>(end - start).count());
    }
    return runtimes;
}

template <typename T, backend_impl I>
std::vector<double> run_benchmark(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions = 101,
    int dry_runs = 1) {
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        T benchmark_object{CellDim, VertexDim, EdgeDim, KDim, ECVDim};
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        T benchmark_object{CellDim, VertexDim, EdgeDim, KDim, ECVDim};
        const auto start = high_resolution_clock::now();
        benchmark_object.template run<I>();
        const auto end = high_resolution_clock::now();
        runtimes.push_back(duration<double>(end - start).count());
    }
    return runtimes;
}

template <typename T, backend_impl I>
std::vector<double> run_benchmark(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions = 101,
    int dry_runs = 1) {
    for (int dry_run{}; dry_run < dry_runs; ++dry_run) {
        T benchmark_object{CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim};
        benchmark_object.template run<I>();
    }
    std::vector<double> runtimes;
    for (int rep{}; rep < repetitions; ++rep) {
        T benchmark_object{CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim};
        const auto start = high_resolution_clock::now();
        benchmark_object.template run<I>();
        const auto end = high_resolution_clock::now();
        runtimes.push_back(duration<double>(end - start).count());
    }
    return runtimes;
}
