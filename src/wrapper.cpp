#include "benchmark.hpp"
#include "nabla4_unstructured.hpp"

template<backend_impl I>
std::vector<double> nabla4_benchmark(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
    return run_benchmark<nabla4_unstructured, I>(nabla4_benchmark_object, repetitions);
}

std::vector<double> nabla4_benchmark_naive(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101) {
    return nabla4_benchmark<naive>(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions);
}

std::vector<double> nabla4_benchmark_cpu_ifirst(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101) {
    return nabla4_benchmark<cpu_ifirst>(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions);
}

std::vector<double> nabla4_benchmark_cpu_kfirst(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101) {
    return nabla4_benchmark<cpu_kfirst>(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions);
}

std::vector<double> nabla4_benchmark_gpu(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101) {
    return nabla4_benchmark<gpu>(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions);
}
