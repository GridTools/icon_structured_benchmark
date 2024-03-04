#include "benchmark.hpp"
#include "nabla4_unstructured.hpp"

std::vector<long long> nabla4_benchmark(std::vector<std::vector<double>> e2c2v, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
    return run_benchmark(nabla4_benchmark_object, 5);
}
