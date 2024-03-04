#include "benchmark.hpp"
#include "nabla4_unstructured.hpp"

std::vector<long long> nabla4_benchmark(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions);
