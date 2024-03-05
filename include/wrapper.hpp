#include "benchmark.hpp"
#include "nabla4_unstructured.hpp"

template<backend_impl I>
std::vector<double> nabla4_benchmark(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101, int dry_runs = 1);

std::vector<double> nabla4_benchmark_naive(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101, int dry_runs = 1);

std::vector<double> nabla4_benchmark_cpu_ifirst(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101, int dry_runs = 1);

std::vector<double> nabla4_benchmark_cpu_kfirst(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101, int dry_runs = 1);

std::vector<double> nabla4_benchmark_gpu(std::vector<std::vector<std::size_t>> e2c2v, std::vector<std::vector<std::size_t>> e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, int repetitions = 101, int dry_runs = 1);
