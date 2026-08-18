#include "wrapper.hpp"
#include "validation.hpp"
#include "wrapper_dispatch.hpp"

std::vector<double> nabla4_interpolate_benchmark_gpu_kloop_roofline(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_interpolate_roofline>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_vertical_interpolate_roofline>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}
