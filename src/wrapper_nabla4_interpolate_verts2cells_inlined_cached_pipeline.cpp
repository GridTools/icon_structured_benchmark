#include "wrapper.hpp"
#include "validation.hpp"
#include "wrapper_dispatch.hpp"

std::vector<double> nabla4_interpolate_verts2cells_benchmark_structured_gpu_kloop_inlined_cached_pipeline(
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<std::vector<WP_TYPE>> nabla4_interpolate_verts2cells_validate_structured_gpu_kloop_inlined_cached_pipeline(
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2,
    const std::vector<std::vector<WP_TYPE>> &ptr_c_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_c_coeff_2) {
    return verts2cells_validate_gridtools<gpu_kloop, nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline>(
        std::make_tuple(CellDim,
            VertexDim,
            EdgeDim,
            KDim,
            ECVDim,
            longitude_dim,
            latitude_dim,
            halo,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            ptr_coeff_1,
            ptr_coeff_2,
            ptr_c_coeff_1,
            ptr_c_coeff_2));
}
