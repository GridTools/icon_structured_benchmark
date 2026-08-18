#include "wrapper.hpp"
#include "validation.hpp"
#include "wrapper_dispatch.hpp"

std::vector<double> nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_vertical_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_vertical_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined_v2v(const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_ifirst, nabla4_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v,
            e2ecv,
            v2e,
            CellDim,
            VertexDim,
            EdgeDim,
            KDim,
            ECVDim,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            ptr_coeff_1,
            ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined_v2v(const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_kfirst, nabla4_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v,
            e2ecv,
            v2e,
            CellDim,
            VertexDim,
            EdgeDim,
            KDim,
            ECVDim,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            ptr_coeff_1,
            ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
nabla4_interpolate_validate_unstructured_gpu_naive_inlined_v2v(const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_naive, nabla4_interpolate_unstructured_inlined_v2v>(std::make_tuple(e2c2v,
        e2ecv,
        v2e,
        CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        ptr_coeff_1,
        ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_naive, nabla4_vertical_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v,
            e2ecv,
            v2e,
            CellDim,
            VertexDim,
            EdgeDim,
            KDim,
            ECVDim,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            ptr_coeff_1,
            ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_v2v(const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_kloop, nabla4_interpolate_unstructured_inlined_v2v>(std::make_tuple(e2c2v,
        e2ecv,
        v2e,
        CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length,
        ptr_coeff_1,
        ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined_v2v(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    const std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_kloop, nabla4_vertical_interpolate_unstructured_inlined_v2v>(
        std::make_tuple(e2c2v,
            e2ecv,
            v2e,
            CellDim,
            VertexDim,
            EdgeDim,
            KDim,
            ECVDim,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            ptr_coeff_1,
            ptr_coeff_2));
}
