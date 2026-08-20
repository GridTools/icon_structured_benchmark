#include "wrapper.hpp"
#include "validation.hpp"
#include "wrapper_dispatch.hpp"

/// TODO: See if we can avoid declaring nabla4_benchmark_* and use templates.
/// Problem with templates are python bindings
nabla4_data<Data::ifirst> get_nabla4_benchmark_validation_data(const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim) {
    nabla4_unstructured<Data::ifirst> nabla4_benchmark_object{e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
    run_benchmark<nabla4_unstructured<Data::ifirst>, naive>(nabla4_benchmark_object, 1, 0);
    return nabla4_benchmark_object.get_validation_data();
}

std::vector<double> nabla4_benchmark_unstructured_naive(const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<naive, nabla4_unstructured>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_ifirst(const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_ifirst, nabla4_unstructured>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_ifirst_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_kfirst(const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_kfirst, nabla4_unstructured>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_kfirst_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu_kloop_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_vertical_benchmark_unstructured_gpu_kloop_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_vertical_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu_naive_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_vertical_benchmark_unstructured_gpu_naive_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_vertical_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu_kloop_cutile(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_unstructured_cutile>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu_naive_cutile(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_unstructured_cutile>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_naive(
    const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<naive, nabla4_unstructured>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_ifirst(
    const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_ifirst, nabla4_unstructured>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_ifirst_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_ifirst, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_kfirst(
    const std::vector<std::array<std::size_t, 4>> &e2c2v,
    const std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_kfirst, nabla4_unstructured>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_kfirst_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_kfirst, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu_kloop_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_kloop, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_vertical_validate_unstructured_gpu_kloop_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_kloop, nabla4_vertical_unstructured_gt>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu_naive_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_naive, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_vertical_validate_unstructured_gpu_naive_gridtools(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_naive, nabla4_vertical_unstructured_gt>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu_kloop_cutile(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_kloop, nabla4_unstructured_cutile>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu_naive_cutile(
    const std::vector<std::array<index_type, 4>> &e2c2v,
    const std::vector<std::array<index_type, 4>> &e2ecv,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_naive, nabla4_unstructured_cutile>(std::make_tuple(e2c2v,
        e2ecv,
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
        inv_primal_edge_length));
}
