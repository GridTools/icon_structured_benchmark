#include "wrapper.hpp"
#include "validation.hpp"
#include "wrapper_dispatch.hpp"

std::vector<double> nabla4_benchmark_structured_simple_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<naive, nabla4_structured_simple>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_ifirst, nabla4_structured_simple>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_kfirst, nabla4_structured_simple>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_naive(std::size_t CellDim,
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
    return nabla4_validate_vector<naive, nabla4_structured_simple>(std::make_tuple(CellDim,
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

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_cpu_ifirst(std::size_t CellDim,
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
    return nabla4_validate_vector<cpu_ifirst, nabla4_structured_simple>(std::make_tuple(CellDim,
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

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_cpu_kfirst(std::size_t CellDim,
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
    return nabla4_validate_vector<cpu_kfirst, nabla4_structured_simple>(std::make_tuple(CellDim,
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

std::vector<double> nabla4_benchmark_structured_torus_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<naive, nabla4_structured_torus>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_ifirst, nabla4_structured_torus>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_ifirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_structured_torus_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_kfirst, nabla4_structured_torus>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_kfirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_structured_torus_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_vertical_benchmark_structured_torus_gpu_kloop_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_vertical_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu_kloop_cutile_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, nabla4_structured_torus_cutile_halo>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu_naive_cutile_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_structured_torus_cutile_halo>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_vertical_benchmark_structured_torus_gpu_naive_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, nabla4_vertical_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<naive, nabla4_structured_torus>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_ifirst, nabla4_structured_torus>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_ifirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_ifirst, nabla4_structured_torus_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_ifirst_gridtools_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_ifirst, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_kfirst, nabla4_structured_torus>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_kfirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    const std::vector<std::vector<VP_TYPE>> &u_vert,
    const std::vector<std::vector<VP_TYPE>> &v_vert,
    const std::vector<WP_TYPE> &primal_normal_vert_v1,
    const std::vector<WP_TYPE> &primal_normal_vert_v2,
    const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    const std::vector<WP_TYPE> &inv_vert_vert_length,
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_kfirst, nabla4_structured_torus_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_kfirst_gridtools_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_kfirst, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu_kloop_gridtools_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_kloop, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_vertical_validate_structured_torus_gpu_kloop_gridtools_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_kloop, nabla4_vertical_structured_torus_halo_gt>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu_kloop_cutile_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_kloop, nabla4_structured_torus_cutile_halo>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu_naive_cutile_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_naive, nabla4_structured_torus_cutile_halo>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu_naive_gridtools_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_naive, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_vertical_validate_structured_torus_gpu_naive_gridtools_halo(index_type CellDim,
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
    const std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu_naive, nabla4_vertical_structured_torus_halo_gt>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}
