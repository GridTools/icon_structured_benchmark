#include "wrapper.hpp"
#include "validation.hpp"
#include "wrapper_dispatch.hpp"

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_cpu_ifirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 6>> &v2e,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_ifirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_cpu_kfirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 6>> &v2e,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_kfirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_gpu_kloop(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 6>> &v2e,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_kloop, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_gpu_naive(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 6>> &v2e,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_naive, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::vector<double> interpolate_benchmark_unstructured_cpu_ifirst(const std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_unstructured_cpu_kfirst(const std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_unstructured_gpu_kloop(const std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_unstructured_gpu_naive(const std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_cpu_ifirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_cpu_kfirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_gpu_kloop(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_gpu_naive(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<std::vector<WP_TYPE>> verts2cells_validate_unstructured_cpu_kfirst(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 3>> &c2v,
    const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
    const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return verts2cells_validate_gridtools<cpu_kfirst, verts2cells_unstructured>(
        std::make_tuple(c2v, VertexDim, CellDim, KDim, p_vert_u_in, p_vert_v_in, ptr_coeff_1, ptr_coeff_2));
}

std::vector<std::vector<WP_TYPE>> verts2cells_validate_structured_cpu_kfirst(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
    const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return verts2cells_validate_gridtools<cpu_kfirst, verts2cells_structured>(std::make_tuple(VertexDim,
        CellDim,
        KDim,
        longitude_dim,
        latitude_dim,
        halo,
        p_vert_u_in,
        p_vert_v_in,
        ptr_coeff_1,
        ptr_coeff_2));
}

std::vector<std::vector<WP_TYPE>> verts2cells_validate_unstructured_gpu_naive(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 3>> &c2v,
    const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
    const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return verts2cells_validate_gridtools<gpu_naive, verts2cells_unstructured>(
        std::make_tuple(c2v, VertexDim, CellDim, KDim, p_vert_u_in, p_vert_v_in, ptr_coeff_1, ptr_coeff_2));
}

std::vector<std::vector<WP_TYPE>> verts2cells_validate_structured_gpu_naive(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
    const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return verts2cells_validate_gridtools<gpu_naive, verts2cells_structured>(std::make_tuple(VertexDim,
        CellDim,
        KDim,
        longitude_dim,
        latitude_dim,
        halo,
        p_vert_u_in,
        p_vert_v_in,
        ptr_coeff_1,
        ptr_coeff_2));
}

std::vector<std::vector<WP_TYPE>> verts2cells_validate_unstructured_gpu_kloop(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    const std::vector<std::array<index_type, 3>> &c2v,
    const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
    const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return verts2cells_validate_gridtools<gpu_kloop, verts2cells_unstructured>(
        std::make_tuple(c2v, VertexDim, CellDim, KDim, p_vert_u_in, p_vert_v_in, ptr_coeff_1, ptr_coeff_2));
}

std::vector<std::vector<WP_TYPE>> verts2cells_validate_structured_gpu_kloop(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
    const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return verts2cells_validate_gridtools<gpu_kloop, verts2cells_structured>(std::make_tuple(VertexDim,
        CellDim,
        KDim,
        longitude_dim,
        latitude_dim,
        halo,
        p_vert_u_in,
        p_vert_v_in,
        ptr_coeff_1,
        ptr_coeff_2));
}

std::vector<double> verts2cells_benchmark_unstructured_cpu_kfirst(const std::vector<std::array<index_type, 3>> &c2v,
    std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, verts2cells_unstructured>(
        std::make_tuple(c2v, VertexDim, CellDim, KDim), repetitions, dry_runs);
}

std::vector<double> verts2cells_benchmark_structured_cpu_kfirst(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, verts2cells_structured>(
        std::make_tuple(VertexDim, CellDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> verts2cells_benchmark_unstructured_gpu_naive(const std::vector<std::array<index_type, 3>> &c2v,
    std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, verts2cells_unstructured>(
        std::make_tuple(c2v, VertexDim, CellDim, KDim), repetitions, dry_runs);
}

std::vector<double> verts2cells_benchmark_structured_gpu_naive(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_naive, verts2cells_structured>(
        std::make_tuple(VertexDim, CellDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> verts2cells_benchmark_unstructured_gpu_kloop(const std::vector<std::array<index_type, 3>> &c2v,
    std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, verts2cells_unstructured>(
        std::make_tuple(c2v, VertexDim, CellDim, KDim), repetitions, dry_runs);
}

std::vector<double> verts2cells_benchmark_structured_gpu_kloop(std::size_t VertexDim,
    std::size_t CellDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu_kloop, verts2cells_structured>(
        std::make_tuple(VertexDim, CellDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_cpu_ifirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_ifirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_cpu_kfirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_kfirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_gpu_kloop(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_kloop, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_gpu_naive(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    const std::vector<std::vector<WP_TYPE>> &p_e_in,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu_naive, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}
