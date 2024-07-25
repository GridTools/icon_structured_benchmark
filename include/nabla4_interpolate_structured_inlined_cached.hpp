#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>

template <typename T>
struct nabla4_interpolate_structured_inlined_cached {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;

  public:
    nabla4_interpolate_structured_inlined_cached(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0),
          interpolate_data(VertexDim, EdgeDim, KDim, y_dim, x_dim, halo + 1, nabla4_data.get_output_gt()){};

    nabla4_interpolate_structured_inlined_cached(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : nabla4_data(CellDim,
              VertexDim,
              EdgeDim,
              KDim,
              ECVDim,
              y_dim,
              x_dim,
              0,
              u_vert,
              v_vert,
              primal_normal_vert_v1,
              primal_normal_vert_v2,
              z_nabla2_e,
              inv_vert_vert_length,
              inv_primal_edge_length),
          interpolate_data(VertexDim,
              EdgeDim,
              KDim,
              y_dim,
              x_dim,
              halo + 1,
              nabla4_data.get_output_gt(),
              ptr_coeff_1,
              ptr_coeff_2){};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    void run_gpu_naive_helper();

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_naive<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_cached_naive =
    get_block_dims_structured_nabla_interpol_inlined_cached_naive<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_inlined_cached_naive.size)
    run_gpu_naive_nabla4_interpolate_inlined_cached_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        index_type inner_domain_size,
        index_type outer_domain_size,
        index_type total_edges,
        index_type shared_mem_inner_domain,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim) {
        return;
    }
    // printf("blockIdx.x: %d, blockIdx.y: %d\n", blockIdx.x, blockIdx.y);
    extern __shared__ WP_TYPE z_nabla4_e2[];
    // printf("i-1: %d, i+blockDim.x+1: %d, x_dim - halo + 2: %d, j-1: %d, j+blockDim.y+2: %d, y_dim - halo + 3: %d\n", i - 1, i + blockDim.x + 1, x_dim - halo + 2, j - 1, j + blockDim.y + 2, y_dim - halo + 3);
    for (auto i_internal_block{i - 1}; i_internal_block < i + blockDim.x + 1 && i_internal_block < x_dim - halo; i_internal_block += min(blockDim.x, x_dim - 2 * halo)) {
        for (auto j_internal_block{j - 1}; j_internal_block < j + blockDim.y + 2 && j_internal_block < y_dim - halo + 1; j_internal_block += min(blockDim.y, y_dim - 2 * halo)) {
            const index_type i_j = j_internal_block * x_dim + i_internal_block;
            const index_type i_jp1 = (j_internal_block + 1) * x_dim + i_internal_block;
            const index_type im1_jp1 = (j_internal_block + 1) * x_dim + i_internal_block - 1;
            const index_type ip1_j = j_internal_block * x_dim + i_internal_block + 1;
            const index_type ip1_jm1 = (j_internal_block - 1) * x_dim + i_internal_block + 1;
            const index_type i_jm1 = (j_internal_block - 1) * x_dim + i_internal_block;
            const index_type E2C2V_0[3] = {i_j, i_j, i_j};
            const index_type E2C2V_1[3] = {i_jp1, ip1_j, ip1_jm1};
            const index_type E2C2V_2[3] = {im1_jp1, i_jp1, ip1_j};
            const index_type E2C2V_3[3] = {ip1_j, ip1_jm1, i_jm1};
            const index_type E2ECV_0[3] = {i_j, i_j + outer_domain_size, i_j + 2 * outer_domain_size};
            const index_type E2ECV_1[3] = {E2ECV_0[0] + total_edges,
                E2ECV_0[1] + total_edges,
                E2ECV_0[2] + total_edges};
            const index_type E2ECV_2[3] = {E2ECV_1[0] + total_edges,
                E2ECV_1[1] + total_edges,
                E2ECV_1[2] + total_edges};
            const index_type E2ECV_3[3] = {E2ECV_2[0] + total_edges,
                E2ECV_2[1] + total_edges,
                E2ECV_2[2] + total_edges};
            const index_type edge_index = i_j;
            // printf("global edge_index: %d, E2C2V_0[0]: %d, E2C2V_1[0]: %d, E2C2V_2[0]: %d, E2C2V_3[0]: %d\n", edge_index, E2C2V_0[0], E2C2V_1[0], E2C2V_2[0], E2C2V_3[0]);
            // printf("global edge_index: %d, E2C2V_0[1]: %d, E2C2V_1[1]: %d, E2C2V_2[1]: %d, E2C2V_3[1]: %d\n", edge_index + outer_domain_size, E2C2V_0[1], E2C2V_1[1], E2C2V_2[1], E2C2V_3[1]);
            // printf("global edge_index: %d, E2C2V_0[2]: %d, E2C2V_1[2]: %d, E2C2V_2[2]: %d, E2C2V_3[2]: %d\n", edge_index + 2 * outer_domain_size, E2C2V_0[2], E2C2V_1[2], E2C2V_2[2], E2C2V_3[2]);
            // printf("global edge_index: %d, E2ECV_0[0]: %d, E2ECV_1[0]: %d, E2ECV_2[0]: %d, E2ECV_3[0]: %d\n", edge_index, E2ECV_0[0], E2ECV_1[0], E2ECV_2[0], E2ECV_3[0]);
            // printf("global edge_index: %d, E2ECV_0[1]: %d, E2ECV_1[1]: %d, E2ECV_2[1]: %d, E2ECV_3[1]: %d\n", edge_index + outer_domain_size, E2ECV_0[1], E2ECV_1[1], E2ECV_2[1], E2ECV_3[1]);
            // printf("global edge_index: %d, E2ECV_0[2]: %d, E2ECV_1[2]: %d, E2ECV_2[2]: %d, E2ECV_3[2]: %d\n", edge_index + 2 * outer_domain_size, E2ECV_0[2], E2ECV_1[2], E2ECV_2[2], E2ECV_3[2]);
            const WP_TYPE primal_normal_vert_v1_0[3] = {primal_normal_vert_v1_gt_tv(E2ECV_0[0]),
                primal_normal_vert_v1_gt_tv(E2ECV_0[1]),
                primal_normal_vert_v1_gt_tv(E2ECV_0[2])};
            const WP_TYPE primal_normal_vert_v1_1[3] = {primal_normal_vert_v1_gt_tv(E2ECV_1[0]),
                primal_normal_vert_v1_gt_tv(E2ECV_1[1]),
                primal_normal_vert_v1_gt_tv(E2ECV_1[2])};
            const WP_TYPE primal_normal_vert_v1_2[3] = {primal_normal_vert_v1_gt_tv(E2ECV_2[0]),
                primal_normal_vert_v1_gt_tv(E2ECV_2[1]),
                primal_normal_vert_v1_gt_tv(E2ECV_2[2])};
            const WP_TYPE primal_normal_vert_v1_3[3] = {primal_normal_vert_v1_gt_tv(E2ECV_3[0]),
                primal_normal_vert_v1_gt_tv(E2ECV_3[1]),
                primal_normal_vert_v1_gt_tv(E2ECV_3[2])};
            const WP_TYPE primal_normal_vert_v2_0[3] = {primal_normal_vert_v2_gt_tv(E2ECV_0[0]),
                primal_normal_vert_v2_gt_tv(E2ECV_0[1]),
                primal_normal_vert_v2_gt_tv(E2ECV_0[2])};
            const WP_TYPE primal_normal_vert_v2_1[3] = {primal_normal_vert_v2_gt_tv(E2ECV_1[0]),
                primal_normal_vert_v2_gt_tv(E2ECV_1[1]),
                primal_normal_vert_v2_gt_tv(E2ECV_1[2])};
            const WP_TYPE primal_normal_vert_v2_2[3] = {primal_normal_vert_v2_gt_tv(E2ECV_2[0]),
                primal_normal_vert_v2_gt_tv(E2ECV_2[1]),
                primal_normal_vert_v2_gt_tv(E2ECV_2[2])};
            const WP_TYPE primal_normal_vert_v2_3[3] = {primal_normal_vert_v2_gt_tv(E2ECV_3[0]),
                primal_normal_vert_v2_gt_tv(E2ECV_3[1]),
                primal_normal_vert_v2_gt_tv(E2ECV_3[2])};
            const WP_TYPE inv_vert_vert_length_sqr[3] = {
                inv_vert_vert_length_gt_tv(edge_index) * inv_vert_vert_length_gt_tv(edge_index),
                inv_vert_vert_length_gt_tv(edge_index + outer_domain_size) *
                    inv_vert_vert_length_gt_tv(edge_index + outer_domain_size),
                inv_vert_vert_length_gt_tv(edge_index + 2 * outer_domain_size) *
                    inv_vert_vert_length_gt_tv(edge_index + 2 * outer_domain_size)};
            const WP_TYPE inv_primal_edge_length_sqr[3] = {
                inv_primal_edge_length_gt_tv(edge_index) * inv_primal_edge_length_gt_tv(edge_index),
                inv_primal_edge_length_gt_tv(edge_index + outer_domain_size) *
                    inv_primal_edge_length_gt_tv(edge_index + outer_domain_size),
                inv_primal_edge_length_gt_tv(edge_index + 2 * outer_domain_size) *
                    inv_primal_edge_length_gt_tv(edge_index + 2 * outer_domain_size)};
#pragma unroll
            for (auto color{0}; color < 3; ++color) {
                const auto E2C2V_0_c = E2C2V_0[color];
                const auto E2C2V_1_c = E2C2V_1[color];
                const auto E2C2V_2_c = E2C2V_2[color];
                const auto E2C2V_3_c = E2C2V_3[color];
                const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0_c, k_index) * primal_normal_vert_v1_0[color] +
                                            v_vert_gt_tv(E2C2V_0_c, k_index) * primal_normal_vert_v2_0[color] +
                                            u_vert_gt_tv(E2C2V_1_c, k_index) * primal_normal_vert_v1_1[color] +
                                            v_vert_gt_tv(E2C2V_1_c, k_index) * primal_normal_vert_v2_1[color];
                const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2_c, k_index) * primal_normal_vert_v1_2[color] +
                                            v_vert_gt_tv(E2C2V_2_c, k_index) * primal_normal_vert_v2_2[color] +
                                            u_vert_gt_tv(E2C2V_3_c, k_index) * primal_normal_vert_v1_3[color] +
                                            v_vert_gt_tv(E2C2V_3_c, k_index) * primal_normal_vert_v2_3[color];
                // printf("i_internal_block: %d, i, %d, j_internal_block: %d, j: %d, blockDim.x: %d, color: %d, shared_mem_inner_domain: %d\n", i_internal_block, i, j_internal_block, j, blockDim.x, color, shared_mem_inner_domain);
                const auto local_edge_index = i_internal_block + 1 - halo + ((j_internal_block + 1 - halo) * (blockDim.x + 1)) + color * shared_mem_inner_domain;
                z_nabla4_e2[local_edge_index] =
                    4.0 *
                    ((nabv_norm_wp - 2.0 * z_nabla2_e_gt_tv(edge_index + color * outer_domain_size, k_index)) * inv_vert_vert_length_sqr[color] +
                        (nabv_tang_wp - 2.0 * z_nabla2_e_gt_tv(edge_index + color * outer_domain_size, k_index)) * inv_primal_edge_length_sqr[color]);
                printf("global edge_index: %d, z_nabla4_e2[%d]: %f\n", edge_index + color * outer_domain_size, local_edge_index, z_nabla4_e2[local_edge_index]);
            };
        }
    }
    __syncthreads();
    const std::array<index_type, 6> v2e{get_v2e_per_orientation(i - halo + 1, j - halo + 1, blockDim.x + 1, blockDim.y + 2)};
    const std::array<index_type, 6> v2e_original{get_v2e_per_orientation(i, j, x_dim, y_dim)};
    std::array<index_type, 6> shared_mem_v2e_indexes;
    for (int idx{0}; idx < 6; ++idx) {
        // const auto i_shared_mem = v2e[i] % x_dim;
        // const auto j_shared_mem = v2e[i] / x_dim;
        // const auto color = v2e[idx] / outer_domain_size;
        // shared_mem_v2e_indexes[idx] = i_shared_mem + (j_shared_mem * (blockDim.x + 1)) + color * shared_mem_inner_domain;
        printf("v2e_original[%d]: %d, v2e_shared[%d]: %d, z_nabla4_e2: %lf\n", idx, v2e_original[idx], idx, v2e[idx], z_nabla4_e2[v2e[idx]]);
    }
    // printf("z_nabla4_e2[v2e[0]]: %lf, v2e[0]: %d, z_nabla4_e2[v2e[1]]: %lf, v2e[1]: %d, z_nabla4_e2[v2e[2]]: %lf, v2e[2]: %d, z_nabla4_e2[v2e[3]]: %lf, v2e[3]: %d, z_nabla4_e2[v2e[4]]: %lf, v2e[4]: %d, z_nabla4_e2[v2e[5]]: %lf, v2e[5]: %d\n", z_nabla4_e2[v2e[0]], v2e[0], z_nabla4_e2[v2e[1]], v2e[1], z_nabla4_e2[v2e[2]], v2e[2], z_nabla4_e2[v2e[3]], v2e[3], z_nabla4_e2[v2e[4]], v2e[4], z_nabla4_e2[v2e[5]], v2e[5]);
    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
    p_u_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2[v2e[0]] * ptr_coeff_1_gt_ctv(vertex_index_internal, 0) +
                                                    z_nabla4_e2[v2e[1]] * ptr_coeff_1_gt_ctv(vertex_index_internal, 1) +
                                                    z_nabla4_e2[v2e[2]] * ptr_coeff_1_gt_ctv(vertex_index_internal, 2) +
                                                    z_nabla4_e2[v2e[3]] * ptr_coeff_1_gt_ctv(vertex_index_internal, 3) +
                                                    z_nabla4_e2[v2e[4]] * ptr_coeff_1_gt_ctv(vertex_index_internal, 4) +
                                                    z_nabla4_e2[v2e[5]] * ptr_coeff_1_gt_ctv(vertex_index_internal, 5);
    p_v_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2[v2e[0]] * ptr_coeff_2_gt_ctv(vertex_index_internal, 0) +
                                                    z_nabla4_e2[v2e[1]] * ptr_coeff_2_gt_ctv(vertex_index_internal, 1) +
                                                    z_nabla4_e2[v2e[2]] * ptr_coeff_2_gt_ctv(vertex_index_internal, 2) +
                                                    z_nabla4_e2[v2e[3]] * ptr_coeff_2_gt_ctv(vertex_index_internal, 3) +
                                                    z_nabla4_e2[v2e[4]] * ptr_coeff_2_gt_ctv(vertex_index_internal, 4) +
                                                    z_nabla4_e2[v2e[5]] * ptr_coeff_2_gt_ctv(vertex_index_internal, 5);
};

template <typename T>
inline void nabla4_interpolate_structured_inlined_cached<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_structured_nabla_interpol_inlined_cached_naive.x,
        block_dims_structured_nabla_interpol_inlined_cached_naive.y,
        block_dims_structured_nabla_interpol_inlined_cached_naive.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x,
        (inner_y_dim + tblocks.y - 1) / tblocks.y,
        (interpolate_data.KDim + tblocks.z - 1) / tblocks.z);
    const index_type shared_mem_inner_domain = (tblocks.x + 1) * (tblocks.y + 2);
    run_gpu_naive_nabla4_interpolate_inlined_cached_structured<<<grid, tblocks, shared_mem_inner_domain * 3 * sizeof(WP_TYPE)>>>(interpolate_data.KDim,
        interpolate_data.x_dim,
        interpolate_data.y_dim,
        interpolate_data.halo,
        inner_domain_size,
        outer_domain_size,
        outer_domain_size * 3,
        shared_mem_inner_domain,
        nabla4_data.u_vert_gt_tv,
        nabla4_data.v_vert_gt_tv,
        nabla4_data.primal_normal_vert_v1_gt_tv,
        nabla4_data.primal_normal_vert_v2_gt_tv,
        nabla4_data.z_nabla2_e_gt_tv,
        nabla4_data.inv_vert_vert_length_gt_tv,
        nabla4_data.inv_primal_edge_length_gt_tv,
        interpolate_data.ptr_coeff_1_gt_ctv,
        interpolate_data.ptr_coeff_2_gt_ctv,
        interpolate_data.p_u_out_gt_tv,
        interpolate_data.p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void nabla4_interpolate_structured_inlined_cached<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
