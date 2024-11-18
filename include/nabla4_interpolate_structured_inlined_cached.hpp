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

    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_cached_kloop<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_cached_kloop =
    get_block_dims_structured_nabla_interpol_inlined_cached_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_inlined_cached_kloop.size)
    run_gpu_kloop_nabla4_interpolate_inlined_cached_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        index_type inner_domain_size,
        index_type outer_domain_size,
        index_type total_edges,
        index_type shared_mem_inner_domain,
        int k_repetitions,
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
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo - 1 - blockIdx.x};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo - 1 - 2 * blockIdx.y};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    extern __shared__ WP_TYPE z_nabla4_e2[];
    if (i < x_dim - halo && j <= y_dim - halo && k_index < KDim) {
        const index_type nabla4_x_dim = x_dim;
        const index_type i_j = j * nabla4_x_dim + i;
        const index_type i_jp1 = (j + 1) * nabla4_x_dim + i;
        const index_type im1_jp1 = (j + 1) * nabla4_x_dim + i - 1;
        const index_type ip1_j = j * nabla4_x_dim + i + 1;
        const index_type ip1_jm1 = (j - 1) * nabla4_x_dim + i + 1;
        const index_type i_jm1 = (j - 1) * nabla4_x_dim + i;
        const index_type E2C2V_0[3] = {i_j, i_j, i_j};
        const index_type E2C2V_1[3] = {i_jp1, ip1_j, ip1_jm1};
        const index_type E2C2V_2[3] = {im1_jp1, i_jp1, ip1_j};
        const index_type E2C2V_3[3] = {ip1_j, ip1_jm1, i_jm1};
        const index_type E2ECV_0[3] = {i_j, i_j + outer_domain_size, i_j + 2 * outer_domain_size};
        const index_type E2ECV_1[3] = {E2ECV_0[0] + total_edges, E2ECV_0[1] + total_edges, E2ECV_0[2] + total_edges};
        const index_type E2ECV_2[3] = {E2ECV_1[0] + total_edges, E2ECV_1[1] + total_edges, E2ECV_1[2] + total_edges};
        const index_type E2ECV_3[3] = {E2ECV_2[0] + total_edges, E2ECV_2[1] + total_edges, E2ECV_2[2] + total_edges};
        const index_type edge_index = i_j;
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
        int k_repetition{0};
        for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_repetition < k_repetitions && k_index < KDim;
             k_index += gridDim.z * blockDim.z) {
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
                const auto k_level_cache_offset =
                    3 * shared_mem_inner_domain * ((threadIdx.z * k_repetitions) + k_repetition);
                const auto local_edge_index =
                    threadIdx.x + threadIdx.y * blockDim.x + color * shared_mem_inner_domain + k_level_cache_offset;
                const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index + color * outer_domain_size, k_index);
                z_nabla4_e2[local_edge_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr[color] +
                              (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr[color]);
            };
            k_repetition++;
        }
    }
    __syncthreads();
    const auto i_interpolate{blockIdx.x * blockDim.x + threadIdx.x + halo - blockIdx.x};
    const auto j_interpolate{blockIdx.y * blockDim.y + threadIdx.y + halo - 2 * blockIdx.y};
    if (i_interpolate >= x_dim - halo || j_interpolate >= y_dim - halo || threadIdx.x > blockDim.x - 2 ||
        threadIdx.y > blockDim.y - 3) {
        return;
    }
    const std::array<index_type, 6> v2e{
        get_v2e_per_orientation(threadIdx.x + 1, threadIdx.y + 1, blockDim.x, blockDim.y)};
    const index_type vertex_index_internal = i_interpolate - halo + (j_interpolate - halo) * (x_dim - 2 * halo);
    const std::array<WP_TYPE, 6> coeff_1{ptr_coeff_1_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 5)};
    const std::array<WP_TYPE, 6> coeff_2{ptr_coeff_2_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 5)};
    int k_repetition{0};
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_repetition < k_repetitions && k_index < KDim;
         k_index += gridDim.z * blockDim.z) {
        const auto k_level_cache_offset = 3 * shared_mem_inner_domain * ((threadIdx.z * k_repetitions) + k_repetition);
        p_u_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2[v2e[0] + k_level_cache_offset] * coeff_1[0] +
                                                        z_nabla4_e2[v2e[1] + k_level_cache_offset] * coeff_1[1] +
                                                        z_nabla4_e2[v2e[2] + k_level_cache_offset] * coeff_1[2] +
                                                        z_nabla4_e2[v2e[3] + k_level_cache_offset] * coeff_1[3] +
                                                        z_nabla4_e2[v2e[4] + k_level_cache_offset] * coeff_1[4] +
                                                        z_nabla4_e2[v2e[5] + k_level_cache_offset] * coeff_1[5];
        p_v_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2[v2e[0] + k_level_cache_offset] * coeff_2[0] +
                                                        z_nabla4_e2[v2e[1] + k_level_cache_offset] * coeff_2[1] +
                                                        z_nabla4_e2[v2e[2] + k_level_cache_offset] * coeff_2[2] +
                                                        z_nabla4_e2[v2e[3] + k_level_cache_offset] * coeff_2[3] +
                                                        z_nabla4_e2[v2e[4] + k_level_cache_offset] * coeff_2[4] +
                                                        z_nabla4_e2[v2e[5] + k_level_cache_offset] * coeff_2[5];
        k_repetition++;
    }
};

template <typename T>
inline void nabla4_interpolate_structured_inlined_cached<T>::run_gpu_kloop_helper() {
    constexpr dim3 tblocks(block_dims_structured_nabla_interpol_inlined_cached_kloop.x,
        block_dims_structured_nabla_interpol_inlined_cached_kloop.y,
        block_dims_structured_nabla_interpol_inlined_cached_kloop.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo + 2;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo + 2;
    constexpr int smemSize{49152}; // GH200
    constexpr index_type shared_mem_inner_domain = tblocks.x * tblocks.y;
    constexpr long unsigned int k_repetitions{smemSize / (shared_mem_inner_domain * 3 * sizeof(WP_TYPE) * tblocks.z)};
    const int KDim_ceil = std::ceil(static_cast<double>(interpolate_data.KDim) / k_repetitions);
    dim3 grid((inner_x_dim - 1 + tblocks.x - 1) / (tblocks.x - 1),
        (inner_y_dim - 2 + tblocks.y - 1) / (tblocks.y - 2),
        (KDim_ceil + tblocks.z - 1) / tblocks.z);
    run_gpu_kloop_nabla4_interpolate_inlined_cached_structured<<<grid,
        tblocks,
        shared_mem_inner_domain * 3 * sizeof(WP_TYPE) * k_repetitions * tblocks.z>>>(interpolate_data.KDim,
        interpolate_data.x_dim,
        interpolate_data.y_dim,
        interpolate_data.halo,
        inner_domain_size,
        outer_domain_size,
        outer_domain_size * 3,
        shared_mem_inner_domain,
        k_repetitions,
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
    return {32, 8, 2, 512};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_cached_naive =
    get_block_dims_structured_nabla_interpol_inlined_cached_naive<index_type>();

__global__ void
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 5)
__launch_bounds__(block_dims_structured_nabla_interpol_inlined_cached_naive.size)
#else
__maxnreg__(64)
#endif
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
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo - 1 - blockIdx.x};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo - 1 - 2 * blockIdx.y};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    extern __shared__ WP_TYPE z_nabla4_e2[];
    if (i < x_dim - halo && j <= y_dim - halo && k_index < KDim) {
        const index_type nabla4_x_dim = x_dim;
        const index_type i_j = j * nabla4_x_dim + i;
        const index_type i_jp1 = (j + 1) * nabla4_x_dim + i;
        const index_type im1_jp1 = (j + 1) * nabla4_x_dim + i - 1;
        const index_type ip1_j = j * nabla4_x_dim + i + 1;
        const index_type ip1_jm1 = (j - 1) * nabla4_x_dim + i + 1;
        const index_type i_jm1 = (j - 1) * nabla4_x_dim + i;
        const index_type E2C2V_0[3] = {i_j, i_j, i_j};
        const index_type E2C2V_1[3] = {i_jp1, ip1_j, ip1_jm1};
        const index_type E2C2V_2[3] = {im1_jp1, i_jp1, ip1_j};
        const index_type E2C2V_3[3] = {ip1_j, ip1_jm1, i_jm1};
        const index_type E2ECV_0[3] = {i_j, i_j + outer_domain_size, i_j + 2 * outer_domain_size};
        const index_type E2ECV_1[3] = {E2ECV_0[0] + total_edges, E2ECV_0[1] + total_edges, E2ECV_0[2] + total_edges};
        const index_type E2ECV_2[3] = {E2ECV_1[0] + total_edges, E2ECV_1[1] + total_edges, E2ECV_1[2] + total_edges};
        const index_type E2ECV_3[3] = {E2ECV_2[0] + total_edges, E2ECV_2[1] + total_edges, E2ECV_2[2] + total_edges};
        const index_type edge_index = i_j;
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
            const auto k_level_cache_offset =
                3 * shared_mem_inner_domain * threadIdx.z;
            const auto local_edge_index =
                threadIdx.x + threadIdx.y * blockDim.x + color * shared_mem_inner_domain + k_level_cache_offset;
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index + color * outer_domain_size, k_index);
            z_nabla4_e2[local_edge_index] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr[color] +
                            (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr[color]);
        };
    }
    __syncthreads();
    const auto i_interpolate{blockIdx.x * blockDim.x + threadIdx.x + halo - blockIdx.x};
    const auto j_interpolate{blockIdx.y * blockDim.y + threadIdx.y + halo - 2 * blockIdx.y};
    if (i_interpolate >= x_dim - halo || j_interpolate >= y_dim - halo || threadIdx.x > blockDim.x - 2 ||
        threadIdx.y > blockDim.y - 3 || k_index >= KDim) {
        return;
    }
    const std::array<index_type, 6> v2e{
        get_v2e_per_orientation(threadIdx.x + 1, threadIdx.y + 1, blockDim.x, blockDim.y)};
    const index_type vertex_index_internal = i_interpolate - halo + (j_interpolate - halo) * (x_dim - 2 * halo);
    const std::array<WP_TYPE, 6> coeff_1{ptr_coeff_1_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 5)};
    const std::array<WP_TYPE, 6> coeff_2{ptr_coeff_2_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 5)};
        const auto k_level_cache_offset = 3 * shared_mem_inner_domain * threadIdx.z;
        p_u_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2[v2e[0] + k_level_cache_offset] * coeff_1[0] +
                                                        z_nabla4_e2[v2e[1] + k_level_cache_offset] * coeff_1[1] +
                                                        z_nabla4_e2[v2e[2] + k_level_cache_offset] * coeff_1[2] +
                                                        z_nabla4_e2[v2e[3] + k_level_cache_offset] * coeff_1[3] +
                                                        z_nabla4_e2[v2e[4] + k_level_cache_offset] * coeff_1[4] +
                                                        z_nabla4_e2[v2e[5] + k_level_cache_offset] * coeff_1[5];
        p_v_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2[v2e[0] + k_level_cache_offset] * coeff_2[0] +
                                                        z_nabla4_e2[v2e[1] + k_level_cache_offset] * coeff_2[1] +
                                                        z_nabla4_e2[v2e[2] + k_level_cache_offset] * coeff_2[2] +
                                                        z_nabla4_e2[v2e[3] + k_level_cache_offset] * coeff_2[3] +
                                                        z_nabla4_e2[v2e[4] + k_level_cache_offset] * coeff_2[4] +
                                                        z_nabla4_e2[v2e[5] + k_level_cache_offset] * coeff_2[5];
};

template <typename T>
inline void nabla4_interpolate_structured_inlined_cached<T>::run_gpu_naive_helper() {
    constexpr dim3 tblocks(block_dims_structured_nabla_interpol_inlined_cached_naive.x,
        block_dims_structured_nabla_interpol_inlined_cached_naive.y,
        block_dims_structured_nabla_interpol_inlined_cached_naive.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo + 2;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo + 2;
    constexpr index_type shared_mem_inner_domain = tblocks.x * tblocks.y;
    dim3 grid((inner_x_dim - 1 + tblocks.x - 1) / (tblocks.x - 1),
        (inner_y_dim - 2 + tblocks.y - 1) / (tblocks.y - 2),
        (interpolate_data.KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_interpolate_inlined_cached_structured<<<grid,
        tblocks,
        shared_mem_inner_domain * 3 * sizeof(WP_TYPE) * tblocks.z>>>(interpolate_data.KDim,
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
