#include <interpolate_unstructured_gridtools.hpp>
#include <nabla4_unstructured_gridtools.hpp>

template <typename T>
struct nabla4_interpolate_unstructured_inlined_cached {
    nabla4_unstructured_gt<T> nabla4_data;
    interpolate_unstructured<T> interpolate_data;
    index_type x_dim;

    nabla4_interpolate_unstructured_inlined_cached(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        std::vector<std::array<index_type, 6>> v2e,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type x_dim)
        : nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim),
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt()),
          x_dim(x_dim) {};

    nabla4_interpolate_unstructured_inlined_cached(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        std::vector<std::array<index_type, 6>> v2e,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type x_dim,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : nabla4_data(e2c2v,
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
              inv_primal_edge_length),
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt(), ptr_coeff_1, ptr_coeff_2),
          x_dim(x_dim) {};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    void run_gpu_kloop_helper();

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_cached_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_cached_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_cached_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_cached_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_cached_kloop<int>() {
    return {128, 1, 1, 128};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_inlined_cached_kloop =
    get_block_dims_unstructured_nabla_interpol_inlined_cached_kloop<index_type>();

constexpr int k_repetitions{4};

__global__ void __launch_bounds__(block_dims_unstructured_nabla_interpol_inlined_cached_kloop.size)
    run_gpu_kloop_nabla4_interpolate_inlined_cached_unstructured(index_type nabla4_output_size,
        index_type interpolate_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type x_dim,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2c2v_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2ecv_gt_tv,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= interpolate_output_size)
        return;
    const index_type i{vertex_index % x_dim};
    const index_type j{vertex_index / x_dim};
    extern __shared__ WP_TYPE z_nabla4_e2[];
    const std::array<index_type, 6> v2e{v2e_gt_tv(vertex_index, 0),
        v2e_gt_tv(vertex_index, 1),
        v2e_gt_tv(vertex_index, 2),
        v2e_gt_tv(vertex_index, 3),
        v2e_gt_tv(vertex_index, 4),
        v2e_gt_tv(vertex_index, 5)};
    //                                 0 == 510   - 1 || 0           == 127            || 0           + 510    >= 128           510   <= 128         || 0             + 510   >= 510 * 590
    // const bool calculate_edges_1_3_5 = i == x_dim - 1 || threadIdx.x == blockDim.x - 1 || (threadIdx.x + x_dim >= blockDim.x && x_dim <= blockDim.x) || i + j * x_dim + x_dim >= interpolate_output_size;
    const bool calculate_1 = i == x_dim - 1 || threadIdx.x == blockDim.x - 1 || vertex_index == interpolate_output_size - 1;
    // const bool calculate_3{true};
    // const bool calculate_5{true};
    const std::array<index_type, 24> e2c2v{e2c2v_gt_tv(v2e[0], 0),
        e2c2v_gt_tv(v2e[0], 1),
        e2c2v_gt_tv(v2e[0], 2),
        e2c2v_gt_tv(v2e[0], 3),
        calculate_1 ? e2c2v_gt_tv(v2e[1], 0) : 0,
        calculate_1 ? e2c2v_gt_tv(v2e[1], 1) : 0,
        calculate_1 ? e2c2v_gt_tv(v2e[1], 2) : 0,
        calculate_1 ? e2c2v_gt_tv(v2e[1], 3) : 0,
        e2c2v_gt_tv(v2e[2], 0),
        e2c2v_gt_tv(v2e[2], 1),
        e2c2v_gt_tv(v2e[2], 2),
        e2c2v_gt_tv(v2e[2], 3),
        e2c2v_gt_tv(v2e[3], 0),
        e2c2v_gt_tv(v2e[3], 1),
        e2c2v_gt_tv(v2e[3], 2),
        e2c2v_gt_tv(v2e[3], 3),
        e2c2v_gt_tv(v2e[4], 0),
        e2c2v_gt_tv(v2e[4], 1),
        e2c2v_gt_tv(v2e[4], 2),
        e2c2v_gt_tv(v2e[4], 3),
        e2c2v_gt_tv(v2e[5], 0),
        e2c2v_gt_tv(v2e[5], 1),
        e2c2v_gt_tv(v2e[5], 2),
        e2c2v_gt_tv(v2e[5], 3)};
    const std::array<index_type, 24> e2ecv{e2ecv_gt_tv(v2e[0], 0),
        e2ecv_gt_tv(v2e[0], 1),
        e2ecv_gt_tv(v2e[0], 2),
        e2ecv_gt_tv(v2e[0], 3),
        calculate_1 ? e2ecv_gt_tv(v2e[1], 0) : 0,
        calculate_1 ? e2ecv_gt_tv(v2e[1], 1) : 0,
        calculate_1 ? e2ecv_gt_tv(v2e[1], 2) : 0,
        calculate_1 ? e2ecv_gt_tv(v2e[1], 3) : 0,
        e2ecv_gt_tv(v2e[2], 0),
        e2ecv_gt_tv(v2e[2], 1),
        e2ecv_gt_tv(v2e[2], 2),
        e2ecv_gt_tv(v2e[2], 3),
        e2ecv_gt_tv(v2e[3], 0),
        e2ecv_gt_tv(v2e[3], 1),
        e2ecv_gt_tv(v2e[3], 2),
        e2ecv_gt_tv(v2e[3], 3),
        e2ecv_gt_tv(v2e[4], 0),
        e2ecv_gt_tv(v2e[4], 1),
        e2ecv_gt_tv(v2e[4], 2),
        e2ecv_gt_tv(v2e[4], 3),
        e2ecv_gt_tv(v2e[5], 0),
        e2ecv_gt_tv(v2e[5], 1),
        e2ecv_gt_tv(v2e[5], 2),
        e2ecv_gt_tv(v2e[5], 3)};
    const std::array<WP_TYPE, 24> primal_normal_vert_v1{primal_normal_vert_v1_gt_tv(e2ecv[0]),
        primal_normal_vert_v1_gt_tv(e2ecv[1]),
        primal_normal_vert_v1_gt_tv(e2ecv[2]),
        primal_normal_vert_v1_gt_tv(e2ecv[3]),
        calculate_1 ? primal_normal_vert_v1_gt_tv(e2ecv[4]) : 0,
        calculate_1 ? primal_normal_vert_v1_gt_tv(e2ecv[5]) : 0,
        calculate_1 ? primal_normal_vert_v1_gt_tv(e2ecv[6]) : 0,
        calculate_1 ? primal_normal_vert_v1_gt_tv(e2ecv[7]) : 0,
        primal_normal_vert_v1_gt_tv(e2ecv[8]),
        primal_normal_vert_v1_gt_tv(e2ecv[9]),
        primal_normal_vert_v1_gt_tv(e2ecv[10]),
        primal_normal_vert_v1_gt_tv(e2ecv[11]),
        primal_normal_vert_v1_gt_tv(e2ecv[12]),
        primal_normal_vert_v1_gt_tv(e2ecv[13]),
        primal_normal_vert_v1_gt_tv(e2ecv[14]),
        primal_normal_vert_v1_gt_tv(e2ecv[15]),
        primal_normal_vert_v1_gt_tv(e2ecv[16]),
        primal_normal_vert_v1_gt_tv(e2ecv[17]),
        primal_normal_vert_v1_gt_tv(e2ecv[18]),
        primal_normal_vert_v1_gt_tv(e2ecv[19]),
        primal_normal_vert_v1_gt_tv(e2ecv[20]),
        primal_normal_vert_v1_gt_tv(e2ecv[21]),
        primal_normal_vert_v1_gt_tv(e2ecv[22]),
        primal_normal_vert_v1_gt_tv(e2ecv[23])};
    const std::array<WP_TYPE, 24> primal_normal_vert_v2{primal_normal_vert_v2_gt_tv(e2ecv[0]),
        primal_normal_vert_v2_gt_tv(e2ecv[1]),
        primal_normal_vert_v2_gt_tv(e2ecv[2]),
        primal_normal_vert_v2_gt_tv(e2ecv[3]),
        calculate_1 ? primal_normal_vert_v2_gt_tv(e2ecv[4]) : 0,
        calculate_1 ? primal_normal_vert_v2_gt_tv(e2ecv[5]) : 0,
        calculate_1 ? primal_normal_vert_v2_gt_tv(e2ecv[6]) : 0,
        calculate_1 ? primal_normal_vert_v2_gt_tv(e2ecv[7]) : 0,
        primal_normal_vert_v2_gt_tv(e2ecv[8]),
        primal_normal_vert_v2_gt_tv(e2ecv[9]),
        primal_normal_vert_v2_gt_tv(e2ecv[10]),
        primal_normal_vert_v2_gt_tv(e2ecv[11]),
        primal_normal_vert_v2_gt_tv(e2ecv[12]),
        primal_normal_vert_v2_gt_tv(e2ecv[13]),
        primal_normal_vert_v2_gt_tv(e2ecv[14]),
        primal_normal_vert_v2_gt_tv(e2ecv[15]),
        primal_normal_vert_v2_gt_tv(e2ecv[16]),
        primal_normal_vert_v2_gt_tv(e2ecv[17]),
        primal_normal_vert_v2_gt_tv(e2ecv[18]),
        primal_normal_vert_v2_gt_tv(e2ecv[19]),
        primal_normal_vert_v2_gt_tv(e2ecv[20]),
        primal_normal_vert_v2_gt_tv(e2ecv[21]),
        primal_normal_vert_v2_gt_tv(e2ecv[22]),
        primal_normal_vert_v2_gt_tv(e2ecv[23])};
    const std::array<WP_TYPE, 6> inv_vert_vert_length{inv_vert_vert_length_gt_tv(v2e[0]),
        calculate_1 ? inv_vert_vert_length_gt_tv(v2e[1]) : 0,
        inv_vert_vert_length_gt_tv(v2e[2]),
        inv_vert_vert_length_gt_tv(v2e[3]),
        inv_vert_vert_length_gt_tv(v2e[4]),
        inv_vert_vert_length_gt_tv(v2e[5])};
    const std::array<WP_TYPE, 6> inv_primal_edge_length{inv_primal_edge_length_gt_tv(v2e[0]),
        calculate_1 ? inv_primal_edge_length_gt_tv(v2e[1]) : 0,
        inv_primal_edge_length_gt_tv(v2e[2]),
        inv_primal_edge_length_gt_tv(v2e[3]),
        inv_primal_edge_length_gt_tv(v2e[4]),
        inv_primal_edge_length_gt_tv(v2e[5])};
    const int initial_k_index{blockIdx.y * blockDim.y + threadIdx.y};
    int k_repetition{0};
    for (int k_index{initial_k_index}; k_index < KDim && k_repetition < k_repetitions; k_index += gridDim.y * blockDim.y) {
        const auto shared_mem_vertex_index{(threadIdx.x + k_repetition * blockDim.x) * 6};
#pragma unroll
        for (int edge_id{0}; edge_id < 6; ++edge_id) {
            if (edge_id == 1 && !calculate_1) {
                continue;
            }
            // if (edge_id == 1) {
            //     printf("[1] threadIdx.x: %d, blockIdx.x: %d, threadIdx.y: %d, shared_mem_vertex_index: %d, edge_id: %d\n", threadIdx.x, blockIdx.x, threadIdx.y, shared_mem_vertex_index, edge_id);
            // }
            // if (edge_id == 3 && !calculate_edges_1_3_5 && !calculate_3) {
            //     continue;
            // }
            // if (edge_id == 5 && !calculate_edges_1_3_5 && !calculate_5) {
            //     continue;
            // }
            const auto edge_index = v2e[edge_id];
            const auto E2C2V_0 = e2c2v[edge_id * 4];
            const auto E2C2V_1 = e2c2v[edge_id * 4 + 1];
            const auto E2C2V_2 = e2c2v[edge_id * 4 + 2];
            const auto E2C2V_3 = e2c2v[edge_id * 4 + 3];
            const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1[4 * edge_id] +
                                        v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2[4 * edge_id] +
                                        u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1[4 * edge_id + 1] +
                                        v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2[4 * edge_id + 1];
            const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1[4 * edge_id + 2] +
                                        v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2[4 * edge_id + 2] +
                                        u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1[4 * edge_id + 3] +
                                        v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2[4 * edge_id + 3];
            const double z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
            z_nabla4_e2[shared_mem_vertex_index + edge_id] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length[edge_id] * inv_vert_vert_length[edge_id] +
                        (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length[edge_id] * inv_primal_edge_length[edge_id]);
            printf("Setting z_nabla4_e2 %d = %lf, i: %d, j: %d, threadIdx.x: %d, edge_id: %d\n", shared_mem_vertex_index + edge_id, z_nabla4_e2[shared_mem_vertex_index + edge_id], i, j, threadIdx.x, edge_id);
            if (edge_id == 0 && threadIdx.x > 0 && i != 0) {
                const auto neighbor_vertex_index{((threadIdx.x - 1) + k_repetition * blockDim.x) * 6 + 1};
                z_nabla4_e2[neighbor_vertex_index] = z_nabla4_e2[shared_mem_vertex_index + edge_id];
                printf("Setting z_nabla4_e2 %d = %lf, i: %d, j: %d, threadIdx.x: %d, edge_id: %d\n", neighbor_vertex_index, z_nabla4_e2[neighbor_vertex_index], i, j, threadIdx.x, edge_id);
                // printf("[0->1] threadIdx.x: %d, blockIdx.x: %d, threadIdx.y: %d, neighbor_vertex_index: %d, shared_mem_vertex_index: %d, edge_id: %d\n", threadIdx.x, blockIdx.x, threadIdx.y, neighbor_vertex_index, shared_mem_vertex_index, edge_id);
            }
            // if (edge_id == 2 && j > 0 && threadIdx.x >= x_dim) {
            //     const auto neighbor_vertex_index{((threadIdx.x - x_dim) + k_repetition * blockDim.x) * 6 + 3};
            //     z_nabla4_e2[neighbor_vertex_index] = z_nabla4_e2[shared_mem_vertex_index + edge_id];
            //     printf("[2->3] threadIdx.x: %d, blockIdx.x: %d, threadIdx.y: %d, neighbor_vertex_index: %d, shared_mem_vertex_index: %d, edge_id: %d\n", threadIdx.x, blockIdx.x, threadIdx.y, neighbor_vertex_index, shared_mem_vertex_index, edge_id);
            // }
            // if (edge_id == 4 && j > 0 && threadIdx.x + 1 >= x_dim && !calculate_5) {
            //     const auto neighbor_vertex_index{((threadIdx.x - x_dim + 1) + k_repetition * blockDim.x) * 6 + 5};
            //     z_nabla4_e2[neighbor_vertex_index] = z_nabla4_e2[shared_mem_vertex_index + edge_id];
            //     printf("[4->5] threadIdx.x: %d, blockIdx.x: %d, threadIdx.y: %d, neighbor_vertex_index: %d, shared_mem_vertex_index: %d, edge_id: %d\n", threadIdx.x, blockIdx.x, threadIdx.y, neighbor_vertex_index, shared_mem_vertex_index, edge_id);
            // }
        }
        k_repetition++;
    }
    __syncthreads();
    const WP_TYPE ptr_coeff_1[6] = {ptr_coeff_1_gt_ctv(vertex_index, 0),
        ptr_coeff_1_gt_ctv(vertex_index, 1),
        ptr_coeff_1_gt_ctv(vertex_index, 2),
        ptr_coeff_1_gt_ctv(vertex_index, 3),
        ptr_coeff_1_gt_ctv(vertex_index, 4),
        ptr_coeff_1_gt_ctv(vertex_index, 5)};
    const WP_TYPE ptr_coeff_2[6] = {ptr_coeff_2_gt_ctv(vertex_index, 0),
        ptr_coeff_2_gt_ctv(vertex_index, 1),
        ptr_coeff_2_gt_ctv(vertex_index, 2),
        ptr_coeff_2_gt_ctv(vertex_index, 3),
        ptr_coeff_2_gt_ctv(vertex_index, 4),
        ptr_coeff_2_gt_ctv(vertex_index, 5)};
    k_repetition = 0;
    for (int k_index{initial_k_index}; k_index < KDim && k_repetition < k_repetitions; k_index += gridDim.y * blockDim.y) {
        const auto shared_mem_vertex_index{(threadIdx.x + k_repetition * blockDim.x) * 6};
        const std::array<WP_TYPE, 6> z_nabla4_e2_wp{z_nabla4_e2[shared_mem_vertex_index],
            z_nabla4_e2[shared_mem_vertex_index + 1],
            z_nabla4_e2[shared_mem_vertex_index + 2],
            z_nabla4_e2[shared_mem_vertex_index + 3],
            z_nabla4_e2[shared_mem_vertex_index + 4],
            z_nabla4_e2[shared_mem_vertex_index + 5]};
        // printf("threadIdx.x: %d, blockIdx.x: %d, threadIdx.y: %d, i: %d, j: %d, shared_mem_vertex_index: %d, shared_mem: [%lf %lf %lf %lf %lf %lf]\n", threadIdx.x, blockIdx.x, threadIdx.y, i, j, shared_mem_vertex_index, z_nabla4_e2_wp[0], z_nabla4_e2_wp[1], z_nabla4_e2_wp[2], z_nabla4_e2_wp[3], z_nabla4_e2_wp[4], z_nabla4_e2_wp[5]);
        p_u_out_gt_tv(vertex_index, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_1[0] + z_nabla4_e2_wp[1] * ptr_coeff_1[1] +
                                               z_nabla4_e2_wp[2] * ptr_coeff_1[2] + z_nabla4_e2_wp[3] * ptr_coeff_1[3] +
                                               z_nabla4_e2_wp[4] * ptr_coeff_1[4] + z_nabla4_e2_wp[5] * ptr_coeff_1[5];
        p_v_out_gt_tv(vertex_index, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_2[0] + z_nabla4_e2_wp[1] * ptr_coeff_2[1] +
                                               z_nabla4_e2_wp[2] * ptr_coeff_2[2] + z_nabla4_e2_wp[3] * ptr_coeff_2[3] +
                                               z_nabla4_e2_wp[4] * ptr_coeff_2[4] + z_nabla4_e2_wp[5] * ptr_coeff_2[5];
        k_repetition++;
    }
};

template <typename T>
inline void nabla4_interpolate_unstructured_inlined_cached<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_unstructured_nabla_interpol_inlined_cached_kloop.x,
        block_dims_unstructured_nabla_interpol_inlined_cached_kloop.y,
        block_dims_unstructured_nabla_interpol_inlined_cached_kloop.z);
    const int KDim_ceil = std::ceil(static_cast<double>(interpolate_data.KDim) / k_repetitions);
    dim3 grid((interpolate_data.output_size + tblocks.x - 1) / tblocks.x, (KDim_ceil + tblocks.y - 1) / tblocks.y , 1);
    const auto shared_mem_size{tblocks.x * k_repetitions * 6 * sizeof(WP_TYPE)};
    std::cout << "x_dim: " << x_dim << std::endl;
    run_gpu_kloop_nabla4_interpolate_inlined_cached_unstructured<<<grid, tblocks, shared_mem_size>>>(nabla4_data.output_size,
        interpolate_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        x_dim,
        nabla4_data.e2c2v_gt_tv,
        nabla4_data.e2ecv_gt_tv,
        interpolate_data.v2e_gt_ctv,
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
inline void nabla4_interpolate_unstructured_inlined_cached<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
