#include <interpolate_unstructured_gridtools.hpp>
#include <nabla4_unstructured_gridtools.hpp>

template <typename T>
struct nabla4_interpolate_unstructured_inlined_v2v_general {
    nabla4_unstructured_gt<T> nabla4_data;
    interpolate_unstructured<T> interpolate_data;

    using neighbors_e2c2v_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 7_c).template type<index_type>().build());
    using neighbors_e2c2v_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 7_c).template type<index_type>().build()->const_target_view());
    using neighbors_e2c2v_indexes_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 24_c).template type<uint8_t>().build());
    using neighbors_e2c2v_indexes_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 24_c).template type<uint8_t>().build()->const_target_view());
    using neighbors_e2ecv_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 24_c).template type<index_type>().build());
    using neighbors_e2ecv_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 24_c).template type<index_type>().build()->const_target_view());
    neighbors_e2c2v_gt_t v2e2c2v_gt;
    neighbors_e2c2v_gt_ctv_t v2e2c2v_gt_ctv;
    neighbors_e2c2v_indexes_gt_t v2e2c2v_indexes_gt;
    neighbors_e2c2v_indexes_gt_ctv_t v2e2c2v_indexes_gt_ctv;
    neighbors_e2ecv_gt_t v2e2ecv_gt;
    neighbors_e2ecv_gt_ctv_t v2e2ecv_gt_ctv;

    nabla4_interpolate_unstructured_inlined_v2v_general(const std::vector<std::array<index_type, 4>> &e2c2v,
        const std::vector<std::array<index_type, 4>> &e2ecv,
        const std::vector<std::array<index_type, 6>> &v2e,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : v2e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 7_c).initializer([&e2c2v, &v2e](int i, int j) {
                switch (j) {
                    case 0:
                        return e2c2v[v2e[i][0]][0]; 
                    case 1:
                        return e2c2v[v2e[i][0]][1];
                    case 2:
                        return e2c2v[v2e[i][0]][2];
                    case 3:
                        return e2c2v[v2e[i][0]][3];
                    case 4:
                        return e2c2v[v2e[i][1]][1];
                    case 5:
                        return e2c2v[v2e[i][1]][2];
                    case 6:
                        return e2c2v[v2e[i][1]][3];
                    default:
                        __builtin_unreachable();
                }
            }).build()),
            v2e2c2v_gt_ctv(v2e2c2v_gt->const_target_view()),
            v2e2c2v_indexes_gt(storage::builder<T>.template type<uint8_t>().dimensions(v2e.size(), 24_c).initializer([](int i, int j) {
                switch (j) {
                    case 0:
                        return 0;
                    case 1:
                        return 1;
                    case 2:
                        return 2;
                    case 3:
                        return 3;
                    case 4:
                        return 1;
                    case 5:
                        return 4;
                    case 6:
                        return 5;
                    case 7:
                        return 6;
                    case 8:
                        return 3;
                    case 9:
                        return 1;
                    case 10:
                        return 0;
                    case 11:
                        return 6;
                    case 12:
                        return 1;
                    case 13:
                        return 5;
                    case 14:
                        return 2;
                    case 15:
                        return 4;
                    case 16:
                        return 1;
                    case 17:
                        return 6;
                    case 18:
                        return 4;
                    case 19:
                        return 3;
                    case 20:
                        return 2;
                    case 21:
                        return 1;
                    case 22:
                        return 5;
                    case 23:
                        return 0;
                    default:
                        __builtin_unreachable();
                }
            }).build()),
            v2e2c2v_indexes_gt_ctv(v2e2c2v_indexes_gt->const_target_view()),
            v2e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 24_c).initializer([&e2ecv, &v2e](int i, int j) { return e2ecv[v2e[i][j/4]][j%4]; }).build()),
            v2e2ecv_gt_ctv(v2e2ecv_gt->const_target_view()),
            nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim),
            interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt()) {};

    nabla4_interpolate_unstructured_inlined_v2v_general(const std::vector<std::array<index_type, 4>> &e2c2v,
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
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : v2e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 7_c).initializer([&e2c2v, &v2e](int i, int j) {
                switch (j) {
                    case 0:
                        return e2c2v[v2e[i][0]][0]; 
                    case 1:
                        return e2c2v[v2e[i][0]][1];
                    case 2:
                        return e2c2v[v2e[i][0]][2];
                    case 3:
                        return e2c2v[v2e[i][0]][3];
                    case 4:
                        return e2c2v[v2e[i][1]][1];
                    case 5:
                        return e2c2v[v2e[i][1]][2];
                    case 6:
                        return e2c2v[v2e[i][1]][3];
                    default:
                        __builtin_unreachable();
                }
            }).build()),
            v2e2c2v_gt_ctv(v2e2c2v_gt->const_target_view()),
            v2e2c2v_indexes_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 24_c).initializer([](int i, int j) {
                switch (j) {
                    case 0:
                        return 0;
                    case 1:
                        return 1;
                    case 2:
                        return 2;
                    case 3:
                        return 3;
                    case 4:
                        return 1;
                    case 5:
                        return 4;
                    case 6:
                        return 5;
                    case 7:
                        return 6;
                    case 8:
                        return 3;
                    case 9:
                        return 1;
                    case 10:
                        return 0;
                    case 11:
                        return 6;
                    case 12:
                        return 1;
                    case 13:
                        return 5;
                    case 14:
                        return 2;
                    case 15:
                        return 4;
                    case 16:
                        return 1;
                    case 17:
                        return 6;
                    case 18:
                        return 4;
                    case 19:
                        return 3;
                    case 20:
                        return 2;
                    case 21:
                        return 1;
                    case 22:
                        return 5;
                    case 23:
                        return 0;
                    default:
                        __builtin_unreachable();
                }
            }).build()),
            v2e2c2v_indexes_gt_ctv(v2e2c2v_indexes_gt->const_target_view()),
            v2e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 24_c).initializer([&e2ecv, &v2e](int i, int j) { return e2ecv[v2e[i][j/4]][j%4]; }).build()),
            v2e2ecv_gt_ctv(v2e2ecv_gt->const_target_view()),
            nabla4_data(e2c2v,
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
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt(), ptr_coeff_1, ptr_coeff_2) {};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop<int>() {
    return {32, 4, 1, 128};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop =
    get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop.size)
    run_gpu_kloop_nabla4_interpolate_inlined_v2v_general_unstructured(index_type nabla4_output_size,
        index_type interpolate_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_interpolate_unstructured_inlined_v2v_general<storage::gpu>::neighbors_e2c2v_gt_ctv_t v2e2c2v_gt_ctv,
        nabla4_interpolate_unstructured_inlined_v2v_general<storage::gpu>::neighbors_e2c2v_indexes_gt_ctv_t
            v2e2c2v_indexes_gt_ctv,
        nabla4_interpolate_unstructured_inlined_v2v_general<storage::gpu>::neighbors_e2ecv_gt_ctv_t v2e2ecv_gt_ctv,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_ctv,
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
    const std::array<index_type, 7> v2e2c2v{v2e2c2v_gt_ctv(vertex_index, 0),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 6)};
    const std::array<uint8_t, 24> v2e2c2v_indexes{v2e2c2v_indexes_gt_ctv(vertex_index, 0),
        v2e2c2v_indexes_gt_ctv(vertex_index, 1),
        v2e2c2v_indexes_gt_ctv(vertex_index, 2),
        v2e2c2v_indexes_gt_ctv(vertex_index, 3),
        v2e2c2v_indexes_gt_ctv(vertex_index, 4),
        v2e2c2v_indexes_gt_ctv(vertex_index, 5),
        v2e2c2v_indexes_gt_ctv(vertex_index, 6),
        v2e2c2v_indexes_gt_ctv(vertex_index, 7),
        v2e2c2v_indexes_gt_ctv(vertex_index, 8),
        v2e2c2v_indexes_gt_ctv(vertex_index, 9),
        v2e2c2v_indexes_gt_ctv(vertex_index, 10),
        v2e2c2v_indexes_gt_ctv(vertex_index, 11),
        v2e2c2v_indexes_gt_ctv(vertex_index, 12),
        v2e2c2v_indexes_gt_ctv(vertex_index, 13),
        v2e2c2v_indexes_gt_ctv(vertex_index, 14),
        v2e2c2v_indexes_gt_ctv(vertex_index, 15),
        v2e2c2v_indexes_gt_ctv(vertex_index, 16),
        v2e2c2v_indexes_gt_ctv(vertex_index, 17),
        v2e2c2v_indexes_gt_ctv(vertex_index, 18),
        v2e2c2v_indexes_gt_ctv(vertex_index, 19),
        v2e2c2v_indexes_gt_ctv(vertex_index, 20),
        v2e2c2v_indexes_gt_ctv(vertex_index, 21),
        v2e2c2v_indexes_gt_ctv(vertex_index, 22),
        v2e2c2v_indexes_gt_ctv(vertex_index, 23)};
    const std::array<index_type, 24> v2e2ecv{v2e2ecv_gt_ctv(vertex_index, 0),
        v2e2ecv_gt_ctv(vertex_index, 1),
        v2e2ecv_gt_ctv(vertex_index, 2),
        v2e2ecv_gt_ctv(vertex_index, 3),
        v2e2ecv_gt_ctv(vertex_index, 4),
        v2e2ecv_gt_ctv(vertex_index, 5),
        v2e2ecv_gt_ctv(vertex_index, 6),
        v2e2ecv_gt_ctv(vertex_index, 7),
        v2e2ecv_gt_ctv(vertex_index, 8),
        v2e2ecv_gt_ctv(vertex_index, 9),
        v2e2ecv_gt_ctv(vertex_index, 10),
        v2e2ecv_gt_ctv(vertex_index, 11),
        v2e2ecv_gt_ctv(vertex_index, 12),
        v2e2ecv_gt_ctv(vertex_index, 13),
        v2e2ecv_gt_ctv(vertex_index, 14),
        v2e2ecv_gt_ctv(vertex_index, 15),
        v2e2ecv_gt_ctv(vertex_index, 16),
        v2e2ecv_gt_ctv(vertex_index, 17),
        v2e2ecv_gt_ctv(vertex_index, 18),
        v2e2ecv_gt_ctv(vertex_index, 19),
        v2e2ecv_gt_ctv(vertex_index, 20),
        v2e2ecv_gt_ctv(vertex_index, 21),
        v2e2ecv_gt_ctv(vertex_index, 22),
        v2e2ecv_gt_ctv(vertex_index, 23)};
    const std::array<index_type, 6> v2e{v2e_gt_ctv(vertex_index, 0),
        v2e_gt_ctv(vertex_index, 1),
        v2e_gt_ctv(vertex_index, 2),
        v2e_gt_ctv(vertex_index, 3),
        v2e_gt_ctv(vertex_index, 4),
        v2e_gt_ctv(vertex_index, 5)};
    const std::array<WP_TYPE, 24> primal_normal_vert_v1{primal_normal_vert_v1_gt_tv(v2e2ecv[0]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[1]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[2]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[3]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[4]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[5]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[6]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[7]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[8]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[9]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[10]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[11]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[12]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[13]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[14]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[15]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[16]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[17]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[18]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[19]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[20]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[21]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[22]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[23])};
    const std::array<WP_TYPE, 24> primal_normal_vert_v2{primal_normal_vert_v2_gt_tv(v2e2ecv[0]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[1]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[2]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[3]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[4]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[5]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[6]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[7]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[8]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[9]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[10]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[11]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[12]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[13]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[14]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[15]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[16]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[17]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[18]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[19]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[20]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[21]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[22]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[23])};
    const std::array<WP_TYPE, 6> inv_vert_vert_length{inv_vert_vert_length_gt_tv(v2e[0]),
        inv_vert_vert_length_gt_tv(v2e[1]),
        inv_vert_vert_length_gt_tv(v2e[2]),
        inv_vert_vert_length_gt_tv(v2e[3]),
        inv_vert_vert_length_gt_tv(v2e[4]),
        inv_vert_vert_length_gt_tv(v2e[5])};
    const std::array<WP_TYPE, 6> inv_primal_edge_length{inv_primal_edge_length_gt_tv(v2e[0]),
        inv_primal_edge_length_gt_tv(v2e[1]),
        inv_primal_edge_length_gt_tv(v2e[2]),
        inv_primal_edge_length_gt_tv(v2e[3]),
        inv_primal_edge_length_gt_tv(v2e[4]),
        inv_primal_edge_length_gt_tv(v2e[5])};
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
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    for (auto k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim; k_index += gridDim.y * blockDim.y) {
        const std::array<WP_TYPE, 7> u_vert{u_vert_gt_tv(v2e2c2v[0], k_index),
            u_vert_gt_tv(v2e2c2v[1], k_index),
            u_vert_gt_tv(v2e2c2v[2], k_index),
            u_vert_gt_tv(v2e2c2v[3], k_index),
            u_vert_gt_tv(v2e2c2v[4], k_index),
            u_vert_gt_tv(v2e2c2v[5], k_index),
            u_vert_gt_tv(v2e2c2v[6], k_index)};
        const std::array<WP_TYPE, 7> v_vert{v_vert_gt_tv(v2e2c2v[0], k_index),
            v_vert_gt_tv(v2e2c2v[1], k_index),
            v_vert_gt_tv(v2e2c2v[2], k_index),
            v_vert_gt_tv(v2e2c2v[3], k_index),
            v_vert_gt_tv(v2e2c2v[4], k_index),
            v_vert_gt_tv(v2e2c2v[5], k_index),
            v_vert_gt_tv(v2e2c2v[6], k_index)};
#pragma unroll 6
        for (auto i{0}; i < 6; ++i) {
            const auto edge_index = v2e[i];
            const auto E2C2V_0_index = v2e2c2v_indexes[i * 4];
            const auto E2C2V_1_index = v2e2c2v_indexes[i * 4 + 1];
            const auto E2C2V_2_index = v2e2c2v_indexes[i * 4 + 2];
            const auto E2C2V_3_index = v2e2c2v_indexes[i * 4 + 3];
            const double nabv_tang_wp = u_vert[E2C2V_0_index] * primal_normal_vert_v1[4 * i] +
                                        v_vert[E2C2V_0_index] * primal_normal_vert_v2[4 * i] +
                                        u_vert[E2C2V_1_index] * primal_normal_vert_v1[4 * i + 1] +
                                        v_vert[E2C2V_1_index] * primal_normal_vert_v2[4 * i + 1];
            const double nabv_norm_wp = u_vert[E2C2V_2_index] * primal_normal_vert_v1[4 * i + 2] +
                                        v_vert[E2C2V_2_index] * primal_normal_vert_v2[4 * i + 2] +
                                        u_vert[E2C2V_3_index] * primal_normal_vert_v1[4 * i + 3] +
                                        v_vert[E2C2V_3_index] * primal_normal_vert_v2[4 * i + 3];
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
            z_nabla4_e2_wp[i] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * (inv_vert_vert_length[i] * inv_vert_vert_length[i]) +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) * (inv_primal_edge_length[i] * inv_primal_edge_length[i]));
        }
        p_u_out_gt_tv(vertex_index, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_1[0] + z_nabla4_e2_wp[1] * ptr_coeff_1[1] +
                                               z_nabla4_e2_wp[2] * ptr_coeff_1[2] + z_nabla4_e2_wp[3] * ptr_coeff_1[3] +
                                               z_nabla4_e2_wp[4] * ptr_coeff_1[4] + z_nabla4_e2_wp[5] * ptr_coeff_1[5];
        p_v_out_gt_tv(vertex_index, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_2[0] + z_nabla4_e2_wp[1] * ptr_coeff_2[1] +
                                               z_nabla4_e2_wp[2] * ptr_coeff_2[2] + z_nabla4_e2_wp[3] * ptr_coeff_2[3] +
                                               z_nabla4_e2_wp[4] * ptr_coeff_2[4] + z_nabla4_e2_wp[5] * ptr_coeff_2[5];
    }
};

template <typename T>
inline void nabla4_interpolate_unstructured_inlined_v2v_general<T>::run_gpu_kloop_helper() {
    constexpr dim3 tblocks(block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop.x,
        block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop.y,
        block_dims_unstructured_nabla_interpol_inlined_v2v_general_kloop.z);
    dim3 grid((interpolate_data.output_size + tblocks.x - 1) / tblocks.x, 2, 1);
    run_gpu_kloop_nabla4_interpolate_inlined_v2v_general_unstructured<<<grid, tblocks>>>(nabla4_data.output_size,
        interpolate_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        v2e2c2v_gt_ctv,
        v2e2c2v_indexes_gt_ctv,
        v2e2ecv_gt_ctv,
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

template <typename T>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive<int>() {
    return {32, 12, 1, 384};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive =
    get_block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive<index_type>();

__global__ void
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 5)
__launch_bounds__(block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive.size)
#else
__maxnreg__(80)
#endif
    run_gpu_naive_nabla4_interpolate_inlined_v2v_general_unstructured(index_type nabla4_output_size,
        index_type interpolate_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_interpolate_unstructured_inlined_v2v_general<storage::gpu>::neighbors_e2c2v_gt_ctv_t v2e2c2v_gt_ctv,
        nabla4_interpolate_unstructured_inlined_v2v_general<storage::gpu>::neighbors_e2c2v_indexes_gt_ctv_t
            v2e2c2v_indexes_gt_ctv,
        nabla4_interpolate_unstructured_inlined_v2v_general<storage::gpu>::neighbors_e2ecv_gt_ctv_t v2e2ecv_gt_ctv,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_ctv,
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
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (vertex_index >= interpolate_output_size || k_index >= KDim)
        return;
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    const std::array<WP_TYPE, 7> u_vert{u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 0), k_index),
        u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 1), k_index),
        u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 2), k_index),
        u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 3), k_index),
        u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 4), k_index),
        u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 5), k_index),
        u_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 6), k_index)};
    const std::array<WP_TYPE, 7> v_vert{v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 0), k_index),
        v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 1), k_index),
        v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 2), k_index),
        v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 3), k_index),
        v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 4), k_index),
        v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 5), k_index),
        v_vert_gt_tv(v2e2c2v_gt_ctv(vertex_index, 6), k_index)};
#pragma unroll 6
    for (int i{0}; i < 6; ++i) {
        const auto E2C2V_0_index = v2e2c2v_indexes_gt_ctv(vertex_index, i * 4);
        const auto E2C2V_1_index = v2e2c2v_indexes_gt_ctv(vertex_index, i * 4 + 1);
        const auto E2C2V_2_index = v2e2c2v_indexes_gt_ctv(vertex_index, i * 4 + 2);
        const auto E2C2V_3_index = v2e2c2v_indexes_gt_ctv(vertex_index, i * 4 + 3);
        const auto E2ECV_0 = v2e2ecv_gt_ctv(vertex_index, i * 4);
        const auto E2ECV_1 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 1);
        const auto E2ECV_2 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 2);
        const auto E2ECV_3 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 3);
        const double nabv_tang_wp = u_vert[E2C2V_0_index] * primal_normal_vert_v1_gt_tv(E2ECV_0) +
                                    v_vert[E2C2V_0_index] * primal_normal_vert_v2_gt_tv(E2ECV_0) +
                                    u_vert[E2C2V_1_index] * primal_normal_vert_v1_gt_tv(E2ECV_1) +
                                    v_vert[E2C2V_1_index] * primal_normal_vert_v2_gt_tv(E2ECV_1);
        const double nabv_norm_wp = u_vert[E2C2V_2_index] * primal_normal_vert_v1_gt_tv(E2ECV_2) +
                                    v_vert[E2C2V_2_index] * primal_normal_vert_v2_gt_tv(E2ECV_2) +
                                    u_vert[E2C2V_3_index] * primal_normal_vert_v1_gt_tv(E2ECV_3) +
                                    v_vert[E2C2V_3_index] * primal_normal_vert_v2_gt_tv(E2ECV_3);
        const auto edge_index = v2e_gt_ctv(vertex_index, i);
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
        const WP_TYPE inv_vert_vert_length = inv_vert_vert_length_gt_tv(edge_index);
        const WP_TYPE inv_vert_vert_length_sqr = inv_vert_vert_length * inv_vert_vert_length;
        const WP_TYPE inv_primal_edge_length = inv_primal_edge_length_gt_tv(edge_index);
        const WP_TYPE inv_primal_edge_length_sqr = inv_primal_edge_length * inv_primal_edge_length;
        z_nabla4_e2_wp[i] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                      (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    }
    p_u_out_gt_tv(vertex_index, k_index) = ptr_coeff_1_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp[0] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp[1] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp[2] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp[3] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp[4] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp[5];
    p_v_out_gt_tv(vertex_index, k_index) = ptr_coeff_2_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp[0] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp[1] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp[2] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp[3] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp[4] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp[5];
};

template <typename T>
inline void nabla4_interpolate_unstructured_inlined_v2v_general<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive.x,
        block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive.y,
        block_dims_unstructured_nabla_interpol_inlined_v2v_general_naive.z);
    dim3 grid((interpolate_data.output_size + tblocks.x - 1) / tblocks.x,
        (interpolate_data.KDim + tblocks.y - 1) / tblocks.y,
        1);
    run_gpu_naive_nabla4_interpolate_inlined_v2v_general_unstructured<<<grid, tblocks>>>(nabla4_data.output_size,
        interpolate_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        v2e2c2v_gt_ctv,
        v2e2c2v_indexes_gt_ctv,
        v2e2ecv_gt_ctv,
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
inline void nabla4_interpolate_unstructured_inlined_v2v_general<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_unstructured_inlined_v2v_general<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
