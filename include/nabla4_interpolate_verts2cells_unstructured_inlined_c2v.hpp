#include <interpolate_unstructured_gridtools.hpp>
#include <nabla4_unstructured_gridtools.hpp>
#include <verts2cells_unstructured_gridtools.hpp>

template <typename T>
struct nabla4_interpolate_verts2cells_unstructured_inlined_c2v {
    nabla4_unstructured_gt<T> nabla4_data;
    interpolate_unstructured<T> interpolate_data;
    verts2cells_unstructured<T> verts2cells_data;

    using neighbors_c2v2e2c2v_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 12_c).template type<index_type>().build());
    using neighbors_c2v2e2c2v_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 12_c).template type<index_type>().build()->const_target_view());
    using neighbors_c2v2e2ecv_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 72_c).template type<index_type>().build());
    using neighbors_c2v2e2ecv_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 72_c).template type<index_type>().build()->const_target_view());
    using neighbors_c2v2e_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 15_c).template type<index_type>().build());
    using neighbors_c2v2e_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 15_c).template type<index_type>().build()->const_target_view());

    neighbors_c2v2e2c2v_gt_t c2v2e2c2v_gt;
    neighbors_c2v2e2c2v_gt_ctv_t c2v2e2c2v_gt_ctv;
    neighbors_c2v2e2ecv_gt_t c2v2e2ecv_gt;
    neighbors_c2v2e2ecv_gt_ctv_t c2v2e2ecv_gt_ctv;
    neighbors_c2v2e_gt_t c2v2e_gt;
    neighbors_c2v2e_gt_ctv_t c2v2e_gt_ctv;

    nabla4_interpolate_verts2cells_unstructured_inlined_c2v(const std::vector<std::array<index_type, 4>> &e2c2v,
        const std::vector<std::array<index_type, 4>> &e2ecv,
        const std::vector<std::array<index_type, 6>> &v2e,
        const std::vector<std::array<index_type, 3>> &c2v,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : c2v2e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(c2v.size(), 12_c).initializer([&e2c2v, &e2ecv, &v2e, &c2v](int i, int j) {
                if (i%2==0) {
                    switch (j) {
                        case 0:
                            return e2c2v[v2e[c2v[i][0]][2]][0];
                        case 1:
                            return e2c2v[v2e[c2v[i][0]][4]][1];
                        case 2:
                            return e2c2v[v2e[c2v[i][2]][4]][1];
                        case 3:
                            return e2c2v[v2e[c2v[i][0]][0]][0];
                        case 4:
                            return e2c2v[v2e[c2v[i][0]][0]][1];
                        case 5:
                            return e2c2v[v2e[c2v[i][0]][1]][1];
                        case 6:
                            return e2c2v[v2e[c2v[i][2]][1]][1];
                        case 7:
                            return e2c2v[v2e[c2v[i][1]][0]][0];
                        case 8:
                            return e2c2v[v2e[c2v[i][1]][0]][1];
                        case 9:
                            return e2c2v[v2e[c2v[i][1]][1]][1];
                        case 10:
                            return e2c2v[v2e[c2v[i][1]][5]][0];
                        case 11:
                            return e2c2v[v2e[c2v[i][1]][3]][1];
                        default:
                            __builtin_unreachable();
                    }
                } else {
                    switch (j) {
                        case 0:
                            return e2c2v[v2e[c2v[i][0]][2]][0];
                        case 1:
                            return e2c2v[v2e[c2v[i][0]][4]][1];
                        case 2:
                            return e2c2v[v2e[c2v[i][0]][0]][0];
                        case 3:
                            return e2c2v[v2e[c2v[i][0]][0]][1];
                        case 4:
                            return e2c2v[v2e[c2v[i][0]][1]][1];
                        case 5:
                            return e2c2v[v2e[c2v[i][1]][0]][0];
                        case 6:
                            return e2c2v[v2e[c2v[i][1]][0]][1];
                        case 7:
                            return e2c2v[v2e[c2v[i][2]][1]][0];
                        case 8:
                            return e2c2v[v2e[c2v[i][2]][1]][1];
                        case 9:
                            return e2c2v[v2e[c2v[i][1]][5]][0];
                        case 10:
                            return e2c2v[v2e[c2v[i][1]][3]][1];
                        case 11:
                            return e2c2v[v2e[c2v[i][2]][3]][1];
                        default:
                            __builtin_unreachable();
                    }
                }
            }).build()),
            c2v2e2c2v_gt_ctv(c2v2e2c2v_gt->const_target_view()),
            c2v2e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(c2v.size(), 72_c).initializer([&e2c2v, &e2ecv, &v2e, &c2v](int i, int j) {
                return e2ecv[v2e[c2v[i][j/24]][(j%24)/4]][j%4];
            }).build()),
            c2v2e2ecv_gt_ctv(c2v2e2ecv_gt->const_target_view()),
            c2v2e_gt(storage::builder<T>.template type<index_type>().dimensions(c2v.size(), 15_c).initializer([&v2e, &c2v](int i, int j) {
                if (i%2==0) {
                    switch (j) {
                        case 0:
                            return v2e[c2v[i][0]][2];
                        case 1:
                            return v2e[c2v[i][2]][2];
                        case 2:
                            return v2e[c2v[i][1]][2];
                        case 3:
                            return v2e[c2v[i][2]][3];
                        case 4:
                            return v2e[c2v[i][1]][3];
                        case 5:
                            return v2e[c2v[i][0]][0];
                        case 6:
                            return v2e[c2v[i][0]][1];
                        case 7:
                            return v2e[c2v[i][2]][1];
                        case 8:
                            return v2e[c2v[i][1]][0];
                        case 9:
                            return v2e[c2v[i][1]][1];
                        case 10:
                            return v2e[c2v[i][0]][4];
                        case 11:
                            return v2e[c2v[i][2]][4];
                        case 12:
                            return v2e[c2v[i][0]][5];
                        case 13:
                            return v2e[c2v[i][2]][5];
                        case 14:
                            return v2e[c2v[i][1]][5];
                        default:
                            __builtin_unreachable();
                    }
                } else {
                    switch (j) {
                        case 0:
                            return v2e[c2v[i][0]][2];
                        case 1:
                            return v2e[c2v[i][1]][2];
                        case 2:
                            return v2e[c2v[i][0]][3];
                        case 3:
                            return v2e[c2v[i][1]][3];
                        case 4:
                            return v2e[c2v[i][2]][3];
                        case 5:
                            return v2e[c2v[i][0]][0];
                        case 6:
                            return v2e[c2v[i][0]][1];
                        case 7:
                            return v2e[c2v[i][1]][0];
                        case 8:
                            return v2e[c2v[i][1]][1];
                        case 9:
                            return v2e[c2v[i][2]][1];
                        case 10:
                            return v2e[c2v[i][0]][4];
                        case 11:
                            return v2e[c2v[i][1]][4];
                        case 12:
                            return v2e[c2v[i][2]][4];
                        case 13:
                            return v2e[c2v[i][1]][5];
                        case 14:
                            return v2e[c2v[i][2]][5];
                        default:
                            __builtin_unreachable();
                    }
                }
            }).build()),
            c2v2e_gt_ctv(c2v2e_gt->const_target_view()),
          nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim),
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt()),
          verts2cells_data(c2v,
              VertexDim,
              CellDim,
              KDim,
              std::get<0>(interpolate_data.get_output_gt()),
              std::get<1>(interpolate_data.get_output_gt())){};

    nabla4_interpolate_verts2cells_unstructured_inlined_c2v(const std::vector<std::array<index_type, 4>> &e2c2v,
        const std::vector<std::array<index_type, 4>> &e2ecv,
        const std::vector<std::array<index_type, 6>> &v2e,
        const std::vector<std::array<index_type, 3>> &c2v,
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
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2,
        const std::vector<std::vector<WP_TYPE>> &ptr_c_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_c_coeff_2)
        : c2v2e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(c2v.size(), 12_c).initializer([&e2c2v, &e2ecv, &v2e, &c2v](int i, int j) {
                if (i%2==0) {
                    switch (j) {
                        case 0:
                            return e2c2v[v2e[c2v[i][0]][2]][0];
                        case 1:
                            return e2c2v[v2e[c2v[i][0]][4]][1];
                        case 2:
                            return e2c2v[v2e[c2v[i][2]][4]][1];
                        case 3:
                            return e2c2v[v2e[c2v[i][0]][0]][0];
                        case 4:
                            return e2c2v[v2e[c2v[i][0]][0]][1];
                        case 5:
                            return e2c2v[v2e[c2v[i][0]][1]][1];
                        case 6:
                            return e2c2v[v2e[c2v[i][2]][1]][1];
                        case 7:
                            return e2c2v[v2e[c2v[i][1]][0]][0];
                        case 8:
                            return e2c2v[v2e[c2v[i][1]][0]][1];
                        case 9:
                            return e2c2v[v2e[c2v[i][1]][1]][1];
                        case 10:
                            return e2c2v[v2e[c2v[i][1]][5]][0];
                        case 11:
                            return e2c2v[v2e[c2v[i][1]][3]][1];
                        default:
                            __builtin_unreachable();
                    }
                } else {
                    switch (j) {
                        case 0:
                            return e2c2v[v2e[c2v[i][0]][2]][0];
                        case 1:
                            return e2c2v[v2e[c2v[i][0]][4]][1];
                        case 2:
                            return e2c2v[v2e[c2v[i][0]][0]][0];
                        case 3:
                            return e2c2v[v2e[c2v[i][0]][0]][1];
                        case 4:
                            return e2c2v[v2e[c2v[i][0]][1]][1];
                        case 5:
                            return e2c2v[v2e[c2v[i][1]][0]][0];
                        case 6:
                            return e2c2v[v2e[c2v[i][1]][0]][1];
                        case 7:
                            return e2c2v[v2e[c2v[i][2]][1]][0];
                        case 8:
                            return e2c2v[v2e[c2v[i][2]][1]][1];
                        case 9:
                            return e2c2v[v2e[c2v[i][1]][5]][0];
                        case 10:
                            return e2c2v[v2e[c2v[i][1]][3]][1];
                        case 11:
                            return e2c2v[v2e[c2v[i][2]][3]][1];
                        default:
                            __builtin_unreachable();
                    }
                }
            }).build()),
            c2v2e2c2v_gt_ctv(c2v2e2c2v_gt->const_target_view()),
            c2v2e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(c2v.size(), 72_c).initializer([&e2c2v, &e2ecv, &v2e, &c2v](int i, int j) {
                return e2ecv[v2e[c2v[i][j/24]][(j%24)/4]][j%4];
            }).build()),
            c2v2e2ecv_gt_ctv(c2v2e2ecv_gt->const_target_view()),
            c2v2e_gt(storage::builder<T>.template type<index_type>().dimensions(c2v.size(), 15_c).initializer([&v2e, &c2v](int i, int j) {
                if (i%2==0) {
                    switch (j) {
                        case 0:
                            return v2e[c2v[i][0]][2];
                        case 1:
                            return v2e[c2v[i][2]][2];
                        case 2:
                            return v2e[c2v[i][1]][2];
                        case 3:
                            return v2e[c2v[i][2]][3];
                        case 4:
                            return v2e[c2v[i][1]][3];
                        case 5:
                            return v2e[c2v[i][0]][0];
                        case 6:
                            return v2e[c2v[i][0]][1];
                        case 7:
                            return v2e[c2v[i][2]][1];
                        case 8:
                            return v2e[c2v[i][1]][0];
                        case 9:
                            return v2e[c2v[i][1]][1];
                        case 10:
                            return v2e[c2v[i][0]][4];
                        case 11:
                            return v2e[c2v[i][2]][4];
                        case 12:
                            return v2e[c2v[i][0]][5];
                        case 13:
                            return v2e[c2v[i][2]][5];
                        case 14:
                            return v2e[c2v[i][1]][5];
                        default:
                            __builtin_unreachable();
                    }
                } else {
                    switch (j) {
                        case 0:
                            return v2e[c2v[i][0]][2];
                        case 1:
                            return v2e[c2v[i][1]][2];
                        case 2:
                            return v2e[c2v[i][0]][3];
                        case 3:
                            return v2e[c2v[i][1]][3];
                        case 4:
                            return v2e[c2v[i][2]][3];
                        case 5:
                            return v2e[c2v[i][0]][0];
                        case 6:
                            return v2e[c2v[i][0]][1];
                        case 7:
                            return v2e[c2v[i][1]][0];
                        case 8:
                            return v2e[c2v[i][1]][1];
                        case 9:
                            return v2e[c2v[i][2]][1];
                        case 10:
                            return v2e[c2v[i][0]][4];
                        case 11:
                            return v2e[c2v[i][1]][4];
                        case 12:
                            return v2e[c2v[i][2]][4];
                        case 13:
                            return v2e[c2v[i][1]][5];
                        case 14:
                            return v2e[c2v[i][2]][5];
                        default:
                            __builtin_unreachable();
                    }
                }
            }).build()),
            c2v2e_gt_ctv(c2v2e_gt->const_target_view()),
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
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt(), ptr_coeff_1, ptr_coeff_2),
          verts2cells_data(c2v,
              VertexDim,
              CellDim,
              KDim,
              std::get<0>(interpolate_data.get_output_gt()),
              std::get<1>(interpolate_data.get_output_gt()),
              ptr_c_coeff_1,
              ptr_c_coeff_2){};

    auto get_output() -> decltype(verts2cells_data.get_output()) { return verts2cells_data.get_output(); }

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
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop<int>() {
    return {32, 4, 1, 128};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop =
    get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop<index_type>();

static __global__ void
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 5)
__launch_bounds__(block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop.size)
#else
__maxnreg__(255)
#endif
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_c2v_unstructured(index_type verts2cells_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_interpolate_verts2cells_unstructured_inlined_c2v<storage::gpu>::neighbors_c2v2e2c2v_gt_ctv_t
            c2v2e2c2v_gt_ctv,
        nabla4_interpolate_verts2cells_unstructured_inlined_c2v<storage::gpu>::neighbors_c2v2e2ecv_gt_ctv_t
            c2v2e2ecv_gt_ctv,
        nabla4_interpolate_verts2cells_unstructured_inlined_c2v<storage::gpu>::neighbors_c2v2e_gt_ctv_t c2v2e_gt_ctv,
        verts2cells_unstructured<storage::gpu>::neighbors_gt_ctv_t c2v_gt_ctv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_c_coeff_1_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_c_coeff_2_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= verts2cells_output_size)
        return;
    const index_type c2v[3]{c2v_gt_ctv(cell_index, 0), c2v_gt_ctv(cell_index, 1), c2v_gt_ctv(cell_index, 2)};
    const bool upward_cell{cell_index % 2 == 0};
    const auto v2e = upward_cell ? std::array<index_type, 18>{c2v2e_gt_ctv(cell_index, 5),
                                       c2v2e_gt_ctv(cell_index, 6),
                                       c2v2e_gt_ctv(cell_index, 0),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 10),
                                       c2v2e_gt_ctv(cell_index, 12),
                                       c2v2e_gt_ctv(cell_index, 8),
                                       c2v2e_gt_ctv(cell_index, 9),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 4),
                                       c2v2e_gt_ctv(cell_index, 13),
                                       c2v2e_gt_ctv(cell_index, 14),
                                       c2v2e_gt_ctv(cell_index, 6),
                                       c2v2e_gt_ctv(cell_index, 7),
                                       c2v2e_gt_ctv(cell_index, 1),
                                       c2v2e_gt_ctv(cell_index, 3),
                                       c2v2e_gt_ctv(cell_index, 11),
                                       c2v2e_gt_ctv(cell_index, 13)}
                                 : std::array<index_type, 18>{c2v2e_gt_ctv(cell_index, 5),
                                       c2v2e_gt_ctv(cell_index, 6),
                                       c2v2e_gt_ctv(cell_index, 0),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 10),
                                       c2v2e_gt_ctv(cell_index, 11),
                                       c2v2e_gt_ctv(cell_index, 7),
                                       c2v2e_gt_ctv(cell_index, 8),
                                       c2v2e_gt_ctv(cell_index, 1),
                                       c2v2e_gt_ctv(cell_index, 3),
                                       c2v2e_gt_ctv(cell_index, 11),
                                       c2v2e_gt_ctv(cell_index, 13),
                                       c2v2e_gt_ctv(cell_index, 8),
                                       c2v2e_gt_ctv(cell_index, 9),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 4),
                                       c2v2e_gt_ctv(cell_index, 12),
                                       c2v2e_gt_ctv(cell_index, 14)};
    const auto e2c2v = upward_cell ? std::array<index_type, 72>{c2v2e2c2v_gt_ctv(cell_index, 3), // 0
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7), // 1
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 4), // 2
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 4)}
                                   : std::array<index_type, 72>{c2v2e2c2v_gt_ctv(cell_index, 2), // 0
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 5), // 1
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 6), // 2
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 6)};
    const std::array<index_type, 72> e2ecv{c2v2e2ecv_gt_ctv(cell_index, 0),
        c2v2e2ecv_gt_ctv(cell_index, 1),
        c2v2e2ecv_gt_ctv(cell_index, 2),
        c2v2e2ecv_gt_ctv(cell_index, 3),
        c2v2e2ecv_gt_ctv(cell_index, 4),
        c2v2e2ecv_gt_ctv(cell_index, 5),
        c2v2e2ecv_gt_ctv(cell_index, 6),
        c2v2e2ecv_gt_ctv(cell_index, 7),
        c2v2e2ecv_gt_ctv(cell_index, 8),
        c2v2e2ecv_gt_ctv(cell_index, 9),
        c2v2e2ecv_gt_ctv(cell_index, 10),
        c2v2e2ecv_gt_ctv(cell_index, 11),
        c2v2e2ecv_gt_ctv(cell_index, 12),
        c2v2e2ecv_gt_ctv(cell_index, 13),
        c2v2e2ecv_gt_ctv(cell_index, 14),
        c2v2e2ecv_gt_ctv(cell_index, 15),
        c2v2e2ecv_gt_ctv(cell_index, 16),
        c2v2e2ecv_gt_ctv(cell_index, 17),
        c2v2e2ecv_gt_ctv(cell_index, 18),
        c2v2e2ecv_gt_ctv(cell_index, 19),
        c2v2e2ecv_gt_ctv(cell_index, 20),
        c2v2e2ecv_gt_ctv(cell_index, 21),
        c2v2e2ecv_gt_ctv(cell_index, 22),
        c2v2e2ecv_gt_ctv(cell_index, 23),
        c2v2e2ecv_gt_ctv(cell_index, 24),
        c2v2e2ecv_gt_ctv(cell_index, 25),
        c2v2e2ecv_gt_ctv(cell_index, 26),
        c2v2e2ecv_gt_ctv(cell_index, 27),
        c2v2e2ecv_gt_ctv(cell_index, 28),
        c2v2e2ecv_gt_ctv(cell_index, 29),
        c2v2e2ecv_gt_ctv(cell_index, 30),
        c2v2e2ecv_gt_ctv(cell_index, 31),
        c2v2e2ecv_gt_ctv(cell_index, 32),
        c2v2e2ecv_gt_ctv(cell_index, 33),
        c2v2e2ecv_gt_ctv(cell_index, 34),
        c2v2e2ecv_gt_ctv(cell_index, 35),
        c2v2e2ecv_gt_ctv(cell_index, 36),
        c2v2e2ecv_gt_ctv(cell_index, 37),
        c2v2e2ecv_gt_ctv(cell_index, 38),
        c2v2e2ecv_gt_ctv(cell_index, 39),
        c2v2e2ecv_gt_ctv(cell_index, 40),
        c2v2e2ecv_gt_ctv(cell_index, 41),
        c2v2e2ecv_gt_ctv(cell_index, 42),
        c2v2e2ecv_gt_ctv(cell_index, 43),
        c2v2e2ecv_gt_ctv(cell_index, 44),
        c2v2e2ecv_gt_ctv(cell_index, 45),
        c2v2e2ecv_gt_ctv(cell_index, 46),
        c2v2e2ecv_gt_ctv(cell_index, 47),
        c2v2e2ecv_gt_ctv(cell_index, 48),
        c2v2e2ecv_gt_ctv(cell_index, 49),
        c2v2e2ecv_gt_ctv(cell_index, 50),
        c2v2e2ecv_gt_ctv(cell_index, 51),
        c2v2e2ecv_gt_ctv(cell_index, 52),
        c2v2e2ecv_gt_ctv(cell_index, 53),
        c2v2e2ecv_gt_ctv(cell_index, 54),
        c2v2e2ecv_gt_ctv(cell_index, 55),
        c2v2e2ecv_gt_ctv(cell_index, 56),
        c2v2e2ecv_gt_ctv(cell_index, 57),
        c2v2e2ecv_gt_ctv(cell_index, 58),
        c2v2e2ecv_gt_ctv(cell_index, 59),
        c2v2e2ecv_gt_ctv(cell_index, 60),
        c2v2e2ecv_gt_ctv(cell_index, 61),
        c2v2e2ecv_gt_ctv(cell_index, 62),
        c2v2e2ecv_gt_ctv(cell_index, 63),
        c2v2e2ecv_gt_ctv(cell_index, 64),
        c2v2e2ecv_gt_ctv(cell_index, 65),
        c2v2e2ecv_gt_ctv(cell_index, 66),
        c2v2e2ecv_gt_ctv(cell_index, 67),
        c2v2e2ecv_gt_ctv(cell_index, 68),
        c2v2e2ecv_gt_ctv(cell_index, 69),
        c2v2e2ecv_gt_ctv(cell_index, 70),
        c2v2e2ecv_gt_ctv(cell_index, 71)};
    const std::array<WP_TYPE, 72> primal_normal_vert_v1{primal_normal_vert_v1_gt_tv(e2ecv[0]),
        primal_normal_vert_v1_gt_tv(e2ecv[1]),
        primal_normal_vert_v1_gt_tv(e2ecv[2]),
        primal_normal_vert_v1_gt_tv(e2ecv[3]),
        primal_normal_vert_v1_gt_tv(e2ecv[4]),
        primal_normal_vert_v1_gt_tv(e2ecv[5]),
        primal_normal_vert_v1_gt_tv(e2ecv[6]),
        primal_normal_vert_v1_gt_tv(e2ecv[7]),
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
        primal_normal_vert_v1_gt_tv(e2ecv[23]),
        primal_normal_vert_v1_gt_tv(e2ecv[24]),
        primal_normal_vert_v1_gt_tv(e2ecv[25]),
        primal_normal_vert_v1_gt_tv(e2ecv[26]),
        primal_normal_vert_v1_gt_tv(e2ecv[27]),
        primal_normal_vert_v1_gt_tv(e2ecv[28]),
        primal_normal_vert_v1_gt_tv(e2ecv[29]),
        primal_normal_vert_v1_gt_tv(e2ecv[30]),
        primal_normal_vert_v1_gt_tv(e2ecv[31]),
        primal_normal_vert_v1_gt_tv(e2ecv[32]),
        primal_normal_vert_v1_gt_tv(e2ecv[33]),
        primal_normal_vert_v1_gt_tv(e2ecv[34]),
        primal_normal_vert_v1_gt_tv(e2ecv[35]),
        primal_normal_vert_v1_gt_tv(e2ecv[36]),
        primal_normal_vert_v1_gt_tv(e2ecv[37]),
        primal_normal_vert_v1_gt_tv(e2ecv[38]),
        primal_normal_vert_v1_gt_tv(e2ecv[39]),
        primal_normal_vert_v1_gt_tv(e2ecv[40]),
        primal_normal_vert_v1_gt_tv(e2ecv[41]),
        primal_normal_vert_v1_gt_tv(e2ecv[42]),
        primal_normal_vert_v1_gt_tv(e2ecv[43]),
        primal_normal_vert_v1_gt_tv(e2ecv[44]),
        primal_normal_vert_v1_gt_tv(e2ecv[45]),
        primal_normal_vert_v1_gt_tv(e2ecv[46]),
        primal_normal_vert_v1_gt_tv(e2ecv[47]),
        primal_normal_vert_v1_gt_tv(e2ecv[48]),
        primal_normal_vert_v1_gt_tv(e2ecv[49]),
        primal_normal_vert_v1_gt_tv(e2ecv[50]),
        primal_normal_vert_v1_gt_tv(e2ecv[51]),
        primal_normal_vert_v1_gt_tv(e2ecv[52]),
        primal_normal_vert_v1_gt_tv(e2ecv[53]),
        primal_normal_vert_v1_gt_tv(e2ecv[54]),
        primal_normal_vert_v1_gt_tv(e2ecv[55]),
        primal_normal_vert_v1_gt_tv(e2ecv[56]),
        primal_normal_vert_v1_gt_tv(e2ecv[57]),
        primal_normal_vert_v1_gt_tv(e2ecv[58]),
        primal_normal_vert_v1_gt_tv(e2ecv[59]),
        primal_normal_vert_v1_gt_tv(e2ecv[60]),
        primal_normal_vert_v1_gt_tv(e2ecv[61]),
        primal_normal_vert_v1_gt_tv(e2ecv[62]),
        primal_normal_vert_v1_gt_tv(e2ecv[63]),
        primal_normal_vert_v1_gt_tv(e2ecv[64]),
        primal_normal_vert_v1_gt_tv(e2ecv[65]),
        primal_normal_vert_v1_gt_tv(e2ecv[66]),
        primal_normal_vert_v1_gt_tv(e2ecv[67]),
        primal_normal_vert_v1_gt_tv(e2ecv[68]),
        primal_normal_vert_v1_gt_tv(e2ecv[69]),
        primal_normal_vert_v1_gt_tv(e2ecv[70]),
        primal_normal_vert_v1_gt_tv(e2ecv[71])};
    const std::array<WP_TYPE, 72> primal_normal_vert_v2{primal_normal_vert_v2_gt_tv(e2ecv[0]),
        primal_normal_vert_v2_gt_tv(e2ecv[1]),
        primal_normal_vert_v2_gt_tv(e2ecv[2]),
        primal_normal_vert_v2_gt_tv(e2ecv[3]),
        primal_normal_vert_v2_gt_tv(e2ecv[4]),
        primal_normal_vert_v2_gt_tv(e2ecv[5]),
        primal_normal_vert_v2_gt_tv(e2ecv[6]),
        primal_normal_vert_v2_gt_tv(e2ecv[7]),
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
        primal_normal_vert_v2_gt_tv(e2ecv[23]),
        primal_normal_vert_v2_gt_tv(e2ecv[24]),
        primal_normal_vert_v2_gt_tv(e2ecv[25]),
        primal_normal_vert_v2_gt_tv(e2ecv[26]),
        primal_normal_vert_v2_gt_tv(e2ecv[27]),
        primal_normal_vert_v2_gt_tv(e2ecv[28]),
        primal_normal_vert_v2_gt_tv(e2ecv[29]),
        primal_normal_vert_v2_gt_tv(e2ecv[30]),
        primal_normal_vert_v2_gt_tv(e2ecv[31]),
        primal_normal_vert_v2_gt_tv(e2ecv[32]),
        primal_normal_vert_v2_gt_tv(e2ecv[33]),
        primal_normal_vert_v2_gt_tv(e2ecv[34]),
        primal_normal_vert_v2_gt_tv(e2ecv[35]),
        primal_normal_vert_v2_gt_tv(e2ecv[36]),
        primal_normal_vert_v2_gt_tv(e2ecv[37]),
        primal_normal_vert_v2_gt_tv(e2ecv[38]),
        primal_normal_vert_v2_gt_tv(e2ecv[39]),
        primal_normal_vert_v2_gt_tv(e2ecv[40]),
        primal_normal_vert_v2_gt_tv(e2ecv[41]),
        primal_normal_vert_v2_gt_tv(e2ecv[42]),
        primal_normal_vert_v2_gt_tv(e2ecv[43]),
        primal_normal_vert_v2_gt_tv(e2ecv[44]),
        primal_normal_vert_v2_gt_tv(e2ecv[45]),
        primal_normal_vert_v2_gt_tv(e2ecv[46]),
        primal_normal_vert_v2_gt_tv(e2ecv[47]),
        primal_normal_vert_v2_gt_tv(e2ecv[48]),
        primal_normal_vert_v2_gt_tv(e2ecv[49]),
        primal_normal_vert_v2_gt_tv(e2ecv[50]),
        primal_normal_vert_v2_gt_tv(e2ecv[51]),
        primal_normal_vert_v2_gt_tv(e2ecv[52]),
        primal_normal_vert_v2_gt_tv(e2ecv[53]),
        primal_normal_vert_v2_gt_tv(e2ecv[54]),
        primal_normal_vert_v2_gt_tv(e2ecv[55]),
        primal_normal_vert_v2_gt_tv(e2ecv[56]),
        primal_normal_vert_v2_gt_tv(e2ecv[57]),
        primal_normal_vert_v2_gt_tv(e2ecv[58]),
        primal_normal_vert_v2_gt_tv(e2ecv[59]),
        primal_normal_vert_v2_gt_tv(e2ecv[60]),
        primal_normal_vert_v2_gt_tv(e2ecv[61]),
        primal_normal_vert_v2_gt_tv(e2ecv[62]),
        primal_normal_vert_v2_gt_tv(e2ecv[63]),
        primal_normal_vert_v2_gt_tv(e2ecv[64]),
        primal_normal_vert_v2_gt_tv(e2ecv[65]),
        primal_normal_vert_v2_gt_tv(e2ecv[66]),
        primal_normal_vert_v2_gt_tv(e2ecv[67]),
        primal_normal_vert_v2_gt_tv(e2ecv[68]),
        primal_normal_vert_v2_gt_tv(e2ecv[69]),
        primal_normal_vert_v2_gt_tv(e2ecv[70]),
        primal_normal_vert_v2_gt_tv(e2ecv[71])};
    const std::array<WP_TYPE, 3> ptr_c_coeff1{
        ptr_c_coeff_1_gt_ctv(cell_index, 0), ptr_c_coeff_1_gt_ctv(cell_index, 1), ptr_c_coeff_1_gt_ctv(cell_index, 2)};
    const std::array<WP_TYPE, 3> ptr_c_coeff2{
        ptr_c_coeff_2_gt_ctv(cell_index, 0), ptr_c_coeff_2_gt_ctv(cell_index, 1), ptr_c_coeff_2_gt_ctv(cell_index, 2)};
    for (auto k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim; k_index += gridDim.y * blockDim.y) {
        std::array<WP_TYPE, 18> z_nabla4_e2;
#pragma unroll 18
        for (int edge_index{}; edge_index < 18; ++edge_index) {
            const auto nabv_tang_wp =
                u_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v1[4 * edge_index] +
                v_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v2[4 * edge_index] +
                u_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v1[4 * edge_index + 1] +
                v_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v2[4 * edge_index + 1];
            const auto nabv_norm_wp =
                u_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) * primal_normal_vert_v1[4 * edge_index + 2] +
                v_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) * primal_normal_vert_v2[4 * edge_index + 2] +
                u_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) * primal_normal_vert_v1[4 * edge_index + 3] +
                v_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) * primal_normal_vert_v2[4 * edge_index + 3];
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(v2e[edge_index], k_index);
            const auto inv_vert_vert_length = inv_vert_vert_length_gt_tv(v2e[edge_index]);
            const auto inv_primal_edge_length = inv_primal_edge_length_gt_tv(v2e[edge_index]);
            z_nabla4_e2[edge_index] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * (inv_vert_vert_length * inv_vert_vert_length) +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) * (inv_primal_edge_length * inv_primal_edge_length));
        }
        std::array<VP_TYPE, 3> p_u_out;
        std::array<VP_TYPE, 3> p_v_out;
#pragma unroll 3
        for (int vertex_index{}; vertex_index < 3; ++vertex_index) {
            p_u_out[vertex_index] = z_nabla4_e2[vertex_index * 6] * ptr_coeff_1_gt_ctv(c2v[vertex_index], 0) +
                                    z_nabla4_e2[vertex_index * 6 + 1] * ptr_coeff_1_gt_ctv(c2v[vertex_index], 1) +
                                    z_nabla4_e2[vertex_index * 6 + 2] * ptr_coeff_1_gt_ctv(c2v[vertex_index], 2) +
                                    z_nabla4_e2[vertex_index * 6 + 3] * ptr_coeff_1_gt_ctv(c2v[vertex_index], 3) +
                                    z_nabla4_e2[vertex_index * 6 + 4] * ptr_coeff_1_gt_ctv(c2v[vertex_index], 4) +
                                    z_nabla4_e2[vertex_index * 6 + 5] * ptr_coeff_1_gt_ctv(c2v[vertex_index], 5);
            p_v_out[vertex_index] = z_nabla4_e2[vertex_index * 6] * ptr_coeff_2_gt_ctv(c2v[vertex_index], 0) +
                                    z_nabla4_e2[vertex_index * 6 + 1] * ptr_coeff_2_gt_ctv(c2v[vertex_index], 1) +
                                    z_nabla4_e2[vertex_index * 6 + 2] * ptr_coeff_2_gt_ctv(c2v[vertex_index], 2) +
                                    z_nabla4_e2[vertex_index * 6 + 3] * ptr_coeff_2_gt_ctv(c2v[vertex_index], 3) +
                                    z_nabla4_e2[vertex_index * 6 + 4] * ptr_coeff_2_gt_ctv(c2v[vertex_index], 4) +
                                    z_nabla4_e2[vertex_index * 6 + 5] * ptr_coeff_2_gt_ctv(c2v[vertex_index], 5);
        }
        p_cell_out_gt_tv(cell_index, k_index) =
            (p_u_out[0] * ptr_c_coeff1[0] + p_u_out[1] * ptr_c_coeff1[1] + p_u_out[2] * ptr_c_coeff1[2] +
                p_v_out[0] * ptr_c_coeff2[0] + p_v_out[1] * ptr_c_coeff2[1] + p_v_out[2] * ptr_c_coeff2[2]) /
            2;
    }
};

template <typename T>
inline void nabla4_interpolate_verts2cells_unstructured_inlined_c2v<T>::run_gpu_kloop_helper() {
    constexpr dim3 tblocks(block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop.x,
        block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop.y,
        block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_kloop.z);
    dim3 grid((verts2cells_data.output_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_c2v_unstructured<<<grid, tblocks>>>(
        verts2cells_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        c2v2e2c2v_gt_ctv,
        c2v2e2ecv_gt_ctv,
        c2v2e_gt_ctv,
        verts2cells_data.c2v_gt_ctv,
        nabla4_data.u_vert_gt_tv,
        nabla4_data.v_vert_gt_tv,
        nabla4_data.primal_normal_vert_v1_gt_tv,
        nabla4_data.primal_normal_vert_v2_gt_tv,
        nabla4_data.z_nabla2_e_gt_tv,
        nabla4_data.inv_vert_vert_length_gt_tv,
        nabla4_data.inv_primal_edge_length_gt_tv,
        interpolate_data.ptr_coeff_1_gt_ctv,
        interpolate_data.ptr_coeff_2_gt_ctv,
        verts2cells_data.ptr_coeff_1_gt_ctv,
        verts2cells_data.ptr_coeff_2_gt_ctv,
        verts2cells_data.p_cell_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

template <typename T>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive<int>() {
    return {32, 4, 1, 128};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive =
    get_block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive<index_type>();

static __global__ void __launch_bounds__(block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive.size)
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_c2v_unstructured(index_type verts2cells_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_interpolate_verts2cells_unstructured_inlined_c2v<storage::gpu>::neighbors_c2v2e2c2v_gt_ctv_t
            c2v2e2c2v_gt_ctv,
        nabla4_interpolate_verts2cells_unstructured_inlined_c2v<storage::gpu>::neighbors_c2v2e2ecv_gt_ctv_t
            c2v2e2ecv_gt_ctv,
        nabla4_interpolate_verts2cells_unstructured_inlined_c2v<storage::gpu>::neighbors_c2v2e_gt_ctv_t c2v2e_gt_ctv,
        verts2cells_unstructured<storage::gpu>::neighbors_gt_ctv_t c2v_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_c_coeff_1_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_c_coeff_2_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (cell_index >= verts2cells_output_size || k_index >= KDim)
        return;
    const index_type c2v[3]{c2v_gt_tv(cell_index, 0), c2v_gt_tv(cell_index, 1), c2v_gt_tv(cell_index, 2)};
    const bool upward_cell{cell_index % 2 == 0};
    const auto v2e = upward_cell ? std::array<index_type, 18>{c2v2e_gt_ctv(cell_index, 5),
                                       c2v2e_gt_ctv(cell_index, 6),
                                       c2v2e_gt_ctv(cell_index, 0),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 10),
                                       c2v2e_gt_ctv(cell_index, 12),
                                       c2v2e_gt_ctv(cell_index, 8),
                                       c2v2e_gt_ctv(cell_index, 9),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 4),
                                       c2v2e_gt_ctv(cell_index, 13),
                                       c2v2e_gt_ctv(cell_index, 14),
                                       c2v2e_gt_ctv(cell_index, 6),
                                       c2v2e_gt_ctv(cell_index, 7),
                                       c2v2e_gt_ctv(cell_index, 1),
                                       c2v2e_gt_ctv(cell_index, 3),
                                       c2v2e_gt_ctv(cell_index, 11),
                                       c2v2e_gt_ctv(cell_index, 13)}
                                 : std::array<index_type, 18>{c2v2e_gt_ctv(cell_index, 5),
                                       c2v2e_gt_ctv(cell_index, 6),
                                       c2v2e_gt_ctv(cell_index, 0),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 10),
                                       c2v2e_gt_ctv(cell_index, 11),
                                       c2v2e_gt_ctv(cell_index, 7),
                                       c2v2e_gt_ctv(cell_index, 8),
                                       c2v2e_gt_ctv(cell_index, 1),
                                       c2v2e_gt_ctv(cell_index, 3),
                                       c2v2e_gt_ctv(cell_index, 11),
                                       c2v2e_gt_ctv(cell_index, 13),
                                       c2v2e_gt_ctv(cell_index, 8),
                                       c2v2e_gt_ctv(cell_index, 9),
                                       c2v2e_gt_ctv(cell_index, 2),
                                       c2v2e_gt_ctv(cell_index, 4),
                                       c2v2e_gt_ctv(cell_index, 12),
                                       c2v2e_gt_ctv(cell_index, 14)};
    const auto e2c2v = upward_cell ? std::array<index_type, 72>{c2v2e2c2v_gt_ctv(cell_index, 3), // 0
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7), // 1 // todo from here
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 4), // 2
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 4)}
                                   : std::array<index_type, 72>{c2v2e2c2v_gt_ctv(cell_index, 2), // 0
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 4), // wrong
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 1),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 0),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 5), // 1 // todo from here
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 2),
                                         c2v2e2c2v_gt_ctv(cell_index, 9),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 5),
                                         c2v2e2c2v_gt_ctv(cell_index, 6), // 2
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 6),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 4),
                                         c2v2e2c2v_gt_ctv(cell_index, 8),
                                         c2v2e2c2v_gt_ctv(cell_index, 3),
                                         c2v2e2c2v_gt_ctv(cell_index, 10),
                                         c2v2e2c2v_gt_ctv(cell_index, 7),
                                         c2v2e2c2v_gt_ctv(cell_index, 11),
                                         c2v2e2c2v_gt_ctv(cell_index, 6)};
    const std::array<index_type, 72> e2ecv{c2v2e2ecv_gt_ctv(cell_index, 0),
        c2v2e2ecv_gt_ctv(cell_index, 1),
        c2v2e2ecv_gt_ctv(cell_index, 2),
        c2v2e2ecv_gt_ctv(cell_index, 3),
        c2v2e2ecv_gt_ctv(cell_index, 4),
        c2v2e2ecv_gt_ctv(cell_index, 5),
        c2v2e2ecv_gt_ctv(cell_index, 6),
        c2v2e2ecv_gt_ctv(cell_index, 7),
        c2v2e2ecv_gt_ctv(cell_index, 8),
        c2v2e2ecv_gt_ctv(cell_index, 9),
        c2v2e2ecv_gt_ctv(cell_index, 10),
        c2v2e2ecv_gt_ctv(cell_index, 11),
        c2v2e2ecv_gt_ctv(cell_index, 12),
        c2v2e2ecv_gt_ctv(cell_index, 13),
        c2v2e2ecv_gt_ctv(cell_index, 14),
        c2v2e2ecv_gt_ctv(cell_index, 15),
        c2v2e2ecv_gt_ctv(cell_index, 16),
        c2v2e2ecv_gt_ctv(cell_index, 17),
        c2v2e2ecv_gt_ctv(cell_index, 18),
        c2v2e2ecv_gt_ctv(cell_index, 19),
        c2v2e2ecv_gt_ctv(cell_index, 20),
        c2v2e2ecv_gt_ctv(cell_index, 21),
        c2v2e2ecv_gt_ctv(cell_index, 22),
        c2v2e2ecv_gt_ctv(cell_index, 23),
        c2v2e2ecv_gt_ctv(cell_index, 24),
        c2v2e2ecv_gt_ctv(cell_index, 25),
        c2v2e2ecv_gt_ctv(cell_index, 26),
        c2v2e2ecv_gt_ctv(cell_index, 27),
        c2v2e2ecv_gt_ctv(cell_index, 28),
        c2v2e2ecv_gt_ctv(cell_index, 29),
        c2v2e2ecv_gt_ctv(cell_index, 30),
        c2v2e2ecv_gt_ctv(cell_index, 31),
        c2v2e2ecv_gt_ctv(cell_index, 32),
        c2v2e2ecv_gt_ctv(cell_index, 33),
        c2v2e2ecv_gt_ctv(cell_index, 34),
        c2v2e2ecv_gt_ctv(cell_index, 35),
        c2v2e2ecv_gt_ctv(cell_index, 36),
        c2v2e2ecv_gt_ctv(cell_index, 37),
        c2v2e2ecv_gt_ctv(cell_index, 38),
        c2v2e2ecv_gt_ctv(cell_index, 39),
        c2v2e2ecv_gt_ctv(cell_index, 40),
        c2v2e2ecv_gt_ctv(cell_index, 41),
        c2v2e2ecv_gt_ctv(cell_index, 42),
        c2v2e2ecv_gt_ctv(cell_index, 43),
        c2v2e2ecv_gt_ctv(cell_index, 44),
        c2v2e2ecv_gt_ctv(cell_index, 45),
        c2v2e2ecv_gt_ctv(cell_index, 46),
        c2v2e2ecv_gt_ctv(cell_index, 47),
        c2v2e2ecv_gt_ctv(cell_index, 48),
        c2v2e2ecv_gt_ctv(cell_index, 49),
        c2v2e2ecv_gt_ctv(cell_index, 50),
        c2v2e2ecv_gt_ctv(cell_index, 51),
        c2v2e2ecv_gt_ctv(cell_index, 52),
        c2v2e2ecv_gt_ctv(cell_index, 53),
        c2v2e2ecv_gt_ctv(cell_index, 54),
        c2v2e2ecv_gt_ctv(cell_index, 55),
        c2v2e2ecv_gt_ctv(cell_index, 56),
        c2v2e2ecv_gt_ctv(cell_index, 57),
        c2v2e2ecv_gt_ctv(cell_index, 58),
        c2v2e2ecv_gt_ctv(cell_index, 59),
        c2v2e2ecv_gt_ctv(cell_index, 60),
        c2v2e2ecv_gt_ctv(cell_index, 61),
        c2v2e2ecv_gt_ctv(cell_index, 62),
        c2v2e2ecv_gt_ctv(cell_index, 63),
        c2v2e2ecv_gt_ctv(cell_index, 64),
        c2v2e2ecv_gt_ctv(cell_index, 65),
        c2v2e2ecv_gt_ctv(cell_index, 66),
        c2v2e2ecv_gt_ctv(cell_index, 67),
        c2v2e2ecv_gt_ctv(cell_index, 68),
        c2v2e2ecv_gt_ctv(cell_index, 69),
        c2v2e2ecv_gt_ctv(cell_index, 70),
        c2v2e2ecv_gt_ctv(cell_index, 71)};
    std::array<WP_TYPE, 18> z_nabla4_e2;
#pragma unroll 18
    for (int edge_index{}; edge_index < 18; ++edge_index) {
        const auto nabv_tang_wp =
            u_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v1_gt_tv(e2ecv[4 * edge_index]) +
            v_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v2_gt_tv(e2ecv[4 * edge_index]) +
            u_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v1_gt_tv(e2ecv[4 * edge_index + 1]) +
            v_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v2_gt_tv(e2ecv[4 * edge_index + 1]);
        const auto nabv_norm_wp =
            u_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) * primal_normal_vert_v1_gt_tv(e2ecv[4 * edge_index + 2]) +
            v_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) * primal_normal_vert_v2_gt_tv(e2ecv[4 * edge_index + 2]) +
            u_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) * primal_normal_vert_v1_gt_tv(e2ecv[4 * edge_index + 3]) +
            v_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) * primal_normal_vert_v2_gt_tv(e2ecv[4 * edge_index + 3]);
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(v2e[edge_index], k_index);
        const WP_TYPE inv_vert_vert_length = inv_vert_vert_length_gt_tv(v2e[edge_index]);
        const WP_TYPE inv_vert_vert_length_sqr = inv_vert_vert_length * inv_vert_vert_length;
        const WP_TYPE inv_primal_edge_length = inv_primal_edge_length_gt_tv(v2e[edge_index]);
        const WP_TYPE inv_primal_edge_length_sqr = inv_primal_edge_length * inv_primal_edge_length;
        z_nabla4_e2[edge_index] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                            (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    }
    std::array<VP_TYPE, 3> p_u_out;
    std::array<VP_TYPE, 3> p_v_out;
#pragma unroll 3
    for (int vertex_index{}; vertex_index < 3; ++vertex_index) {
        const std::array<WP_TYPE, 6> ptr_coeff_1{ptr_coeff_1_gt_ctv(c2v[vertex_index], 0),
            ptr_coeff_1_gt_ctv(c2v[vertex_index], 1),
            ptr_coeff_1_gt_ctv(c2v[vertex_index], 2),
            ptr_coeff_1_gt_ctv(c2v[vertex_index], 3),
            ptr_coeff_1_gt_ctv(c2v[vertex_index], 4),
            ptr_coeff_1_gt_ctv(c2v[vertex_index], 5)};
        const std::array<WP_TYPE, 6> ptr_coeff_2{ptr_coeff_2_gt_ctv(c2v[vertex_index], 0),
            ptr_coeff_2_gt_ctv(c2v[vertex_index], 1),
            ptr_coeff_2_gt_ctv(c2v[vertex_index], 2),
            ptr_coeff_2_gt_ctv(c2v[vertex_index], 3),
            ptr_coeff_2_gt_ctv(c2v[vertex_index], 4),
            ptr_coeff_2_gt_ctv(c2v[vertex_index], 5)};
        p_u_out[vertex_index] =
            z_nabla4_e2[vertex_index * 6] * ptr_coeff_1[0] + z_nabla4_e2[vertex_index * 6 + 1] * ptr_coeff_1[1] +
            z_nabla4_e2[vertex_index * 6 + 2] * ptr_coeff_1[2] + z_nabla4_e2[vertex_index * 6 + 3] * ptr_coeff_1[3] +
            z_nabla4_e2[vertex_index * 6 + 4] * ptr_coeff_1[4] + z_nabla4_e2[vertex_index * 6 + 5] * ptr_coeff_1[5];
        p_v_out[vertex_index] =
            z_nabla4_e2[vertex_index * 6] * ptr_coeff_2[0] + z_nabla4_e2[vertex_index * 6 + 1] * ptr_coeff_2[1] +
            z_nabla4_e2[vertex_index * 6 + 2] * ptr_coeff_2[2] + z_nabla4_e2[vertex_index * 6 + 3] * ptr_coeff_2[3] +
            z_nabla4_e2[vertex_index * 6 + 4] * ptr_coeff_2[4] + z_nabla4_e2[vertex_index * 6 + 5] * ptr_coeff_2[5];
    }
    p_cell_out_gt_tv(cell_index, k_index) =
        (p_u_out[0] * ptr_c_coeff_1_gt_ctv(cell_index, 0) + p_u_out[1] * ptr_c_coeff_1_gt_ctv(cell_index, 1) +
            p_u_out[2] * ptr_c_coeff_1_gt_ctv(cell_index, 2) + p_v_out[0] * ptr_c_coeff_2_gt_ctv(cell_index, 0) +
            p_v_out[1] * ptr_c_coeff_2_gt_ctv(cell_index, 1) + p_v_out[2] * ptr_c_coeff_2_gt_ctv(cell_index, 2)) /
        2;
};

template <typename T>
inline void nabla4_interpolate_verts2cells_unstructured_inlined_c2v<T>::run_gpu_naive_helper() {
    constexpr dim3 tblocks(block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive.x,
        block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive.y,
        block_dims_unstructured_nabla_interpol_v2c_inlined_c2v_naive.z);
    dim3 grid((verts2cells_data.output_size + tblocks.x - 1) / tblocks.x,
        (verts2cells_data.KDim + tblocks.y - 1) / tblocks.y,
        1);
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_c2v_unstructured<<<grid, tblocks>>>(
        verts2cells_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        c2v2e2c2v_gt_ctv,
        c2v2e2ecv_gt_ctv,
        c2v2e_gt_ctv,
        verts2cells_data.c2v_gt_ctv,
        nabla4_data.u_vert_gt_tv,
        nabla4_data.v_vert_gt_tv,
        nabla4_data.primal_normal_vert_v1_gt_tv,
        nabla4_data.primal_normal_vert_v2_gt_tv,
        nabla4_data.z_nabla2_e_gt_tv,
        nabla4_data.inv_vert_vert_length_gt_tv,
        nabla4_data.inv_primal_edge_length_gt_tv,
        interpolate_data.ptr_coeff_1_gt_ctv,
        interpolate_data.ptr_coeff_2_gt_ctv,
        verts2cells_data.ptr_coeff_1_gt_ctv,
        verts2cells_data.ptr_coeff_2_gt_ctv,
        verts2cells_data.p_cell_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void nabla4_interpolate_verts2cells_unstructured_inlined_c2v<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_verts2cells_unstructured_inlined_c2v<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
