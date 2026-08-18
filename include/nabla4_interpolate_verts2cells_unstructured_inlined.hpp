#include <interpolate_unstructured_gridtools.hpp>
#include <nabla4_unstructured_gridtools.hpp>
#include <verts2cells_unstructured_gridtools.hpp>

template <typename T>
struct nabla4_interpolate_verts2cells_unstructured_inlined {
    nabla4_unstructured_gt<T> nabla4_data;
    interpolate_unstructured<T> interpolate_data;
    verts2cells_unstructured<T> verts2cells_data;

    nabla4_interpolate_verts2cells_unstructured_inlined(const std::vector<std::array<index_type, 4>> &e2c2v,
        const std::vector<std::array<index_type, 4>> &e2ecv,
        const std::vector<std::array<index_type, 6>> &v2e,
        const std::vector<std::array<index_type, 3>> &c2v,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim),
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt()),
          verts2cells_data(c2v,
              VertexDim,
              CellDim,
              KDim,
              std::get<0>(interpolate_data.get_output_gt()),
              std::get<1>(interpolate_data.get_output_gt())){};

    nabla4_interpolate_verts2cells_unstructured_inlined(const std::vector<std::array<index_type, 4>> &e2c2v,
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
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_kloop<int>() {
    return {32, 6, 1, 192};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_v2c_inlined_kloop =
    get_block_dims_unstructured_nabla_interpol_v2c_inlined_kloop<index_type>();

static __global__ void __launch_bounds__(block_dims_unstructured_nabla_interpol_v2c_inlined_kloop.size)
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_unstructured(index_type verts2cells_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2c2v_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2ecv_gt_tv,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_tv,
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
    if (cell_index >= verts2cells_output_size)
        return;
    const index_type c2v[3]{c2v_gt_tv(cell_index, 0), c2v_gt_tv(cell_index, 1), c2v_gt_tv(cell_index, 2)};
    const index_type v2e[18]{v2e_gt_tv(c2v[0], 0),
        v2e_gt_tv(c2v[0], 1),
        v2e_gt_tv(c2v[0], 2),
        v2e_gt_tv(c2v[0], 3),
        v2e_gt_tv(c2v[0], 4),
        v2e_gt_tv(c2v[0], 5),
        v2e_gt_tv(c2v[1], 0),
        v2e_gt_tv(c2v[1], 1),
        v2e_gt_tv(c2v[1], 2),
        v2e_gt_tv(c2v[1], 3),
        v2e_gt_tv(c2v[1], 4),
        v2e_gt_tv(c2v[1], 5),
        v2e_gt_tv(c2v[2], 0),
        v2e_gt_tv(c2v[2], 1),
        v2e_gt_tv(c2v[2], 2),
        v2e_gt_tv(c2v[2], 3),
        v2e_gt_tv(c2v[2], 4),
        v2e_gt_tv(c2v[2], 5)};
    const index_type e2c2v[72]{e2c2v_gt_tv(v2e[0], 0),
        e2c2v_gt_tv(v2e[0], 1),
        e2c2v_gt_tv(v2e[0], 2),
        e2c2v_gt_tv(v2e[0], 3),
        e2c2v_gt_tv(v2e[1], 0),
        e2c2v_gt_tv(v2e[1], 1),
        e2c2v_gt_tv(v2e[1], 2),
        e2c2v_gt_tv(v2e[1], 3),
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
        e2c2v_gt_tv(v2e[5], 3),
        e2c2v_gt_tv(v2e[6], 0),
        e2c2v_gt_tv(v2e[6], 1),
        e2c2v_gt_tv(v2e[6], 2),
        e2c2v_gt_tv(v2e[6], 3),
        e2c2v_gt_tv(v2e[7], 0),
        e2c2v_gt_tv(v2e[7], 1),
        e2c2v_gt_tv(v2e[7], 2),
        e2c2v_gt_tv(v2e[7], 3),
        e2c2v_gt_tv(v2e[8], 0),
        e2c2v_gt_tv(v2e[8], 1),
        e2c2v_gt_tv(v2e[8], 2),
        e2c2v_gt_tv(v2e[8], 3),
        e2c2v_gt_tv(v2e[9], 0),
        e2c2v_gt_tv(v2e[9], 1),
        e2c2v_gt_tv(v2e[9], 2),
        e2c2v_gt_tv(v2e[9], 3),
        e2c2v_gt_tv(v2e[10], 0),
        e2c2v_gt_tv(v2e[10], 1),
        e2c2v_gt_tv(v2e[10], 2),
        e2c2v_gt_tv(v2e[10], 3),
        e2c2v_gt_tv(v2e[11], 0),
        e2c2v_gt_tv(v2e[11], 1),
        e2c2v_gt_tv(v2e[11], 2),
        e2c2v_gt_tv(v2e[11], 3),
        e2c2v_gt_tv(v2e[12], 0),
        e2c2v_gt_tv(v2e[12], 1),
        e2c2v_gt_tv(v2e[12], 2),
        e2c2v_gt_tv(v2e[12], 3),
        e2c2v_gt_tv(v2e[13], 0),
        e2c2v_gt_tv(v2e[13], 1),
        e2c2v_gt_tv(v2e[13], 2),
        e2c2v_gt_tv(v2e[13], 3),
        e2c2v_gt_tv(v2e[14], 0),
        e2c2v_gt_tv(v2e[14], 1),
        e2c2v_gt_tv(v2e[14], 2),
        e2c2v_gt_tv(v2e[14], 3),
        e2c2v_gt_tv(v2e[15], 0),
        e2c2v_gt_tv(v2e[15], 1),
        e2c2v_gt_tv(v2e[15], 2),
        e2c2v_gt_tv(v2e[15], 3),
        e2c2v_gt_tv(v2e[16], 0),
        e2c2v_gt_tv(v2e[16], 1),
        e2c2v_gt_tv(v2e[16], 2),
        e2c2v_gt_tv(v2e[16], 3),
        e2c2v_gt_tv(v2e[17], 0),
        e2c2v_gt_tv(v2e[17], 1),
        e2c2v_gt_tv(v2e[17], 2),
        e2c2v_gt_tv(v2e[17], 3)};
    const std::array<index_type, 72> e2ecv{e2ecv_gt_tv(v2e[0], 0),
        e2ecv_gt_tv(v2e[0], 1),
        e2ecv_gt_tv(v2e[0], 2),
        e2ecv_gt_tv(v2e[0], 3),
        e2ecv_gt_tv(v2e[1], 0),
        e2ecv_gt_tv(v2e[1], 1),
        e2ecv_gt_tv(v2e[1], 2),
        e2ecv_gt_tv(v2e[1], 3),
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
        e2ecv_gt_tv(v2e[5], 3),
        e2ecv_gt_tv(v2e[6], 0),
        e2ecv_gt_tv(v2e[6], 1),
        e2ecv_gt_tv(v2e[6], 2),
        e2ecv_gt_tv(v2e[6], 3),
        e2ecv_gt_tv(v2e[7], 0),
        e2ecv_gt_tv(v2e[7], 1),
        e2ecv_gt_tv(v2e[7], 2),
        e2ecv_gt_tv(v2e[7], 3),
        e2ecv_gt_tv(v2e[8], 0),
        e2ecv_gt_tv(v2e[8], 1),
        e2ecv_gt_tv(v2e[8], 2),
        e2ecv_gt_tv(v2e[8], 3),
        e2ecv_gt_tv(v2e[9], 0),
        e2ecv_gt_tv(v2e[9], 1),
        e2ecv_gt_tv(v2e[9], 2),
        e2ecv_gt_tv(v2e[9], 3),
        e2ecv_gt_tv(v2e[10], 0),
        e2ecv_gt_tv(v2e[10], 1),
        e2ecv_gt_tv(v2e[10], 2),
        e2ecv_gt_tv(v2e[10], 3),
        e2ecv_gt_tv(v2e[11], 0),
        e2ecv_gt_tv(v2e[11], 1),
        e2ecv_gt_tv(v2e[11], 2),
        e2ecv_gt_tv(v2e[11], 3),
        e2ecv_gt_tv(v2e[12], 0),
        e2ecv_gt_tv(v2e[12], 1),
        e2ecv_gt_tv(v2e[12], 2),
        e2ecv_gt_tv(v2e[12], 3),
        e2ecv_gt_tv(v2e[13], 0),
        e2ecv_gt_tv(v2e[13], 1),
        e2ecv_gt_tv(v2e[13], 2),
        e2ecv_gt_tv(v2e[13], 3),
        e2ecv_gt_tv(v2e[14], 0),
        e2ecv_gt_tv(v2e[14], 1),
        e2ecv_gt_tv(v2e[14], 2),
        e2ecv_gt_tv(v2e[14], 3),
        e2ecv_gt_tv(v2e[15], 0),
        e2ecv_gt_tv(v2e[15], 1),
        e2ecv_gt_tv(v2e[15], 2),
        e2ecv_gt_tv(v2e[15], 3),
        e2ecv_gt_tv(v2e[16], 0),
        e2ecv_gt_tv(v2e[16], 1),
        e2ecv_gt_tv(v2e[16], 2),
        e2ecv_gt_tv(v2e[16], 3),
        e2ecv_gt_tv(v2e[17], 0),
        e2ecv_gt_tv(v2e[17], 1),
        e2ecv_gt_tv(v2e[17], 2),
        e2ecv_gt_tv(v2e[17], 3)};
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
    const std::array<WP_TYPE, 18> inv_vert_vert_length{inv_vert_vert_length_gt_tv(v2e[0]),
        inv_vert_vert_length_gt_tv(v2e[1]),
        inv_vert_vert_length_gt_tv(v2e[2]),
        inv_vert_vert_length_gt_tv(v2e[3]),
        inv_vert_vert_length_gt_tv(v2e[4]),
        inv_vert_vert_length_gt_tv(v2e[5]),
        inv_vert_vert_length_gt_tv(v2e[6]),
        inv_vert_vert_length_gt_tv(v2e[7]),
        inv_vert_vert_length_gt_tv(v2e[8]),
        inv_vert_vert_length_gt_tv(v2e[9]),
        inv_vert_vert_length_gt_tv(v2e[10]),
        inv_vert_vert_length_gt_tv(v2e[11]),
        inv_vert_vert_length_gt_tv(v2e[12]),
        inv_vert_vert_length_gt_tv(v2e[13]),
        inv_vert_vert_length_gt_tv(v2e[14]),
        inv_vert_vert_length_gt_tv(v2e[15]),
        inv_vert_vert_length_gt_tv(v2e[16]),
        inv_vert_vert_length_gt_tv(v2e[17])};
    const std::array<WP_TYPE, 18> inv_primal_edge_length{inv_primal_edge_length_gt_tv(v2e[0]),
        inv_primal_edge_length_gt_tv(v2e[1]),
        inv_primal_edge_length_gt_tv(v2e[2]),
        inv_primal_edge_length_gt_tv(v2e[3]),
        inv_primal_edge_length_gt_tv(v2e[4]),
        inv_primal_edge_length_gt_tv(v2e[5]),
        inv_primal_edge_length_gt_tv(v2e[6]),
        inv_primal_edge_length_gt_tv(v2e[7]),
        inv_primal_edge_length_gt_tv(v2e[8]),
        inv_primal_edge_length_gt_tv(v2e[9]),
        inv_primal_edge_length_gt_tv(v2e[10]),
        inv_primal_edge_length_gt_tv(v2e[11]),
        inv_primal_edge_length_gt_tv(v2e[12]),
        inv_primal_edge_length_gt_tv(v2e[13]),
        inv_primal_edge_length_gt_tv(v2e[14]),
        inv_primal_edge_length_gt_tv(v2e[15]),
        inv_primal_edge_length_gt_tv(v2e[16]),
        inv_primal_edge_length_gt_tv(v2e[17])};
    const std::array<WP_TYPE, 18> ptr_coeff_1{ptr_coeff_1_gt_ctv(c2v[0], 0),
        ptr_coeff_1_gt_ctv(c2v[0], 1),
        ptr_coeff_1_gt_ctv(c2v[0], 2),
        ptr_coeff_1_gt_ctv(c2v[0], 3),
        ptr_coeff_1_gt_ctv(c2v[0], 4),
        ptr_coeff_1_gt_ctv(c2v[0], 5),
        ptr_coeff_1_gt_ctv(c2v[1], 0),
        ptr_coeff_1_gt_ctv(c2v[1], 1),
        ptr_coeff_1_gt_ctv(c2v[1], 2),
        ptr_coeff_1_gt_ctv(c2v[1], 3),
        ptr_coeff_1_gt_ctv(c2v[1], 4),
        ptr_coeff_1_gt_ctv(c2v[1], 5),
        ptr_coeff_1_gt_ctv(c2v[2], 0),
        ptr_coeff_1_gt_ctv(c2v[2], 1),
        ptr_coeff_1_gt_ctv(c2v[2], 2),
        ptr_coeff_1_gt_ctv(c2v[2], 3),
        ptr_coeff_1_gt_ctv(c2v[2], 4),
        ptr_coeff_1_gt_ctv(c2v[2], 5)};
    const std::array<WP_TYPE, 18> ptr_coeff_2{ptr_coeff_2_gt_ctv(c2v[0], 0),
        ptr_coeff_2_gt_ctv(c2v[0], 1),
        ptr_coeff_2_gt_ctv(c2v[0], 2),
        ptr_coeff_2_gt_ctv(c2v[0], 3),
        ptr_coeff_2_gt_ctv(c2v[0], 4),
        ptr_coeff_2_gt_ctv(c2v[0], 5),
        ptr_coeff_2_gt_ctv(c2v[1], 0),
        ptr_coeff_2_gt_ctv(c2v[1], 1),
        ptr_coeff_2_gt_ctv(c2v[1], 2),
        ptr_coeff_2_gt_ctv(c2v[1], 3),
        ptr_coeff_2_gt_ctv(c2v[1], 4),
        ptr_coeff_2_gt_ctv(c2v[1], 5),
        ptr_coeff_2_gt_ctv(c2v[2], 0),
        ptr_coeff_2_gt_ctv(c2v[2], 1),
        ptr_coeff_2_gt_ctv(c2v[2], 2),
        ptr_coeff_2_gt_ctv(c2v[2], 3),
        ptr_coeff_2_gt_ctv(c2v[2], 4),
        ptr_coeff_2_gt_ctv(c2v[2], 5)};
    const std::array<WP_TYPE, 3> ptr_c_coeff1{
        ptr_c_coeff_1_gt_ctv(cell_index, 0), ptr_c_coeff_1_gt_ctv(cell_index, 1), ptr_c_coeff_1_gt_ctv(cell_index, 2)};
    const std::array<WP_TYPE, 3> ptr_c_coeff2{
        ptr_c_coeff_2_gt_ctv(cell_index, 0), ptr_c_coeff_2_gt_ctv(cell_index, 1), ptr_c_coeff_2_gt_ctv(cell_index, 2)};
    for (auto k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim; k_index += gridDim.y * blockDim.y) {
        std::array<WP_TYPE, 18> z_nabla4_e2;
#pragma unroll 18
        for (int edge_index{}; edge_index < 18; ++edge_index) {
            double nabv_tang_wp =
                u_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v1[4 * edge_index] +
                v_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v2[4 * edge_index] +
                u_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v1[4 * edge_index + 1] +
                v_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v2[4 * edge_index + 1];
            double nabv_norm_wp =
                u_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) * primal_normal_vert_v1[4 * edge_index + 2] +
                v_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) * primal_normal_vert_v2[4 * edge_index + 2] +
                u_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) * primal_normal_vert_v1[4 * edge_index + 3] +
                v_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) * primal_normal_vert_v2[4 * edge_index + 3];
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(v2e[edge_index], k_index);
            z_nabla4_e2[edge_index] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) *
                              (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) *
                              (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
        }
        std::array<VP_TYPE, 3> p_u_out;
        std::array<VP_TYPE, 3> p_v_out;
#pragma unroll 3
        for (int vertex_index{}; vertex_index < 3; ++vertex_index) {
            p_u_out[vertex_index] = z_nabla4_e2[vertex_index * 6] * ptr_coeff_1[6 * vertex_index] +
                                    z_nabla4_e2[vertex_index * 6 + 1] * ptr_coeff_1[6 * vertex_index + 1] +
                                    z_nabla4_e2[vertex_index * 6 + 2] * ptr_coeff_1[6 * vertex_index + 2] +
                                    z_nabla4_e2[vertex_index * 6 + 3] * ptr_coeff_1[6 * vertex_index + 3] +
                                    z_nabla4_e2[vertex_index * 6 + 4] * ptr_coeff_1[6 * vertex_index + 4] +
                                    z_nabla4_e2[vertex_index * 6 + 5] * ptr_coeff_1[6 * vertex_index + 5];
            p_v_out[vertex_index] = z_nabla4_e2[vertex_index * 6] * ptr_coeff_2[6 * vertex_index] +
                                    z_nabla4_e2[vertex_index * 6 + 1] * ptr_coeff_2[6 * vertex_index + 1] +
                                    z_nabla4_e2[vertex_index * 6 + 2] * ptr_coeff_2[6 * vertex_index + 2] +
                                    z_nabla4_e2[vertex_index * 6 + 3] * ptr_coeff_2[6 * vertex_index + 3] +
                                    z_nabla4_e2[vertex_index * 6 + 4] * ptr_coeff_2[6 * vertex_index + 4] +
                                    z_nabla4_e2[vertex_index * 6 + 5] * ptr_coeff_2[6 * vertex_index + 5];
        }
        p_cell_out_gt_tv(cell_index, k_index) =
            (p_u_out[0] * ptr_c_coeff1[0] + p_u_out[1] * ptr_c_coeff1[1] + p_u_out[2] * ptr_c_coeff1[2] +
                p_v_out[0] * ptr_c_coeff2[0] + p_v_out[1] * ptr_c_coeff2[1] + p_v_out[2] * ptr_c_coeff2[2]) /
            2;
    }
};

template <typename T>
inline void nabla4_interpolate_verts2cells_unstructured_inlined<T>::run_gpu_kloop_helper() {
    constexpr dim3 tblocks(block_dims_unstructured_nabla_interpol_v2c_inlined_kloop.x,
        block_dims_unstructured_nabla_interpol_v2c_inlined_kloop.y,
        block_dims_unstructured_nabla_interpol_v2c_inlined_kloop.z);
    dim3 grid((verts2cells_data.output_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_unstructured<<<grid, tblocks>>>(verts2cells_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        nabla4_data.e2c2v_gt_tv,
        nabla4_data.e2ecv_gt_tv,
        interpolate_data.v2e_gt_ctv,
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
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_v2c_inlined_naive<int>() {
    return {32, 4, 1, 128};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_v2c_inlined_naive =
    get_block_dims_unstructured_nabla_interpol_v2c_inlined_naive<index_type>();

static __global__ void __launch_bounds__(block_dims_unstructured_nabla_interpol_v2c_inlined_naive.size)
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_unstructured(index_type verts2cells_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2c2v_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2ecv_gt_tv,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_tv,
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
    const index_type v2e[18]{v2e_gt_tv(c2v[0], 0),
        v2e_gt_tv(c2v[0], 1),
        v2e_gt_tv(c2v[0], 2),
        v2e_gt_tv(c2v[0], 3),
        v2e_gt_tv(c2v[0], 4),
        v2e_gt_tv(c2v[0], 5),
        v2e_gt_tv(c2v[1], 0),
        v2e_gt_tv(c2v[1], 1),
        v2e_gt_tv(c2v[1], 2),
        v2e_gt_tv(c2v[1], 3),
        v2e_gt_tv(c2v[1], 4),
        v2e_gt_tv(c2v[1], 5),
        v2e_gt_tv(c2v[2], 0),
        v2e_gt_tv(c2v[2], 1),
        v2e_gt_tv(c2v[2], 2),
        v2e_gt_tv(c2v[2], 3),
        v2e_gt_tv(c2v[2], 4),
        v2e_gt_tv(c2v[2], 5)};
    const index_type e2c2v[72]{e2c2v_gt_tv(v2e[0], 0),
        e2c2v_gt_tv(v2e[0], 1),
        e2c2v_gt_tv(v2e[0], 2),
        e2c2v_gt_tv(v2e[0], 3),
        e2c2v_gt_tv(v2e[1], 0),
        e2c2v_gt_tv(v2e[1], 1),
        e2c2v_gt_tv(v2e[1], 2),
        e2c2v_gt_tv(v2e[1], 3),
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
        e2c2v_gt_tv(v2e[5], 3),
        e2c2v_gt_tv(v2e[6], 0),
        e2c2v_gt_tv(v2e[6], 1),
        e2c2v_gt_tv(v2e[6], 2),
        e2c2v_gt_tv(v2e[6], 3),
        e2c2v_gt_tv(v2e[7], 0),
        e2c2v_gt_tv(v2e[7], 1),
        e2c2v_gt_tv(v2e[7], 2),
        e2c2v_gt_tv(v2e[7], 3),
        e2c2v_gt_tv(v2e[8], 0),
        e2c2v_gt_tv(v2e[8], 1),
        e2c2v_gt_tv(v2e[8], 2),
        e2c2v_gt_tv(v2e[8], 3),
        e2c2v_gt_tv(v2e[9], 0),
        e2c2v_gt_tv(v2e[9], 1),
        e2c2v_gt_tv(v2e[9], 2),
        e2c2v_gt_tv(v2e[9], 3),
        e2c2v_gt_tv(v2e[10], 0),
        e2c2v_gt_tv(v2e[10], 1),
        e2c2v_gt_tv(v2e[10], 2),
        e2c2v_gt_tv(v2e[10], 3),
        e2c2v_gt_tv(v2e[11], 0),
        e2c2v_gt_tv(v2e[11], 1),
        e2c2v_gt_tv(v2e[11], 2),
        e2c2v_gt_tv(v2e[11], 3),
        e2c2v_gt_tv(v2e[12], 0),
        e2c2v_gt_tv(v2e[12], 1),
        e2c2v_gt_tv(v2e[12], 2),
        e2c2v_gt_tv(v2e[12], 3),
        e2c2v_gt_tv(v2e[13], 0),
        e2c2v_gt_tv(v2e[13], 1),
        e2c2v_gt_tv(v2e[13], 2),
        e2c2v_gt_tv(v2e[13], 3),
        e2c2v_gt_tv(v2e[14], 0),
        e2c2v_gt_tv(v2e[14], 1),
        e2c2v_gt_tv(v2e[14], 2),
        e2c2v_gt_tv(v2e[14], 3),
        e2c2v_gt_tv(v2e[15], 0),
        e2c2v_gt_tv(v2e[15], 1),
        e2c2v_gt_tv(v2e[15], 2),
        e2c2v_gt_tv(v2e[15], 3),
        e2c2v_gt_tv(v2e[16], 0),
        e2c2v_gt_tv(v2e[16], 1),
        e2c2v_gt_tv(v2e[16], 2),
        e2c2v_gt_tv(v2e[16], 3),
        e2c2v_gt_tv(v2e[17], 0),
        e2c2v_gt_tv(v2e[17], 1),
        e2c2v_gt_tv(v2e[17], 2),
        e2c2v_gt_tv(v2e[17], 3)};
    const index_type e2ecv[72]{e2ecv_gt_tv(v2e[0], 0),
        e2ecv_gt_tv(v2e[0], 1),
        e2ecv_gt_tv(v2e[0], 2),
        e2ecv_gt_tv(v2e[0], 3),
        e2ecv_gt_tv(v2e[1], 0),
        e2ecv_gt_tv(v2e[1], 1),
        e2ecv_gt_tv(v2e[1], 2),
        e2ecv_gt_tv(v2e[1], 3),
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
        e2ecv_gt_tv(v2e[5], 3),
        e2ecv_gt_tv(v2e[6], 0),
        e2ecv_gt_tv(v2e[6], 1),
        e2ecv_gt_tv(v2e[6], 2),
        e2ecv_gt_tv(v2e[6], 3),
        e2ecv_gt_tv(v2e[7], 0),
        e2ecv_gt_tv(v2e[7], 1),
        e2ecv_gt_tv(v2e[7], 2),
        e2ecv_gt_tv(v2e[7], 3),
        e2ecv_gt_tv(v2e[8], 0),
        e2ecv_gt_tv(v2e[8], 1),
        e2ecv_gt_tv(v2e[8], 2),
        e2ecv_gt_tv(v2e[8], 3),
        e2ecv_gt_tv(v2e[9], 0),
        e2ecv_gt_tv(v2e[9], 1),
        e2ecv_gt_tv(v2e[9], 2),
        e2ecv_gt_tv(v2e[9], 3),
        e2ecv_gt_tv(v2e[10], 0),
        e2ecv_gt_tv(v2e[10], 1),
        e2ecv_gt_tv(v2e[10], 2),
        e2ecv_gt_tv(v2e[10], 3),
        e2ecv_gt_tv(v2e[11], 0),
        e2ecv_gt_tv(v2e[11], 1),
        e2ecv_gt_tv(v2e[11], 2),
        e2ecv_gt_tv(v2e[11], 3),
        e2ecv_gt_tv(v2e[12], 0),
        e2ecv_gt_tv(v2e[12], 1),
        e2ecv_gt_tv(v2e[12], 2),
        e2ecv_gt_tv(v2e[12], 3),
        e2ecv_gt_tv(v2e[13], 0),
        e2ecv_gt_tv(v2e[13], 1),
        e2ecv_gt_tv(v2e[13], 2),
        e2ecv_gt_tv(v2e[13], 3),
        e2ecv_gt_tv(v2e[14], 0),
        e2ecv_gt_tv(v2e[14], 1),
        e2ecv_gt_tv(v2e[14], 2),
        e2ecv_gt_tv(v2e[14], 3),
        e2ecv_gt_tv(v2e[15], 0),
        e2ecv_gt_tv(v2e[15], 1),
        e2ecv_gt_tv(v2e[15], 2),
        e2ecv_gt_tv(v2e[15], 3),
        e2ecv_gt_tv(v2e[16], 0),
        e2ecv_gt_tv(v2e[16], 1),
        e2ecv_gt_tv(v2e[16], 2),
        e2ecv_gt_tv(v2e[16], 3),
        e2ecv_gt_tv(v2e[17], 0),
        e2ecv_gt_tv(v2e[17], 1),
        e2ecv_gt_tv(v2e[17], 2),
        e2ecv_gt_tv(v2e[17], 3)};
    std::array<WP_TYPE, 18> z_nabla4_e2;
#pragma unroll 18
    for (int edge_index{}; edge_index < 18; ++edge_index) {
        double nabv_tang_wp =
            u_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v1_gt_tv(e2ecv[4 * edge_index]) +
            v_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v2_gt_tv(e2ecv[4 * edge_index]) +
            u_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v1_gt_tv(e2ecv[4 * edge_index + 1]) +
            v_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) * primal_normal_vert_v2_gt_tv(e2ecv[4 * edge_index + 1]);
        double nabv_norm_wp =
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
inline void nabla4_interpolate_verts2cells_unstructured_inlined<T>::run_gpu_naive_helper() {
    constexpr dim3 tblocks(block_dims_unstructured_nabla_interpol_v2c_inlined_naive.x,
        block_dims_unstructured_nabla_interpol_v2c_inlined_naive.y,
        block_dims_unstructured_nabla_interpol_v2c_inlined_naive.z);
    dim3 grid((verts2cells_data.output_size + tblocks.x - 1) / tblocks.x,
        (verts2cells_data.KDim + tblocks.y - 1) / tblocks.y,
        1);
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_unstructured<<<grid, tblocks>>>(verts2cells_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        nabla4_data.e2c2v_gt_tv,
        nabla4_data.e2ecv_gt_tv,
        interpolate_data.v2e_gt_ctv,
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
inline void nabla4_interpolate_verts2cells_unstructured_inlined<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_verts2cells_unstructured_inlined<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
