#include <utility>

#include <interpolate_structured_gridtools.hpp>
#include <nabla4_interpolate_structured_inlined.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>
#include <verts2cells_structured_gridtools.hpp>

GT_FORCE_INLINE constexpr const std::array<index_type, 14> get_c2v2e2c2v_compressed(
    const index_type i, const index_type j, const index_type x_dim) {
    std::array<index_type, 14> c2v2e2c2v{};
    c2v2e2c2v[0] = get_index(i, j - 1, x_dim);
    c2v2e2c2v[1] = get_index(i + 1, j - 1, x_dim);
    c2v2e2c2v[2] = get_index(i + 2, j - 1, x_dim);
    c2v2e2c2v[3] = get_index(i - 1, j, x_dim);
    c2v2e2c2v[4] = get_index(i, j, x_dim);
    c2v2e2c2v[5] = get_index(i + 1, j, x_dim);
    c2v2e2c2v[6] = get_index(i + 2, j, x_dim);
    c2v2e2c2v[7] = get_index(i - 1, j + 1, x_dim);
    c2v2e2c2v[8] = get_index(i, j + 1, x_dim);
    c2v2e2c2v[9] = get_index(i + 1, j + 1, x_dim);
    c2v2e2c2v[10] = get_index(i + 2, j + 1, x_dim);
    c2v2e2c2v[11] = get_index(i - 1, j + 2, x_dim);
    c2v2e2c2v[12] = get_index(i, j + 2, x_dim);
    c2v2e2c2v[13] = get_index(i + 1, j + 2, x_dim);
    return c2v2e2c2v;
}

template <typename T>
struct nabla4_interpolate_verts2cells_structured_inlined {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;
    verts2cells_structured<T> verts2cells_data;

  public:
    nabla4_interpolate_verts2cells_structured_inlined(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0),
          interpolate_data(
              VertexDim, EdgeDim, KDim, y_dim - 2 * halo, x_dim - 2 * halo, 1, nabla4_data.get_output_gt()),
          verts2cells_data(VertexDim,
              CellDim,
              KDim,
              y_dim - 2 * (halo + 1),
              x_dim - 2 * (halo + 1),
              1,
              std::get<0>(interpolate_data.get_output_gt()),
              std::get<1>(interpolate_data.get_output_gt())){};

    nabla4_interpolate_verts2cells_structured_inlined(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
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
              y_dim - 2 * halo,
              x_dim - 2 * halo,
              1,
              nabla4_data.get_output_gt(),
              ptr_coeff_1,
              ptr_coeff_2),
          verts2cells_data(VertexDim,
              CellDim,
              KDim,
              y_dim - 2 * (halo + 1),
              x_dim - 2 * (halo + 1),
              1,
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
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_naive<int>() {
    return {32, 1, 4, 128};
};

constexpr block_dims block_dims_structured_nabla_interpol_v2c_inlined_naive =
    get_block_dims_structured_nabla_interpol_v2c_inlined_naive<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_v2c_inlined_naive.size)
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_structured(index_type KDim,
        index_type x_dim_verts2cells,
        index_type y_dim_verts2cells,
        index_type halo_verts2cells,
        index_type x_dim_interpolate,
        index_type y_dim_interpolate,
        index_type halo_interpolate,
        index_type x_dim_nabla4,
        index_type y_dim_nabla4,
        index_type halo_nabla4,
        index_type verts2cells_output_domain_size,
        index_type interpolate_output_domain_size,
        index_type nabla4_output_domain_size,
        index_type outer_domain_size,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_c_coeff_1_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_c_coeff_2_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo_verts2cells};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo_verts2cells};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i >= x_dim_verts2cells - halo_verts2cells || j >= y_dim_verts2cells - halo_verts2cells || k_index >= KDim) {
        return;
    }
    const index_type index_internal =
        i - halo_verts2cells + (j - halo_verts2cells) * (x_dim_verts2cells - 2 * halo_verts2cells);
    const index_type cell_index_internal_upward{2 * index_internal};
    const index_type cell_index_internal_downward{2 * index_internal + 1};
    const std::array<index_type, 14> c2v2e2c2v_compressed{
        get_c2v2e2c2v_compressed(i + halo_nabla4 + halo_interpolate, j + halo_nabla4 + halo_interpolate, x_dim_nabla4)};
    const std::array<index_type, 6> c2v{get_index(i, j, x_dim_verts2cells),
        get_index(i, j + 1, x_dim_verts2cells),
        get_index(i + 1, j, x_dim_verts2cells),
        get_index(i + 1, j, x_dim_verts2cells),
        get_index(i, j + 1, x_dim_verts2cells),
        get_index(i + 1, j + 1, x_dim_verts2cells)};
    const index_type i_v2e{static_cast<index_type>(i) + halo_nabla4 + halo_interpolate};
    const index_type j_v2e{static_cast<index_type>(j) + halo_nabla4 + halo_interpolate};
    const auto v2e_upward_0{get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)};
    const auto v2e_upward_1{get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)};
    const auto v2e_upward_2{get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)};
    const auto v2e_downward_2{get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)};
    const std::array<index_type, 36> c2v2e{v2e_upward_0[0], // upward
        v2e_upward_0[1],
        v2e_upward_0[2],
        v2e_upward_0[3],
        v2e_upward_0[4],
        v2e_upward_0[5],
        v2e_upward_1[0],
        v2e_upward_1[1],
        v2e_upward_1[2],
        v2e_upward_1[3],
        v2e_upward_1[4],
        v2e_upward_1[5],
        v2e_upward_2[0],
        v2e_upward_2[1],
        v2e_upward_2[2],
        v2e_upward_2[3],
        v2e_upward_2[4],
        v2e_upward_2[5],
        v2e_upward_2[0], // downward
        v2e_upward_2[1],
        v2e_upward_2[2],
        v2e_upward_2[3],
        v2e_upward_2[4],
        v2e_upward_2[5],
        v2e_upward_1[0],
        v2e_upward_1[1],
        v2e_upward_1[2],
        v2e_upward_1[3],
        v2e_upward_1[4],
        v2e_upward_1[5],
        v2e_downward_2[0],
        v2e_downward_2[1],
        v2e_downward_2[2],
        v2e_downward_2[3],
        v2e_downward_2[4],
        v2e_downward_2[5]};
    const index_type e2c2v[144]{c2v2e2c2v_compressed[3], // 0 // upward
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[7],
        c2v2e2c2v_compressed[0],
        c2v2e2c2v_compressed[4], // 1
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[0], // 2
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[3],
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[4], // 3
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[7],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[4], // 4
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[0],
        c2v2e2c2v_compressed[7], // 5
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[3],
        c2v2e2c2v_compressed[7], // 0
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[11],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[8], // 1
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[4], // 2
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[7],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[8], // 3
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[11],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[8], // 4
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[11], // 5
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[7],
        c2v2e2c2v_compressed[4], // 0
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[5], // 1
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[2],
        c2v2e2c2v_compressed[1], // 2
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[2],
        c2v2e2c2v_compressed[5], // 3
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[5], // 4
        c2v2e2c2v_compressed[2],
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[8], // 5
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[4], // 0 // downward
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[5], // 1
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[2],
        c2v2e2c2v_compressed[1], // 2
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[2],
        c2v2e2c2v_compressed[5], // 3
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[5], // 4
        c2v2e2c2v_compressed[2],
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[1],
        c2v2e2c2v_compressed[8], // 5
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[7], // 0
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[11],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[8], // 1
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[4], // 2
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[7],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[8], // 3
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[11],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[8], // 4
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[4],
        c2v2e2c2v_compressed[11], // 5
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[7],
        c2v2e2c2v_compressed[8], // 0
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[9], // 1
        c2v2e2c2v_compressed[10],
        c2v2e2c2v_compressed[13],
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[5], // 2
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[8],
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[9], // 3
        c2v2e2c2v_compressed[13],
        c2v2e2c2v_compressed[12],
        c2v2e2c2v_compressed[10],
        c2v2e2c2v_compressed[9], // 4
        c2v2e2c2v_compressed[6],
        c2v2e2c2v_compressed[10],
        c2v2e2c2v_compressed[5],
        c2v2e2c2v_compressed[12], // 5
        c2v2e2c2v_compressed[9],
        c2v2e2c2v_compressed[13],
        c2v2e2c2v_compressed[8]};
    const std::array<index_type, 144> c2v2e2ecv{c2v2e[0], // 0 // upward
        c2v2e[0] + outer_domain_size * 3,
        c2v2e[0] + outer_domain_size * 6,
        c2v2e[0] + outer_domain_size * 9,
        c2v2e[1],
        c2v2e[1] + outer_domain_size * 3,
        c2v2e[1] + outer_domain_size * 6,
        c2v2e[1] + outer_domain_size * 9,
        c2v2e[2],
        c2v2e[2] + outer_domain_size * 3,
        c2v2e[2] + outer_domain_size * 6,
        c2v2e[2] + outer_domain_size * 9,
        c2v2e[3],
        c2v2e[3] + outer_domain_size * 3,
        c2v2e[3] + outer_domain_size * 6,
        c2v2e[3] + outer_domain_size * 9,
        c2v2e[4],
        c2v2e[4] + outer_domain_size * 3,
        c2v2e[4] + outer_domain_size * 6,
        c2v2e[4] + outer_domain_size * 9,
        c2v2e[5],
        c2v2e[5] + outer_domain_size * 3,
        c2v2e[5] + outer_domain_size * 6,
        c2v2e[5] + outer_domain_size * 9,
        c2v2e[6],
        c2v2e[6] + outer_domain_size * 3,
        c2v2e[6] + outer_domain_size * 6,
        c2v2e[6] + outer_domain_size * 9,
        c2v2e[7],
        c2v2e[7] + outer_domain_size * 3,
        c2v2e[7] + outer_domain_size * 6,
        c2v2e[7] + outer_domain_size * 9,
        c2v2e[8],
        c2v2e[8] + outer_domain_size * 3,
        c2v2e[8] + outer_domain_size * 6,
        c2v2e[8] + outer_domain_size * 9,
        c2v2e[9],
        c2v2e[9] + outer_domain_size * 3,
        c2v2e[9] + outer_domain_size * 6,
        c2v2e[9] + outer_domain_size * 9,
        c2v2e[10],
        c2v2e[10] + outer_domain_size * 3,
        c2v2e[10] + outer_domain_size * 6,
        c2v2e[10] + outer_domain_size * 9,
        c2v2e[11],
        c2v2e[11] + outer_domain_size * 3,
        c2v2e[11] + outer_domain_size * 6,
        c2v2e[11] + outer_domain_size * 9,
        c2v2e[12],
        c2v2e[12] + outer_domain_size * 3,
        c2v2e[12] + outer_domain_size * 6,
        c2v2e[12] + outer_domain_size * 9,
        c2v2e[13],
        c2v2e[13] + outer_domain_size * 3,
        c2v2e[13] + outer_domain_size * 6,
        c2v2e[13] + outer_domain_size * 9,
        c2v2e[14],
        c2v2e[14] + outer_domain_size * 3,
        c2v2e[14] + outer_domain_size * 6,
        c2v2e[14] + outer_domain_size * 9,
        c2v2e[15],
        c2v2e[15] + outer_domain_size * 3,
        c2v2e[15] + outer_domain_size * 6,
        c2v2e[15] + outer_domain_size * 9,
        c2v2e[16],
        c2v2e[16] + outer_domain_size * 3,
        c2v2e[16] + outer_domain_size * 6,
        c2v2e[16] + outer_domain_size * 9,
        c2v2e[17],
        c2v2e[17] + outer_domain_size * 3,
        c2v2e[17] + outer_domain_size * 6,
        c2v2e[17] + outer_domain_size * 9,
        c2v2e[18],
        c2v2e[18] + outer_domain_size * 3,
        c2v2e[18] + outer_domain_size * 6,
        c2v2e[18] + outer_domain_size * 9,
        c2v2e[19],
        c2v2e[19] + outer_domain_size * 3,
        c2v2e[19] + outer_domain_size * 6,
        c2v2e[19] + outer_domain_size * 9,
        c2v2e[20],
        c2v2e[20] + outer_domain_size * 3,
        c2v2e[20] + outer_domain_size * 6,
        c2v2e[20] + outer_domain_size * 9,
        c2v2e[21],
        c2v2e[21] + outer_domain_size * 3,
        c2v2e[21] + outer_domain_size * 6,
        c2v2e[21] + outer_domain_size * 9,
        c2v2e[22],
        c2v2e[22] + outer_domain_size * 3,
        c2v2e[22] + outer_domain_size * 6,
        c2v2e[22] + outer_domain_size * 9,
        c2v2e[23],
        c2v2e[23] + outer_domain_size * 3,
        c2v2e[23] + outer_domain_size * 6,
        c2v2e[23] + outer_domain_size * 9,
        c2v2e[24],
        c2v2e[24] + outer_domain_size * 3,
        c2v2e[24] + outer_domain_size * 6,
        c2v2e[24] + outer_domain_size * 9,
        c2v2e[25],
        c2v2e[25] + outer_domain_size * 3,
        c2v2e[25] + outer_domain_size * 6,
        c2v2e[25] + outer_domain_size * 9,
        c2v2e[26],
        c2v2e[26] + outer_domain_size * 3,
        c2v2e[26] + outer_domain_size * 6,
        c2v2e[26] + outer_domain_size * 9,
        c2v2e[27],
        c2v2e[27] + outer_domain_size * 3,
        c2v2e[27] + outer_domain_size * 6,
        c2v2e[27] + outer_domain_size * 9,
        c2v2e[28],
        c2v2e[28] + outer_domain_size * 3,
        c2v2e[28] + outer_domain_size * 6,
        c2v2e[28] + outer_domain_size * 9,
        c2v2e[29],
        c2v2e[29] + outer_domain_size * 3,
        c2v2e[29] + outer_domain_size * 6,
        c2v2e[29] + outer_domain_size * 9,
        c2v2e[30],
        c2v2e[30] + outer_domain_size * 3,
        c2v2e[30] + outer_domain_size * 6,
        c2v2e[30] + outer_domain_size * 9,
        c2v2e[31],
        c2v2e[31] + outer_domain_size * 3,
        c2v2e[31] + outer_domain_size * 6,
        c2v2e[31] + outer_domain_size * 9,
        c2v2e[32],
        c2v2e[32] + outer_domain_size * 3,
        c2v2e[32] + outer_domain_size * 6,
        c2v2e[32] + outer_domain_size * 9,
        c2v2e[33],
        c2v2e[33] + outer_domain_size * 3,
        c2v2e[33] + outer_domain_size * 6,
        c2v2e[33] + outer_domain_size * 9,
        c2v2e[34],
        c2v2e[34] + outer_domain_size * 3,
        c2v2e[34] + outer_domain_size * 6,
        c2v2e[34] + outer_domain_size * 9,
        c2v2e[35],
        c2v2e[35] + outer_domain_size * 3,
        c2v2e[35] + outer_domain_size * 6,
        c2v2e[35] + outer_domain_size * 9};
    std::array<WP_TYPE, 36> z_nabla4_e2;
#pragma unroll 36
    for (int edge_index{}; edge_index < 36; ++edge_index) {
        double nabv_tang_wp =
            u_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v1_gt_tv(c2v2e2ecv[4 * edge_index]) +
            v_vert_gt_tv(e2c2v[4 * edge_index], k_index) * primal_normal_vert_v2_gt_tv(c2v2e2ecv[4 * edge_index]) +
            u_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) *
                primal_normal_vert_v1_gt_tv(c2v2e2ecv[4 * edge_index + 1]) +
            v_vert_gt_tv(e2c2v[4 * edge_index + 1], k_index) *
                primal_normal_vert_v2_gt_tv(c2v2e2ecv[4 * edge_index + 1]);
        double nabv_norm_wp = u_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) *
                                  primal_normal_vert_v1_gt_tv(c2v2e2ecv[4 * edge_index + 2]) +
                              v_vert_gt_tv(e2c2v[4 * edge_index + 2], k_index) *
                                  primal_normal_vert_v2_gt_tv(c2v2e2ecv[4 * edge_index + 2]) +
                              u_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) *
                                  primal_normal_vert_v1_gt_tv(c2v2e2ecv[4 * edge_index + 3]) +
                              v_vert_gt_tv(e2c2v[4 * edge_index + 3], k_index) *
                                  primal_normal_vert_v2_gt_tv(c2v2e2ecv[4 * edge_index + 3]);
        const index_type edge_id{c2v2e[edge_index]};
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_id, k_index);
        const WP_TYPE inv_vert_vert_length = inv_vert_vert_length_gt_tv(edge_id);
        const WP_TYPE inv_vert_vert_length_sqr = inv_vert_vert_length * inv_vert_vert_length;
        const WP_TYPE inv_primal_edge_length = inv_primal_edge_length_gt_tv(edge_id);
        const WP_TYPE inv_primal_edge_length_sqr = inv_primal_edge_length * inv_primal_edge_length;
        z_nabla4_e2[edge_index] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                            (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    }
    std::array<VP_TYPE, 6> p_u_out;
    std::array<VP_TYPE, 6> p_v_out;
#pragma unroll 6
    for (int vertex_index{}; vertex_index < 6; ++vertex_index) {
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
    p_cell_out_gt_tv(cell_index_internal_upward, k_index) =
        (p_u_out[0] * ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 0) +
            p_u_out[1] * ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 1) +
            p_u_out[2] * ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 2) +
            p_v_out[0] * ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 0) +
            p_v_out[1] * ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 1) +
            p_v_out[2] * ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 2)) /
        2;
    p_cell_out_gt_tv(cell_index_internal_downward, k_index) =
        (p_u_out[3] * ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 0) +
            p_u_out[4] * ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 1) +
            p_u_out[5] * ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 2) +
            p_v_out[3] * ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 0) +
            p_v_out[4] * ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 1) +
            p_v_out[5] * ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 2)) /
        2;
};

template <typename T>
inline void nabla4_interpolate_verts2cells_structured_inlined<T>::run_gpu_naive_helper() {
    constexpr dim3 tblocks(block_dims_structured_nabla_interpol_v2c_inlined_naive.x,
        block_dims_structured_nabla_interpol_v2c_inlined_naive.y,
        block_dims_structured_nabla_interpol_v2c_inlined_naive.z);
    const index_type verts2cells_output_domain_size =
        (verts2cells_data.x_dim - 2 * verts2cells_data.halo) * (verts2cells_data.y_dim - 2 * verts2cells_data.halo);
    const index_type interpolate_output_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type nabla4_output_domain_size =
        (nabla4_data.x_dim - 2 * nabla4_data.halo) * (nabla4_data.y_dim - 2 * nabla4_data.halo);
    const index_type outer_domain_size = nabla4_data.x_dim * nabla4_data.y_dim;
    const index_type verts2cells_output_x_dim = verts2cells_data.x_dim - 2 * verts2cells_data.halo;
    const index_type verts2cells_output_y_dim = verts2cells_data.y_dim - 2 * verts2cells_data.halo;
    dim3 grid((verts2cells_output_x_dim + tblocks.x - 1) / tblocks.x,
        (verts2cells_output_y_dim + tblocks.y - 1) / tblocks.y,
        (verts2cells_data.KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_structured<<<grid, tblocks>>>(verts2cells_data.KDim,
        verts2cells_data.x_dim,
        verts2cells_data.y_dim,
        verts2cells_data.halo,
        interpolate_data.x_dim,
        interpolate_data.y_dim,
        interpolate_data.halo,
        nabla4_data.x_dim,
        nabla4_data.y_dim,
        2,
        verts2cells_output_domain_size,
        interpolate_output_domain_size,
        nabla4_output_domain_size,
        outer_domain_size,
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
inline void nabla4_interpolate_verts2cells_structured_inlined<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_verts2cells_structured_inlined<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
