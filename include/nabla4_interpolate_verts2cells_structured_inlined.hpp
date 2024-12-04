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
    return {32, 4, 1, 128};
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
    // printf("s[%d %d %d:%d %d %d] i: %d j: %d k: %d\n",
    //     blockIdx.x,
    //     blockIdx.y,
    //     blockIdx.z,
    //     threadIdx.x,
    //     threadIdx.y,
    //     threadIdx.z,
    //     i,
    //     j,
    //     k_index);
    // printf("s[%d %d %d:%d %d %d] c2v: %d %d %d %d %d %d\n",
    //     blockIdx.x,
    //     blockIdx.y,
    //     blockIdx.z,
    //     threadIdx.x,
    //     threadIdx.y,
    //     threadIdx.z,
    //     c2v[0],
    //     c2v[1],
    //     c2v[2],
    //     c2v[3],
    //     c2v[4],
    //     c2v[5]);
    const index_type i_v2e{i + halo_nabla4 + halo_interpolate};
    const index_type j_v2e{j + halo_nabla4 + halo_interpolate};
    const std::array<index_type, 36> c2v2e{
        get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)[0], // upward
        get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)[1],
        get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)[2],
        get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)[3],
        get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)[4],
        get_v2e_per_orientation(i_v2e, j_v2e, x_dim_nabla4, y_dim_nabla4)[5],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[0],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[1],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[2],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[3],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[4],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[5],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[0],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[1],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[2],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[3],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[4],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[5],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[0], // downward
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[1],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[2],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[3],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[4],
        get_v2e_per_orientation(i_v2e + 1, j_v2e, x_dim_nabla4, y_dim_nabla4)[5],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[0],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[1],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[2],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[3],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[4],
        get_v2e_per_orientation(i_v2e, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[5],
        get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[0],
        get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[1],
        get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[2],
        get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[3],
        get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[4],
        get_v2e_per_orientation(i_v2e + 1, j_v2e + 1, x_dim_nabla4, y_dim_nabla4)[5]};
    // printf("s[%d %d %d:%d %d %d] v2e: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    // %d "
    //        "%d %d %d %d %d %d %d %d %d\n",
    //     blockIdx.x,
    //     blockIdx.y,
    //     blockIdx.z,
    //     threadIdx.x,
    //     threadIdx.y,
    //     threadIdx.z,
    //     c2v2e[0],
    //     c2v2e[1],
    //     c2v2e[2],
    //     c2v2e[3],
    //     c2v2e[4],
    //     c2v2e[5],
    //     c2v2e[6],
    //     c2v2e[7],
    //     c2v2e[8],
    //     c2v2e[9],
    //     c2v2e[10],
    //     c2v2e[11],
    //     c2v2e[12],
    //     c2v2e[13],
    //     c2v2e[14],
    //     c2v2e[15],
    //     c2v2e[16],
    //     c2v2e[17],
    //     c2v2e[18],
    //     c2v2e[19],
    //     c2v2e[20],
    //     c2v2e[21],
    //     c2v2e[22],
    //     c2v2e[23],
    //     c2v2e[24],
    //     c2v2e[25],
    //     c2v2e[26],
    //     c2v2e[27],
    //     c2v2e[28],
    //     c2v2e[29],
    //     c2v2e[30],
    //     c2v2e[31],
    //     c2v2e[32],
    //     c2v2e[33],
    //     c2v2e[34],
    //     c2v2e[35]);
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
    // printf("s[%d %d %d:%d %d %d] e2c2v: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    // "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    //        "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    //        "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    //        "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d\n",
    //     blockIdx.x,
    //     blockIdx.y,
    //     blockIdx.z,
    //     threadIdx.x,
    //     threadIdx.y,
    //     threadIdx.z,
    //     e2c2v[0],
    //     e2c2v[1],
    //     e2c2v[2],
    //     e2c2v[3],
    //     e2c2v[4],
    //     e2c2v[5],
    //     e2c2v[6],
    //     e2c2v[7],
    //     e2c2v[8],
    //     e2c2v[9],
    //     e2c2v[10],
    //     e2c2v[11],
    //     e2c2v[12],
    //     e2c2v[13],
    //     e2c2v[14],
    //     e2c2v[15],
    //     e2c2v[16],
    //     e2c2v[17],
    //     e2c2v[18],
    //     e2c2v[19],
    //     e2c2v[20],
    //     e2c2v[21],
    //     e2c2v[22],
    //     e2c2v[23],
    //     e2c2v[24],
    //     e2c2v[25],
    //     e2c2v[26],
    //     e2c2v[27],
    //     e2c2v[28],
    //     e2c2v[29],
    //     e2c2v[30],
    //     e2c2v[31],
    //     e2c2v[32],
    //     e2c2v[33],
    //     e2c2v[34],
    //     e2c2v[35],
    //     e2c2v[36],
    //     e2c2v[37],
    //     e2c2v[38],
    //     e2c2v[39],
    //     e2c2v[40],
    //     e2c2v[41],
    //     e2c2v[42],
    //     e2c2v[43],
    //     e2c2v[44],
    //     e2c2v[45],
    //     e2c2v[46],
    //     e2c2v[47],
    //     e2c2v[48],
    //     e2c2v[49],
    //     e2c2v[50],
    //     e2c2v[51],
    //     e2c2v[52],
    //     e2c2v[53],
    //     e2c2v[54],
    //     e2c2v[55],
    //     e2c2v[56],
    //     e2c2v[57],
    //     e2c2v[58],
    //     e2c2v[59],
    //     e2c2v[60],
    //     e2c2v[61],
    //     e2c2v[62],
    //     e2c2v[63],
    //     e2c2v[64],
    //     e2c2v[65],
    //     e2c2v[66],
    //     e2c2v[67],
    //     e2c2v[68],
    //     e2c2v[69],
    //     e2c2v[70],
    //     e2c2v[71],
    //     e2c2v[72],
    //     e2c2v[73],
    //     e2c2v[74],
    //     e2c2v[75],
    //     e2c2v[76],
    //     e2c2v[77],
    //     e2c2v[78],
    //     e2c2v[79],
    //     e2c2v[80],
    //     e2c2v[81],
    //     e2c2v[82],
    //     e2c2v[83],
    //     e2c2v[84],
    //     e2c2v[85],
    //     e2c2v[86],
    //     e2c2v[87],
    //     e2c2v[88],
    //     e2c2v[89],
    //     e2c2v[90],
    //     e2c2v[91],
    //     e2c2v[92],
    //     e2c2v[93],
    //     e2c2v[94],
    //     e2c2v[95],
    //     e2c2v[96],
    //     e2c2v[97],
    //     e2c2v[98],
    //     e2c2v[99],
    //     e2c2v[100],
    //     e2c2v[101],
    //     e2c2v[102],
    //     e2c2v[103],
    //     e2c2v[104],
    //     e2c2v[105],
    //     e2c2v[106],
    //     e2c2v[107],
    //     e2c2v[108],
    //     e2c2v[109],
    //     e2c2v[110],
    //     e2c2v[111],
    //     e2c2v[112],
    //     e2c2v[113],
    //     e2c2v[114],
    //     e2c2v[115],
    //     e2c2v[116],
    //     e2c2v[117],
    //     e2c2v[118],
    //     e2c2v[119],
    //     e2c2v[120],
    //     e2c2v[121],
    //     e2c2v[122],
    //     e2c2v[123],
    //     e2c2v[124],
    //     e2c2v[125],
    //     e2c2v[126],
    //     e2c2v[127],
    //     e2c2v[128],
    //     e2c2v[129],
    //     e2c2v[130],
    //     e2c2v[131],
    //     e2c2v[132],
    //     e2c2v[133],
    //     e2c2v[134],
    //     e2c2v[135],
    //     e2c2v[136],
    //     e2c2v[137],
    //     e2c2v[138],
    //     e2c2v[139],
    //     e2c2v[140],
    //     e2c2v[141],
    //     e2c2v[142],
    //     e2c2v[143]);
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
    // printf("s[%d %d %d:%d %d %d] e2ecv: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    // "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    //        "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    //        "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d
    //        "
    //        "%d %d %d %d %d %d %d %d %d %d %d %d %d\n",
    //     blockIdx.x,
    //     blockIdx.y,
    //     blockIdx.z,
    //     threadIdx.x,
    //     threadIdx.y,
    //     threadIdx.z,
    //     c2v2e2ecv[0],
    //     c2v2e2ecv[1],
    //     c2v2e2ecv[2],
    //     c2v2e2ecv[3],
    //     c2v2e2ecv[4],
    //     c2v2e2ecv[5],
    //     c2v2e2ecv[6],
    //     c2v2e2ecv[7],
    //     c2v2e2ecv[8],
    //     c2v2e2ecv[9],
    //     c2v2e2ecv[10],
    //     c2v2e2ecv[11],
    //     c2v2e2ecv[12],
    //     c2v2e2ecv[13],
    //     c2v2e2ecv[14],
    //     c2v2e2ecv[15],
    //     c2v2e2ecv[16],
    //     c2v2e2ecv[17],
    //     c2v2e2ecv[18],
    //     c2v2e2ecv[19],
    //     c2v2e2ecv[20],
    //     c2v2e2ecv[21],
    //     c2v2e2ecv[22],
    //     c2v2e2ecv[23],
    //     c2v2e2ecv[24],
    //     c2v2e2ecv[25],
    //     c2v2e2ecv[26],
    //     c2v2e2ecv[27],
    //     c2v2e2ecv[28],
    //     c2v2e2ecv[29],
    //     c2v2e2ecv[30],
    //     c2v2e2ecv[31],
    //     c2v2e2ecv[32],
    //     c2v2e2ecv[33],
    //     c2v2e2ecv[34],
    //     c2v2e2ecv[35],
    //     c2v2e2ecv[36],
    //     c2v2e2ecv[37],
    //     c2v2e2ecv[38],
    //     c2v2e2ecv[39],
    //     c2v2e2ecv[40],
    //     c2v2e2ecv[41],
    //     c2v2e2ecv[42],
    //     c2v2e2ecv[43],
    //     c2v2e2ecv[44],
    //     c2v2e2ecv[45],
    //     c2v2e2ecv[46],
    //     c2v2e2ecv[47],
    //     c2v2e2ecv[48],
    //     c2v2e2ecv[49],
    //     c2v2e2ecv[50],
    //     c2v2e2ecv[51],
    //     c2v2e2ecv[52],
    //     c2v2e2ecv[53],
    //     c2v2e2ecv[54],
    //     c2v2e2ecv[55],
    //     c2v2e2ecv[56],
    //     c2v2e2ecv[57],
    //     c2v2e2ecv[58],
    //     c2v2e2ecv[59],
    //     c2v2e2ecv[60],
    //     c2v2e2ecv[61],
    //     c2v2e2ecv[62],
    //     c2v2e2ecv[63],
    //     c2v2e2ecv[64],
    //     c2v2e2ecv[65],
    //     c2v2e2ecv[66],
    //     c2v2e2ecv[67],
    //     c2v2e2ecv[68],
    //     c2v2e2ecv[69],
    //     c2v2e2ecv[70],
    //     c2v2e2ecv[71],
    //     c2v2e2ecv[72],
    //     c2v2e2ecv[73],
    //     c2v2e2ecv[74],
    //     c2v2e2ecv[75],
    //     c2v2e2ecv[76],
    //     c2v2e2ecv[77],
    //     c2v2e2ecv[78],
    //     c2v2e2ecv[79],
    //     c2v2e2ecv[80],
    //     c2v2e2ecv[81],
    //     c2v2e2ecv[82],
    //     c2v2e2ecv[83],
    //     c2v2e2ecv[84],
    //     c2v2e2ecv[85],
    //     c2v2e2ecv[86],
    //     c2v2e2ecv[87],
    //     c2v2e2ecv[88],
    //     c2v2e2ecv[89],
    //     c2v2e2ecv[90],
    //     c2v2e2ecv[91],
    //     c2v2e2ecv[92],
    //     c2v2e2ecv[93],
    //     c2v2e2ecv[94],
    //     c2v2e2ecv[95],
    //     c2v2e2ecv[96],
    //     c2v2e2ecv[97],
    //     c2v2e2ecv[98],
    //     c2v2e2ecv[99],
    //     c2v2e2ecv[100],
    //     c2v2e2ecv[101],
    //     c2v2e2ecv[102],
    //     c2v2e2ecv[103],
    //     c2v2e2ecv[104],
    //     c2v2e2ecv[105],
    //     c2v2e2ecv[106],
    //     c2v2e2ecv[107],
    //     c2v2e2ecv[108],
    //     c2v2e2ecv[109],
    //     c2v2e2ecv[110],
    //     c2v2e2ecv[111],
    //     c2v2e2ecv[112],
    //     c2v2e2ecv[113],
    //     c2v2e2ecv[114],
    //     c2v2e2ecv[115],
    //     c2v2e2ecv[116],
    //     c2v2e2ecv[117],
    //     c2v2e2ecv[118],
    //     c2v2e2ecv[119],
    //     c2v2e2ecv[120],
    //     c2v2e2ecv[121],
    //     c2v2e2ecv[122],
    //     c2v2e2ecv[123],
    //     c2v2e2ecv[124],
    //     c2v2e2ecv[125],
    //     c2v2e2ecv[126],
    //     c2v2e2ecv[127],
    //     c2v2e2ecv[128],
    //     c2v2e2ecv[129],
    //     c2v2e2ecv[130],
    //     c2v2e2ecv[131],
    //     c2v2e2ecv[132],
    //     c2v2e2ecv[133],
    //     c2v2e2ecv[134],
    //     c2v2e2ecv[135],
    //     c2v2e2ecv[136],
    //     c2v2e2ecv[137],
    //     c2v2e2ecv[138],
    //     c2v2e2ecv[139],
    //     c2v2e2ecv[140],
    //     c2v2e2ecv[141],
    //     c2v2e2ecv[142],
    //     c2v2e2ecv[143]);
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
        // printf("s[%d %d %d:%d %d %d] edge_id: %d z_nabla2_e: %lf inv_vert_vert_length: %lf inv_primal_edge_length:
        // %lf\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, edge_id, z_nabla2_e,
        // inv_vert_vert_length, inv_primal_edge_length);
        z_nabla4_e2[edge_index] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                            (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    }
    printf(
        "s[%d %d %d:%d %d %d] z_nabla4_e2: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        z_nabla4_e2[18],
        z_nabla4_e2[19],
        z_nabla4_e2[20],
        z_nabla4_e2[21],
        z_nabla4_e2[22],
        z_nabla4_e2[23],
        z_nabla4_e2[24],
        z_nabla4_e2[25],
        z_nabla4_e2[26],
        z_nabla4_e2[27],
        z_nabla4_e2[28],
        z_nabla4_e2[29],
        z_nabla4_e2[30],
        z_nabla4_e2[31],
        z_nabla4_e2[32],
        z_nabla4_e2[33],
        z_nabla4_e2[34],
        z_nabla4_e2[35]);
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
    printf("verts2cells_output_domain_size: %d\n", verts2cells_output_domain_size);
    printf("interpolate_output_domain_size: %d\n", interpolate_output_domain_size);
    printf("nabla4_output_domain_size: %d\n", nabla4_output_domain_size);
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
