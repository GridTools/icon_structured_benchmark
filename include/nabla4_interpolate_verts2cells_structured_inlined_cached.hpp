#include <utility>

#include <nabla4_interpolate_verts2cells_structured_inlined.hpp>

template <typename T>
struct nabla4_interpolate_verts2cells_structured_inlined_cached {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;
    verts2cells_structured<T> verts2cells_data;

  public:
    nabla4_interpolate_verts2cells_structured_inlined_cached(index_type CellDim,
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

    nabla4_interpolate_verts2cells_structured_inlined_cached(index_type CellDim,
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
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop<int>() {
    return {32, 2, 4, 256};
};

constexpr block_dims block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop =
    get_block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop.size)
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_cached_structured(index_type KDim,
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
        verts2cells_structured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv){};

template <typename T>
inline void nabla4_interpolate_verts2cells_structured_inlined_cached<T>::run_gpu_kloop_helper() {
    constexpr dim3 tblocks(block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop.x,
        block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop.y,
        block_dims_structured_nabla_interpol_v2c_inlined_cached_kloop.z);
    const index_type verts2cells_output_domain_size =
        (verts2cells_data.x_dim - 2 * verts2cells_data.halo) * (verts2cells_data.y_dim - 2 * verts2cells_data.halo);
    const index_type interpolate_output_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type nabla4_output_domain_size =
        (nabla4_data.x_dim - 2 * nabla4_data.halo) * (nabla4_data.y_dim - 2 * nabla4_data.halo);
    const index_type outer_domain_size = nabla4_data.x_dim * nabla4_data.y_dim;
    const index_type verts2cells_output_x_dim = verts2cells_data.x_dim - 2 * verts2cells_data.halo;
    const index_type verts2cells_output_y_dim = verts2cells_data.y_dim - 2 * verts2cells_data.halo;
    constexpr int smemSize{49152}; // GH200
    constexpr index_type shared_mem_inner_domain = tblocks.x * tblocks.y;
    constexpr long unsigned int k_repetitions{smemSize / (shared_mem_inner_domain * 3 * sizeof(WP_TYPE) * tblocks.z)};
    const int KDim_ceil = std::ceil(static_cast<double>(interpolate_data.KDim) / k_repetitions);
    dim3 grid((verts2cells_output_x_dim + tblocks.x - 1) / tblocks.x,
        (verts2cells_output_y_dim + tblocks.y - 1) / tblocks.y,
        (KDim_ceil + tblocks.z - 1) / tblocks.z);
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_cached_structured<<<grid, tblocks>>>(verts2cells_data.KDim,
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

template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_naive<int>() {
    return {32, 8, 2, 512};
};

constexpr block_dims block_dims_structured_nabla_interpol_v2c_inlined_cached_naive =
    get_block_dims_structured_nabla_interpol_v2c_inlined_cached_naive<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_v2c_inlined_cached_naive.size)
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_cached_structured(index_type KDim,
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
    const auto i_nabla4{blockIdx.x * blockDim.x + threadIdx.x + halo_nabla4 - blockIdx.x};
    const auto j_nabla4{blockIdx.y * blockDim.y + threadIdx.y + halo_nabla4 - 2 * blockIdx.y};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i_nabla4 >= x_dim_nabla4 - halo_nabla4 || j_nabla4 >= y_dim_nabla4 - halo_nabla4 || k_index >= KDim) {
        return;
    }
    extern __shared__ WP_TYPE shared_mem[];
    constexpr auto block_horizontal_dim{block_dims_structured_nabla_interpol_v2c_inlined_cached_naive.x *
                                        block_dims_structured_nabla_interpol_v2c_inlined_cached_naive.y};
    const index_type z_nabla4_offset{static_cast<index_type>(5 * block_horizontal_dim * threadIdx.z)};
    const index_type p_u_out_offset{
        static_cast<index_type>(3 * block_horizontal_dim + 5 * block_horizontal_dim * threadIdx.z)};
    const index_type p_v_out_offset{
        static_cast<index_type>(4 * block_horizontal_dim + 5 * block_horizontal_dim * threadIdx.z)};
    const index_type i_j = j_nabla4 * x_dim_nabla4 + i_nabla4;
    const index_type i_jp1 = (j_nabla4 + 1) * x_dim_nabla4 + i_nabla4;
    const index_type im1_jp1 = (j_nabla4 + 1) * x_dim_nabla4 + i_nabla4 - 1;
    const index_type ip1_j = j_nabla4 * x_dim_nabla4 + i_nabla4 + 1;
    const index_type ip1_jm1 = (j_nabla4 - 1) * x_dim_nabla4 + i_nabla4 + 1;
    const index_type i_jm1 = (j_nabla4 - 1) * x_dim_nabla4 + i_nabla4;
    const index_type E2C2V_0[3] = {i_j, i_j, i_j};
    const index_type E2C2V_1[3] = {i_jp1, ip1_j, ip1_jm1};
    const index_type E2C2V_2[3] = {im1_jp1, i_jp1, ip1_j};
    const index_type E2C2V_3[3] = {ip1_j, ip1_jm1, i_jm1};
    const index_type E2ECV_0[3] = {i_j, i_j + outer_domain_size, i_j + 2 * outer_domain_size};
    const index_type total_edges{3 * outer_domain_size};
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
        const auto local_edge_index =
            threadIdx.x + threadIdx.y * blockDim.x + color * block_horizontal_dim + z_nabla4_offset;
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index + color * outer_domain_size, k_index);
        shared_mem[local_edge_index] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr[color] +
                                                 (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr[color]);
        printf("[%d %d %d:%d %d %d] i_nabla4: %d j_nabla4: %d x_dim_nabla4: %d shared_mem[%d] = %lf\n",
            blockIdx.x,
            blockIdx.y,
            blockIdx.z,
            threadIdx.x,
            threadIdx.y,
            threadIdx.z,
            i_nabla4,
            j_nabla4,
            x_dim_nabla4,
            local_edge_index,
            shared_mem[local_edge_index]);
    };
    __syncthreads();
    const auto i_interpolate{blockIdx.x * blockDim.x + threadIdx.x + halo_nabla4 + halo_interpolate - blockIdx.x};
    const auto j_interpolate{blockIdx.y * blockDim.y + threadIdx.y + halo_nabla4 + halo_interpolate - 2 * blockIdx.y};
    if (i_interpolate >= x_dim_nabla4 - halo_interpolate - halo_nabla4 ||
        j_interpolate >= y_dim_nabla4 - halo_interpolate - halo_nabla4 || threadIdx.x >= blockDim.x - 1 ||
        threadIdx.y >= blockDim.y - 2 || k_index >= KDim) {
        return;
    }
    const std::array<index_type, 6> v2e{get_v2e_per_orientation(
        threadIdx.x + halo_interpolate, threadIdx.y + halo_interpolate, blockDim.x, blockDim.y)};
    const index_type vertex_index_internal =
        i_interpolate - (halo_nabla4 + halo_interpolate) +
        (j_interpolate - (halo_nabla4 + halo_interpolate)) * (x_dim_nabla4 - 2 * (halo_nabla4 + halo_interpolate));
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
    const auto local_vertex_index = threadIdx.x + threadIdx.y * (blockDim.x - 1);
    shared_mem[local_vertex_index + p_u_out_offset] =
        shared_mem[v2e[0] + z_nabla4_offset] * coeff_1[0] + shared_mem[v2e[1] + z_nabla4_offset] * coeff_1[1] +
        shared_mem[v2e[2] + z_nabla4_offset] * coeff_1[2] + shared_mem[v2e[3] + z_nabla4_offset] * coeff_1[3] +
        shared_mem[v2e[4] + z_nabla4_offset] * coeff_1[4] + shared_mem[v2e[5] + z_nabla4_offset] * coeff_1[5];
    shared_mem[local_vertex_index + p_v_out_offset] =
        shared_mem[v2e[0] + z_nabla4_offset] * coeff_2[0] + shared_mem[v2e[1] + z_nabla4_offset] * coeff_2[1] +
        shared_mem[v2e[2] + z_nabla4_offset] * coeff_2[2] + shared_mem[v2e[3] + z_nabla4_offset] * coeff_2[3] +
        shared_mem[v2e[4] + z_nabla4_offset] * coeff_2[4] + shared_mem[v2e[5] + z_nabla4_offset] * coeff_2[5];
    printf("[%d %d %d:%d %d %d] i_interpolate: %d j_interpolate: %d x_dim_interpolate: %d vertex_index_internal: %d "
           "v2e: %d %d %d %d %d %d "
           "coef1: %lf %lf %lf %lf %lf %lf shared_mem_u[%d] = %lf shared_mem_v[%d] = %lf\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        i_interpolate,
        j_interpolate,
        x_dim_nabla4,
        vertex_index_internal,
        v2e[0],
        v2e[1],
        v2e[2],
        v2e[3],
        v2e[4],
        v2e[5],
        coeff_1[0],
        coeff_1[1],
        coeff_1[2],
        coeff_1[3],
        coeff_1[4],
        coeff_1[5],
        local_vertex_index,
        shared_mem[local_vertex_index + p_u_out_offset],
        local_vertex_index,
        shared_mem[local_vertex_index + p_v_out_offset]);
    __syncthreads();
    const auto i_verts2cells{blockIdx.x * blockDim.x + threadIdx.x + halo_nabla4 + halo_interpolate - blockIdx.x};
    const auto j_verts2cells{blockIdx.y * blockDim.y + threadIdx.y + halo_nabla4 + halo_interpolate - 2 * blockIdx.y};
    if (i_verts2cells >= x_dim_nabla4 - halo_interpolate - halo_verts2cells - halo_nabla4 ||
        j_verts2cells >= y_dim_nabla4 - halo_interpolate - halo_verts2cells - halo_nabla4 ||
        threadIdx.x >= blockDim.x - 2 || threadIdx.y >= blockDim.y - 3 || k_index >= KDim) {
        return;
    }
    const index_type cell_index_internal = i_verts2cells - (halo_nabla4 + halo_interpolate) +
                                           (j_verts2cells - (halo_nabla4 + halo_interpolate)) *
                                               (x_dim_nabla4 - 2 * (halo_nabla4 + halo_interpolate) - halo_verts2cells);
    const index_type cell_index_internal_upward{2 * cell_index_internal};
    const index_type cell_index_internal_downward{2 * cell_index_internal + 1};
    const index_type i_c2v{static_cast<index_type>(threadIdx.x)};
    const index_type j_c2v{static_cast<index_type>(threadIdx.y)};
    const std::array<index_type, 6> c2v{get_index(i_c2v, j_c2v, blockDim.x - 1),
        get_index(i_c2v, j_c2v + 1, blockDim.x - 1),
        get_index(i_c2v + 1, j_c2v, blockDim.x - 1),
        get_index(i_c2v + 1, j_c2v, blockDim.x - 1),
        get_index(i_c2v, j_c2v + 1, blockDim.x - 1),
        get_index(i_c2v + 1, j_c2v + 1, blockDim.x - 1)};
    printf(
        "[%d %d %d:%d %d %d] i_verts2cells: %d j_verts2cells: %d x_dim_verts2cells: %d cell_index_internal: %d c2v: %d "
        "%d %d %d %d %d shared_mem_u: %lf %lf %lf %lf %lf %lf coef1_up[%d]: %lf %lf %lf coef1_down[%d]: %lf %lf %lf\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        i_verts2cells,
        j_verts2cells,
        blockDim.x,
        cell_index_internal,
        c2v[0],
        c2v[1],
        c2v[2],
        c2v[3],
        c2v[4],
        c2v[5],
        shared_mem[c2v[0] + p_u_out_offset],
        shared_mem[c2v[1] + p_u_out_offset],
        shared_mem[c2v[2] + p_u_out_offset],
        shared_mem[c2v[3] + p_u_out_offset],
        shared_mem[c2v[4] + p_u_out_offset],
        shared_mem[c2v[5] + p_u_out_offset],
        cell_index_internal_upward,
        ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 0),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 1),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 2),
        cell_index_internal_downward,
        ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 0),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 1),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 2));
    p_cell_out_gt_tv(cell_index_internal_upward, k_index) =
        (shared_mem[c2v[0] + p_u_out_offset] * ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 0) +
            shared_mem[c2v[1] + p_u_out_offset] * ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 1) +
            shared_mem[c2v[2] + p_u_out_offset] * ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 2) +
            shared_mem[c2v[0] + p_v_out_offset] * ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 0) +
            shared_mem[c2v[1] + p_v_out_offset] * ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 1) +
            shared_mem[c2v[2] + p_v_out_offset] * ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 2)) /
        2;
    p_cell_out_gt_tv(cell_index_internal_downward, k_index) =
        (shared_mem[c2v[3] + p_u_out_offset] * ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 0) +
            shared_mem[c2v[4] + p_u_out_offset] * ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 1) +
            shared_mem[c2v[5] + p_u_out_offset] * ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 2) +
            shared_mem[c2v[3] + p_v_out_offset] * ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 0) +
            shared_mem[c2v[4] + p_v_out_offset] * ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 1) +
            shared_mem[c2v[5] + p_v_out_offset] * ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 2)) /
        2;
};

template <typename T>
inline void nabla4_interpolate_verts2cells_structured_inlined_cached<T>::run_gpu_naive_helper() {
    constexpr dim3 tblocks(block_dims_structured_nabla_interpol_v2c_inlined_cached_naive.x,
        block_dims_structured_nabla_interpol_v2c_inlined_cached_naive.y,
        block_dims_structured_nabla_interpol_v2c_inlined_cached_naive.z);
    const index_type verts2cells_output_domain_size =
        (verts2cells_data.x_dim - 2 * verts2cells_data.halo) * (verts2cells_data.y_dim - 2 * verts2cells_data.halo);
    const index_type interpolate_output_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type nabla4_output_domain_size =
        (nabla4_data.x_dim - 2 * nabla4_data.halo) * (nabla4_data.y_dim - 2 * nabla4_data.halo);
    const index_type outer_domain_size = nabla4_data.x_dim * nabla4_data.y_dim;
    const index_type verts2cells_output_x_dim = verts2cells_data.x_dim - 2 * verts2cells_data.halo;
    const index_type verts2cells_output_y_dim = verts2cells_data.y_dim - 2 * verts2cells_data.halo;
    constexpr index_type shared_mem_inner_domain = tblocks.x * tblocks.y;
    const index_type outer_x_dim = nabla4_data.x_dim - 2 * 2;
    const index_type outer_y_dim = nabla4_data.y_dim - 2 * 2;
    dim3 grid((outer_x_dim + (tblocks.x - 2) - 1) / (tblocks.x - 2),
        (outer_y_dim + (tblocks.y - 3) - 1) / (tblocks.y - 3),
        (verts2cells_data.KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_interpolate_verts2cells_inlined_cached_structured<<<grid,
        tblocks,
        shared_mem_inner_domain * 5 * sizeof(WP_TYPE) * tblocks.z>>>(verts2cells_data.KDim,
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
inline void nabla4_interpolate_verts2cells_structured_inlined_cached<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_verts2cells_structured_inlined_cached<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
