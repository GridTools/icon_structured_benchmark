#include <utility>

#include <nabla4_interpolate_verts2cells_structured_inlined.hpp>

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cuda/pipeline>
#endif

template <typename T>
struct nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;
    verts2cells_structured<T> verts2cells_data;

  public:
    nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline(index_type CellDim,
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

    nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline(index_type CellDim,
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
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop<int>() {
    return {32, 6, 1, 192};
};

constexpr block_dims block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop =
    get_block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.size)
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_cached_pipeline_structured(index_type KDim,
        index_type x_dim_verts2cells,
        index_type halo_verts2cells,
        index_type halo_interpolate,
        index_type x_dim_nabla4,
        index_type y_dim_nabla4,
        index_type halo_nabla4,
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
    const auto i_nabla4{blockIdx.x * blockDim.x + threadIdx.x + halo_nabla4 - 2 * blockIdx.x};
    const auto j_nabla4{blockIdx.y * blockDim.y + threadIdx.y + halo_nabla4 - 3 * blockIdx.y};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i_nabla4 >= x_dim_nabla4 - halo_nabla4 || j_nabla4 >= y_dim_nabla4 - halo_nabla4 || k_index >= KDim) {
        return;
    }
    constexpr auto block_horizontal_dim_nabla4{
        block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x *
        block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y};
    constexpr auto shared_mem_k_offset{block_horizontal_dim_nabla4 * 3};
    __shared__ WP_TYPE z_nabla4_shared_mem[shared_mem_k_offset *
                                           block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.z];
    const index_type z_nabla4_offset{static_cast<index_type>(shared_mem_k_offset * threadIdx.z)};
    const index_type i_j = j_nabla4 * x_dim_nabla4 + i_nabla4;
    const index_type i_smem = threadIdx.x + 1;
    const index_type j_smem = threadIdx.y + 1;
    constexpr auto smem_x = block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x + 2;
    constexpr auto smem_y = block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y + 2;
    const auto smem_x_dim_scaled =
        (blockIdx.x + 1) * (block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x) - 2 * blockIdx.x +
                    halo_nabla4 >=
                x_dim_nabla4 - halo_nabla4
            ? x_dim_nabla4 - halo_nabla4 -
                  blockIdx.x * (block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x) +
                  2 * blockIdx.x - halo_nabla4
            : block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x;
    const auto smem_y_dim_scaled =
        (blockIdx.y + 1) * (block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y) - 3 * blockIdx.y +
                    halo_nabla4 >=
                y_dim_nabla4 - halo_nabla4
            ? y_dim_nabla4 - halo_nabla4 -
                  blockIdx.y * (block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y) +
                  3 * blockIdx.y - halo_nabla4
            : block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y;
    printf("[%d %d %d:%d %d %d] smem_x_dim_scaled: %d smem_y_dim_scaled: %d\n",
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,
        smem_x_dim_scaled,
        smem_y_dim_scaled);
    const index_type i_j_smem = j_smem * smem_x + i_smem;
    const index_type i_jp1_smem = (j_smem + 1) * smem_x + i_smem;
    const index_type im1_jp1_smem = (j_smem + 1) * smem_x + i_smem - 1;
    const index_type ip1_j_smem = j_smem * smem_x + i_smem + 1;
    const index_type ip1_jm1_smem = (j_smem - 1) * smem_x + i_smem + 1;
    const index_type i_jm1_smem = (j_smem - 1) * smem_x + i_smem;
    const index_type E2C2V_0_smem[3] = {i_j_smem, i_j_smem, i_j_smem};
    const index_type E2C2V_1_smem[3] = {i_jp1_smem, ip1_j_smem, ip1_jm1_smem};
    const index_type E2C2V_2_smem[3] = {im1_jp1_smem, i_jp1_smem, ip1_j_smem};
    const index_type E2C2V_3_smem[3] = {ip1_j_smem, ip1_jm1_smem, i_jm1_smem};
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
    const WP_TYPE inv_vert_vert_length[3] = {inv_vert_vert_length_gt_tv(edge_index),
        inv_vert_vert_length_gt_tv(edge_index + outer_domain_size),
        inv_vert_vert_length_gt_tv(edge_index + 2 * outer_domain_size)};
    const WP_TYPE inv_primal_edge_length[3] = {inv_primal_edge_length_gt_tv(edge_index),
        inv_primal_edge_length_gt_tv(edge_index + outer_domain_size),
        inv_primal_edge_length_gt_tv(edge_index + 2 * outer_domain_size)};
    const auto i_verts2cells{blockIdx.x * blockDim.x + threadIdx.x + halo_nabla4 + halo_interpolate - 2 * blockIdx.x};
    const auto j_verts2cells{blockIdx.y * blockDim.y + threadIdx.y + halo_nabla4 + halo_interpolate - 3 * blockIdx.y};
    const bool thread_calculate_verts2cells{
        i_verts2cells < x_dim_nabla4 - halo_interpolate - halo_verts2cells - halo_nabla4 &&
        j_verts2cells < y_dim_nabla4 - halo_interpolate - halo_verts2cells - halo_nabla4 &&
        threadIdx.x < blockDim.x - 2 && threadIdx.y < blockDim.y - 3};
    const index_type cell_index_internal = i_verts2cells - (halo_nabla4 + halo_interpolate) +
                                           (j_verts2cells - (halo_nabla4 + halo_interpolate)) *
                                               (x_dim_nabla4 - 2 * (halo_nabla4 + halo_interpolate) - halo_verts2cells);
    const index_type cell_index_internal_upward{2 * cell_index_internal};
    const index_type cell_index_internal_downward{2 * cell_index_internal + 1};
    const index_type i_c2v{static_cast<index_type>(threadIdx.x) + halo_verts2cells};
    const index_type j_c2v{static_cast<index_type>(threadIdx.y) + halo_verts2cells};
    const auto v2e_upward_0_shared{get_v2e_per_orientation(i_c2v, j_c2v, blockDim.x, blockDim.y)};
    const auto v2e_upward_1_shared{get_v2e_per_orientation(i_c2v, j_c2v + 1, blockDim.x, blockDim.y)};
    const auto v2e_upward_2_shared{get_v2e_per_orientation(i_c2v + 1, j_c2v, blockDim.x, blockDim.y)};
    const auto v2e_downward_2_shared{get_v2e_per_orientation(i_c2v + 1, j_c2v + 1, blockDim.x, blockDim.y)};
    const std::array<index_type, 36> c2v2e_shared{v2e_upward_0_shared[0], // upward
        v2e_upward_0_shared[1],
        v2e_upward_0_shared[2],
        v2e_upward_0_shared[3],
        v2e_upward_0_shared[4],
        v2e_upward_0_shared[5],
        v2e_upward_1_shared[0],
        v2e_upward_1_shared[1],
        v2e_upward_1_shared[2],
        v2e_upward_1_shared[3],
        v2e_upward_1_shared[4],
        v2e_upward_1_shared[5],
        v2e_upward_2_shared[0],
        v2e_upward_2_shared[1],
        v2e_upward_2_shared[2],
        v2e_upward_2_shared[3],
        v2e_upward_2_shared[4],
        v2e_upward_2_shared[5],
        v2e_upward_2_shared[0], // downward
        v2e_upward_2_shared[1],
        v2e_upward_2_shared[2],
        v2e_upward_2_shared[3],
        v2e_upward_2_shared[4],
        v2e_upward_2_shared[5],
        v2e_upward_1_shared[0],
        v2e_upward_1_shared[1],
        v2e_upward_1_shared[2],
        v2e_upward_1_shared[3],
        v2e_upward_1_shared[4],
        v2e_upward_1_shared[5],
        v2e_downward_2_shared[0],
        v2e_downward_2_shared[1],
        v2e_downward_2_shared[2],
        v2e_downward_2_shared[3],
        v2e_downward_2_shared[4],
        v2e_downward_2_shared[5]};
    const auto i_interpolate{blockIdx.x * blockDim.x + threadIdx.x - 2 * blockIdx.x};
    const auto j_interpolate{blockIdx.y * blockDim.y + threadIdx.y - 3 * blockIdx.y};
    const std::array<index_type, 6> c2v_global{get_index(i_interpolate, j_interpolate, x_dim_verts2cells),
        get_index(i_interpolate, j_interpolate + 1, x_dim_verts2cells),
        get_index(i_interpolate + 1, j_interpolate, x_dim_verts2cells),
        get_index(i_interpolate + 1, j_interpolate, x_dim_verts2cells),
        get_index(i_interpolate, j_interpolate + 1, x_dim_verts2cells),
        get_index(i_interpolate + 1, j_interpolate + 1, x_dim_verts2cells)};
    const std::array<WP_TYPE, 6> ptr_c_coeff_1{ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 0),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 1),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_upward, 2),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 0),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 1),
        ptr_c_coeff_1_gt_ctv(cell_index_internal_downward, 2)};
    const std::array<WP_TYPE, 6> ptr_c_coeff_2{ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 0),
        ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 1),
        ptr_c_coeff_2_gt_ctv(cell_index_internal_upward, 2),
        ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 0),
        ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 1),
        ptr_c_coeff_2_gt_ctv(cell_index_internal_downward, 2)};
    auto thread = cooperative_groups::this_thread();
    constexpr auto smem_size{(block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x + 2) *
                             (block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y + 2)};
    __shared__ WP_TYPE
        u_vert_smem[smem_size * block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.z];
    __shared__ WP_TYPE
        v_vert_smem[smem_size * block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.z];
    __shared__ WP_TYPE z_nabla2_e_smem[block_horizontal_dim_nabla4 *
                                       block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.z * 3];
    auto pipeline = cuda::make_pipeline();
    constexpr auto shared_mem_offset{
        block_horizontal_dim_nabla4 * block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.z};
    const auto shared_mem_index_z_nabla2{
        threadIdx.z * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x};
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
        pipeline.producer_acquire();
        int j_global_new = j_nabla4 - 1;
        for (int j_new{static_cast<int>(threadIdx.y)}; j_new < smem_y && j_global_new < y_dim_nabla4;
             j_new += smem_y_dim_scaled) {
            int i_global_new = i_nabla4 - 1;
            for (int i_new{static_cast<int>(threadIdx.x)}; i_new < smem_x && i_global_new < x_dim_nabla4;
                 i_new += smem_x_dim_scaled) {
                const auto shared_mem_index_uv{i_new + j_new * smem_x};
                const auto i_j_global_new{i_global_new + j_global_new * x_dim_nabla4};
                cuda::memcpy_async(thread,
                    &u_vert_smem[shared_mem_index_uv],
                    &(u_vert_gt_tv(i_j_global_new, k_index)),
                    cuda::aligned_size_t<8>(sizeof(WP_TYPE)),
                    pipeline);
                printf(
                    "[%d %d %d:%d %d %d] smem_y: %d smem_x: %d y_dim_nabla4: %d x_dim_nabla4: %d smem_y_dim_scaled: %d "
                    "smem_x_dim_scaled: %d j_new: %d i_new: %d shared_mem_index_uv: %d u_smem: %lf j_global_new: %d "
                    "i_global_new: %d "
                    "i_j_global_new: %d u_global: %lf\n",
                    blockIdx.x,
                    blockIdx.y,
                    blockIdx.z,
                    threadIdx.x,
                    threadIdx.y,
                    threadIdx.z,
                    smem_y,
                    smem_x,
                    y_dim_nabla4,
                    x_dim_nabla4,
                    smem_y_dim_scaled,
                    smem_x_dim_scaled,
                    j_new,
                    i_new,
                    shared_mem_index_uv,
                    u_vert_smem[shared_mem_index_uv],
                    j_global_new,
                    i_global_new,
                    i_j_global_new,
                    u_vert_gt_tv(i_j_global_new, k_index));
                cuda::memcpy_async(thread,
                    &v_vert_smem[shared_mem_index_uv],
                    &(v_vert_gt_tv(i_j_global_new, k_index)),
                    cuda::aligned_size_t<8>(sizeof(WP_TYPE)),
                    pipeline);
                i_global_new += smem_x_dim_scaled;
            }
            j_global_new += smem_y_dim_scaled;
            printf("[%d %d %d:%d %d %d] j_global_new: %d\n",
                blockIdx.x,
                blockIdx.y,
                blockIdx.z,
                threadIdx.x,
                threadIdx.y,
                threadIdx.z,
                j_global_new);
        }
#pragma unroll 3
        for (auto color{0}; color < 3; ++color) {
            const auto z_nabla2_e_ptr{&(z_nabla2_e_gt_tv(edge_index + color * outer_domain_size, k_index))};
            cuda::memcpy_async(thread,
                &z_nabla2_e_smem[shared_mem_index_z_nabla2 + color * shared_mem_offset],
                z_nabla2_e_ptr,
                cuda::aligned_size_t<8>(sizeof(WP_TYPE)),
                pipeline);
        }
        pipeline.producer_commit();
        pipeline.consumer_wait();
        __syncthreads();
#pragma unroll 3
        for (auto color{0}; color < 3; ++color) {
            const auto E2C2V_0_c = E2C2V_0_smem[color];
            const auto E2C2V_1_c = E2C2V_1_smem[color];
            const auto E2C2V_2_c = E2C2V_2_smem[color];
            const auto E2C2V_3_c = E2C2V_3_smem[color];
            printf("[%d %d %d:%d %d %d] color: %d E2C2V_0: %d E2C2V_1: %d E2C2V_2: %d E2C2V_3: %d\n",
                blockIdx.x,
                blockIdx.y,
                blockIdx.z,
                threadIdx.x,
                threadIdx.y,
                threadIdx.z,
                color,
                E2C2V_0_c,
                E2C2V_1_c,
                E2C2V_2_c,
                E2C2V_3_c);
            const double nabv_tang_wp = u_vert_smem[E2C2V_0_c] * primal_normal_vert_v1_0[color] +
                                        v_vert_smem[E2C2V_0_c] * primal_normal_vert_v2_0[color] +
                                        u_vert_smem[E2C2V_1_c] * primal_normal_vert_v1_1[color] +
                                        v_vert_smem[E2C2V_1_c] * primal_normal_vert_v2_1[color];
            const double nabv_norm_wp = u_vert_smem[E2C2V_2_c] * primal_normal_vert_v1_2[color] +
                                        v_vert_smem[E2C2V_2_c] * primal_normal_vert_v2_2[color] +
                                        u_vert_smem[E2C2V_3_c] * primal_normal_vert_v1_3[color] +
                                        v_vert_smem[E2C2V_3_c] * primal_normal_vert_v2_3[color];
            const WP_TYPE z_nabla2_e = z_nabla2_e_smem[shared_mem_index_z_nabla2 + color * shared_mem_offset];
            printf("[%d %d %d:%d %d %d] color: %d u_vert[0]: %lf v_vert[0]: %lf u_vert[1]: %lf v_vert[1]: %lf "
                   "u_vert[2]: %lf v_vert[2]: %lf u_vert[3]: %lf v_vert[3]: %lf z_nabla2_e: %lf\n",
                blockIdx.x,
                blockIdx.y,
                blockIdx.z,
                threadIdx.x,
                threadIdx.y,
                threadIdx.z,
                color,
                u_vert_smem[E2C2V_0_c],
                v_vert_smem[E2C2V_0_c],
                u_vert_smem[E2C2V_1_c],
                v_vert_smem[E2C2V_1_c],
                u_vert_smem[E2C2V_2_c],
                v_vert_smem[E2C2V_2_c],
                u_vert_smem[E2C2V_3_c],
                v_vert_smem[E2C2V_3_c],
                z_nabla2_e);
            const auto local_edge_index =
                threadIdx.x + threadIdx.y * blockDim.x + color * block_horizontal_dim_nabla4 + z_nabla4_offset;
            z_nabla4_shared_mem[local_edge_index] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * (inv_vert_vert_length[color] * inv_vert_vert_length[color]) +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) *
                              (inv_primal_edge_length[color] * inv_primal_edge_length[color]));
        }
        pipeline.consumer_release();
        __syncthreads();
        if (thread_calculate_verts2cells && k_index < KDim) {
            WP_TYPE p_u_out[6];
            WP_TYPE p_v_out[6];
#pragma unroll 6
            for (int vertex_index{}; vertex_index < 6; ++vertex_index) {
                const std::array<WP_TYPE, 6> ptr_coeff_1{ptr_coeff_1_gt_ctv(c2v_global[vertex_index], 0),
                    ptr_coeff_1_gt_ctv(c2v_global[vertex_index], 1),
                    ptr_coeff_1_gt_ctv(c2v_global[vertex_index], 2),
                    ptr_coeff_1_gt_ctv(c2v_global[vertex_index], 3),
                    ptr_coeff_1_gt_ctv(c2v_global[vertex_index], 4),
                    ptr_coeff_1_gt_ctv(c2v_global[vertex_index], 5)};
                const std::array<WP_TYPE, 6> ptr_coeff_2{ptr_coeff_2_gt_ctv(c2v_global[vertex_index], 0),
                    ptr_coeff_2_gt_ctv(c2v_global[vertex_index], 1),
                    ptr_coeff_2_gt_ctv(c2v_global[vertex_index], 2),
                    ptr_coeff_2_gt_ctv(c2v_global[vertex_index], 3),
                    ptr_coeff_2_gt_ctv(c2v_global[vertex_index], 4),
                    ptr_coeff_2_gt_ctv(c2v_global[vertex_index], 5)};
                p_u_out[vertex_index] =
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6]] * ptr_coeff_1[0] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 1]] * ptr_coeff_1[1] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 2]] * ptr_coeff_1[2] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 3]] * ptr_coeff_1[3] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 4]] * ptr_coeff_1[4] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 5]] * ptr_coeff_1[5];
                p_v_out[vertex_index] =
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6]] * ptr_coeff_2[0] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 1]] * ptr_coeff_2[1] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 2]] * ptr_coeff_2[2] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 3]] * ptr_coeff_2[3] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 4]] * ptr_coeff_2[4] +
                    z_nabla4_shared_mem[z_nabla4_offset + c2v2e_shared[vertex_index * 6 + 5]] * ptr_coeff_2[5];
            };
            __stcs(static_cast<double2 *>(static_cast<void *>(&p_cell_out_gt_tv(cell_index_internal_upward, k_index))),
                make_double2(
                    (p_u_out[0] * ptr_c_coeff_1[0] + p_u_out[1] * ptr_c_coeff_1[1] + p_u_out[2] * ptr_c_coeff_1[2] +
                        p_v_out[0] * ptr_c_coeff_2[0] + p_v_out[1] * ptr_c_coeff_2[1] + p_v_out[2] * ptr_c_coeff_2[2]) /
                        2,
                    (p_u_out[3] * ptr_c_coeff_1[3] + p_u_out[4] * ptr_c_coeff_1[4] + p_u_out[5] * ptr_c_coeff_1[5] +
                        p_v_out[3] * ptr_c_coeff_2[3] + p_v_out[4] * ptr_c_coeff_2[4] + p_v_out[5] * ptr_c_coeff_2[5]) /
                        2));
        }
        __syncthreads();
    }
};

template <typename T>
inline void nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline<T>::run_gpu_kloop_helper() {
    constexpr dim3 tblocks(block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.x,
        block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.y,
        block_dims_structured_nabla_interpol_v2c_inlined_cached_pipeline_kloop.z);
    const index_type outer_domain_size = nabla4_data.x_dim * nabla4_data.y_dim;
    const index_type verts2cells_output_x_dim = verts2cells_data.x_dim - 2 * verts2cells_data.halo;
    const index_type verts2cells_output_y_dim = verts2cells_data.y_dim - 2 * verts2cells_data.halo;
    constexpr index_type shared_mem_elements = (tblocks.x * tblocks.y) * 3;
    const index_type outer_x_dim = nabla4_data.x_dim - 2 * 2;
    const index_type outer_y_dim = nabla4_data.y_dim - 2 * 2;
    dim3 grid((outer_x_dim + (tblocks.x - 2) - 1) / (tblocks.x - 2),
        (outer_y_dim + (tblocks.y - 3) - 1) / (tblocks.y - 3),
        4);
    run_gpu_kloop_nabla4_interpolate_verts2cells_inlined_cached_pipeline_structured<<<grid, tblocks>>>(
        verts2cells_data.KDim,
        verts2cells_data.x_dim,
        verts2cells_data.halo,
        interpolate_data.halo,
        nabla4_data.x_dim,
        nabla4_data.y_dim,
        2,
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
inline void nabla4_interpolate_verts2cells_structured_inlined_cached_pipeline<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
