#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>

#if defined(__CUDACC__)
#include <cooperative_groups.h>
#include <cuda/pipeline>
#endif

template <typename T>
struct nabla4_interpolate_structured_inlined_pipeline {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;

  public:
    nabla4_interpolate_structured_inlined_pipeline(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0),
          interpolate_data(VertexDim, EdgeDim, KDim, y_dim, x_dim, halo + 1, nabla4_data.get_output_gt()){};

    nabla4_interpolate_structured_inlined_pipeline(index_type CellDim,
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
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_pipeline_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_pipeline_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_pipeline_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_pipeline_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_pipeline_kloop<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_pipeline_kloop =
    get_block_dims_structured_nabla_interpol_inlined_pipeline_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_inlined_pipeline_kloop.size)
    run_gpu_kloop_nabla4_interpolate_inlined_pipeline_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        index_type inner_domain_size,
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
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x + halo >= x_dim - halo ? x_dim - halo - 1 : blockIdx.x * blockDim.x + threadIdx.x + halo;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y + halo >= y_dim - halo ? y_dim - halo - 1 : blockIdx.y * blockDim.y + threadIdx.y + halo;
    const auto skip_write = blockIdx.x * blockDim.x + threadIdx.x >= x_dim - 2 * halo ||
                            blockIdx.y * blockDim.y + threadIdx.y >= y_dim - 2 * halo;
    const std::array<index_type, 7> v2e2c2v_compressed{get_v2e2c2v(i, j, x_dim)};
    const std::array<index_type, 6> v2e{get_v2e_per_orientation(i, j, x_dim, y_dim)};
    const std::array<index_type, 24> v2e2ecv{v2e[0],
        v2e[0] + outer_domain_size * 3,
        v2e[0] + outer_domain_size * 6,
        v2e[0] + outer_domain_size * 9,
        v2e[1],
        v2e[1] + outer_domain_size * 3,
        v2e[1] + outer_domain_size * 6,
        v2e[1] + outer_domain_size * 9,
        v2e[2],
        v2e[2] + outer_domain_size * 3,
        v2e[2] + outer_domain_size * 6,
        v2e[2] + outer_domain_size * 9,
        v2e[3],
        v2e[3] + outer_domain_size * 3,
        v2e[3] + outer_domain_size * 6,
        v2e[3] + outer_domain_size * 9,
        v2e[4],
        v2e[4] + outer_domain_size * 3,
        v2e[4] + outer_domain_size * 6,
        v2e[4] + outer_domain_size * 9,
        v2e[5],
        v2e[5] + outer_domain_size * 3,
        v2e[5] + outer_domain_size * 6,
        v2e[5] + outer_domain_size * 9};
    const std::array<index_type, 24> v2e2c2v{v2e2c2v_compressed[0],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[0],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[0]};
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
    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
    const std::array<WP_TYPE, 6> ptr_coeff_1{ptr_coeff_1_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 5)};
    const std::array<WP_TYPE, 6> ptr_coeff_2{ptr_coeff_2_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 5)};
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    auto thread = cooperative_groups::this_thread();
    extern __shared__ WP_TYPE smem[];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        1
    > shared_state;
    auto block = cooperative_groups::this_thread_block();
    auto pipeline_z_nabla2_e = cuda::make_pipeline(block, &shared_state);
    const auto shared_mem_offset{blockDim.z * blockDim.x * blockDim.y};
    const auto shared_mem_index_offset0{
        threadIdx.z * blockDim.x * blockDim.y + threadIdx.x + threadIdx.y * blockDim.x};
    const auto shared_mem_index_offset1{shared_mem_index_offset0 + shared_mem_offset};
    const auto shared_mem_index_offset2{shared_mem_index_offset1 + shared_mem_offset};
    const auto shared_mem_index_offset3{shared_mem_index_offset2 + shared_mem_offset};
    const auto shared_mem_index_offset4{shared_mem_index_offset3 + shared_mem_offset};
    const auto shared_mem_index_offset5{shared_mem_index_offset4 + shared_mem_offset};
    const auto shared_mem_index_offset6{shared_mem_index_offset5 + shared_mem_offset};
    const auto shared_mem_index_offset7{shared_mem_index_offset6 + shared_mem_offset};
    const auto shared_mem_index_offset8{shared_mem_index_offset7 + shared_mem_offset};
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
#pragma unroll 6
        for (auto i{0}; i < 6; ++i) {
            const auto edge_index = v2e[i];
            const auto E2C2V_0 = v2e2c2v[i * 4];
            const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
            const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
            const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
            const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1[4 * i] +
                                        v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2[4 * i] +
                                        u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1[4 * i + 1] +
                                        v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2[4 * i + 1];
            const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1[4 * i + 2] +
                                        v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2[4 * i + 2] +
                                        u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1[4 * i + 3] +
                                        v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2[4 * i + 3];
            pipeline_z_nabla2_e.producer_acquire();
            const WP_TYPE* z_nabla2_e_ptr{&(z_nabla2_e_gt_tv(edge_index, k_index))};
            cuda::memcpy_async(block,
                &smem[shared_mem_offset * 8],
                z_nabla2_e_ptr,
                sizeof(WP_TYPE) * block.size(),
                pipeline_z_nabla2_e);
            pipeline_z_nabla2_e.producer_commit();
            pipeline_z_nabla2_e.consumer_wait();
            const WP_TYPE z_nabla2_e = smem[shared_mem_index_offset8];
            z_nabla4_e2_wp[i] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * (inv_vert_vert_length[i] * inv_vert_vert_length[i]) +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) * (inv_primal_edge_length[i] * inv_primal_edge_length[i]));
            pipeline_z_nabla2_e.consumer_release();
        }
        if (skip_write) {
            continue;
        }
        p_u_out_gt_tv(vertex_index_internal, k_index) =
            z_nabla4_e2_wp[0] * ptr_coeff_1[0] + z_nabla4_e2_wp[1] * ptr_coeff_1[1] +
            z_nabla4_e2_wp[2] * ptr_coeff_1[2] + z_nabla4_e2_wp[3] * ptr_coeff_1[3] +
            z_nabla4_e2_wp[4] * ptr_coeff_1[4] + z_nabla4_e2_wp[5] * ptr_coeff_1[5];
        p_v_out_gt_tv(vertex_index_internal, k_index) =
            z_nabla4_e2_wp[0] * ptr_coeff_2[0] + z_nabla4_e2_wp[1] * ptr_coeff_2[1] +
            z_nabla4_e2_wp[2] * ptr_coeff_2[2] + z_nabla4_e2_wp[3] * ptr_coeff_2[3] +
            z_nabla4_e2_wp[4] * ptr_coeff_2[4] + z_nabla4_e2_wp[5] * ptr_coeff_2[5];
    }
};

template <typename T>
inline void nabla4_interpolate_structured_inlined_pipeline<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_structured_nabla_interpol_inlined_pipeline_kloop.x,
        block_dims_structured_nabla_interpol_inlined_pipeline_kloop.y,
        block_dims_structured_nabla_interpol_inlined_pipeline_kloop.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x, (inner_y_dim + tblocks.y - 1) / tblocks.y, 1);
    const auto shared_mem_size = block_dims_structured_nabla_interpol_inlined_pipeline_kloop.x * block_dims_structured_nabla_interpol_inlined_pipeline_kloop.y * block_dims_structured_nabla_interpol_inlined_pipeline_kloop.z * 9 * sizeof(WP_TYPE);
    run_gpu_kloop_nabla4_interpolate_inlined_pipeline_structured<<<grid, tblocks, shared_mem_size>>>(interpolate_data.KDim,
        interpolate_data.x_dim,
        interpolate_data.y_dim,
        interpolate_data.halo,
        inner_domain_size,
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
        interpolate_data.p_u_out_gt_tv,
        interpolate_data.p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

#else
template <typename T>
inline void nabla4_interpolate_structured_inlined_pipeline<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
