#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>
#include <nabla4_vertical_structured_torus_gridtools_halo.hpp>

template <typename T>
struct nabla4_vertical_interpolate_roofline {
    nabla4_vertical_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;

  public:
    nabla4_vertical_interpolate_roofline(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0),
          interpolate_data(VertexDim, EdgeDim, KDim, y_dim, x_dim, halo + 1, nabla4_data.get_output_gt()){};

    nabla4_vertical_interpolate_roofline(index_type CellDim,
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
constexpr block_dims get_block_dims_nabla_interpol_roofline_kloop_vertical() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_nabla_interpol_roofline_kloop_vertical<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_nabla_interpol_roofline_kloop_vertical<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_nabla_interpol_roofline_kloop_vertical<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_nabla_interpol_roofline_kloop_vertical<int>() {
    return {32, 1, 4, 128};
};

constexpr block_dims block_dims_nabla_interpol_roofline_kloop_vertical =
    get_block_dims_nabla_interpol_roofline_kloop_vertical<index_type>();

__global__ void __launch_bounds__(block_dims_nabla_interpol_roofline_kloop_vertical.size)
    run_gpu_kloop_nabla4_vertical_interpolate_roofline(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        index_type inner_domain_size,
        index_type outer_domain_size,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim) {
        return;
    }
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
    WP_TYPE sum{0};
#pragma unroll 6
        for (auto index{0}; index < 6; ++index) {
            sum += ptr_coeff_1[index] + ptr_coeff_2[index];
        }
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
        const std::array<WP_TYPE, 12> primal_normal_vert_v1{primal_normal_vert_v1_gt_tv(vertex_index_internal, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 2 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 3 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 4 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 5 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 6 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 7 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 8 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 9 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 10 * inner_domain_size, k_index),
            primal_normal_vert_v1_gt_tv(vertex_index_internal + 11 * inner_domain_size, k_index)};
        const std::array<WP_TYPE, 12> primal_normal_vert_v2{primal_normal_vert_v2_gt_tv(vertex_index_internal, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 2 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 3 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 4 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 5 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 6 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 7 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 8 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 9 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 10 * inner_domain_size, k_index),
            primal_normal_vert_v2_gt_tv(vertex_index_internal + 11 * inner_domain_size, k_index)};
        const std::array<WP_TYPE, 3> inv_vert_vert_length{inv_vert_vert_length_gt_tv(vertex_index_internal, k_index),
            inv_vert_vert_length_gt_tv(vertex_index_internal + inner_domain_size, k_index),
            inv_vert_vert_length_gt_tv(vertex_index_internal + 2 * inner_domain_size, k_index)};
        const std::array<WP_TYPE, 3> inv_primal_edge_length{inv_primal_edge_length_gt_tv(vertex_index_internal, k_index),
            inv_primal_edge_length_gt_tv(vertex_index_internal + inner_domain_size, k_index),
            inv_primal_edge_length_gt_tv(vertex_index_internal + 2 * inner_domain_size, k_index)};
        const auto u_vert = u_vert_gt_tv(vertex_index_internal, k_index);
        const auto v_vert = v_vert_gt_tv(vertex_index_internal, k_index);
        auto vertical_sum{sum};
#pragma unroll 3
        for (auto color{0}; color < 3; ++color) {
            vertical_sum += z_nabla2_e_gt_tv(vertex_index_internal + color * inner_domain_size, k_index);
        }
#pragma unroll 3
        for (auto color{0}; color < 3; ++color) {
            vertical_sum += inv_vert_vert_length[color] + inv_primal_edge_length[color];
        }
#pragma unroll 12
        for (auto index{0}; index < 12; ++index) {
            vertical_sum += primal_normal_vert_v1[index] + primal_normal_vert_v2[index];
        }
        vertical_sum += u_vert + v_vert;
        p_u_out_gt_tv(vertex_index_internal, k_index) = vertical_sum;
        p_v_out_gt_tv(vertex_index_internal, k_index) = vertical_sum;
    }
};

template <typename T>
inline void nabla4_vertical_interpolate_roofline<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_nabla_interpol_roofline_kloop_vertical.x,
        block_dims_nabla_interpol_roofline_kloop_vertical.y,
        block_dims_nabla_interpol_roofline_kloop_vertical.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x, (inner_y_dim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_kloop_nabla4_vertical_interpolate_roofline<<<grid, tblocks>>>(interpolate_data.KDim,
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
inline void nabla4_vertical_interpolate_roofline<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
