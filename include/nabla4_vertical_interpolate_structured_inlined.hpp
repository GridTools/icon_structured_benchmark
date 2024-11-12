#include "hip/hip_runtime.h"
#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>
#include <nabla4_vertical_structured_torus_gridtools_halo.hpp>

template <typename T>
struct nabla4_vertical_interpolate_structured_inlined {
    nabla4_vertical_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;

  public:
    nabla4_vertical_interpolate_structured_inlined(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0),
          interpolate_data(VertexDim, EdgeDim, KDim, y_dim, x_dim, halo + 1, nabla4_data.get_output_gt()){};

    nabla4_vertical_interpolate_structured_inlined(index_type CellDim,
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

#if defined(__HIPCC__)
template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop_vertical() {
    return {};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop_vertical<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop_vertical<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop_vertical<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop_vertical<int>() {
    return {32, 1, 4, 128};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_kloop_vertical =
    get_block_dims_structured_nabla_interpol_inlined_kloop_vertical<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_inlined_kloop_vertical.size)
    run_gpu_kloop_nabla4_vertical_interpolate_inlined_structured(index_type KDim,
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
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
#pragma unroll 6
        for (auto i{0}; i < 6; ++i) {
            const auto edge_index = v2e[i];
            const auto E2C2V_0 = v2e2c2v[i * 4];
            const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
            const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
            const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
            const double nabv_tang_wp =
                u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_tv(v2e2ecv[4 * i], k_index) +
                v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_tv(v2e2ecv[4 * i], k_index) +
                u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_tv(v2e2ecv[4 * i + 1], k_index) +
                v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_tv(v2e2ecv[4 * i + 1], k_index);
            const double nabv_norm_wp =
                u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_tv(v2e2ecv[4 * i + 2], k_index) +
                v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_tv(v2e2ecv[4 * i + 2], k_index) +
                u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_tv(v2e2ecv[4 * i + 3], k_index) +
                v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_tv(v2e2ecv[4 * i + 3], k_index);
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
            const auto inv_vert_vert_length = inv_vert_vert_length_gt_tv(edge_index, k_index);
            const auto inv_primal_edge_length = inv_primal_edge_length_gt_tv(edge_index, k_index);
            const auto inv_vert_vert_length_sqr = inv_vert_vert_length * inv_vert_vert_length;
            const auto inv_primal_edge_length_sqr = inv_primal_edge_length * inv_primal_edge_length;
            z_nabla4_e2_wp[i] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                          (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
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
inline void nabla4_vertical_interpolate_structured_inlined<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_structured_nabla_interpol_inlined_kloop_vertical.x,
        block_dims_structured_nabla_interpol_inlined_kloop_vertical.y,
        block_dims_structured_nabla_interpol_inlined_kloop_vertical.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x, (inner_y_dim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_kloop_nabla4_vertical_interpolate_inlined_structured<<<grid, tblocks>>>(interpolate_data.KDim,
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
    GT_CUDA_CHECK(hipGetLastError());
};

template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive_vertical() {
    return {};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive_vertical<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive_vertical<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive_vertical<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive_vertical<int>() {
    return {32, 1, 18, 576}; // Not optimized
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_naive_vertical =
    get_block_dims_structured_nabla_interpol_inlined_naive_vertical<index_type>();

__global__ void
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 5)
__launch_bounds__(block_dims_structured_nabla_interpol_inlined_naive_vertical.size)
#else
__maxnreg__(56)
#endif
    run_gpu_naive_nabla4_vertical_interpolate_inlined_structured(index_type KDim,
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
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
#pragma unroll 6
    for (auto i{0}; i < 6; ++i) {
        const auto edge_index = v2e[i];
        const auto E2C2V_0 = v2e2c2v[i * 4];
        const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
        const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
        const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
        const auto E2ECV_0 = v2e2ecv[i * 4];
        const auto E2ECV_1 = v2e2ecv[i * 4 + 1];
        const auto E2ECV_2 = v2e2ecv[i * 4 + 2];
        const auto E2ECV_3 = v2e2ecv[i * 4 + 3];
        const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0, k_index) +
                                    v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0, k_index) +
                                    u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1, k_index) +
                                    v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1, k_index);
        const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2, k_index) +
                                    v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2, k_index) +
                                    u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3, k_index) +
                                    v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3, k_index);
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
        const WP_TYPE inv_vert_vert_length = inv_vert_vert_length_gt_tv(edge_index, k_index);
        const WP_TYPE inv_vert_vert_length_sqr = inv_vert_vert_length * inv_vert_vert_length;
        const WP_TYPE inv_primal_edge_length = inv_primal_edge_length_gt_tv(edge_index, k_index);
        const WP_TYPE inv_primal_edge_length_sqr = inv_primal_edge_length * inv_primal_edge_length;
        z_nabla4_e2_wp[i] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                      (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    }
    p_u_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_1_gt_ctv(vertex_index_internal, 0) +
                                                    z_nabla4_e2_wp[1] * ptr_coeff_1_gt_ctv(vertex_index_internal, 1) +
                                                    z_nabla4_e2_wp[2] * ptr_coeff_1_gt_ctv(vertex_index_internal, 2) +
                                                    z_nabla4_e2_wp[3] * ptr_coeff_1_gt_ctv(vertex_index_internal, 3) +
                                                    z_nabla4_e2_wp[4] * ptr_coeff_1_gt_ctv(vertex_index_internal, 4) +
                                                    z_nabla4_e2_wp[5] * ptr_coeff_1_gt_ctv(vertex_index_internal, 5);
    p_v_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_2_gt_ctv(vertex_index_internal, 0) +
                                                    z_nabla4_e2_wp[1] * ptr_coeff_2_gt_ctv(vertex_index_internal, 1) +
                                                    z_nabla4_e2_wp[2] * ptr_coeff_2_gt_ctv(vertex_index_internal, 2) +
                                                    z_nabla4_e2_wp[3] * ptr_coeff_2_gt_ctv(vertex_index_internal, 3) +
                                                    z_nabla4_e2_wp[4] * ptr_coeff_2_gt_ctv(vertex_index_internal, 4) +
                                                    z_nabla4_e2_wp[5] * ptr_coeff_2_gt_ctv(vertex_index_internal, 5);
};

template <typename T>
inline void nabla4_vertical_interpolate_structured_inlined<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_structured_nabla_interpol_inlined_naive_vertical.x,
        block_dims_structured_nabla_interpol_inlined_naive_vertical.y,
        block_dims_structured_nabla_interpol_inlined_naive_vertical.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x,
        (inner_y_dim + tblocks.y - 1) / tblocks.y,
        (interpolate_data.KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_vertical_interpolate_inlined_structured<<<grid, tblocks>>>(interpolate_data.KDim,
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
    GT_CUDA_CHECK(hipGetLastError());
};
#else
template <typename T>
inline void nabla4_vertical_interpolate_structured_inlined<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_vertical_interpolate_structured_inlined<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
