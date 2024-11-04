#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "nabla4_vertical_gridtools.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_structured_kloop_vertical() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_kloop_vertical<std::size_t>() {
    // Increased thread block size to limit register usage and increase occupancy
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_kloop_vertical<std::int64_t>() {
    return {32, 4, 3, 384};
};

template <>
constexpr block_dims get_block_dims_structured_kloop_vertical<std::uint32_t>() {
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_kloop_vertical<int>() {
    return {32, 2, 4, 672};
};

constexpr block_dims block_dims_structured_kloop_vertical = get_block_dims_structured_kloop_vertical<index_type>();

template <typename T>
constexpr block_dims get_block_dims_structured_naive_vertical() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_naive_vertical<std::size_t>() {
    // Increased thread block size to limit register usage and increase occupancy
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_naive_vertical<std::int64_t>() {
    return {32, 4, 3, 384};
};

template <>
constexpr block_dims get_block_dims_structured_naive_vertical<std::uint32_t>() {
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_naive_vertical<int>() {
    return {32, 2, 4, 672};
};

constexpr block_dims block_dims_structured_naive_vertical = get_block_dims_structured_naive_vertical<index_type>();
#endif

template <typename T>
class nabla4_vertical_structured_torus_halo_gt : public nabla4_vertical_gt_data<T> {
  public:
    using nabla4_vertical_gt_data<T>::CellDim;
    using nabla4_vertical_gt_data<T>::EdgeDim;
    using nabla4_vertical_gt_data<T>::VertexDim;
    using nabla4_vertical_gt_data<T>::KDim;
    using nabla4_vertical_gt_data<T>::ECVDim;
    using nabla4_vertical_gt_data<T>::u_vert_gt_tv;
    using nabla4_vertical_gt_data<T>::v_vert_gt_tv;
    using nabla4_vertical_gt_data<T>::primal_normal_vert_v1_gt_tv;
    using nabla4_vertical_gt_data<T>::primal_normal_vert_v2_gt_tv;
    using nabla4_vertical_gt_data<T>::z_nabla2_e_gt_tv;
    using nabla4_vertical_gt_data<T>::inv_vert_vert_length_gt_tv;
    using nabla4_vertical_gt_data<T>::inv_primal_edge_length_gt_tv;
    using nabla4_vertical_gt_data<T>::z_nabla4_e2_wp_gt_tv;

    const index_type y_dim;
    const index_type x_dim;
    const index_type halo;

    /// Constructor with all the necessary information for \c nabla4_vertical compute
    /// kernel execution
    nabla4_vertical_structured_torus_halo_gt(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : y_dim(y_dim), x_dim(x_dim),
          halo(halo), nabla4_vertical_gt_data<T>(
                          CellDim, VertexDim, EdgeDim, KDim, ECVDim, (x_dim - 2 * halo) * (y_dim - halo * 2) * 3){};
    nabla4_vertical_structured_torus_halo_gt(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        // std::vector<std::array<index_type, 4>> e2c2v, // 510 * 590 * 4 * 8 B = 9.6288 MB
        // std::vector<std::array<index_type, 4>> e2ecv, // 510 * 590 * 4 * 8 B = 9.6288 MB
        std::vector<std::vector<VP_TYPE>> &u_vert,     // 510 * 590 * 3 * 65 * 8 B = 469.404 MB
        std::vector<std::vector<VP_TYPE>> &v_vert,     // 510 * 590 * 3 * 65 * 8 B = 469.404 MB
        std::vector<WP_TYPE> &primal_normal_vert_v1,   // 510 * 590 * 3 * 8 B = 7.2216 MB
        std::vector<WP_TYPE> &primal_normal_vert_v2,   // 510 * 590 * 3 * 8 B = 7.2216 MB
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e, // 510 * 590 * 3 * 65 * 8 B = 469.404 MB
        std::vector<WP_TYPE> &inv_vert_vert_length,    // 510 * 590 * 3 * 8 B = 7.2216 MB
        std::vector<WP_TYPE> &inv_primal_edge_length)  // 510 * 590 * 3 * 8 B = 7.2216 MB
        : y_dim(y_dim), x_dim(x_dim), halo(halo), nabla4_vertical_gt_data<T>(CellDim,
                                                      VertexDim,
                                                      EdgeDim,
                                                      KDim,
                                                      ECVDim,
                                                      (x_dim - 2 * halo) * (y_dim - halo * 2) * 3,
                                                      u_vert,
                                                      v_vert,
                                                      primal_normal_vert_v1,
                                                      primal_normal_vert_v2,
                                                      z_nabla2_e,
                                                      inv_vert_vert_length,
                                                      inv_primal_edge_length){};

  private:
    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};

#if defined(__CUDACC__)
__global__ void __launch_bounds__(block_dims_structured_kloop_vertical.size)
    run_gpu_kloop_nabla4_vertical_structured(index_type KDim,
        index_type x_dim,
        index_type x_dim_inner,
        index_type y_dim,
        index_type y_dim_inner,
        index_type halo,
        index_type total_grid_size,
        index_type inner_grid_size,
        index_type global_edges_per_orientation,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t inv_primal_edge_length_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo};
    if (i >= x_dim - halo || j >= y_dim - halo) {
        return;
    }
    const index_type i_j = j * x_dim + i;
    const index_type i_jp1 = (j + 1) * x_dim + i;
    const index_type im1_jp1 = (j + 1) * x_dim + i - 1;
    const index_type ip1_j = j * x_dim + i + 1;
    const index_type ip1_jm1 = (j - 1) * x_dim + i + 1;
    const index_type i_jm1 = (j - 1) * x_dim + i;
    const index_type E2C2V_0[3] = {i_j, i_j, i_j};
    const index_type E2C2V_1[3] = {i_jp1, ip1_j, ip1_jm1};
    const index_type E2C2V_2[3] = {im1_jp1, i_jp1, ip1_j};
    const index_type E2C2V_3[3] = {ip1_j, ip1_jm1, i_jm1};
    const index_type local_edge_index_start = (j - halo) * x_dim_inner + i - halo;
    const index_type E2ECV_0[3] = {i_j, i_j + total_grid_size, i_j + 2 * total_grid_size};
    const index_type E2ECV_1[3] = {E2ECV_0[0] + global_edges_per_orientation,
        E2ECV_0[1] + global_edges_per_orientation,
        E2ECV_0[2] + global_edges_per_orientation};
    const index_type E2ECV_2[3] = {E2ECV_1[0] + global_edges_per_orientation,
        E2ECV_1[1] + global_edges_per_orientation,
        E2ECV_1[2] + global_edges_per_orientation};
    const index_type E2ECV_3[3] = {E2ECV_2[0] + global_edges_per_orientation,
        E2ECV_2[1] + global_edges_per_orientation,
        E2ECV_2[2] + global_edges_per_orientation};
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
#pragma unroll
        for (auto color{0}; color < 3; ++color) {
            const auto E2C2V_0_c = E2C2V_0[color];
            const auto E2C2V_1_c = E2C2V_1[color];
            const auto E2C2V_2_c = E2C2V_2[color];
            const auto E2C2V_3_c = E2C2V_3[color];
            const double nabv_tang_wp =
                u_vert_gt_tv(E2C2V_0_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0[color], k_index) +
                v_vert_gt_tv(E2C2V_0_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0[color], k_index) +
                u_vert_gt_tv(E2C2V_1_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1[color], k_index) +
                v_vert_gt_tv(E2C2V_1_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1[color], k_index);
            const double nabv_norm_wp =
                u_vert_gt_tv(E2C2V_2_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2[color], k_index) +
                v_vert_gt_tv(E2C2V_2_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2[color], k_index) +
                u_vert_gt_tv(E2C2V_3_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3[color], k_index) +
                v_vert_gt_tv(E2C2V_3_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3[color], k_index);
            const auto local_edge_index = local_edge_index_start + color * inner_grid_size;
            const auto inv_vert_vert_length_sqr = inv_vert_vert_length_gt_tv(local_edge_index, k_index) *
                                                  inv_vert_vert_length_gt_tv(local_edge_index, k_index);
            const auto inv_primal_edge_length_sqr = inv_primal_edge_length_gt_tv(local_edge_index, k_index) *
                                                    inv_primal_edge_length_gt_tv(local_edge_index, k_index);
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(local_edge_index, k_index);
            z_nabla4_e2_wp_gt_tv(local_edge_index, k_index) =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
        };
    };
};

template <typename T>
inline void nabla4_vertical_structured_torus_halo_gt<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_structured_kloop_vertical.x,
        block_dims_structured_kloop_vertical.y,
        block_dims_structured_kloop_vertical.z);
    const index_type inner_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    dim3 grid((x_dim - 2 * halo + tblocks.x - 1) / tblocks.x, (y_dim - 2 * halo + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_kloop_nabla4_vertical_structured<<<grid, tblocks>>>(KDim,
        x_dim,
        x_dim - 2 * halo,
        y_dim,
        y_dim - 2 * halo,
        halo,
        x_dim * y_dim,
        inner_grid_size,
        x_dim * y_dim * 3,
        u_vert_gt_tv,
        v_vert_gt_tv,
        primal_normal_vert_v1_gt_tv,
        primal_normal_vert_v2_gt_tv,
        z_nabla2_e_gt_tv,
        inv_vert_vert_length_gt_tv,
        inv_primal_edge_length_gt_tv,
        z_nabla4_e2_wp_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

__global__ void __launch_bounds__(block_dims_structured_naive_vertical.size)
    run_gpu_naive_nabla4_vertical_structured(index_type KDim,
        index_type x_dim,
        index_type x_dim_inner,
        index_type y_dim,
        index_type y_dim_inner,
        index_type halo,
        index_type total_grid_size,
        index_type inner_grid_size,
        index_type global_edges_per_orientation,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t inv_primal_edge_length_gt_tv,
        nabla4_vertical_structured_torus_halo_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim) {
        return;
    }
    const index_type i_j = j * x_dim + i;
    const index_type i_jp1 = (j + 1) * x_dim + i;
    const index_type im1_jp1 = (j + 1) * x_dim + i - 1;
    const index_type ip1_j = j * x_dim + i + 1;
    const index_type ip1_jm1 = (j - 1) * x_dim + i + 1;
    const index_type i_jm1 = (j - 1) * x_dim + i;
    const index_type E2C2V_0[3] = {i_j, i_j, i_j};
    const index_type E2C2V_1[3] = {i_jp1, ip1_j, ip1_jm1};
    const index_type E2C2V_2[3] = {im1_jp1, i_jp1, ip1_j};
    const index_type E2C2V_3[3] = {ip1_j, ip1_jm1, i_jm1};
    const index_type local_edge_index_start = (j - halo) * x_dim_inner + i - halo;
    const index_type E2ECV_0[3] = {i_j, i_j + total_grid_size, i_j + 2 * total_grid_size};
    const index_type E2ECV_1[3] = {E2ECV_0[0] + global_edges_per_orientation,
        E2ECV_0[1] + global_edges_per_orientation,
        E2ECV_0[2] + global_edges_per_orientation};
    const index_type E2ECV_2[3] = {E2ECV_1[0] + global_edges_per_orientation,
        E2ECV_1[1] + global_edges_per_orientation,
        E2ECV_1[2] + global_edges_per_orientation};
    const index_type E2ECV_3[3] = {E2ECV_2[0] + global_edges_per_orientation,
        E2ECV_2[1] + global_edges_per_orientation,
        E2ECV_2[2] + global_edges_per_orientation};
#pragma unroll
    for (auto color{0}; color < 3; ++color) {
        const auto E2C2V_0_c = E2C2V_0[color];
        const auto E2C2V_1_c = E2C2V_1[color];
        const auto E2C2V_2_c = E2C2V_2[color];
        const auto E2C2V_3_c = E2C2V_3[color];
        const double nabv_tang_wp =
            u_vert_gt_tv(E2C2V_0_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0[color], k_index) +
            v_vert_gt_tv(E2C2V_0_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0[color], k_index) +
            u_vert_gt_tv(E2C2V_1_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1[color], k_index) +
            v_vert_gt_tv(E2C2V_1_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1[color], k_index);
        const double nabv_norm_wp =
            u_vert_gt_tv(E2C2V_2_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2[color], k_index) +
            v_vert_gt_tv(E2C2V_2_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2[color], k_index) +
            u_vert_gt_tv(E2C2V_3_c, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3[color], k_index) +
            v_vert_gt_tv(E2C2V_3_c, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3[color], k_index);
        const auto local_edge_index = local_edge_index_start + color * inner_grid_size;
        const auto inv_vert_vert_length_sqr = inv_vert_vert_length_gt_tv(local_edge_index, k_index) *
                                              inv_vert_vert_length_gt_tv(local_edge_index, k_index);
        const auto inv_primal_edge_length_sqr = inv_primal_edge_length_gt_tv(local_edge_index, k_index) *
                                                inv_primal_edge_length_gt_tv(local_edge_index, k_index);
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(local_edge_index, k_index);
        z_nabla4_e2_wp_gt_tv(local_edge_index, k_index) =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                      (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    };
};

template <typename T>
inline void nabla4_vertical_structured_torus_halo_gt<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_structured_naive_vertical.x,
        block_dims_structured_naive_vertical.y,
        block_dims_structured_naive_vertical.z);
    const index_type inner_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    dim3 grid((x_dim - 2 * halo + tblocks.x - 1) / tblocks.x,
        (y_dim - 2 * halo + tblocks.y - 1) / tblocks.y,
        (KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_vertical_structured<<<grid, tblocks>>>(KDim,
        x_dim,
        x_dim - 2 * halo,
        y_dim,
        y_dim - 2 * halo,
        halo,
        x_dim * y_dim,
        inner_grid_size,
        x_dim * y_dim * 3,
        u_vert_gt_tv,
        v_vert_gt_tv,
        primal_normal_vert_v1_gt_tv,
        primal_normal_vert_v2_gt_tv,
        z_nabla2_e_gt_tv,
        inv_vert_vert_length_gt_tv,
        inv_primal_edge_length_gt_tv,
        z_nabla4_e2_wp_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void nabla4_vertical_structured_torus_halo_gt<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};

template <typename T>
inline void nabla4_vertical_structured_torus_halo_gt<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
