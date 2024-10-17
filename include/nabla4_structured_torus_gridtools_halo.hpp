#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "nabla4_gridtools.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_structured_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_kloop<std::size_t>() {
    // Increased thread block size to limit register usage and increase occupancy
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_kloop<std::int64_t>() {
    return {32, 4, 3, 384};
};

template <>
constexpr block_dims get_block_dims_structured_kloop<std::uint32_t>() {
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_kloop<int>() {
    return {32, 2, 4, 672};
};

constexpr block_dims block_dims_structured_kloop = get_block_dims_structured_kloop<index_type>();

template <typename T>
constexpr block_dims get_block_dims_structured_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_naive<std::size_t>() {
    // Increased thread block size to limit register usage and increase occupancy
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_naive<std::int64_t>() {
    return {32, 4, 3, 384};
};

template <>
constexpr block_dims get_block_dims_structured_naive<std::uint32_t>() {
    return {32, 2, 4, 672};
};

template <>
constexpr block_dims get_block_dims_structured_naive<int>() {
    return {32, 2, 4, 672};
};

constexpr block_dims block_dims_structured_naive = get_block_dims_structured_naive<index_type>();
#endif

template <typename T>
class nabla4_structured_torus_halo_gt : public nabla4_gt_data<T> {
  public:
    using nabla4_gt_data<T>::CellDim;
    using nabla4_gt_data<T>::EdgeDim;
    using nabla4_gt_data<T>::VertexDim;
    using nabla4_gt_data<T>::KDim;
    using nabla4_gt_data<T>::ECVDim;
    using nabla4_gt_data<T>::u_vert_gt_tv;
    using nabla4_gt_data<T>::v_vert_gt_tv;
    using nabla4_gt_data<T>::primal_normal_vert_v1_gt_tv;
    using nabla4_gt_data<T>::primal_normal_vert_v2_gt_tv;
    using nabla4_gt_data<T>::z_nabla2_e_gt_tv;
    using nabla4_gt_data<T>::inv_vert_vert_length_gt_tv;
    using nabla4_gt_data<T>::inv_primal_edge_length_gt_tv;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt_tv;

    const index_type y_dim;
    const index_type x_dim;
    const index_type halo;

    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_structured_torus_halo_gt(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : y_dim(y_dim), x_dim(x_dim),
          halo(halo), nabla4_gt_data<T>(
                          CellDim, VertexDim, EdgeDim, KDim, ECVDim, (x_dim - 2 * halo) * (y_dim - halo * 2) * 3){};
    nabla4_structured_torus_halo_gt(index_type CellDim,
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
        : y_dim(y_dim), x_dim(x_dim), halo(halo), nabla4_gt_data<T>(CellDim,
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
    inline __attribute__((always_inline)) void inner_kernel(
        const std::array<index_type, 4> &e2c2v_vec, index_type edge_index, index_type k_index, index_type e2c2v_index) {
        const auto E2C2V_0 = e2c2v_vec[0];
        const auto E2C2V_1 = e2c2v_vec[1];
        const auto E2C2V_2 = e2c2v_vec[2];
        const auto E2C2V_3 = e2c2v_vec[3];
        const auto E2ECV_0 = e2c2v_index * 4;
        const auto E2ECV_1 = e2c2v_index * 4 + 1;
        const auto E2ECV_2 = e2c2v_index * 4 + 2;
        const auto E2ECV_3 = e2c2v_index * 4 + 3;
        double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0) +
                              v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0) +
                              u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1) +
                              v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1);
        double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2) +
                              v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2) +
                              u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3) +
                              v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3);
        z_nabla4_e2_wp_gt_tv(edge_index, k_index) =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_gt_tv(edge_index, k_index)) *
                          (inv_vert_vert_length_gt_tv(edge_index) * inv_vert_vert_length_gt_tv(edge_index)) +
                      (nabv_tang_wp - 2.0 * z_nabla2_e_gt_tv(edge_index, k_index)) *
                          (inv_primal_edge_length_gt_tv(edge_index) * inv_primal_edge_length_gt_tv(edge_index)));
    }

#if defined(__GNUC__)
    __attribute__((optimize("unroll-loops")))
#endif
    void
    run_cpu_ifirst() {
        for (index_type k_index{}; k_index < KDim; ++k_index) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type j = halo; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type i = halo; i < x_dim - halo; ++i) {
                    const auto local_edge_index = ((j - halo) * (x_dim - 2 * halo) + (i - halo)) * 3;
                    const auto global_edge_index = (j * x_dim + i) * 3;
                    const auto i_j = j * x_dim + i;
                    const auto ip1_j = j * x_dim + i + 1;
                    const auto i_jp1 = (j + 1) * x_dim + i;
                    const auto i_jm1 = (j - 1) * x_dim + i;
                    const auto ip1_jm1 = (j - 1) * x_dim + i + 1;
                    const auto im1_jp1 = (j + 1) * x_dim + i - 1;
                    inner_kernel({i_j, i_jp1, im1_jp1, ip1_j}, local_edge_index, k_index, global_edge_index);
                    inner_kernel({i_j, ip1_j, i_jp1, ip1_jm1}, local_edge_index + 1, k_index, global_edge_index + 1);
                    inner_kernel({i_j, ip1_jm1, ip1_j, i_jm1}, local_edge_index + 2, k_index, global_edge_index + 2);
                }
            }
        }
    };

#if defined(__GNUC__)
    __attribute__((optimize("unroll-loops")))
#endif
    void
    run_cpu_kfirst() {
        for (index_type j = halo; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#endif
            for (index_type i = halo; i < x_dim - halo; ++i) {
                const auto local_edge_index = ((j - halo) * (x_dim - 2 * halo) + (i - halo)) * 3;
                const auto global_edge_index = (j * x_dim + i) * 3;
                const auto i_j = j * x_dim + i;
                const auto ip1_j = j * x_dim + i + 1;
                const auto i_jp1 = (j + 1) * x_dim + i;
                const auto i_jm1 = (j - 1) * x_dim + i;
                const auto ip1_jm1 = (j - 1) * x_dim + i + 1;
                const auto im1_jp1 = (j + 1) * x_dim + i - 1;
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type k_index{}; k_index < KDim; ++k_index) {
                    inner_kernel({i_j, i_jp1, im1_jp1, ip1_j}, local_edge_index, k_index, global_edge_index);
                    inner_kernel({i_j, ip1_j, i_jp1, ip1_jm1}, local_edge_index + 1, k_index, global_edge_index + 1);
                    inner_kernel({i_j, ip1_jm1, ip1_j, i_jm1}, local_edge_index + 2, k_index, global_edge_index + 2);
                }
            }
        }
    };
    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};

#if defined(__CUDACC__)
__global__ void __launch_bounds__(block_dims_structured_kloop.size) run_gpu_kloop_nabla4_structured(index_type KDim,
    index_type x_dim,
    index_type x_dim_inner,
    index_type y_dim,
    index_type y_dim_inner,
    index_type halo,
    index_type total_grid_size,
    index_type inner_grid_size,
    index_type global_edges_per_orientation,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
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
        inv_vert_vert_length_gt_tv(local_edge_index_start) * inv_vert_vert_length_gt_tv(local_edge_index_start),
        inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size) *
            inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size),
        inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size) *
            inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size)};
    const WP_TYPE inv_primal_edge_length_sqr[3] = {
        inv_primal_edge_length_gt_tv(local_edge_index_start) * inv_primal_edge_length_gt_tv(local_edge_index_start),
        inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size) *
            inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size),
        inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size) *
            inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size)};
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
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
            const auto local_edge_index = local_edge_index_start + color * inner_grid_size;
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(local_edge_index, k_index);
            z_nabla4_e2_wp_gt_tv(local_edge_index, k_index) =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr[color] +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr[color]);
        };
    };
};

template <typename T>
inline void nabla4_structured_torus_halo_gt<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_structured_kloop.x, block_dims_structured_kloop.y, block_dims_structured_kloop.z);
    const index_type inner_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    dim3 grid((x_dim - 2 * halo + tblocks.x - 1) / tblocks.x, (y_dim - 2 * halo + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_kloop_nabla4_structured<<<grid, tblocks>>>(KDim,
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

__global__ void __launch_bounds__(block_dims_structured_naive.size) run_gpu_naive_nabla4_structured(index_type KDim,
    index_type x_dim,
    index_type x_dim_inner,
    index_type y_dim,
    index_type y_dim_inner,
    index_type halo,
    index_type total_grid_size,
    index_type inner_grid_size,
    index_type global_edges_per_orientation,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
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
        inv_vert_vert_length_gt_tv(local_edge_index_start) * inv_vert_vert_length_gt_tv(local_edge_index_start),
        inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size) *
            inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size),
        inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size) *
            inv_vert_vert_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size)};
    const WP_TYPE inv_primal_edge_length_sqr[3] = {
        inv_primal_edge_length_gt_tv(local_edge_index_start) * inv_primal_edge_length_gt_tv(local_edge_index_start),
        inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size) *
            inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size),
        inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size) *
            inv_primal_edge_length_gt_tv(local_edge_index_start + inner_grid_size + inner_grid_size)};
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
        const auto local_edge_index = local_edge_index_start + color * inner_grid_size;
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(local_edge_index, k_index);
        z_nabla4_e2_wp_gt_tv(local_edge_index, k_index) =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr[color] +
                      (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr[color]);
    };
};

template <typename T>
inline void nabla4_structured_torus_halo_gt<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_structured_naive.x, block_dims_structured_naive.y, block_dims_structured_naive.z);
    const index_type inner_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    dim3 grid((x_dim - 2 * halo + tblocks.x - 1) / tblocks.x,
        (y_dim - 2 * halo + tblocks.y - 1) / tblocks.y,
        (KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_structured<<<grid, tblocks>>>(KDim,
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
inline void nabla4_structured_torus_halo_gt<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};

template <typename T>
inline void nabla4_structured_torus_halo_gt<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
