#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "nabla4_gridtools.hpp"
#include "random_init.hpp"

template <typename T>
class nabla4_structured_torus_halo_gt : public nabla4_gt_data<T> {

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

  public:
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
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length)
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
    void run_gpu_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else if constexpr (I == backend_impl::gpu) {
            run_gpu_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_structured() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured<std::size_t>() {
    return {32, 3, 4, 384};
};

template <>
constexpr block_dims get_block_dims_structured<std::uint32_t>() {
    return {32, 3, 4, 384};
};

template <>
constexpr block_dims get_block_dims_structured<int>() {
    return {32, 3, 4, 384};
};

constexpr block_dims block_dims_structured = get_block_dims_structured<index_type>();

__global__ void __launch_bounds__(block_dims_structured.size, 2) run_gpu(index_type KDim,
    index_type x_dim,
    index_type y_dim,
    index_type halo,
    index_type total_grid_size,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
    nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    __shared__ index_type E2C2V[3][block_dims_structured.x][5];
    __shared__ index_type is[block_dims_structured.x];
    __shared__ index_type js[block_dims_structured.x];
    const auto x_dim_without_halo = x_dim - 2 * halo;
    for (auto edge_index{blockIdx.x * blockDim.x + threadIdx.x}; edge_index < total_grid_size;
         edge_index += blockDim.x * gridDim.x) {
        const auto i = edge_index % x_dim_without_halo + halo;
        const auto j = (edge_index / x_dim_without_halo) + halo;
        is[threadIdx.x] = i;
        js[threadIdx.x] = j;
        const auto i_j = j * x_dim + i;
        if (threadIdx.y == 0) {
            const auto i_jp1 = (j + 1) * x_dim + i;
            const auto im1_jp1 = (j + 1) * x_dim + i - 1;
            const auto ip1_j = j * x_dim + i + 1;
            E2C2V[0][threadIdx.x][0] = i_j;
            E2C2V[0][threadIdx.x][1] = i_jp1;
            E2C2V[0][threadIdx.x][2] = im1_jp1;
            E2C2V[0][threadIdx.x][3] = ip1_j;
        } else if (threadIdx.y == 1) {
            const auto ip1_j = j * x_dim + i + 1;
            const auto i_jp1 = (j + 1) * x_dim + i;
            const auto ip1_jm1 = (j - 1) * x_dim + i + 1;
            E2C2V[1][threadIdx.x][0] = i_j;
            E2C2V[1][threadIdx.x][1] = ip1_j;
            E2C2V[1][threadIdx.x][2] = i_jp1;
            E2C2V[1][threadIdx.x][3] = ip1_jm1;
        } else {
            const auto ip1_jm1 = (j - 1) * x_dim + i + 1;
            const auto ip1_j = j * x_dim + i + 1;
            const auto i_jm1 = (j - 1) * x_dim + i;
            E2C2V[2][threadIdx.x][0] = i_j;
            E2C2V[2][threadIdx.x][1] = ip1_jm1;
            E2C2V[2][threadIdx.x][2] = ip1_j;
            E2C2V[2][threadIdx.x][3] = i_jm1;
        };
    };
    __syncthreads();
    const auto global_edges_per_orientation = x_dim * y_dim * 4;
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += blockDim.z * gridDim.z) {
        for (auto orientation_id{blockIdx.y * blockDim.y + threadIdx.y}; orientation_id < 3;
             orientation_id += blockDim.y * gridDim.y) {
            for (auto edge_index{blockIdx.x * blockDim.x + threadIdx.x}; edge_index < total_grid_size;
                 edge_index += blockDim.x * gridDim.x) {
                const auto E2C2V_0 = E2C2V[orientation_id][threadIdx.x][0];
                const auto E2C2V_1 = E2C2V[orientation_id][threadIdx.x][1];
                const auto E2C2V_2 = E2C2V[orientation_id][threadIdx.x][2];
                const auto E2C2V_3 = E2C2V[orientation_id][threadIdx.x][3];
                const auto i = is[threadIdx.x];
                const auto j = js[threadIdx.x];
                const auto global_edge_index = (j * x_dim + i) * 4;
                const auto E2ECV_0 = global_edge_index + orientation_id * global_edges_per_orientation;
                const auto E2ECV_1 = E2ECV_0 + 1;
                const auto E2ECV_2 = E2ECV_0 + 2;
                const auto E2ECV_3 = E2ECV_0 + 3;
                const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0) +
                                            v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0) +
                                            u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1) +
                                            v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1);
                const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2) +
                                            v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2) +
                                            u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3) +
                                            v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3);
                const auto local_edge_index = edge_index + orientation_id * total_grid_size;
                z_nabla4_e2_wp_gt_tv(local_edge_index, k_index) =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_gt_tv(local_edge_index, k_index)) *
                                  (inv_vert_vert_length_gt_tv(local_edge_index) *
                                      inv_vert_vert_length_gt_tv(local_edge_index)) +
                              (nabv_tang_wp - 2.0 * z_nabla2_e_gt_tv(local_edge_index, k_index)) *
                                  (inv_primal_edge_length_gt_tv(local_edge_index) *
                                      inv_primal_edge_length_gt_tv(local_edge_index)));
            };
        };
    };
};

template <typename T>
void nabla4_structured_torus_halo_gt<T>::run_gpu_helper() {
    const index_type total_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    cudaError_t errSync, errAsync;
    dim3 tblocks(block_dims_structured.x, block_dims_structured.y, block_dims_structured.z);
    dim3 grid((total_grid_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu<<<grid, tblocks>>>(KDim,
        x_dim,
        y_dim,
        halo,
        total_grid_size,
        u_vert_gt_tv,
        v_vert_gt_tv,
        primal_normal_vert_v1_gt_tv,
        primal_normal_vert_v2_gt_tv,
        z_nabla2_e_gt_tv,
        inv_vert_vert_length_gt_tv,
        inv_primal_edge_length_gt_tv,
        z_nabla4_e2_wp_gt_tv);
    errSync = cudaGetLastError();
    errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        const auto error_string = "Sync error: " + static_cast<std::string>(cudaGetErrorString(errSync));
        throw std::runtime_error(error_string);
    }
    if (errAsync != cudaSuccess) {
        const auto error_string = "Async error: " + static_cast<std::string>(cudaGetErrorString(errSync));
        throw std::runtime_error(error_string);
    }
};
#else
template <typename T>
void nabla4_structured_torus_halo_gt<T>::run_gpu_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
