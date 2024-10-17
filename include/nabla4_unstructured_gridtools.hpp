#pragma once
#include <array>
#include <iostream>
#include <vector>

#include "nabla4_gridtools.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_unstructured_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_kloop<std::size_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_kloop<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_kloop<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_kloop<int>() {
    return {32, 9, 1, 288};
};

constexpr block_dims block_dims_unstructured_kloop = get_block_dims_unstructured_kloop<index_type>();

template <typename T>
constexpr block_dims get_block_dims_unstructured_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_naive<std::size_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_naive<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_naive<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_naive<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_unstructured_naive = get_block_dims_unstructured_naive<index_type>();
#endif

template <typename T>
class nabla4_unstructured_gt : public nabla4_gt_data<T> {
  public:
    using nabla4_gt_data<T>::CellDim;
    using nabla4_gt_data<T>::EdgeDim;
    using nabla4_gt_data<T>::VertexDim;
    using nabla4_gt_data<T>::KDim;
    using nabla4_gt_data<T>::ECVDim;
    using nabla4_gt_data<T>::output_size;
    using nabla4_gt_data<T>::u_vert_gt;
    using nabla4_gt_data<T>::v_vert_gt;
    using nabla4_gt_data<T>::primal_normal_vert_v1_gt;
    using nabla4_gt_data<T>::primal_normal_vert_v2_gt;
    using nabla4_gt_data<T>::z_nabla2_e_gt;
    using nabla4_gt_data<T>::inv_vert_vert_length_gt;
    using nabla4_gt_data<T>::inv_primal_edge_length_gt;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt;
    using nabla4_gt_data<T>::u_vert_gt_tv;
    using nabla4_gt_data<T>::v_vert_gt_tv;
    using nabla4_gt_data<T>::primal_normal_vert_v1_gt_tv;
    using nabla4_gt_data<T>::primal_normal_vert_v2_gt_tv;
    using nabla4_gt_data<T>::z_nabla2_e_gt_tv;
    using nabla4_gt_data<T>::inv_vert_vert_length_gt_tv;
    using nabla4_gt_data<T>::inv_primal_edge_length_gt_tv;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt_tv;

    using neighbors_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<index_type>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<index_type>().build()->const_target_view());
    neighbors_gt_t e2c2v_gt;
    neighbors_gt_ctv_t e2c2v_gt_tv;
    neighbors_gt_t e2ecv_gt;
    neighbors_gt_ctv_t e2ecv_gt_tv;

    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_unstructured_gt(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(e2c2v.size(), 4_c).initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; }).build()),
        e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(e2ecv.size(), 4_c).initializer([&e2ecv](int i, int j) { return e2ecv[i][j]; }).build()),
        e2c2v_gt_tv(e2c2v_gt->const_target_view()),
        e2ecv_gt_tv(e2ecv_gt->const_target_view()),
        nabla4_gt_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size()){};

    nabla4_unstructured_gt(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length)
        : e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(e2c2v.size(), 4_c).initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; }).build()),
        e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(e2ecv.size(), 4_c).initializer([&e2ecv](int i, int j) { return e2ecv[i][j]; }).build()),
        e2c2v_gt_tv(e2c2v_gt->const_target_view()),
        e2ecv_gt_tv(e2ecv_gt->const_target_view()),
        nabla4_gt_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size(), u_vert,
                                          v_vert,
                                          primal_normal_vert_v1,
                                          primal_normal_vert_v2,
                                          z_nabla2_e,
                                          inv_vert_vert_length,
                                          inv_primal_edge_length){};

    inline __attribute__((always_inline)) void inner_kernel(index_type edge_index,
        index_type k_index,
        const std::array<index_type, 4> &e2c2v_vec,
        const std::array<index_type, 4> &e2ecv_vec) {
        const auto E2C2V_0 = e2c2v_vec[0];
        const auto E2C2V_1 = e2c2v_vec[1];
        const auto E2C2V_2 = e2c2v_vec[2];
        const auto E2C2V_3 = e2c2v_vec[3];
        const auto E2ECV_0 = e2ecv_vec[0];
        const auto E2ECV_1 = e2ecv_vec[1];
        const auto E2ECV_2 = e2ecv_vec[2];
        const auto E2ECV_3 = e2ecv_vec[3];
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
    };

  private:
    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < KDim; ++k_index) {
            const auto edges = e2c2v_gt_tv.lengths()[0];
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type edge_index = 0; edge_index < edges; ++edge_index) {
                const auto E2C2V_0 = e2c2v_gt_tv(edge_index, 0);
                const auto E2C2V_1 = e2c2v_gt_tv(edge_index, 1);
                const auto E2C2V_2 = e2c2v_gt_tv(edge_index, 2);
                const auto E2C2V_3 = e2c2v_gt_tv(edge_index, 3);
                const auto E2ECV_0 = e2ecv_gt_tv(edge_index, 0);
                const auto E2ECV_1 = e2ecv_gt_tv(edge_index, 1);
                const auto E2ECV_2 = e2ecv_gt_tv(edge_index, 2);
                const auto E2ECV_3 = e2ecv_gt_tv(edge_index, 3);
                inner_kernel(
                    edge_index, k_index, {E2C2V_0, E2C2V_1, E2C2V_2, E2C2V_3}, {E2ECV_0, E2ECV_1, E2ECV_2, E2ECV_3});
            };
        };
    };

    void run_cpu_kfirst() {
        for (index_type edge_index{}; edge_index < e2c2v_gt_tv.lengths()[0]; ++edge_index) {
            const auto E2C2V_0 = e2c2v_gt_tv(edge_index, 0);
            const auto E2C2V_1 = e2c2v_gt_tv(edge_index, 1);
            const auto E2C2V_2 = e2c2v_gt_tv(edge_index, 2);
            const auto E2C2V_3 = e2c2v_gt_tv(edge_index, 3);
            const auto E2ECV_0 = e2ecv_gt_tv(edge_index, 0);
            const auto E2ECV_1 = e2ecv_gt_tv(edge_index, 1);
            const auto E2ECV_2 = e2ecv_gt_tv(edge_index, 2);
            const auto E2ECV_3 = e2ecv_gt_tv(edge_index, 3);
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type k_index = 0; k_index < KDim; ++k_index) {
                inner_kernel(
                    edge_index, k_index, {E2C2V_0, E2C2V_1, E2C2V_2, E2C2V_3}, {E2ECV_0, E2ECV_1, E2ECV_2, E2ECV_3});
            };
        };
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
    }
};

#if defined(__CUDACC__)
__global__ void __launch_bounds__(block_dims_unstructured_kloop.size, 2)
    run_gpu_kloop_nabla4_unstructured(index_type EdgeDim,
        index_type KDim,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2c2v_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2ecv_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    const auto edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= EdgeDim) {
        return;
    }
    const auto E2C2V_0 = e2c2v_gt_tv(edge_index, 0);
    const auto E2C2V_1 = e2c2v_gt_tv(edge_index, 1);
    const auto E2C2V_2 = e2c2v_gt_tv(edge_index, 2);
    const auto E2C2V_3 = e2c2v_gt_tv(edge_index, 3);
    const auto E2ECV_0 = e2ecv_gt_tv(edge_index, 0);
    const auto E2ECV_1 = e2ecv_gt_tv(edge_index, 1);
    const auto E2ECV_2 = e2ecv_gt_tv(edge_index, 2);
    const auto E2ECV_3 = e2ecv_gt_tv(edge_index, 3);
    const WP_TYPE primal_normal_vert_v1_0 = primal_normal_vert_v1_gt_tv(E2ECV_0);
    const WP_TYPE primal_normal_vert_v2_0 = primal_normal_vert_v2_gt_tv(E2ECV_0);
    const WP_TYPE primal_normal_vert_v1_1 = primal_normal_vert_v1_gt_tv(E2ECV_1);
    const WP_TYPE primal_normal_vert_v2_1 = primal_normal_vert_v2_gt_tv(E2ECV_1);
    const WP_TYPE primal_normal_vert_v1_2 = primal_normal_vert_v1_gt_tv(E2ECV_2);
    const WP_TYPE primal_normal_vert_v2_2 = primal_normal_vert_v2_gt_tv(E2ECV_2);
    const WP_TYPE primal_normal_vert_v1_3 = primal_normal_vert_v1_gt_tv(E2ECV_3);
    const WP_TYPE primal_normal_vert_v2_3 = primal_normal_vert_v2_gt_tv(E2ECV_3);
    const WP_TYPE inv_vert_vert_length_sqr =
        inv_vert_vert_length_gt_tv(edge_index) * inv_vert_vert_length_gt_tv(edge_index);
    const WP_TYPE inv_primal_edge_length_sqr =
        inv_primal_edge_length_gt_tv(edge_index) * inv_primal_edge_length_gt_tv(edge_index);
    for (auto k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim; k_index += gridDim.y * blockDim.y) {
        const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_0 +
                                    v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_0 +
                                    u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_1 +
                                    v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_1;
        const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_2 +
                                    v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_2 +
                                    u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_3 +
                                    v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_3;
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
        z_nabla4_e2_wp_gt_tv(edge_index, k_index) =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                      (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    };
};

template <typename T>
inline void nabla4_unstructured_gt<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_unstructured_kloop.x, block_dims_unstructured_kloop.y, block_dims_unstructured_kloop.z);
    dim3 grid((output_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu_kloop_nabla4_unstructured<<<grid, tblocks>>>(output_size,
        KDim,
        e2c2v_gt_tv,
        e2ecv_gt_tv,
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

__global__ void __launch_bounds__(block_dims_unstructured_naive.size)
    run_gpu_naive_nabla4_unstructured(index_type EdgeDim,
        index_type KDim,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2c2v_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::neighbors_gt_ctv_t e2ecv_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    const auto edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (edge_index >= EdgeDim || k_index >= KDim)
        return;
    const auto E2C2V_0 = e2c2v_gt_tv(edge_index, 0);
    const auto E2C2V_1 = e2c2v_gt_tv(edge_index, 1);
    const auto E2C2V_2 = e2c2v_gt_tv(edge_index, 2);
    const auto E2C2V_3 = e2c2v_gt_tv(edge_index, 3);
    const auto E2ECV_0 = e2ecv_gt_tv(edge_index, 0);
    const auto E2ECV_1 = e2ecv_gt_tv(edge_index, 1);
    const auto E2ECV_2 = e2ecv_gt_tv(edge_index, 2);
    const auto E2ECV_3 = e2ecv_gt_tv(edge_index, 3);
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
};

template <typename T>
inline void nabla4_unstructured_gt<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_unstructured_naive.x, block_dims_unstructured_naive.y, block_dims_unstructured_naive.z);
    dim3 grid((output_size + tblocks.x - 1) / tblocks.x, (KDim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_naive_nabla4_unstructured<<<grid, tblocks>>>(output_size,
        KDim,
        e2c2v_gt_tv,
        e2ecv_gt_tv,
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
inline void nabla4_unstructured_gt<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};

template <typename T>
inline void nabla4_unstructured_gt<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
