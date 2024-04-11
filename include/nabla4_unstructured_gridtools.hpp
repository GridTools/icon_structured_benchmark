#pragma once
#include <array>
#include <iostream>
#include <vector>

#include "nabla4_gridtools.hpp"

template <typename T>
class nabla4_unstructured_gt : public nabla4_gt_data<T> {

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

    using neighbors_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<std::size_t>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<std::size_t>().build()->const_target_view());
    neighbors_gt_t e2c2v_gt;
    neighbors_gt_ctv_t e2c2v_gt_tv;
    neighbors_gt_t e2ecv_gt;
    neighbors_gt_ctv_t e2ecv_gt_tv;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_unstructured_gt(std::vector<std::array<std::size_t, 4>> e2c2v,
        std::vector<std::array<std::size_t, 4>> e2ecv,
        std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim)
        : e2c2v_gt(storage::builder<T>.template type<std::size_t>().dimensions(e2c2v.size(), 4_c).initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; }).build()),
        e2ecv_gt(storage::builder<T>.template type<std::size_t>().dimensions(e2ecv.size(), 4_c).initializer([&e2ecv](int i, int j) { return e2ecv[i][j]; }).build()),
        e2c2v_gt_tv(e2c2v_gt->const_target_view()),
        e2ecv_gt_tv(e2ecv_gt->const_target_view()),
        nabla4_gt_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size()){};

    nabla4_unstructured_gt(std::vector<std::array<std::size_t, 4>> e2c2v,
        std::vector<std::array<std::size_t, 4>> e2ecv,
        std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length)
        : e2c2v_gt(storage::builder<T>.template type<std::size_t>().dimensions(e2c2v.size(), 4_c).initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; }).build()),
        e2ecv_gt(storage::builder<T>.template type<std::size_t>().dimensions(e2ecv.size(), 4_c).initializer([&e2ecv](int i, int j) { return e2ecv[i][j]; }).build()),
        e2c2v_gt_tv(e2c2v_gt->const_target_view()),
        e2ecv_gt_tv(e2ecv_gt->const_target_view()),
        nabla4_gt_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size(), u_vert,
                                          v_vert,
                                          primal_normal_vert_v1,
                                          primal_normal_vert_v2,
                                          z_nabla2_e,
                                          inv_vert_vert_length,
                                          inv_primal_edge_length){};

  private:
    inline __attribute__((always_inline)) void inner_kernel(std::size_t edge_index,
        std::size_t k_index,
        const std::array<std::size_t, 4> &e2c2v_vec,
        const std::array<std::size_t, 4> &e2ecv_vec) {
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

    void run_cpu_ifirst() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (std::size_t edge_index = 0; edge_index < e2c2v_gt_tv.lengths()[0]; ++edge_index) {
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
        for (std::size_t edge_index{}; edge_index < e2c2v_gt_tv.lengths()[0]; ++edge_index) {
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
            for (std::size_t k_index = 0; k_index < KDim; ++k_index) {
                inner_kernel(
                    edge_index, k_index, {E2C2V_0, E2C2V_1, E2C2V_2, E2C2V_3}, {E2ECV_0, E2ECV_1, E2ECV_2, E2ECV_3});
            };
        };
    };

#ifdef __NVCC__
    __global__ __device__ void run_gpu() {
        const auto edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
        for (std::size_t edge_index{blockIdx.x * blockDim.x + threadIdx.x}; edge_index < e2c2v_gt_tv.lengths()[0];
             edge_index += blockDim.x * gridDim.x) {
            for (std::size_t k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim;
                 k_index += blockDim.y * gridDim.y) {
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
#endif

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
#ifdef __NVCC__
        } else if constexpr (I == backend_impl::gpu) {
            dim3 tblocks(32, 32, 1);
            dim3 grid((e2c2v_gt_tv.lengths()[0] + tblocks.x - 1) / tblocks.x, (KDim + tblocks.y - 1) / tblocks.y, 1);
            run_gpu<<<grid, tblocks>>>();
#endif
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};
