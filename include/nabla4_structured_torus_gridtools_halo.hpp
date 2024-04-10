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
    using nabla4_gt_data<T>::u_vert_gt_hv;
    using nabla4_gt_data<T>::v_vert_gt_hv;
    using nabla4_gt_data<T>::primal_normal_vert_v1_gt_hv;
    using nabla4_gt_data<T>::primal_normal_vert_v2_gt_hv;
    using nabla4_gt_data<T>::z_nabla2_e_gt_hv;
    using nabla4_gt_data<T>::inv_vert_vert_length_gt_hv;
    using nabla4_gt_data<T>::inv_primal_edge_length_gt_hv;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt_hv;

    const std::size_t y_dim;
    const std::size_t x_dim;
    const std::size_t halo;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_structured_torus_halo_gt(std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim,
        std::size_t y_dim,
        std::size_t x_dim,
        std::size_t halo)
        : y_dim(y_dim), x_dim(x_dim),
          halo(halo), nabla4_gt_data<T>(
                          CellDim, VertexDim, EdgeDim, KDim, ECVDim, (x_dim - 2 * halo) * (y_dim - halo * 2) * 3){};
    nabla4_structured_torus_halo_gt(std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim,
        std::size_t y_dim,
        std::size_t x_dim,
        std::size_t halo,
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
    inline __attribute__((always_inline)) void inner_kernel(const std::array<ARRAY_TYPE, 4> &e2c2v_vec,
        std::size_t edge_index,
        std::size_t k_index,
        std::size_t e2c2v_index) {
        const auto E2C2V_0 = e2c2v_vec[0];
        const auto E2C2V_1 = e2c2v_vec[1];
        const auto E2C2V_2 = e2c2v_vec[2];
        const auto E2C2V_3 = e2c2v_vec[3];
        const auto E2ECV_0 = e2c2v_index * 4;
        const auto E2ECV_1 = e2c2v_index * 4 + 1;
        const auto E2ECV_2 = e2c2v_index * 4 + 2;
        const auto E2ECV_3 = e2c2v_index * 4 + 3;
        double nabv_tang_wp = u_vert_gt_hv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_hv(E2ECV_0) +
                              v_vert_gt_hv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_hv(E2ECV_0) +
                              u_vert_gt_hv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_hv(E2ECV_1) +
                              v_vert_gt_hv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_hv(E2ECV_1);
        double nabv_norm_wp = u_vert_gt_hv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_hv(E2ECV_2) +
                              v_vert_gt_hv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_hv(E2ECV_2) +
                              u_vert_gt_hv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_hv(E2ECV_3) +
                              v_vert_gt_hv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_hv(E2ECV_3);
        z_nabla4_e2_wp_gt_hv(edge_index, k_index) =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_gt_hv(edge_index, k_index)) *
                          (inv_vert_vert_length_gt_hv(edge_index) * inv_vert_vert_length_gt_hv(edge_index)) +
                      (nabv_tang_wp - 2.0 * z_nabla2_e_gt_hv(edge_index, k_index)) *
                          (inv_primal_edge_length_gt_hv(edge_index) * inv_primal_edge_length_gt_hv(edge_index)));
    }

#if defined(__GNUC__)
    __attribute__((optimize("unroll-loops")))
#endif
    void
    run_cpu_ifirst() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (std::size_t j = halo; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (std::size_t i = halo; i < x_dim - halo; ++i) {
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
        for (std::size_t j = halo; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable)
#endif
            for (std::size_t i = halo; i < x_dim - halo; ++i) {
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
                for (std::size_t k_index{}; k_index < KDim; ++k_index) {
                    inner_kernel({i_j, i_jp1, im1_jp1, ip1_j}, local_edge_index, k_index, global_edge_index);
                    inner_kernel({i_j, ip1_j, i_jp1, ip1_jm1}, local_edge_index + 1, k_index, global_edge_index + 1);
                    inner_kernel({i_j, ip1_jm1, ip1_j, i_jm1}, local_edge_index + 2, k_index, global_edge_index + 2);
                }
            }
        }
    };

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};
