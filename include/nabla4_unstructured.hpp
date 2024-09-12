#pragma once
#include <array>
#include <iostream>
#include <vector>
#include <execution>
#include <ranges>

#include <experimental/simd>

#include "common.hpp"

namespace stdx = std::experimental;

template <Data T>
class nabla4_unstructured : private nabla4_data<T> {

    using nabla4_data<T>::CellDim;
    using nabla4_data<T>::EdgeDim;
    using nabla4_data<T>::VertexDim;
    using nabla4_data<T>::KDim;
    using nabla4_data<T>::ECVDim;
    using nabla4_data<T>::u_vert;
    using nabla4_data<T>::v_vert;
    using nabla4_data<T>::primal_normal_vert_v1;
    using nabla4_data<T>::primal_normal_vert_v2;
    using nabla4_data<T>::z_nabla2_e;
    using nabla4_data<T>::inv_vert_vert_length;
    using nabla4_data<T>::inv_primal_edge_length;
    using nabla4_data<T>::z_nabla4_e2_wp;
    using nabla4_data<T>::get_data;

    const std::vector<std::array<std::size_t, 4>> e2c2v;
    const std::vector<std::array<std::size_t, 4>> e2ecv;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_unstructured(std::vector<std::array<std::size_t, 4>> e2c2v,
        std::vector<std::array<std::size_t, 4>> e2ecv,
        std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim)
        : e2c2v(e2c2v), e2ecv(e2ecv), nabla4_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size()){};

    /// Constructor for validation
    nabla4_unstructured(std::vector<std::array<std::size_t, 4>> &e2c2v,
        std::vector<std::array<std::size_t, 4>> &e2ecv,
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
        : e2c2v(e2c2v), e2ecv(e2ecv), nabla4_data<T>(CellDim,
                                          VertexDim,
                                          EdgeDim,
                                          KDim,
                                          ECVDim,
                                          e2c2v.size(),
                                          u_vert,
                                          v_vert,
                                          primal_normal_vert_v1,
                                          primal_normal_vert_v2,
                                          z_nabla2_e,
                                          inv_vert_vert_length,
                                          inv_primal_edge_length){};

    std::vector<std::vector<VP_TYPE>> get_output() { return z_nabla4_e2_wp; }

    nabla4_data<T> get_validation_data() { return get_data(); }

    void run_naive() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            for (std::size_t edge_index{}; edge_index < e2c2v.size(); ++edge_index) {
                const auto E2C2V_0 = e2c2v[edge_index][0];
                const auto E2C2V_1 = e2c2v[edge_index][1];
                const auto E2C2V_2 = e2c2v[edge_index][2];
                const auto E2C2V_3 = e2c2v[edge_index][3];
                const auto E2ECV_0 = e2ecv[edge_index][0];
                const auto E2ECV_1 = e2ecv[edge_index][1];
                const auto E2ECV_2 = e2ecv[edge_index][2];
                const auto E2ECV_3 = e2ecv[edge_index][3];
                double nabv_tang_wp = u_vert[k_index][E2C2V_0] * primal_normal_vert_v1[E2ECV_0] +
                                      v_vert[k_index][E2C2V_0] * primal_normal_vert_v2[E2ECV_0] +
                                      u_vert[k_index][E2C2V_1] * primal_normal_vert_v1[E2ECV_1] +
                                      v_vert[k_index][E2C2V_1] * primal_normal_vert_v2[E2ECV_1];
                double nabv_norm_wp = u_vert[k_index][E2C2V_2] * primal_normal_vert_v1[E2ECV_2] +
                                      v_vert[k_index][E2C2V_2] * primal_normal_vert_v2[E2ECV_2] +
                                      u_vert[k_index][E2C2V_3] * primal_normal_vert_v1[E2ECV_3] +
                                      v_vert[k_index][E2C2V_3] * primal_normal_vert_v2[E2ECV_3];
                z_nabla4_e2_wp[k_index][edge_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[k_index][edge_index]) *
                                  (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                              (nabv_tang_wp - 2.0 * z_nabla2_e[k_index][edge_index]) *
                                  (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
            };
        };
    };

    void run_cpu_ifirst() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            for (std::size_t edge_index = 0; edge_index < e2c2v.size(); ++edge_index) {
                const auto E2C2V_0 = e2c2v[edge_index][0];
                const auto E2C2V_1 = e2c2v[edge_index][1];
                const auto E2C2V_2 = e2c2v[edge_index][2];
                const auto E2C2V_3 = e2c2v[edge_index][3];
                const auto E2ECV_0 = e2ecv[edge_index][0];
                const auto E2ECV_1 = e2ecv[edge_index][1];
                const auto E2ECV_2 = e2ecv[edge_index][2];
                const auto E2ECV_3 = e2ecv[edge_index][3];
                double nabv_tang_wp = u_vert[k_index][E2C2V_0] * primal_normal_vert_v1[E2ECV_0] +
                                      v_vert[k_index][E2C2V_0] * primal_normal_vert_v2[E2ECV_0] +
                                      u_vert[k_index][E2C2V_1] * primal_normal_vert_v1[E2ECV_1] +
                                      v_vert[k_index][E2C2V_1] * primal_normal_vert_v2[E2ECV_1];
                double nabv_norm_wp = u_vert[k_index][E2C2V_2] * primal_normal_vert_v1[E2ECV_2] +
                                      v_vert[k_index][E2C2V_2] * primal_normal_vert_v2[E2ECV_2] +
                                      u_vert[k_index][E2C2V_3] * primal_normal_vert_v1[E2ECV_3] +
                                      v_vert[k_index][E2C2V_3] * primal_normal_vert_v2[E2ECV_3];
                z_nabla4_e2_wp[k_index][edge_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[k_index][edge_index]) *
                                  (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                              (nabv_tang_wp - 2.0 * z_nabla2_e[k_index][edge_index]) *
                                  (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
            };
        };
    };

    void run_cpu_kfirst() {
        for (std::size_t edge_index{}; edge_index < e2c2v.size(); ++edge_index) {
            const auto E2C2V_0 = e2c2v[edge_index][0];
            const auto E2C2V_1 = e2c2v[edge_index][1];
            const auto E2C2V_2 = e2c2v[edge_index][2];
            const auto E2C2V_3 = e2c2v[edge_index][3];
            const auto E2ECV_0 = e2ecv[edge_index][0];
            const auto E2ECV_1 = e2ecv[edge_index][1];
            const auto E2ECV_2 = e2ecv[edge_index][2];
            const auto E2ECV_3 = e2ecv[edge_index][3];
#ifdef __clang__
#pragma clang loop unroll_count(8) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (std::size_t k_index = 0; k_index < KDim; ++k_index) {
                double nabv_tang_wp = u_vert[E2C2V_0][k_index] * primal_normal_vert_v1[E2ECV_0] +
                                      v_vert[E2C2V_0][k_index] * primal_normal_vert_v2[E2ECV_0] +
                                      u_vert[E2C2V_1][k_index] * primal_normal_vert_v1[E2ECV_1] +
                                      v_vert[E2C2V_1][k_index] * primal_normal_vert_v2[E2ECV_1];
                double nabv_norm_wp = u_vert[E2C2V_2][k_index] * primal_normal_vert_v1[E2ECV_2] +
                                      v_vert[E2C2V_2][k_index] * primal_normal_vert_v2[E2ECV_2] +
                                      u_vert[E2C2V_3][k_index] * primal_normal_vert_v1[E2ECV_3] +
                                      v_vert[E2C2V_3][k_index] * primal_normal_vert_v2[E2ECV_3];
                z_nabla4_e2_wp[edge_index][k_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                                  (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                              (nabv_tang_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                                  (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
            };
        };
    };

    void run_cpu_kfirst_for_each() {
        for (std::size_t edge_index{}; edge_index < e2c2v.size(); ++edge_index) {
            const auto E2C2V_0 = e2c2v[edge_index][0];
            const auto E2C2V_1 = e2c2v[edge_index][1];
            const auto E2C2V_2 = e2c2v[edge_index][2];
            const auto E2C2V_3 = e2c2v[edge_index][3];
            const auto E2ECV_0 = e2ecv[edge_index][0];
            const auto E2ECV_1 = e2ecv[edge_index][1];
            const auto E2ECV_2 = e2ecv[edge_index][2];
            const auto E2ECV_3 = e2ecv[edge_index][3];
            auto kernel_fields = std::views::zip(u_vert[E2C2V_0], u_vert[E2C2V_1], u_vert[E2C2V_2], u_vert[E2C2V_3],
                v_vert[E2C2V_0], v_vert[E2C2V_1], v_vert[E2C2V_2], v_vert[E2C2V_3], z_nabla2_e[edge_index]);
            const auto inv_vert_vert_length_sqr = inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index];
            const auto inv_primal_edge_length_sqr = inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index];
            const auto kernel_input_fields = std::views::enumerate(kernel_fields);
            std::for_each(kernel_input_fields.begin(), kernel_input_fields.end(),
                [&](const auto& pair) {
                    const auto k_index = get<0>(pair);
                    const auto kernel_fields = get<1>(pair);
                    const auto u_vert_e2c2v_0 = get<0>(kernel_fields);
                    const auto u_vert_e2c2v_1 = get<1>(kernel_fields);
                    const auto u_vert_e2c2v_2 = get<2>(kernel_fields);
                    const auto u_vert_e2c2v_3 = get<3>(kernel_fields);
                    const auto v_vert_e2c2v_0 = get<4>(kernel_fields);
                    const auto v_vert_e2c2v_1 = get<5>(kernel_fields);
                    const auto v_vert_e2c2v_2 = get<6>(kernel_fields);
                    const auto v_vert_e2c2v_3 = get<7>(kernel_fields);
                    double nabv_tang_wp = u_vert_e2c2v_0 * primal_normal_vert_v1[E2ECV_0] +
                                          v_vert_e2c2v_0 * primal_normal_vert_v2[E2ECV_0] +
                                          u_vert_e2c2v_1 * primal_normal_vert_v1[E2ECV_1] +
                                          v_vert_e2c2v_1 * primal_normal_vert_v2[E2ECV_1];
                    double nabv_norm_wp = u_vert_e2c2v_2 * primal_normal_vert_v1[E2ECV_2] +
                                            v_vert_e2c2v_2 * primal_normal_vert_v2[E2ECV_2] +
                                            u_vert_e2c2v_3 * primal_normal_vert_v1[E2ECV_3] +
                                            v_vert_e2c2v_3 * primal_normal_vert_v2[E2ECV_3];
                    const auto z_nabla2_e_edge_index = get<8>(kernel_fields);
                    z_nabla4_e2_wp[edge_index][k_index] =
                        4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_edge_index) *
                                      inv_vert_vert_length_sqr +
                                  (nabv_tang_wp - 2.0 * z_nabla2_e_edge_index) *
                                      inv_primal_edge_length_sqr);
                }
            );
        };
    };

    void run_cpu_kfirst_for_each_unseq() {
        for (std::size_t edge_index{}; edge_index < e2c2v.size(); ++edge_index) {
            const auto E2C2V_0 = e2c2v[edge_index][0];
            const auto E2C2V_1 = e2c2v[edge_index][1];
            const auto E2C2V_2 = e2c2v[edge_index][2];
            const auto E2C2V_3 = e2c2v[edge_index][3];
            const auto E2ECV_0 = e2ecv[edge_index][0];
            const auto E2ECV_1 = e2ecv[edge_index][1];
            const auto E2ECV_2 = e2ecv[edge_index][2];
            const auto E2ECV_3 = e2ecv[edge_index][3];
            auto kernel_fields = std::views::zip(u_vert[E2C2V_0], u_vert[E2C2V_1], u_vert[E2C2V_2], u_vert[E2C2V_3],
                v_vert[E2C2V_0], v_vert[E2C2V_1], v_vert[E2C2V_2], v_vert[E2C2V_3], z_nabla2_e[edge_index]);
            const auto inv_vert_vert_length_sqr = inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index];
            const auto inv_primal_edge_length_sqr = inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index];
            const auto kernel_input_fields = std::views::enumerate(kernel_fields);
            std::for_each(std::execution::unseq, kernel_input_fields.begin(), kernel_input_fields.end(),
                [&](const auto& pair) {
                    const auto k_index = get<0>(pair);
                    const auto kernel_fields = get<1>(pair);
                    const auto u_vert_e2c2v_0 = get<0>(kernel_fields);
                    const auto u_vert_e2c2v_1 = get<1>(kernel_fields);
                    const auto u_vert_e2c2v_2 = get<2>(kernel_fields);
                    const auto u_vert_e2c2v_3 = get<3>(kernel_fields);
                    const auto v_vert_e2c2v_0 = get<4>(kernel_fields);
                    const auto v_vert_e2c2v_1 = get<5>(kernel_fields);
                    const auto v_vert_e2c2v_2 = get<6>(kernel_fields);
                    const auto v_vert_e2c2v_3 = get<7>(kernel_fields);
                    double nabv_tang_wp = u_vert_e2c2v_0 * primal_normal_vert_v1[E2ECV_0] +
                                          v_vert_e2c2v_0 * primal_normal_vert_v2[E2ECV_0] +
                                          u_vert_e2c2v_1 * primal_normal_vert_v1[E2ECV_1] +
                                          v_vert_e2c2v_1 * primal_normal_vert_v2[E2ECV_1];
                    double nabv_norm_wp = u_vert_e2c2v_2 * primal_normal_vert_v1[E2ECV_2] +
                                            v_vert_e2c2v_2 * primal_normal_vert_v2[E2ECV_2] +
                                            u_vert_e2c2v_3 * primal_normal_vert_v1[E2ECV_3] +
                                            v_vert_e2c2v_3 * primal_normal_vert_v2[E2ECV_3];
                    const auto z_nabla2_e_edge_index = get<8>(kernel_fields);
                    z_nabla4_e2_wp[edge_index][k_index] =
                        4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_edge_index) *
                                      inv_vert_vert_length_sqr +
                                  (nabv_tang_wp - 2.0 * z_nabla2_e_edge_index) *
                                      inv_primal_edge_length_sqr);
                }
            );
        };
    };

    void run_cpu_kfirst_simd() {
        for (std::size_t edge_index{}; edge_index < e2c2v.size(); ++edge_index) {
            const auto E2C2V_0 = e2c2v[edge_index][0];
            const auto E2C2V_1 = e2c2v[edge_index][1];
            const auto E2C2V_2 = e2c2v[edge_index][2];
            const auto E2C2V_3 = e2c2v[edge_index][3];
            const auto E2ECV_0 = e2ecv[edge_index][0];
            const auto E2ECV_1 = e2ecv[edge_index][1];
            const auto E2ECV_2 = e2ecv[edge_index][2];
            const auto E2ECV_3 = e2ecv[edge_index][3];
            stdx::native_simd<VP_TYPE> u_vert_e2c2v_0, u_vert_e2c2v_1, u_vert_e2c2v_2, u_vert_e2c2v_3;
            stdx::native_simd<VP_TYPE> v_vert_e2c2v_0, v_vert_e2c2v_1, v_vert_e2c2v_2, v_vert_e2c2v_3;
            stdx::native_simd<WP_TYPE> z_nabla2_e_edge_index;
            std::size_t k_index{};
            for (; k_index < KDim; k_index += stdx::native_simd<VP_TYPE>::size()) {
                u_vert_e2c2v_0.copy_from(&u_vert[E2C2V_0][k_index], stdx::element_aligned);
                u_vert_e2c2v_1.copy_from(&u_vert[E2C2V_1][k_index], stdx::element_aligned);
                u_vert_e2c2v_2.copy_from(&u_vert[E2C2V_2][k_index], stdx::element_aligned);
                u_vert_e2c2v_3.copy_from(&u_vert[E2C2V_3][k_index], stdx::element_aligned);
                v_vert_e2c2v_0.copy_from(&v_vert[E2C2V_0][k_index], stdx::element_aligned);
                v_vert_e2c2v_1.copy_from(&v_vert[E2C2V_1][k_index], stdx::element_aligned);
                v_vert_e2c2v_2.copy_from(&v_vert[E2C2V_2][k_index], stdx::element_aligned);
                v_vert_e2c2v_3.copy_from(&v_vert[E2C2V_3][k_index], stdx::element_aligned);
                const auto nabv_tang_wp = u_vert_e2c2v_0 * primal_normal_vert_v1[E2ECV_0] +
                                      v_vert_e2c2v_0 * primal_normal_vert_v2[E2ECV_0] +
                                      u_vert_e2c2v_1 * primal_normal_vert_v1[E2ECV_1] +
                                      v_vert_e2c2v_1 * primal_normal_vert_v2[E2ECV_1];
                const auto nabv_norm_wp = u_vert_e2c2v_2 * primal_normal_vert_v1[E2ECV_2] +
                                      v_vert_e2c2v_2 * primal_normal_vert_v2[E2ECV_2] +
                                      u_vert_e2c2v_3 * primal_normal_vert_v1[E2ECV_3] +
                                      v_vert_e2c2v_3 * primal_normal_vert_v2[E2ECV_3];
                z_nabla2_e_edge_index.copy_from(&z_nabla2_e[edge_index][k_index], stdx::element_aligned);
                const auto z_nabla4_e2_wp_v = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_edge_index) *
                                  (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                              (nabv_tang_wp - 2.0 * z_nabla2_e_edge_index) *
                                  (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
                z_nabla2_e_edge_index.copy_to(&z_nabla4_e2_wp[edge_index][k_index], stdx::element_aligned);
            };
            for (; k_index < KDim; ++k_index) {
                double nabv_tang_wp = u_vert[E2C2V_0][k_index] * primal_normal_vert_v1[E2ECV_0] +
                                      v_vert[E2C2V_0][k_index] * primal_normal_vert_v2[E2ECV_0] +
                                      u_vert[E2C2V_1][k_index] * primal_normal_vert_v1[E2ECV_1] +
                                      v_vert[E2C2V_1][k_index] * primal_normal_vert_v2[E2ECV_1];
                double nabv_norm_wp = u_vert[E2C2V_2][k_index] * primal_normal_vert_v1[E2ECV_2] +
                                      v_vert[E2C2V_2][k_index] * primal_normal_vert_v2[E2ECV_2] +
                                      u_vert[E2C2V_3][k_index] * primal_normal_vert_v1[E2ECV_3] +
                                      v_vert[E2C2V_3][k_index] * primal_normal_vert_v2[E2ECV_3];
                z_nabla4_e2_wp[edge_index][k_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                                  (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                              (nabv_tang_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                                  (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
            };
        };
    };

    /// Compute function timed for benchmarking
    template <backend_impl I>
    void run() {
        if constexpr (I == backend_impl::naive) {
            run_naive();
        } else if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else if constexpr (I == backend_impl::cpu_kfirst_for_each) {
            run_cpu_kfirst_for_each();
        } else if constexpr (I == backend_impl::cpu_kfirst_for_each_unseq) {
            run_cpu_kfirst_for_each_unseq();
        } else if constexpr (I == backend_impl::cpu_kfirst_simd) {
            run_cpu_kfirst_simd();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};
