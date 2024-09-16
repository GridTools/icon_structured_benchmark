#pragma once
#include <array>
#include <execution>
#include <iostream>
#include <ranges>
#include <vector>

#include <experimental/simd>

#include "common.hpp"
#include "nabla4_data_simd.hpp"

template <Data T>
class nabla4_unstructured_simd : private nabla4_data_simd<T> {

    using nabla4_data_simd<T>::CellDim;
    using nabla4_data_simd<T>::EdgeDim;
    using nabla4_data_simd<T>::VertexDim;
    using nabla4_data_simd<T>::KDim;
    using nabla4_data_simd<T>::ECVDim;
    using nabla4_data_simd<T>::u_vert;
    using nabla4_data_simd<T>::v_vert;
    using nabla4_data_simd<T>::primal_normal_vert_v1;
    using nabla4_data_simd<T>::primal_normal_vert_v2;
    using nabla4_data_simd<T>::z_nabla2_e;
    using nabla4_data_simd<T>::inv_vert_vert_length;
    using nabla4_data_simd<T>::inv_primal_edge_length;
    using nabla4_data_simd<T>::z_nabla4_e2_wp;
    // using nabla4_data_simd<T>::get_data;

    const std::vector<std::array<std::size_t, 4>> e2c2v;
    const std::vector<std::array<std::size_t, 4>> e2ecv;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_unstructured_simd(std::vector<std::array<std::size_t, 4>> e2c2v,
        std::vector<std::array<std::size_t, 4>> e2ecv,
        std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim)
        : e2c2v(e2c2v), e2ecv(e2ecv), nabla4_data_simd<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size()){};

    /// Constructor for validation
    nabla4_unstructured_simd(std::vector<std::array<std::size_t, 4>> &e2c2v,
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
        : e2c2v(e2c2v), e2ecv(e2ecv), nabla4_data_simd<T>(CellDim,
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

    std::vector<std::vector<VP_TYPE>> get_output() {
        std::vector<std::vector<VP_TYPE>> output;
        output.resize(z_nabla4_e2_wp.size());
        for (std::size_t edge_index{}; edge_index < e2c2v.size(); ++edge_index) {
            output[edge_index].resize(KDim);
            for (int k_index{}; k_index < KDim / stdx::native_simd<VP_TYPE>::size(); ++k_index) {
                z_nabla4_e2_wp[edge_index][k_index].copy_to(&output[edge_index][k_index * stdx::native_simd<VP_TYPE>::size()], stdx::element_aligned);
            };
            for (int k_index{}; k_index < KDim % stdx::native_simd<VP_TYPE>::size(); ++k_index) {
                const auto final_k_index = (KDim / stdx::native_simd<VP_TYPE>::size()) * stdx::native_simd<VP_TYPE>::size() + k_index;
                output[edge_index][final_k_index] = z_nabla4_e2_wp[edge_index][(KDim / stdx::native_simd<VP_TYPE>::size())][k_index];
            };
        };
        return output;
    }

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
            const std::array<VP_TYPE, 4> primal_normal_vert_v1_e2ecv = {primal_normal_vert_v1[E2ECV_0],
                primal_normal_vert_v1[E2ECV_1],
                primal_normal_vert_v1[E2ECV_2],
                primal_normal_vert_v1[E2ECV_3]};
            const std::array<VP_TYPE, 4> primal_normal_vert_v2_e2ecv = {primal_normal_vert_v2[E2ECV_0],
                primal_normal_vert_v2[E2ECV_1],
                primal_normal_vert_v2[E2ECV_2],
                primal_normal_vert_v2[E2ECV_3]};
            const auto inv_vert_vert_length_sqr = inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index];
            const auto inv_primal_edge_length_sqr =
                inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index];
            for (int k_index{}; k_index < KDim / stdx::native_simd<VP_TYPE>::size(); ++k_index) {
                const auto nabv_tang_wp = u_vert[E2C2V_0][k_index] * primal_normal_vert_v1_e2ecv[0] +
                                      v_vert[E2C2V_0][k_index] * primal_normal_vert_v2_e2ecv[0] +
                                      u_vert[E2C2V_1][k_index] * primal_normal_vert_v1_e2ecv[1] +
                                      v_vert[E2C2V_1][k_index] * primal_normal_vert_v2_e2ecv[1];
                const auto nabv_norm_wp = u_vert[E2C2V_2][k_index] * primal_normal_vert_v1_e2ecv[2] +
                                      v_vert[E2C2V_2][k_index] * primal_normal_vert_v2_e2ecv[2] +
                                      u_vert[E2C2V_3][k_index] * primal_normal_vert_v1_e2ecv[3] +
                                      v_vert[E2C2V_3][k_index] * primal_normal_vert_v2_e2ecv[3];
                z_nabla4_e2_wp[edge_index][k_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                                  inv_vert_vert_length_sqr +
                              (nabv_tang_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                                  inv_primal_edge_length_sqr);
            };
            const auto k_index_epilogue = KDim / stdx::native_simd<VP_TYPE>::size();
            for (int k_index{}; k_index < KDim % stdx::native_simd<VP_TYPE>::size(); ++k_index) {
                const auto nabv_tang_wp = u_vert[E2C2V_0][k_index_epilogue][k_index] * primal_normal_vert_v1_e2ecv[0] +
                                      v_vert[E2C2V_0][k_index_epilogue][k_index] * primal_normal_vert_v2_e2ecv[0] +
                                      u_vert[E2C2V_1][k_index_epilogue][k_index] * primal_normal_vert_v1_e2ecv[1] +
                                      v_vert[E2C2V_1][k_index_epilogue][k_index] * primal_normal_vert_v2_e2ecv[1];
                const auto nabv_norm_wp = u_vert[E2C2V_2][k_index_epilogue][k_index] * primal_normal_vert_v1_e2ecv[2] +
                                      v_vert[E2C2V_2][k_index_epilogue][k_index] * primal_normal_vert_v2_e2ecv[2] +
                                      u_vert[E2C2V_3][k_index_epilogue][k_index] * primal_normal_vert_v1_e2ecv[3] +
                                      v_vert[E2C2V_3][k_index_epilogue][k_index] * primal_normal_vert_v2_e2ecv[3];
                z_nabla4_e2_wp[edge_index][k_index_epilogue][k_index] =
                    4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[edge_index][k_index_epilogue][k_index]) *
                                  inv_vert_vert_length_sqr +
                              (nabv_tang_wp - 2.0 * z_nabla2_e[edge_index][k_index_epilogue][k_index]) *
                                  inv_primal_edge_length_sqr);
            };
        };
    };

    /// Compute function timed for benchmarking
    template <backend_impl I>
    void run() {
        if constexpr (I == backend_impl::cpu_kfirst_simd) {
            run_cpu_kfirst_simd();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};
