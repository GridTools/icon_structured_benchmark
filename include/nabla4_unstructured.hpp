#pragma once
#include <array>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "random_init.hpp"

struct nabla4_validation_data {
    std::size_t CellDim;
    std::size_t EdgeDim;
    std::size_t VertexDim;
    std::size_t KDim;
    std::size_t ECVDim;
    std::vector<std::vector<VP_TYPE>> u_vert;
    std::vector<std::vector<VP_TYPE>> v_vert;
    std::vector<double> primal_normal_vert_v1;
    std::vector<double> primal_normal_vert_v2;
    std::vector<std::vector<double>> z_nabla2_e;
    std::vector<double> inv_vert_vert_length;
    std::vector<double> inv_primal_edge_length;
    std::vector<std::vector<VP_TYPE>> z_nabla4_e2_wp;
};

template <Data T>
class nabla4_unstructured {
  private:
    std::vector<std::array<std::size_t, 4>> e2c2v;
    std::vector<std::array<std::size_t, 4>> e2ecv;
    std::size_t CellDim;
    std::size_t EdgeDim;
    std::size_t VertexDim;
    std::size_t KDim;
    std::size_t ECVDim;
    std::vector<std::vector<VP_TYPE>> u_vert;
    std::vector<std::vector<VP_TYPE>> v_vert;
    std::vector<double> primal_normal_vert_v1;
    std::vector<double> primal_normal_vert_v2;
    std::vector<std::vector<double>> z_nabla2_e;
    std::vector<double> inv_vert_vert_length;
    std::vector<double> inv_primal_edge_length;
    std::vector<std::vector<VP_TYPE>> z_nabla4_e2_wp;

    /// Random number utilities
    RandomUniformUtils rand_utils{-1.0, 1.0};

    /// Initialize vectors needed to execute kernel with random numbers
    void init_ifirst() {
        u_vert = rand_utils.random_init_vec_2d<VP_TYPE>(KDim, VertexDim);
        v_vert = rand_utils.random_init_vec_2d<VP_TYPE>(KDim, VertexDim);
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        z_nabla2_e = rand_utils.random_init_vec_2d(KDim, EdgeDim);
        inv_vert_vert_length = rand_utils.random_init_vec_1d(EdgeDim);
        inv_primal_edge_length = rand_utils.random_init_vec_1d(EdgeDim);
        z_nabla4_e2_wp.resize(KDim);
        for (std::size_t i{}; i < KDim; ++i) {
            z_nabla4_e2_wp[i].resize(EdgeDim);
        }
    }

    void init_kfirst() {
        u_vert = rand_utils.random_init_vec_2d<VP_TYPE>(VertexDim, KDim);
        v_vert = rand_utils.random_init_vec_2d<VP_TYPE>(VertexDim, KDim);
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        z_nabla2_e = rand_utils.random_init_vec_2d(EdgeDim, KDim);
        inv_vert_vert_length = rand_utils.random_init_vec_1d(EdgeDim);
        inv_primal_edge_length = rand_utils.random_init_vec_1d(EdgeDim);
        z_nabla4_e2_wp.resize(EdgeDim);
        for (std::size_t i{}; i < EdgeDim; ++i) {
            z_nabla4_e2_wp[i].resize(KDim);
        }
    }

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
        : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim),
          ECVDim(ECVDim) {
        if constexpr (T == Data::ifirst) {
            init_ifirst();
        } else if constexpr (T == Data::kfirst) {
            init_kfirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

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
        std::vector<double> &primal_normal_vert_v1,
        std::vector<double> &primal_normal_vert_v2,
        std::vector<std::vector<double>> &z_nabla2_e,
        std::vector<double> &inv_vert_vert_length,
        std::vector<double> &inv_primal_edge_length)
        : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim),
          ECVDim(ECVDim), u_vert(u_vert), v_vert(v_vert), primal_normal_vert_v1(primal_normal_vert_v1),
          primal_normal_vert_v2(primal_normal_vert_v2), z_nabla2_e(z_nabla2_e),
          inv_vert_vert_length(inv_vert_vert_length), inv_primal_edge_length(inv_primal_edge_length) {
        if constexpr (T == Data::ifirst) {
            z_nabla4_e2_wp.resize(KDim);
            for (std::size_t i{}; i < KDim; ++i) {
                z_nabla4_e2_wp[i].resize(EdgeDim);
            }
        } else if constexpr (T == Data::kfirst) {
            z_nabla4_e2_wp.resize(EdgeDim);
            for (std::size_t i{}; i < EdgeDim; ++i) {
                z_nabla4_e2_wp[i].resize(KDim);
            }
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

    std::vector<std::vector<VP_TYPE>> get_output() { return z_nabla4_e2_wp; }

    nabla4_validation_data get_validation_data() {
        return nabla4_validation_data{CellDim,
            EdgeDim,
            VertexDim,
            KDim,
            ECVDim,
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
            z_nabla4_e2_wp};
    }

    void run_naive() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            for (std::size_t edge_index{}; edge_index < EdgeDim; ++edge_index) {
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
#pragma omp simd
            for (std::size_t edge_index = 0; edge_index < EdgeDim; ++edge_index) {
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
        for (std::size_t edge_index{}; edge_index < EdgeDim; ++edge_index) {
            const auto E2C2V_0 = e2c2v[edge_index][0];
            const auto E2C2V_1 = e2c2v[edge_index][1];
            const auto E2C2V_2 = e2c2v[edge_index][2];
            const auto E2C2V_3 = e2c2v[edge_index][3];
            const auto E2ECV_0 = e2ecv[edge_index][0];
            const auto E2ECV_1 = e2ecv[edge_index][1];
            const auto E2ECV_2 = e2ecv[edge_index][2];
            const auto E2ECV_3 = e2ecv[edge_index][3];
#pragma omp simd
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

    /// Compute function timed for benchmarking
    template <backend_impl I>
    void run() {
        if constexpr (I == backend_impl::naive) {
            run_naive();
        } else if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };
};
