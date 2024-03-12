#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "random_init.hpp"

#define ARRAY_TYPE std::size_t

class nabla4_structured {
  private:
    std::vector<std::vector<std::size_t>> e2c2v;
    std::vector<std::vector<std::size_t>> e2ecv;
    std::size_t CellDim;
    std::size_t EdgeDim;
    std::size_t VertexDim;
    std::size_t KDim;
    std::size_t ECVDim;
    std::vector<std::vector<float>> u_vert;
    std::vector<std::vector<float>> v_vert;
    std::vector<double> primal_normal_vert_v1;
    std::vector<double> primal_normal_vert_v2;
    std::vector<std::vector<double>> z_nabla2_e;
    std::vector<double> inv_vert_vert_length;
    std::vector<double> inv_primal_edge_length;
    std::vector<std::vector<float>> z_nabla4_e2_wp;

    /// Random number utilities
    RandomUniformUtils rand_utils{-2.0, 2.0};

    std::size_t longitude_dim{3};
    std::size_t latitude_dim{3};

    /// Initialize vectors needed to execute kernel with random numbers
    void init() {
        // std::cout << "Initializing vectors" << std::endl;
        u_vert = rand_utils.random_init_vec_2d<float>(VertexDim, KDim);
        v_vert = rand_utils.random_init_vec_2d<float>(VertexDim, KDim);
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(ECVDim);
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
    nabla4_structured(std::vector<std::vector<std::size_t>> &e2c2v,
        std::vector<std::vector<std::size_t>> &e2ecv,
        std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim)
        : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim),
          ECVDim(ECVDim) {
        init();
    };

    /// Constructor for validation
    nabla4_structured(std::vector<std::vector<std::size_t>> &e2c2v,
        std::vector<std::vector<std::size_t>> &e2ecv,
        std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim,
        std::vector<std::vector<float>> &u_vert,
        std::vector<std::vector<float>> &v_vert,
        std::vector<double> &primal_normal_vert_v1,
        std::vector<double> &primal_normal_vert_v2,
        std::vector<std::vector<double>> &z_nabla2_e,
        std::vector<double> &inv_vert_vert_length,
        std::vector<double> &inv_primal_edge_length)
        : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim),
          ECVDim(ECVDim), u_vert(u_vert), v_vert(v_vert), primal_normal_vert_v1(primal_normal_vert_v1),
          primal_normal_vert_v2(primal_normal_vert_v2), z_nabla2_e(z_nabla2_e),
          inv_vert_vert_length(inv_vert_vert_length), inv_primal_edge_length(inv_primal_edge_length) {
        z_nabla4_e2_wp.resize(EdgeDim);
        for (std::size_t i{}; i < EdgeDim; ++i) {
            z_nabla4_e2_wp[i].resize(KDim);
        }
    };

    std::vector<std::vector<float>> get_output() { return z_nabla4_e2_wp; }

    inline std::array<ARRAY_TYPE, 4> get_e2c2v_offsets0(std::size_t edge_index) {
        std::array<ARRAY_TYPE, 4> e2c2v_ret{};
        const std::size_t edges_per_index{3};
        e2c2v_ret[0] = edge_index / edges_per_index;
        auto latitude = e2c2v_ret[0] % latitude_dim;
        auto longitude = e2c2v_ret[0] / latitude_dim;
        auto latitude_p1 = (latitude + 1) % latitude_dim;
        auto longitude_p1 = (longitude + 1) % longitude_dim;
        auto longitude_m1 = (longitude_dim + (longitude - 1)) % longitude_dim;
        e2c2v_ret[1] = longitude * latitude_dim + latitude_p1;
        e2c2v_ret[2] = longitude_p1 * latitude_dim + latitude_p1;
        e2c2v_ret[3] = longitude_m1 * latitude_dim + latitude;
        return e2c2v_ret;
    }

    inline std::array<ARRAY_TYPE, 4> get_e2c2v_offsets1(std::size_t edge_index) {
        std::array<ARRAY_TYPE, 4> e2c2v_ret{};
        const std::size_t edges_per_index{3};
        e2c2v_ret[0] = edge_index / edges_per_index;
        auto latitude = e2c2v_ret[0] % latitude_dim;
        auto longitude = e2c2v_ret[0] / latitude_dim;
        auto latitude_p1 = (latitude + 1) % latitude_dim;
        auto longitude_p1 = (longitude + 1) % longitude_dim;
        e2c2v_ret[1] = longitude_p1 * latitude_dim + latitude_p1;
        e2c2v_ret[2] = longitude * latitude_dim + latitude_p1;
        e2c2v_ret[3] = longitude_p1 * latitude_dim + latitude;
        return e2c2v_ret;
    }

    inline std::array<ARRAY_TYPE, 4> get_e2c2v_offsets2(std::size_t edge_index) {
        std::array<ARRAY_TYPE, 4> e2c2v_ret{};
        const std::size_t edges_per_index{3};
        e2c2v_ret[0] = edge_index / edges_per_index;
        auto latitude = e2c2v_ret[0] % latitude_dim;
        auto longitude = e2c2v_ret[0] / latitude_dim;
        auto latitude_p1 = (latitude + 1) % latitude_dim;
        auto latitude_m1 = (latitude_dim + (latitude - 1)) % latitude_dim;
        auto longitude_p1 = (longitude + 1) % longitude_dim;
        e2c2v_ret[1] = longitude_p1 * latitude_dim + latitude;
        e2c2v_ret[2] = longitude * latitude_dim + latitude_m1;
        e2c2v_ret[3] = longitude_p1 * latitude_dim + latitude_p1;
        return e2c2v_ret;
    }

    template <auto f>
    inline void inner_kernel(ARRAY_TYPE edge_index, std::size_t k_index) {
        const auto e2c2v_vec = (this->*f)(edge_index);
        const auto E2C2V_0 = e2c2v_vec[0];
        const auto E2C2V_1 = e2c2v_vec[1];
        const auto E2C2V_2 = e2c2v_vec[2];
        const auto E2C2V_3 = e2c2v_vec[3];
        // if (k_index == 0) {
        //     std::cout << "E2C2V[" << edge_index << "]: [" << E2C2V_0 << ", " <<
        //     E2C2V_1 << ", " << E2C2V_2 << ", " << E2C2V_3 << "]" << std::endl;
        // }
        const auto E2ECV_0 = edge_index * 4;
        const auto E2ECV_1 = edge_index * 4 + 1;
        const auto E2ECV_2 = edge_index * 4 + 2;
        const auto E2ECV_3 = edge_index * 4 + 3;
        double nabv_tang_wp = static_cast<double>(u_vert[E2C2V_0][k_index]) * primal_normal_vert_v1[E2ECV_0] +
                              static_cast<double>(v_vert[E2C2V_0][k_index]) * primal_normal_vert_v2[E2ECV_0] +
                              static_cast<double>(u_vert[E2C2V_1][k_index]) * primal_normal_vert_v1[E2ECV_1] +
                              static_cast<double>(v_vert[E2C2V_1][k_index]) * primal_normal_vert_v2[E2ECV_1];
        double nabv_norm_wp = static_cast<double>(u_vert[E2C2V_2][k_index]) * primal_normal_vert_v1[E2ECV_2] +
                              static_cast<double>(v_vert[E2C2V_2][k_index]) * primal_normal_vert_v2[E2ECV_2] +
                              static_cast<double>(u_vert[E2C2V_3][k_index]) * primal_normal_vert_v1[E2ECV_3] +
                              static_cast<double>(v_vert[E2C2V_3][k_index]) * primal_normal_vert_v2[E2ECV_3];
        z_nabla4_e2_wp[edge_index][k_index] =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                          (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index]) +
                      (nabv_tang_wp - 2.0 * z_nabla2_e[edge_index][k_index]) *
                          (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index]));
    }

    void run_naive() {
        // std::cout << "Running naive nabla4_unstructured benchmark" << std::endl;
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            for (std::size_t edge_index{0}; edge_index < EdgeDim; edge_index += 3) {
                inner_kernel<&nabla4_structured::get_e2c2v_offsets0>(edge_index, k_index);
                inner_kernel<&nabla4_structured::get_e2c2v_offsets1>(edge_index + 1, k_index);
                inner_kernel<&nabla4_structured::get_e2c2v_offsets2>(edge_index + 2, k_index);
            };
        };
    };

    void run_cpu_ifirst() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
#pragma omp simd
            for (std::size_t edge_index = 0; edge_index < EdgeDim; edge_index += 3) {
                inner_kernel<&nabla4_structured::get_e2c2v_offsets0>(edge_index, k_index);
                inner_kernel<&nabla4_structured::get_e2c2v_offsets1>(edge_index + 1, k_index);
                inner_kernel<&nabla4_structured::get_e2c2v_offsets2>(edge_index + 2, k_index);
            };
        };
    };

    void run_cpu_kfirst() {
        for (std::size_t edge_index{}; edge_index < EdgeDim; edge_index += 3) {
#pragma omp simd
            for (std::size_t k_index = 0; k_index < KDim; ++k_index) {
                inner_kernel<&nabla4_structured::get_e2c2v_offsets0>(edge_index, k_index);
                inner_kernel<&nabla4_structured::get_e2c2v_offsets1>(edge_index + 1, k_index);
                inner_kernel<&nabla4_structured::get_e2c2v_offsets2>(edge_index + 2, k_index);
            };
        };
    };

    template <backend_impl I>
    /// Compute function timed for benchmarking
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
