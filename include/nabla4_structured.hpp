#pragma once
#include <cmath>
#include <vector>
#include <iostream>

#include "random_init.hpp"

enum backend_impl {naive = 0, cpu_ifirst, cpu_kfirst, gpu};
class nabla4_structured
{
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
    nabla4_structured(std::vector<std::vector<std::size_t>>& e2c2v, std::vector<std::vector<std::size_t>>& e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim) : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim) {
        init();
    };

    /// Constructor for validation
    nabla4_structured(std::vector<std::vector<std::size_t>>& e2c2v, std::vector<std::vector<std::size_t>>& e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, std::vector<std::vector<float>>& u_vert, std::vector<std::vector<float>>& v_vert, std::vector<double>& primal_normal_vert_v1, std::vector<double>& primal_normal_vert_v2, std::vector<std::vector<double>>& z_nabla2_e, std::vector<double>& inv_vert_vert_length, std::vector<double>& inv_primal_edge_length) : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim), u_vert(u_vert), v_vert(v_vert), primal_normal_vert_v1(primal_normal_vert_v1), primal_normal_vert_v2(primal_normal_vert_v2), z_nabla2_e(z_nabla2_e), inv_vert_vert_length(inv_vert_vert_length), inv_primal_edge_length(inv_primal_edge_length)  {
        z_nabla4_e2_wp.resize(EdgeDim);
        for (std::size_t i{}; i < EdgeDim; ++i) {
            z_nabla4_e2_wp[i].resize(KDim);
        }
    };

    std::vector<std::vector<float>> get_output() {
        return z_nabla4_e2_wp;
    }

    template<backend_impl I>
    /// Compute function timed for benchmarking
    void run() {
        throw std::runtime_error("Undefined backend implementation");
    };

    inline std::array<std::size_t, 4> get_e2c2v_offsets(std::size_t edge_index) {
        std::array<std::size_t, 4> e2c2v_ret{};
        const std::size_t edges_per_index = 3;
        e2c2v_ret[0] = edge_index/edges_per_index;
        auto edge_direction = edge_index%edges_per_index;
        auto total_domain_size = latitude_dim*longitude_dim;
        auto latitude = edge_index % latitude_dim;
        auto longitude = edge_index / latitude_dim;
        auto latitude_m1 = (latitude_dim + (latitude - 1)) % latitude_dim;
        auto latitude_p1 = (latitude + 1) % latitude_dim;
        // auto longitude_m1 = longitude > 1 ? (e2c2v_ret[0] - latitude_dim)/latitude_dim : longitude_dim - 1;
        auto longitude_m1 = (longitude_dim + (longitude - 1)) % longitude_dim;
        auto longitude_p1 = (longitude + 1) % longitude_dim;
        /// TODO: Adjust below for border edges
        /// i.e. edges 18, 25
        if (edge_direction == 0) {  // east edge
            e2c2v_ret[1] = longitude * latitude_dim + latitude_p1;
            e2c2v_ret[2] = longitude_m1 * latitude_dim + latitude;
            e2c2v_ret[3] = longitude_p1 * latitude_dim + latitude_p1;
        } else if (edge_direction == 1) {  // northeast edge
            e2c2v_ret[1] = longitude_p1 * latitude_dim + latitude_p1;
            e2c2v_ret[2] = longitude_p1 * latitude_dim + latitude;
            e2c2v_ret[3] = longitude * latitude_dim + latitude_p1;
        } else {  // north edge
            e2c2v_ret[1] = longitude_p1 * latitude_dim + latitude;
            e2c2v_ret[2] = longitude_p1 * latitude_dim + latitude_m1;
            e2c2v_ret[3] = longitude * latitude_dim + latitude_p1;
        }
        return e2c2v_ret;
    }

    template<>
    /// Compute function timed for benchmarking
    void run<naive>() {
        // std::cout << "Running naive nabla4_unstructured benchmark" << std::endl;
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            for (std::size_t edge_index{}; edge_index < EdgeDim; ++edge_index) {
                const auto e2c2v_vec = get_e2c2v_offsets(edge_index);
                const auto E2C2V_0 = e2c2v_vec[0];
                const auto E2C2V_1 = e2c2v_vec[1];
                const auto E2C2V_2 = e2c2v_vec[2];
                const auto E2C2V_3 = e2c2v_vec[3];
                if (k_index == 0) {
                    std::cout << "E2C2V[" << edge_index << "]: [" << E2C2V_0 << E2C2V_1 << E2C2V_2 << E2C2V_3 << "]" << std::endl;
                }
                const auto E2ECV_0 = e2ecv[edge_index][0];
                const auto E2ECV_1 = e2ecv[edge_index][1];
                const auto E2ECV_2 = e2ecv[edge_index][2];
                const auto E2ECV_3 = e2ecv[edge_index][3];
                double nabv_tang_wp = static_cast<double>(u_vert[E2C2V_0][k_index]) * primal_normal_vert_v1[E2ECV_0]
                                    + static_cast<double>(v_vert[E2C2V_0][k_index]) * primal_normal_vert_v2[E2ECV_0]
                                    + static_cast<double>(u_vert[E2C2V_1][k_index]) * primal_normal_vert_v1[E2ECV_1]
                                    + static_cast<double>(v_vert[E2C2V_1][k_index]) * primal_normal_vert_v2[E2ECV_1];
                double nabv_norm_wp = static_cast<double>(u_vert[E2C2V_2][k_index]) * primal_normal_vert_v1[E2ECV_2]
                                    + static_cast<double>(v_vert[E2C2V_2][k_index]) * primal_normal_vert_v2[E2ECV_2]
                                    + static_cast<double>(u_vert[E2C2V_3][k_index]) * primal_normal_vert_v1[E2ECV_3]
                                    + static_cast<double>(v_vert[E2C2V_3][k_index]) * primal_normal_vert_v2[E2ECV_3];
                z_nabla4_e2_wp[edge_index][k_index] = 4.0 * (
                    (nabv_norm_wp - 2.0 * z_nabla2_e[edge_index][k_index]) * (inv_vert_vert_length[edge_index] * inv_vert_vert_length[edge_index])
                    + (nabv_tang_wp - 2.0 * z_nabla2_e[edge_index][k_index]) * (inv_primal_edge_length[edge_index] * inv_primal_edge_length[edge_index])
                    );
            };
        };
    };

    template<>
    /// Compute function timed for benchmarking
    void run<cpu_ifirst>() {
        throw std::runtime_error("Undefined backend implementation");
    };

    template<>
    /// Compute function timed for benchmarking
    void run<cpu_kfirst>() {
        throw std::runtime_error("Undefined backend implementation");
    };
};
