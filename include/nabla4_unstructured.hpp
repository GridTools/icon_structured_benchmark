#include <vector>
#include <iostream>

#include "random_init.hpp"

class nabla4_unstructured
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

public:

    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_unstructured(std::vector<std::vector<std::size_t>>& e2c2v, std::vector<std::vector<std::size_t>>& e2ecv, std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim) : e2c2v(e2c2v), e2ecv(e2ecv), CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim) {};

    /// Initialize vectors needed to execute kernel with random numbers
    void init() {
        std::cout << "Initializing vectors" << std::endl;
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

    /// Compute function timed for benchmarking
    void run() {
        std::cout << "Running nabla4_unstructured benchmark" << std::endl;
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
};
