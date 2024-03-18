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
    std::vector<std::vector<float>> u_vert;
    std::vector<std::vector<float>> v_vert;
    std::vector<double> primal_normal_vert_v1;
    std::vector<double> primal_normal_vert_v2;
    std::vector<std::vector<double>> z_nabla2_e;
    std::vector<double> inv_vert_vert_length;
    std::vector<double> inv_primal_edge_length;
    std::vector<std::vector<float>> z_nabla4_e2_wp;
};

class nabla4_unstructured {
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
    RandomUniformUtils rand_utils{-1.0, 1.0};

    /// Initialize vectors needed to execute kernel with random numbers
    void init() {
        // std::cout << "Initializing vectors" << std::endl;
        u_vert = rand_utils.random_init_vec_2d<float>(VertexDim, KDim);
        v_vert = rand_utils.random_init_vec_2d<float>(VertexDim, KDim);
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
    nabla4_unstructured(std::vector<std::vector<std::size_t>> &e2c2v,
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
    nabla4_unstructured(std::vector<std::vector<std::size_t>> &e2c2v,
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
        // std::cout << "=== Initializing validation ===" << std::endl;
        // print_all();
    };

    std::vector<std::vector<float>> get_output() {
        // print_z_nabla4_e2_wp();
        return z_nabla4_e2_wp;
    }

    void print_all() {
        print_e2c2v();
        print_e2ecv();
        print_u_vert();
        print_v_vert();
        print_primal_normal_vert_v1();
        print_primal_normal_vert_v2();
        print_z_nabla2_e();
        print_inv_vert_vert_length();
        print_inv_primal_edge_length();
        print_z_nabla4_e2_wp();
    }

    void print_e2c2v() {
        for (int i{}; i < e2c2v.size(); ++i) {
            std::cout << "e2c2v[" << i << "]: [";
            for (int j{}; j < e2c2v[0].size(); ++j) {
                std::cout << e2c2v[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    void print_e2ecv() {
        for (int i{}; i < e2ecv.size(); ++i) {
            std::cout << "e2ecv[" << i << "]: [";
            for (int j{}; j < e2ecv[0].size(); ++j) {
                std::cout << e2ecv[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    void print_u_vert() {
        for (int i{}; i < u_vert.size(); ++i) {
            std::cout << "u_vert[" << i << "]: [";
            for (int j{}; j < u_vert[0].size(); ++j) {
                std::cout << u_vert[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    void print_v_vert() {
        for (int i{}; i < v_vert.size(); ++i) {
            std::cout << "v_vert[" << i << "]: [";
            for (int j{}; j < v_vert[0].size(); ++j) {
                std::cout << v_vert[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    void print_primal_normal_vert_v1() {
        std::cout << "primal_normal_vert_v1: [";
        for (int i{}; i < primal_normal_vert_v1.size(); ++i) {
            std::cout << primal_normal_vert_v1[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    void print_primal_normal_vert_v2() {
        std::cout << "primal_normal_vert_v2: [";
        for (int i{}; i < primal_normal_vert_v2.size(); ++i) {
            std::cout << primal_normal_vert_v2[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    void print_z_nabla2_e() {
        for (int i{}; i < z_nabla2_e.size(); ++i) {
            std::cout << "z_nabla2_e[" << i << "]: [";
            for (int j{}; j < z_nabla2_e[0].size(); ++j) {
                std::cout << z_nabla2_e[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    void print_inv_vert_vert_length() {
        std::cout << "inv_vert_vert_length: [";
        for (int i{}; i < inv_vert_vert_length.size(); ++i) {
            std::cout << inv_vert_vert_length[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    void print_inv_primal_edge_length() {
        std::cout << "inv_primal_edge_length: [";
        for (int i{}; i < inv_primal_edge_length.size(); ++i) {
            std::cout << inv_primal_edge_length[i] << " ";
        }
        std::cout << "]" << std::endl;
    }

    void print_z_nabla4_e2_wp() {
        for (int i{}; i < EdgeDim; ++i) {
            std::cout << "z_nabla4_e2_wp[" << i << "]: [";
            for (int j{}; j < KDim; ++j) {
                std::cout << z_nabla4_e2_wp[i][j] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    nabla4_validation_data get_validation_data() {
        // std::cout << "=== Getting validation data ===" << std::endl;
        // print_all();
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
        // std::cout << "Running naive nabla4_unstructured benchmark" << std::endl;
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
                // if (k_index == 0) {
                //     std::cout << "E2C2V[" << edge_index << "]: [" << E2C2V_0 <<
                //     ", "
                //     << E2C2V_1 << ", " << E2C2V_2 << ", " << E2C2V_3 << "]" <<
                //     std::endl;
                // }
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
            };
        };
    };

    void run_cpu_ifirst() {
        // std::cout << "Running cpu_ifirst nabla4_unstructured benchmark" <<
        // std::endl;
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
            };
        };
    };

    void run_cpu_kfirst() {
        // std::cout << "Running cpu_kfirst nabla4_unstructured benchmark" <<
        // std::endl;
        for (std::size_t edge_index{}; edge_index < EdgeDim; ++edge_index) {
            const auto E2C2V_0 = e2c2v[edge_index][0];
            const auto E2C2V_1 = e2c2v[edge_index][1];
            const auto E2C2V_2 = e2c2v[edge_index][2];
            const auto E2C2V_3 = e2c2v[edge_index][3];
            const auto E2ECV_0 = e2ecv[edge_index][0];
            const auto E2ECV_1 = e2ecv[edge_index][1];
            const auto E2ECV_2 = e2ecv[edge_index][2];
            const auto E2ECV_3 = e2ecv[edge_index][3];
            for (std::size_t k_index{}; k_index < KDim; ++k_index) {
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
