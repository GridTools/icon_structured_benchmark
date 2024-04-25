#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "common.hpp"
#include "random_init.hpp"

template <Data T>
class nabla4_structured_torus : private nabla4_data<T> {

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

    const std::size_t longitude_dim;
    const std::size_t latitude_dim;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    nabla4_structured_torus(std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim,
        std::size_t longitude_dim,
        std::size_t latitude_dim)
        : longitude_dim(longitude_dim),
          latitude_dim(latitude_dim), nabla4_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, EdgeDim){};

    /// Constructor for validation
    nabla4_structured_torus(std::size_t CellDim,
        std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::size_t ECVDim,
        std::size_t longitude_dim,
        std::size_t latitude_dim,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length)
        : longitude_dim(longitude_dim), latitude_dim(latitude_dim), nabla4_data<T>(CellDim,
                                                                        VertexDim,
                                                                        EdgeDim,
                                                                        KDim,
                                                                        ECVDim,
                                                                        EdgeDim,
                                                                        u_vert,
                                                                        v_vert,
                                                                        primal_normal_vert_v1,
                                                                        primal_normal_vert_v2,
                                                                        z_nabla2_e,
                                                                        inv_vert_vert_length,
                                                                        inv_primal_edge_length){};

    std::vector<std::vector<VP_TYPE>> get_output() { return z_nabla4_e2_wp; }

    inline __attribute__((always_inline)) std::size_t modulo(int a, int b) { return a % b; }

    inline __attribute__((always_inline)) std::array<index_type, 4> get_e2c2v_vertices_north_edge(
        std::size_t edge_index,
        std::size_t parent_vertex,
        std::size_t latitude,
        std::size_t longitude,
        std::size_t latitude_p1,
        std::size_t latitude_m1,
        std::size_t longitude_p1,
        std::size_t longitude_m1,
        std::size_t longitude_pstride_latitude_p1,
        std::size_t logitude_pstride_m1_latitude_p1) {
        std::array<index_type, 4> e2c2v_ret{};
        e2c2v_ret[0] = parent_vertex;
        e2c2v_ret[1] = longitude_p1 * latitude_dim + latitude;
        e2c2v_ret[2] =
            modulo((((latitude == 0) * ((2 * longitude_dim - latitude_dim) / 2)) + longitude + 1), longitude_dim) *
                latitude_dim +
            latitude_m1;
        e2c2v_ret[3] = longitude_pstride_latitude_p1;
        return e2c2v_ret;
    }

    inline __attribute__((always_inline)) std::array<index_type, 4> get_e2c2v_vertices_east_edge(std::size_t edge_index,
        std::size_t parent_vertex,
        std::size_t latitude,
        std::size_t longitude,
        std::size_t latitude_p1,
        std::size_t latitude_m1,
        std::size_t longitude_p1,
        std::size_t longitude_m1,
        std::size_t longitude_pstride_latitude_p1,
        std::size_t logitude_pstride_m1_latitude_p1) {
        std::array<index_type, 4> e2c2v_ret{};
        e2c2v_ret[0] = parent_vertex;
        e2c2v_ret[1] = longitude_pstride_latitude_p1;
        e2c2v_ret[2] = longitude_p1 * latitude_dim + latitude;
        e2c2v_ret[3] = logitude_pstride_m1_latitude_p1;
        return e2c2v_ret;
    }

    inline __attribute__((always_inline)) std::array<index_type, 4> get_e2c2v_vertices_southeast_edge(
        std::size_t edge_index,
        std::size_t parent_vertex,
        std::size_t latitude,
        std::size_t longitude,
        std::size_t latitude_p1,
        std::size_t latitude_m1,
        std::size_t longitude_p1,
        std::size_t longitude_m1,
        std::size_t longitude_pstride_latitude_p1,
        std::size_t logitude_pstride_m1_latitude_p1) {
        std::array<index_type, 4> e2c2v_ret{};
        e2c2v_ret[0] = parent_vertex;
        e2c2v_ret[1] = logitude_pstride_m1_latitude_p1;
        e2c2v_ret[2] = longitude_pstride_latitude_p1;
        e2c2v_ret[3] = longitude_m1 * latitude_dim + latitude;
        return e2c2v_ret;
    }

    template <auto f>
    inline __attribute__((always_inline)) const std::array<index_type, 4> get_e2c2v(index_type edge_index) {
        const std::size_t edges_per_index{3};
        const auto starting_vertex = edge_index / edges_per_index;
        const auto latitude = modulo(starting_vertex, latitude_dim);
        const auto longitude = starting_vertex / latitude_dim;
        const auto latitude_p1 = modulo((latitude + 1), latitude_dim);
        const auto latitude_m1 = modulo((latitude_dim + (latitude - 1)), latitude_dim);
        const auto longitude_p1 = modulo((longitude + 1), longitude_dim);
        const auto longitude_m1 = modulo((longitude_dim + (longitude - 1)), longitude_dim);
        const auto longitude_pstride_latitude_p1 =
            modulo((((latitude == latitude_dim - 1) * (latitude_dim / 2)) + longitude), longitude_dim) * latitude_dim +
            latitude_p1;
        const auto logitude_pstride_m1_latitude_p1 =
            modulo((((latitude == latitude_dim - 1) * (latitude_dim / 2)) + longitude_dim + longitude - 1),
                longitude_dim) *
                latitude_dim +
            latitude_p1;
        const auto e2c2v_vec = (this->*f)(edge_index,
            starting_vertex,
            latitude,
            longitude,
            latitude_p1,
            latitude_m1,
            longitude_p1,
            longitude_m1,
            longitude_pstride_latitude_p1,
            logitude_pstride_m1_latitude_p1);
        return e2c2v_vec;
    }

    inline __attribute__((always_inline)) void inner_kernel_ifirst(
        const std::array<index_type, 4> &e2c2v_vec, std::size_t edge_index, std::size_t k_index) {
        const auto E2C2V_0 = e2c2v_vec[0];
        const auto E2C2V_1 = e2c2v_vec[1];
        const auto E2C2V_2 = e2c2v_vec[2];
        const auto E2C2V_3 = e2c2v_vec[3];
        const auto E2ECV_0 = edge_index * 4;
        const auto E2ECV_1 = edge_index * 4 + 1;
        const auto E2ECV_2 = edge_index * 4 + 2;
        const auto E2ECV_3 = edge_index * 4 + 3;
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
    }

    inline __attribute__((always_inline)) void inner_kernel_kfirst(
        const std::array<index_type, 4> &e2c2v_vec, std::size_t edge_index, std::size_t k_index) {
        const auto E2C2V_0 = e2c2v_vec[0];
        const auto E2C2V_1 = e2c2v_vec[1];
        const auto E2C2V_2 = e2c2v_vec[2];
        const auto E2C2V_3 = e2c2v_vec[3];
        const auto E2ECV_0 = edge_index * 4;
        const auto E2ECV_1 = edge_index * 4 + 1;
        const auto E2ECV_2 = edge_index * 4 + 2;
        const auto E2ECV_3 = edge_index * 4 + 3;
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
    }

    void run_naive() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            for (std::size_t edge_index{0}; edge_index < EdgeDim; edge_index += 3) {
                const auto e2c2v_vec_north =
                    get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_north_edge>(edge_index);
                inner_kernel_ifirst(e2c2v_vec_north, edge_index, k_index);
                const auto e2c2v_vec_east =
                    get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_east_edge>(edge_index);
                inner_kernel_ifirst(e2c2v_vec_east, edge_index + 1, k_index);
                const auto e2c2v_vec_southeast =
                    get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_southeast_edge>(edge_index);
                inner_kernel_ifirst(e2c2v_vec_southeast, edge_index + 2, k_index);
            }
        };
    };

    void run_cpu_ifirst() {
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
#pragma omp simd
            for (std::size_t edge_index = 0; edge_index < EdgeDim; edge_index += 3) {
                const auto e2c2v_vec_north =
                    get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_north_edge>(edge_index);
                inner_kernel_ifirst(e2c2v_vec_north, edge_index, k_index);
                const auto e2c2v_vec_east =
                    get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_east_edge>(edge_index);
                inner_kernel_ifirst(e2c2v_vec_east, edge_index + 1, k_index);
                const auto e2c2v_vec_southeast =
                    get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_southeast_edge>(edge_index);
                inner_kernel_ifirst(e2c2v_vec_southeast, edge_index + 2, k_index);
            }
        }
    };

    void run_cpu_kfirst() {
        for (std::size_t edge_index{}; edge_index < EdgeDim; edge_index += 3) {
            const auto e2c2v_vec_north = get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_north_edge>(edge_index);
            const auto e2c2v_vec_east = get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_east_edge>(edge_index);
            const auto e2c2v_vec_southeast =
                get_e2c2v<&nabla4_structured_torus::get_e2c2v_vertices_southeast_edge>(edge_index);
#pragma omp simd
            for (std::size_t k_index = 0; k_index < KDim; ++k_index) {
                inner_kernel_kfirst(e2c2v_vec_north, edge_index, k_index);
                inner_kernel_kfirst(e2c2v_vec_east, edge_index + 1, k_index);
                inner_kernel_kfirst(e2c2v_vec_southeast, edge_index + 2, k_index);
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
