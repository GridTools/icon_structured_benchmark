#pragma once

#include "common.hpp"

template <Data T>
struct nabla4_data_simd {
    index_type CellDim;
    index_type VertexDim;
    index_type EdgeDim;
    index_type KDim;
    index_type ECVDim;
    index_type output_size;
    std::vector<std::vector<stdx::native_simd<VP_TYPE>>> u_vert;
    std::vector<std::vector<stdx::native_simd<VP_TYPE>>> v_vert;
    std::vector<WP_TYPE> primal_normal_vert_v1;
    std::vector<WP_TYPE> primal_normal_vert_v2;
    std::vector<std::vector<stdx::native_simd<VP_TYPE>>> z_nabla2_e;
    std::vector<WP_TYPE> inv_vert_vert_length;
    std::vector<WP_TYPE> inv_primal_edge_length;
    std::vector<std::vector<stdx::native_simd<VP_TYPE>>> z_nabla4_e2_wp;

    void init_kfirst() {
        u_vert = rand_utils.random_init_vec_2d_simd<VP_TYPE>(
            VertexDim, std::ceil(static_cast<double>(KDim) / stdx::native_simd<VP_TYPE>::size()));
        v_vert = rand_utils.random_init_vec_2d_simd<VP_TYPE>(
            VertexDim, std::ceil(static_cast<double>(KDim) / stdx::native_simd<VP_TYPE>::size()));
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        z_nabla2_e = rand_utils.random_init_vec_2d_simd(
            output_size, std::ceil(static_cast<double>(KDim) / stdx::native_simd<VP_TYPE>::size()));
        inv_vert_vert_length = rand_utils.random_init_vec_1d(output_size);
        inv_primal_edge_length = rand_utils.random_init_vec_1d(output_size);
        z_nabla4_e2_wp.resize(output_size);
        for (index_type i{}; i < output_size; ++i) {
            z_nabla4_e2_wp[i].resize(std::ceil(static_cast<double>(KDim) / stdx::native_simd<VP_TYPE>::size()));
        }
    };

    nabla4_data_simd(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type output_size)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim),
          output_size(output_size) {
        if constexpr (T == Data::kfirst) {
            init_kfirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

    template <typename D>
    std::vector<std::vector<stdx::native_simd<D>>> convert_scalar_simd_vec(const std::vector<std::vector<D>> &vec) {
        std::vector<std::vector<stdx::native_simd<D>>> vec_simd;
        vec_simd.resize(vec.size());
        for (index_type i{}; i < vec.size(); ++i) {
            const int k_size = std::ceil(static_cast<double>(vec[i].size()) / stdx::native_simd<D>::size());
            vec_simd[i].resize(k_size);
            for (index_type j{}; j < vec[i].size() / stdx::native_simd<D>::size(); ++j) {
                vec_simd[i][j].copy_from(&vec[i][j * stdx::native_simd<D>::size()], stdx::element_aligned);
            }
            for (index_type j{}; j < vec[i].size() % stdx::native_simd<D>::size(); ++j) {
                vec_simd[i][k_size - 1][j] =
                    vec[i][j + (vec[i].size() / stdx::native_simd<D>::size()) * stdx::native_simd<D>::size()];
            }
        }
        return vec_simd;
    };

    /// Constructor for validation
    nabla4_data_simd(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type output_size,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim),
          output_size(output_size), u_vert(convert_scalar_simd_vec(u_vert)), v_vert(convert_scalar_simd_vec(v_vert)),
          primal_normal_vert_v1(primal_normal_vert_v1), primal_normal_vert_v2(primal_normal_vert_v2),
          z_nabla2_e(convert_scalar_simd_vec(z_nabla2_e)), inv_vert_vert_length(inv_vert_vert_length),
          inv_primal_edge_length(inv_primal_edge_length) {
        if constexpr (T == Data::kfirst) {
            z_nabla4_e2_wp.resize(EdgeDim);
            for (index_type i{}; i < EdgeDim; ++i) {
                z_nabla4_e2_wp[i].resize(std::ceil(static_cast<double>(KDim) / stdx::native_simd<VP_TYPE>::size()));
            }
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

    nabla4_data_simd get_data() { return *this; };
};
