#pragma once

#include "random_init.hpp"
#include <experimental/simd>

namespace stdx = std::experimental;

#if defined(INDEX_TYPE_SIZE_T)
using index_type = std::size_t;
#elif defined(INDEX_TYPE_INT64)
using index_type = std::int64_t;
#elif defined(INDEX_TYPE_UINT32_T)
using index_type = std::uint32_t;
#elif defined(INDEX_TYPE_INT)
using index_type = int;
#endif

#define VP_TYPE double
#define WP_TYPE double

enum backend_impl {
    naive = 0,
    cpu_ifirst,
    cpu_kfirst,
    gpu_naive,
    gpu_kloop,
    cpu_kfirst_pragma,
    cpu_kfirst_for_each,
    cpu_kfirst_for_each_unseq,
    cpu_kfirst_simd
};

enum class Data { ifirst, kfirst };

struct block_dims {
    index_type x;
    index_type y;
    index_type z;
    index_type size;
};

/// Random number utilities
static RandomUniformUtils rand_utils{-1.0, 1.0};

template <Data T>
struct nabla4_data {
    index_type CellDim;
    index_type VertexDim;
    index_type EdgeDim;
    index_type KDim;
    index_type ECVDim;
    index_type output_size;
    std::vector<std::vector<VP_TYPE>> u_vert;
    std::vector<std::vector<VP_TYPE>> v_vert;
    std::vector<WP_TYPE> primal_normal_vert_v1;
    std::vector<WP_TYPE> primal_normal_vert_v2;
    std::vector<std::vector<WP_TYPE>> z_nabla2_e;
    std::vector<WP_TYPE> inv_vert_vert_length;
    std::vector<WP_TYPE> inv_primal_edge_length;
    std::vector<std::vector<VP_TYPE>> z_nabla4_e2_wp;

    /// Initialize vectors needed to execute kernel with random numbers
    void init_ifirst() {
        u_vert = rand_utils.random_init_vec_2d<VP_TYPE>(KDim, VertexDim);
        v_vert = rand_utils.random_init_vec_2d<VP_TYPE>(KDim, VertexDim);
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        z_nabla2_e = rand_utils.random_init_vec_2d(KDim, output_size);
        inv_vert_vert_length = rand_utils.random_init_vec_1d(output_size);
        inv_primal_edge_length = rand_utils.random_init_vec_1d(output_size);
        z_nabla4_e2_wp.resize(KDim);
        for (index_type i{}; i < KDim; ++i) {
            z_nabla4_e2_wp[i].resize(output_size);
        }
    }

    void init_kfirst() {
        u_vert = rand_utils.random_init_vec_2d<VP_TYPE>(VertexDim, KDim);
        v_vert = rand_utils.random_init_vec_2d<VP_TYPE>(VertexDim, KDim);
        primal_normal_vert_v1 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        primal_normal_vert_v2 = rand_utils.random_init_vec_1d(EdgeDim * ECVDim);
        z_nabla2_e = rand_utils.random_init_vec_2d(output_size, KDim);
        inv_vert_vert_length = rand_utils.random_init_vec_1d(output_size);
        inv_primal_edge_length = rand_utils.random_init_vec_1d(output_size);
        z_nabla4_e2_wp.resize(output_size);
        for (index_type i{}; i < output_size; ++i) {
            z_nabla4_e2_wp[i].resize(KDim);
        }
    }

    nabla4_data(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type output_size)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim),
          output_size(output_size) {
        if constexpr (T == Data::ifirst) {
            init_ifirst();
        } else if constexpr (T == Data::kfirst) {
            init_kfirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

    /// Constructor for validation
    nabla4_data(index_type CellDim,
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
          output_size(output_size), u_vert(u_vert), v_vert(v_vert), primal_normal_vert_v1(primal_normal_vert_v1),
          primal_normal_vert_v2(primal_normal_vert_v2), z_nabla2_e(z_nabla2_e),
          inv_vert_vert_length(inv_vert_vert_length), inv_primal_edge_length(inv_primal_edge_length) {
        if constexpr (T == Data::ifirst) {
            z_nabla4_e2_wp.resize(KDim);
            for (index_type i{}; i < KDim; ++i) {
                z_nabla4_e2_wp[i].resize(EdgeDim);
            }
        } else if constexpr (T == Data::kfirst) {
            z_nabla4_e2_wp.resize(EdgeDim);
            for (index_type i{}; i < EdgeDim; ++i) {
                z_nabla4_e2_wp[i].resize(KDim);
            }
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

    nabla4_data get_data() { return *this; }
};
