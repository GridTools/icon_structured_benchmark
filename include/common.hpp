#pragma once

#include <gridtools/storage/builder.hpp>

using namespace gridtools;

#ifdef GT_CUDACC
#include <gridtools/storage/gpu.hpp>
using traits_t = storage::gpu;
#else
#include <gridtools/storage/cpu_ifirst.hpp>
using traits_t = storage::cpu_ifirst;
#endif

#include "random_init.hpp"

#define ARRAY_TYPE std::size_t

#define VP_TYPE double
#define WP_TYPE double

enum backend_impl { naive = 0, cpu_ifirst, cpu_kfirst, gpu };

enum class Data { ifirst, kfirst };

/// Random number utilities
static RandomUniformUtils rand_utils{-1.0, 1.0};

template <Data T>
struct nabla4_data {
    std::size_t CellDim;
    std::size_t VertexDim;
    std::size_t EdgeDim;
    std::size_t KDim;
    std::size_t ECVDim;

    using data_store_2d_t =
        decltype(gridtools::storage::builder<traits_t>.dimensions(0, 0).template type<double>().build());
    using data_store_1d_t =
        decltype(gridtools::storage::builder<traits_t>.dimensions(0).template type<double>().build());
    using data_store_2d_tv_t =
        decltype(gridtools::storage::builder<traits_t>.dimensions(0, 0).template type<double>().build()->target_view());
    using data_store_1d_tv_t =
        decltype(gridtools::storage::builder<traits_t>.dimensions(0).template type<double>().build()->target_view());
    // using data_store_2d_t =
    //     std::__1::shared_ptr<gridtools::storage::data_store_impl_::data_store<gridtools::storage::cpu_ifirst,
    //         double,
    //         gridtools::storage::info_impl_::info<gridtools::tuple<int, int>,
    //             gridtools::tuple<gridtools::integral_constant<int, 1>, int>,
    //             std::__1::integer_sequence<unsigned long, 0ul, 1ul>>,
    //         gridtools::meta::list<gridtools::tuple<gridtools::integral_constant<int, 1>, int> const &,
    //             gridtools::layout_map_impl::layout_map<1, 0>,
    //             void,
    //             gridtools::integral_constant<int, 8>>,
    //         false,
    //         true>>;
    // using data_store_1d_t =
    //     std::shared_ptr<gridtools::storage::data_store_impl_::data_store<gridtools::storage::cpu_ifirst,
    //         double,
    //         gridtools::storage::info_impl_::info<gridtools::tuple<int>,
    //             gridtools::tuple<gridtools::integral_constant<int, 1>>,
    //             std::integer_sequence<unsigned long, 0ul>>,
    //         gridtools::tuple<gridtools::integral_constant<int, 1>> const &,
    //         false,
    //         true>>;

    data_store_2d_t u_vert_gt;
    data_store_2d_t v_vert_gt;
    data_store_1d_t primal_normal_vert_v1_gt;
    data_store_1d_t primal_normal_vert_v2_gt;
    data_store_2d_t z_nabla2_e_gt;
    data_store_1d_t inv_vert_vert_length_gt;
    data_store_1d_t inv_primal_edge_length_gt;
    data_store_2d_t z_nabla4_e2_wp_gt;

    data_store_2d_tv_t u_vert_gt_tv;
    data_store_2d_tv_t v_vert_gt_tv;
    data_store_1d_tv_t primal_normal_vert_v1_gt_tv;
    data_store_1d_tv_t primal_normal_vert_v2_gt_tv;
    data_store_2d_tv_t z_nabla2_e_gt_tv;
    data_store_1d_tv_t inv_vert_vert_length_gt_tv;
    data_store_1d_tv_t inv_primal_edge_length_gt_tv;
    data_store_2d_tv_t z_nabla4_e2_wp_gt_tv;

    std::vector<std::vector<VP_TYPE>> u_vert;
    std::vector<std::vector<VP_TYPE>> v_vert;
    std::vector<WP_TYPE> primal_normal_vert_v1;
    std::vector<WP_TYPE> primal_normal_vert_v2;
    std::vector<std::vector<WP_TYPE>> z_nabla2_e;
    std::vector<WP_TYPE> inv_vert_vert_length;
    std::vector<WP_TYPE> inv_primal_edge_length;
    std::vector<std::vector<WP_TYPE>> z_nabla4_e2_wp;

    /// Initialize vectors needed to execute kernel with random numbers
    void init_ifirst() {
        const auto storage_builder = storage::builder<traits_t>.template type<double>();
        u_vert_gt = storage_builder.dimensions(KDim, VertexDim).build();
        v_vert_gt = storage_builder.dimensions(KDim, VertexDim).build();
        // std::cout << "v_vert_gt " << debug::print_type<decltype(v_vert_gt)>("") << std::endl;
        primal_normal_vert_v1_gt = storage_builder.dimensions(EdgeDim * ECVDim).build();
        // std::cout << "primal_normal_vert_v1_gt " << debug::print_type<decltype(primal_normal_vert_v1_gt)>("") <<
        // std::endl;
        primal_normal_vert_v2_gt = storage_builder.dimensions(EdgeDim * ECVDim).build();
        z_nabla2_e_gt = storage_builder.dimensions(KDim, EdgeDim).build();
        inv_vert_vert_length_gt = storage_builder.dimensions(EdgeDim).build();
        inv_primal_edge_length_gt = storage_builder.dimensions(EdgeDim).build();
        z_nabla4_e2_wp_gt = storage_builder.dimensions(KDim, EdgeDim).build();
        u_vert_gt_tv = u_vert_gt->target_view();
        v_vert_gt_tv = v_vert_gt->target_view();
        primal_normal_vert_v1_gt_tv = primal_normal_vert_v1_gt->target_view();
        primal_normal_vert_v2_gt_tv = primal_normal_vert_v2_gt->target_view();
        z_nabla2_e_gt_tv = z_nabla2_e_gt->target_view();
        inv_vert_vert_length_gt_tv = inv_vert_vert_length_gt->target_view();
        inv_primal_edge_length_gt_tv = inv_primal_edge_length_gt->target_view();
        z_nabla4_e2_wp_gt_tv = z_nabla4_e2_wp_gt->target_view();
        // Old vectors
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

    nabla4_data(std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim) {
        if constexpr (T == Data::ifirst) {
            init_ifirst();
        } else if constexpr (T == Data::kfirst) {
            init_kfirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    };

    /// Constructor for validation
    nabla4_data(std::size_t CellDim,
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
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim), u_vert(u_vert),
          v_vert(v_vert), primal_normal_vert_v1(primal_normal_vert_v1), primal_normal_vert_v2(primal_normal_vert_v2),
          z_nabla2_e(z_nabla2_e), inv_vert_vert_length(inv_vert_vert_length),
          inv_primal_edge_length(inv_primal_edge_length) {
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

    nabla4_data get_data() { return *this; }
};
