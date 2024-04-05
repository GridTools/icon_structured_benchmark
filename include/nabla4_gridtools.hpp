#pragma once

#include <gridtools/storage/builder.hpp>

using namespace gridtools;

#include <gridtools/storage/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_kfirst.hpp>

#include "common.hpp"

template <typename T>
struct nabla4_gt_data {
    std::size_t CellDim;
    std::size_t VertexDim;
    std::size_t EdgeDim;
    std::size_t KDim;
    std::size_t ECVDim;

    using data_store_2d_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<VP_TYPE>().build());
    using data_store_2d_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build());
    using data_store_1d_VP_t = decltype(gridtools::storage::builder<T>.dimensions(0).template type<VP_TYPE>().build());
    using data_store_1d_WP_t = decltype(gridtools::storage::builder<T>.dimensions(0).template type<WP_TYPE>().build());
    using data_store_2d_hv_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<VP_TYPE>().build()->host_view());
    using data_store_2d_hv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build()->host_view());
    using data_store_1d_hv_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0).template type<VP_TYPE>().build()->host_view());
    using data_store_1d_hv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0).template type<WP_TYPE>().build()->host_view());

    const data_store_2d_VP_t u_vert_gt;
    const data_store_2d_VP_t v_vert_gt;
    const data_store_1d_WP_t primal_normal_vert_v1_gt;
    const data_store_1d_WP_t primal_normal_vert_v2_gt;
    const data_store_2d_WP_t z_nabla2_e_gt;
    const data_store_1d_WP_t inv_vert_vert_length_gt;
    const data_store_1d_WP_t inv_primal_edge_length_gt;
    const data_store_2d_VP_t z_nabla4_e2_wp_gt;
    const data_store_2d_hv_VP_t u_vert_gt_hv;
    const data_store_2d_hv_VP_t v_vert_gt_hv;
    const data_store_1d_hv_WP_t primal_normal_vert_v1_gt_hv;
    const data_store_1d_hv_WP_t primal_normal_vert_v2_gt_hv;
    const data_store_2d_hv_WP_t z_nabla2_e_gt_hv;
    const data_store_1d_hv_WP_t inv_vert_vert_length_gt_hv;
    const data_store_1d_hv_WP_t inv_primal_edge_length_gt_hv;
    const data_store_2d_hv_VP_t z_nabla4_e2_wp_gt_hv;

    nabla4_gt_data(std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim),
        u_vert_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(KDim, VertexDim).initializer([](int i, int j) { return rand_utils.template get<VP_TYPE>(); }).build()),
        v_vert_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(KDim, VertexDim).initializer([](int i, int j) { return rand_utils.template get<VP_TYPE>(); }).build()),
        primal_normal_vert_v1_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim * ECVDim).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        primal_normal_vert_v2_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim * ECVDim).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        z_nabla2_e_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(KDim, EdgeDim).initializer([](int i, int j) { return rand_utils.template get<WP_TYPE>(); }).build()),
        inv_vert_vert_length_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        inv_primal_edge_length_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        z_nabla4_e2_wp_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(KDim, EdgeDim).initializer([](int i, int j) { return rand_utils.template get<VP_TYPE>(); }).build()),
        u_vert_gt_hv(u_vert_gt->host_view()),
        v_vert_gt_hv(v_vert_gt->host_view()),
        primal_normal_vert_v1_gt_hv(primal_normal_vert_v1_gt->host_view()),
        primal_normal_vert_v2_gt_hv(primal_normal_vert_v2_gt->host_view()),
        z_nabla2_e_gt_hv(z_nabla2_e_gt->host_view()),
        inv_vert_vert_length_gt_hv(inv_vert_vert_length_gt->host_view()),
        inv_primal_edge_length_gt_hv(inv_primal_edge_length_gt->host_view()),
        z_nabla4_e2_wp_gt_hv(z_nabla4_e2_wp_gt->host_view())
    {};
};
