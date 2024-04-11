#pragma once

#include <gridtools/storage/builder.hpp>

using namespace gridtools;

#include <gridtools/storage/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_kfirst.hpp>

#include "common.hpp"

using namespace literals;

template <typename T>
struct nabla4_gt_data {
    std::size_t CellDim;
    std::size_t VertexDim;
    std::size_t EdgeDim;
    std::size_t KDim;
    std::size_t ECVDim;
    std::size_t output_size;

    using data_store_2d_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<VP_TYPE>().build());
    using data_store_2d_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build());
    using data_store_1d_VP_t = decltype(gridtools::storage::builder<T>.dimensions(0).template type<VP_TYPE>().build());
    using data_store_1d_WP_t = decltype(gridtools::storage::builder<T>.dimensions(0).template type<WP_TYPE>().build());
    using data_store_2d_tv_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<VP_TYPE>().build()->target_view());
    using data_store_2d_ctv_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<VP_TYPE>().build()->const_target_view());
    using data_store_2d_ctv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build()->const_target_view());
    using data_store_1d_ctv_VP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0).template type<VP_TYPE>().build()->const_target_view());
    using data_store_1d_ctv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0).template type<WP_TYPE>().build()->const_target_view());

    const data_store_2d_VP_t u_vert_gt;
    const data_store_2d_VP_t v_vert_gt;
    const data_store_1d_WP_t primal_normal_vert_v1_gt;
    const data_store_1d_WP_t primal_normal_vert_v2_gt;
    const data_store_2d_WP_t z_nabla2_e_gt;
    const data_store_1d_WP_t inv_vert_vert_length_gt;
    const data_store_1d_WP_t inv_primal_edge_length_gt;
    const data_store_2d_VP_t z_nabla4_e2_wp_gt;
    const data_store_2d_ctv_VP_t u_vert_gt_tv;
    const data_store_2d_ctv_VP_t v_vert_gt_tv;
    const data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv;
    const data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv;
    const data_store_2d_ctv_WP_t z_nabla2_e_gt_tv;
    const data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv;
    const data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv;
    const data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv;

    nabla4_gt_data(std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, std::size_t output_size)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim), output_size(output_size),
        u_vert_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(VertexDim, KDim).initializer([](int i, int j) { return rand_utils.template get<VP_TYPE>(); }).build()),
        v_vert_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(VertexDim, KDim).initializer([](int i, int j) { return rand_utils.template get<VP_TYPE>(); }).build()),
        primal_normal_vert_v1_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim * ECVDim).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        primal_normal_vert_v2_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim * ECVDim).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        z_nabla2_e_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(output_size, KDim).initializer([](int i, int j) { return rand_utils.template get<WP_TYPE>(); }).build()),
        inv_vert_vert_length_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(output_size).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        inv_primal_edge_length_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(output_size).initializer([](int i) { return rand_utils.template get<WP_TYPE>(); }).build()),
        z_nabla4_e2_wp_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(output_size, KDim).initializer([](int i, int j) { return 0; }).build()),
        u_vert_gt_tv(u_vert_gt->const_target_view()),
        v_vert_gt_tv(v_vert_gt->const_target_view()),
        primal_normal_vert_v1_gt_tv(primal_normal_vert_v1_gt->const_target_view()),
        primal_normal_vert_v2_gt_tv(primal_normal_vert_v2_gt->const_target_view()),
        z_nabla2_e_gt_tv(z_nabla2_e_gt->const_target_view()),
        inv_vert_vert_length_gt_tv(inv_vert_vert_length_gt->const_target_view()),
        inv_primal_edge_length_gt_tv(inv_primal_edge_length_gt->const_target_view()),
        z_nabla4_e2_wp_gt_tv(z_nabla4_e2_wp_gt->target_view())
    {};

    nabla4_gt_data(std::size_t CellDim, std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, std::size_t ECVDim, std::size_t output_size, std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length)
        : CellDim(CellDim), VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim), ECVDim(ECVDim), output_size(output_size),
        u_vert_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(VertexDim, KDim).initializer([&u_vert](int i, int j) { return u_vert[i][j]; }).build()),
        v_vert_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(VertexDim, KDim).initializer([&v_vert](int i, int j) { return v_vert[i][j]; }).build()),
        primal_normal_vert_v1_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim * ECVDim).initializer([&primal_normal_vert_v1](int i) { return primal_normal_vert_v1[i]; }).build()),
        primal_normal_vert_v2_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim * ECVDim).initializer([&primal_normal_vert_v2](int i) { return primal_normal_vert_v2[i]; }).build()),
        z_nabla2_e_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(output_size, KDim).initializer([&z_nabla2_e](int i, int j) { return z_nabla2_e[i][j]; }).build()),
        inv_vert_vert_length_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(output_size).initializer([&inv_vert_vert_length](int i) { return inv_vert_vert_length[i]; }).build()),
        inv_primal_edge_length_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(output_size).initializer([&inv_primal_edge_length](int i) { return inv_primal_edge_length[i]; }).build()),
        z_nabla4_e2_wp_gt(storage::builder<T>.template type<VP_TYPE>().dimensions(output_size, KDim).initializer([](int i, int j) { return 0; }).build()),
        u_vert_gt_tv(u_vert_gt->const_target_view()),
        v_vert_gt_tv(v_vert_gt->const_target_view()),
        primal_normal_vert_v1_gt_tv(primal_normal_vert_v1_gt->const_target_view()),
        primal_normal_vert_v2_gt_tv(primal_normal_vert_v2_gt->const_target_view()),
        z_nabla2_e_gt_tv(z_nabla2_e_gt->const_target_view()),
        inv_vert_vert_length_gt_tv(inv_vert_vert_length_gt->const_target_view()),
        inv_primal_edge_length_gt_tv(inv_primal_edge_length_gt->const_target_view()),
        z_nabla4_e2_wp_gt_tv(z_nabla4_e2_wp_gt->target_view())
    {};

  public:
    std::vector<std::vector<VP_TYPE>> get_output() {
        const auto z_nabla4_e2_wp_gt_chv = z_nabla4_e2_wp_gt->const_host_view();
        std::vector<std::vector<VP_TYPE>> result;
        result.resize(KDim);
        for (std::size_t k_index{}; k_index < KDim; ++k_index) {
            result[k_index].reserve(output_size);
            for (std::size_t edge_index{}; edge_index < output_size; ++edge_index) {
                result[k_index].push_back(z_nabla4_e2_wp_gt_chv(edge_index, k_index));
            }
        }
        return result;
    }
};
