#pragma once

#include <utility>
#include <vector>

#include <gridtools/storage/builder.hpp>

using namespace gridtools;

#include <gridtools/storage/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_kfirst.hpp>

#ifdef __CUDACC__
#include <gridtools/storage/gpu.hpp>
#endif

#include "common.hpp"

using namespace literals;

template <typename T>
struct mo_intp_rbf_rbf_vec_interpol_vertex {
    std::size_t VertexDim;
    std::size_t EdgeDim;
    std::size_t KDim;
    const static std::size_t V2EDim{6};

    using data_store_2d_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build());
    using data_store_2d_coef_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 6_c).template type<WP_TYPE>().build());
    using data_store_2d_ctv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build()->const_target_view());
    using data_store_2d_coef_ctv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 6_c).template type<WP_TYPE>().build()->const_target_view());
    using data_store_2d_tv_WP_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 0).template type<WP_TYPE>().build()->target_view());

    const data_store_2d_WP_t p_e_in_gt;
    const data_store_2d_coef_WP_t ptr_coeff_1_gt;
    const data_store_2d_coef_WP_t ptr_coeff_2_gt;
    const data_store_2d_WP_t p_u_out_gt;
    const data_store_2d_WP_t p_v_out_gt;
    const data_store_2d_ctv_WP_t p_e_in_gt_ctv;
    const data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv;
    const data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv;
    const data_store_2d_tv_WP_t p_u_out_gt_ctv;
    const data_store_2d_tv_WP_t p_v_out_gt_ctv;

  public:
    mo_intp_rbf_rbf_vec_interpol_vertex(std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim)
        : VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim),
        p_e_in_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim, KDim).initializer([](int i, int j) { return rand_utils.template get<WP_TYPE>(); }).build()),
        ptr_coeff_1_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, 6_c).initializer([](int i, int j) { return rand_utils.template get<WP_TYPE>(); }).build()),
        ptr_coeff_2_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, 6_c).initializer([](int i, int j) { return rand_utils.template get<WP_TYPE>(); }).build()),
        p_u_out_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, KDim).initializer([](int i, int j) { return 0; }).build()),
        p_v_out_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, KDim).initializer([](int i, int j) { return 0; }).build()),
        p_e_in_gt_ctv(p_e_in_gt->const_target_view()),
        ptr_coeff_1_gt_ctv(ptr_coeff_1_gt->const_target_view()),
        ptr_coeff_2_gt_ctv(ptr_coeff_2_gt->const_target_view()),
        p_u_out_gt_ctv(p_u_out_gt->target_view()),
        p_v_out_gt_ctv(p_v_out_gt->target_view())
    {};

    mo_intp_rbf_rbf_vec_interpol_vertex(std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim,
        std::vector<std::vector<WP_TYPE>> p_e_in,
        std::vector<std::vector<WP_TYPE>> ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> ptr_coeff_2) : VertexDim(VertexDim), EdgeDim(EdgeDim), KDim(KDim),
        p_e_in_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(EdgeDim, KDim).initializer([&p_e_in](int i, int j) { return p_e_in[i][j]; }).build()),
        ptr_coeff_1_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, 6_c).initializer([&ptr_coeff_1](int i, int j) { return ptr_coeff_1[i][j]; }).build()),
        ptr_coeff_2_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, 6_c).initializer([&ptr_coeff_2](int i, int j) { return ptr_coeff_2[i][j]; }).build()),
        p_u_out_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, KDim).initializer([](int i, int j) { return 0; }).build()),
        p_v_out_gt(storage::builder<T>.template type<WP_TYPE>().dimensions(VertexDim, KDim).initializer([](int i, int j) { return 0; }).build()),
        p_e_in_gt_ctv(p_e_in_gt->const_target_view()),
        ptr_coeff_1_gt_ctv(ptr_coeff_1_gt->const_target_view()),
        ptr_coeff_2_gt_ctv(ptr_coeff_2_gt->const_target_view()),
        p_u_out_gt_ctv(p_u_out_gt->target_view()),
        p_v_out_gt_ctv(p_v_out_gt->target_view())
    {};

    std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>> get_output() {
        auto p_u_out = std::vector<std::vector<WP_TYPE>>(VertexDim, std::vector<WP_TYPE>(KDim));
        auto p_v_out = std::vector<std::vector<WP_TYPE>>(VertexDim, std::vector<WP_TYPE>(KDim));
        for (int i = 0; i < VertexDim; i++) {
            for (int j = 0; j < KDim; j++) {
                p_u_out[i][j] = p_u_out_gt->const_host_view()(i, j);
                p_v_out[i][j] = p_v_out_gt->const_host_view()(i, j);
            }
        }
        return std::make_pair(p_u_out, p_v_out);
    }
};
