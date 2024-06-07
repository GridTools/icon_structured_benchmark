#pragma once

#include "mo_intp_rbf_rbf_vec_interpol_vertex.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_unstructured() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured<std::size_t>() {
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_unstructured<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured<int>() {
    return {32, 9, 1, 288};
};

constexpr block_dims block_dims_unstructured = get_block_dims_unstructured<index_type>();
#endif

template <typename T>
class interpolate_unstructured : public mo_intp_rbf_rbf_vec_interpol_vertex<T> {
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::KDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::VertexDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::EdgeDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::p_e_in_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::ptr_coeff_1_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::ptr_coeff_2_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::p_u_out_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<T>::p_v_out_gt_ctv;

  public:
    using neighbors_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 6_c).template type<index_type>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 6_c).template type<index_type>().build()->const_target_view());
    neighbors_gt_t v2e_gt;
    neighbors_gt_ctv_t v2e_gt_ctv;

    interpolate_unstructured(std::vector<std::array<index_type, 6>> v2e,
        std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim)
        : mo_intp_rbf_rbf_vec_interpol_vertex<T>(VertexDim, EdgeDim, KDim),
        v2e_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 6_c).initializer([&v2e](int i, int j) { return v2e[i][j]; }).build()),
        v2e_gt_ctv(v2e_gt->const_target_view())
    {};

    interpolate_unstructured(std::vector<std::array<index_type, 6>> v2e,
        std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim,
        std::vector<std::vector<WP_TYPE>> &p_e_in,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) : mo_intp_rbf_rbf_vec_interpol_vertex<T>(VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2),
        v2e_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 6_c).initializer([&v2e](int i, int j) { return v2e[i][j]; }).build()),
        v2e_gt_ctv(v2e_gt->const_target_view())
    {};

  private:
    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < KDim; ++k_index) {
            const auto vertices = v2e_gt_ctv.lengths()[0];
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type vertex_index = 0; vertex_index < vertices; ++vertex_index) {
                std::array<WP_TYPE, 6> u;
                std::array<WP_TYPE, 6> v;
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (auto i{0}; i < 6; ++i) {
                    u[i] = p_e_in_gt_ctv(v2e_gt_ctv(vertex_index, i), k_index) * ptr_coeff_1_gt_ctv(vertex_index, i);
                    v[i] = p_e_in_gt_ctv(v2e_gt_ctv(vertex_index, i), k_index) * ptr_coeff_2_gt_ctv(vertex_index, i);
                }
                p_u_out_gt_ctv(vertex_index, k_index) = std::accumulate(u.begin(), u.end(), 0.0);
                p_v_out_gt_ctv(vertex_index, k_index) = std::accumulate(v.begin(), v.end(), 0.0);
            };
        };
    };
    void run_cpu_kfirst() {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
        for (index_type vertex_index = 0; vertex_index < v2e_gt_ctv.lengths()[0]; ++vertex_index) {
            std::array<WP_TYPE, 6> coeff1;
            std::array<WP_TYPE, 6> coeff2;
            std::array<index_type, 6> v2e;
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (auto i{0}; i < 6; ++i) {
                coeff1[i] = ptr_coeff_1_gt_ctv(vertex_index, i);
                coeff2[i] = ptr_coeff_2_gt_ctv(vertex_index, i);
                v2e[i] = v2e_gt_ctv(vertex_index, i);
            }
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type k_index{}; k_index < KDim; ++k_index) {
                std::array<WP_TYPE, 6> u;
                std::array<WP_TYPE, 6> v;
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (auto i{0}; i < 6; ++i) {
                    u[i] = p_e_in_gt_ctv(v2e[i], k_index) * coeff1[i];
                    v[i] = p_e_in_gt_ctv(v2e[i], k_index) * coeff2[i];
                }
                p_u_out_gt_ctv(vertex_index, k_index) = std::accumulate(u.begin(), u.end(), 0.0);
                p_v_out_gt_ctv(vertex_index, k_index) = std::accumulate(v.begin(), v.end(), 0.0);
            };
        };
    };
    void run_gpu_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else if constexpr (I == backend_impl::gpu) {
            run_gpu_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
__global__ void __launch_bounds__(block_dims_unstructured.size, 2) run_gpu(){};

template <typename T>
inline void interpolate_unstructured<T>::run_gpu_helper() {
    dim3 tblocks(block_dims_unstructured.x, block_dims_unstructured.y, block_dims_unstructured.z);
    dim3 grid((e2c2v_gt->const_host_view().lengths()[0] + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu<<<grid, tblocks>>>();
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void interpolate_unstructured<T>::run_gpu_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
