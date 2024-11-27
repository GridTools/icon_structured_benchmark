#pragma once

#include "mo_intp_rbf_rbf_vec_interpol_vertex.hpp"

#if defined(__CUDACC__)
template <typename S>
constexpr block_dims get_block_dims_unstructured_interpol_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_kloop<std::size_t>() {
    // not optimized
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_kloop<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_unstructured_interpol_kloop = get_block_dims_unstructured_interpol_kloop<index_type>();

template <typename S>
constexpr block_dims get_block_dims_unstructured_interpol_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_naive<std::size_t>() {
    // not optimized
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_interpol_naive<int>() {
    return {32, 9, 1, 288};
};

constexpr block_dims block_dims_unstructured_interpol_naive = get_block_dims_unstructured_interpol_naive<index_type>();
#endif

template <typename S>
class interpolate_unstructured : public mo_intp_rbf_rbf_vec_interpol_vertex<S> {
  public:
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::KDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::VertexDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::EdgeDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::output_size;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::p_e_in_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::ptr_coeff_1_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::ptr_coeff_2_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::p_u_out_gt_tv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::p_v_out_gt_tv;
    using input_type = typename mo_intp_rbf_rbf_vec_interpol_vertex<S>::data_store_2d_WP_t;

    using neighbors_gt_t =
        decltype(gridtools::storage::builder<S>.dimensions(0, 6_c).template type<index_type>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<S>.dimensions(0, 6_c).template type<index_type>().build()->const_target_view());
    neighbors_gt_t v2e_gt;
    neighbors_gt_ctv_t v2e_gt_ctv;

    interpolate_unstructured(const std::vector<std::array<index_type, 6>> &v2e,
        std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim)
        : mo_intp_rbf_rbf_vec_interpol_vertex<S>(VertexDim, EdgeDim, KDim, v2e.size()),
        v2e_gt(storage::builder<S>.template type<index_type>().dimensions(v2e.size(), 6_c).initializer([&v2e](int i, int j) { return v2e[i][j]; }).build()),
        v2e_gt_ctv(v2e_gt->const_target_view())
    {};

    interpolate_unstructured(const std::vector<std::array<index_type, 6>> &v2e,
        std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim, input_type p_e_in_gt)
        : mo_intp_rbf_rbf_vec_interpol_vertex<S>(VertexDim, EdgeDim, KDim, p_e_in_gt, v2e.size()),
        v2e_gt(storage::builder<S>.template type<index_type>().dimensions(v2e.size(), 6_c).initializer([&v2e](int i, int j) { return v2e[i][j]; }).build()),
        v2e_gt_ctv(v2e_gt->const_target_view())
    {};

    interpolate_unstructured(const std::vector<std::array<index_type, 6>> &v2e,
        std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim,
        const std::vector<std::vector<WP_TYPE>> &p_e_in,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) : mo_intp_rbf_rbf_vec_interpol_vertex<S>(VertexDim, EdgeDim, KDim, v2e.size(), p_e_in, ptr_coeff_1, ptr_coeff_2),
        v2e_gt(storage::builder<S>.template type<index_type>().dimensions(v2e.size(), 6_c).initializer([&v2e](int i, int j) { return v2e[i][j]; }).build()),
        v2e_gt_ctv(v2e_gt->const_target_view())
    {};

    interpolate_unstructured(const std::vector<std::array<index_type, 6>> &v2e,
        std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim,
        const input_type &p_e_in,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) : mo_intp_rbf_rbf_vec_interpol_vertex<S>(VertexDim, EdgeDim, KDim, v2e.size(), p_e_in, ptr_coeff_1, ptr_coeff_2),
        v2e_gt(storage::builder<S>.template type<index_type>().dimensions(v2e.size(), 6_c).initializer([&v2e](int i, int j) { return v2e[i][j]; }).build()),
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
#pragma clang loop unroll(full)
#elif defined(__NVCC__)
#pragma unroll 6
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                for (auto i{0}; i < 6; ++i) {
                    u[i] = p_e_in_gt_ctv(v2e_gt_ctv(vertex_index, i), k_index) * ptr_coeff_1_gt_ctv(vertex_index, i);
                    v[i] = p_e_in_gt_ctv(v2e_gt_ctv(vertex_index, i), k_index) * ptr_coeff_2_gt_ctv(vertex_index, i);
                }
                p_u_out_gt_tv(vertex_index, k_index) = std::accumulate(u.begin(), u.end(), 0.0);
                p_v_out_gt_tv(vertex_index, k_index) = std::accumulate(v.begin(), v.end(), 0.0);
            };
        };
    };
    void run_cpu_kfirst() {
        for (index_type vertex_index = 0; vertex_index < v2e_gt_ctv.lengths()[0]; ++vertex_index) {
            std::array<WP_TYPE, 6> coeff1;
            std::array<WP_TYPE, 6> coeff2;
            std::array<index_type, 6> v2e;
#ifdef __clang__
#pragma clang loop unroll(full)
#elif defined(__NVCC__)
#pragma unroll 6
#elif defined(__GNUC__)
#pragma GCC unroll 6
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
#pragma clang loop unroll(full)
#elif defined(__NVCC__)
#pragma unroll 6
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                for (auto i{0}; i < 6; ++i) {
                    u[i] = p_e_in_gt_ctv(v2e[i], k_index) * coeff1[i];
                    v[i] = p_e_in_gt_ctv(v2e[i], k_index) * coeff2[i];
                }
                p_u_out_gt_tv(vertex_index, k_index) = std::accumulate(u.begin(), u.end(), 0.0);
                p_v_out_gt_tv(vertex_index, k_index) = std::accumulate(v.begin(), v.end(), 0.0);
            };
        };
    };
    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
__global__ void __launch_bounds__(block_dims_unstructured_interpol_kloop.size)
    run_gpu_kloop_interpol_unstructured(index_type VertexDim,
        index_type KDim,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_e_in_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= VertexDim)
        return;
    std::array<WP_TYPE, 6> coeff1;
    std::array<WP_TYPE, 6> coeff2;
    std::array<index_type, 6> v2e;
    for (int i{0}; i < 6; ++i) {
        coeff1[i] = ptr_coeff_1_gt_ctv(vertex_index, i);
        coeff2[i] = ptr_coeff_2_gt_ctv(vertex_index, i);
        v2e[i] = v2e_gt_ctv(vertex_index, i);
    }
    for (auto k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim; k_index += gridDim.y * blockDim.y) {
        double u{0.0};
        double v{0.0};
#pragma unroll
        for (int i{0}; i < 6; ++i) {
            u += p_e_in_gt_ctv(v2e[i], k_index) * coeff1[i];
            v += p_e_in_gt_ctv(v2e[i], k_index) * coeff2[i];
        }
        p_u_out_gt_tv(vertex_index, k_index) = u;
        p_v_out_gt_tv(vertex_index, k_index) = v;
    }
};

template <typename S>
inline void interpolate_unstructured<S>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_unstructured_interpol_kloop.x,
        block_dims_unstructured_interpol_kloop.y,
        block_dims_unstructured_interpol_kloop.z);
    dim3 grid((output_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu_kloop_interpol_unstructured<<<grid, tblocks>>>(output_size,
        KDim,
        v2e_gt_ctv,
        p_e_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_u_out_gt_tv,
        p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

__global__ void __launch_bounds__(block_dims_unstructured_interpol_naive.size)
    run_gpu_naive_interpol_unstructured(index_type VertexDim,
        index_type KDim,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_e_in_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (vertex_index >= VertexDim || k_index >= KDim)
        return;
    double u{0.0};
    double v{0.0};
#pragma unroll
    for (int i{0}; i < 6; ++i) {
        u += p_e_in_gt_ctv(v2e_gt_ctv(vertex_index, i), k_index) * ptr_coeff_1_gt_ctv(vertex_index, i);
        v += p_e_in_gt_ctv(v2e_gt_ctv(vertex_index, i), k_index) * ptr_coeff_2_gt_ctv(vertex_index, i);
    }
    p_u_out_gt_tv(vertex_index, k_index) = u;
    p_v_out_gt_tv(vertex_index, k_index) = v;
};

template <typename S>
inline void interpolate_unstructured<S>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_unstructured_interpol_naive.x,
        block_dims_unstructured_interpol_naive.y,
        block_dims_unstructured_interpol_naive.z);
    dim3 grid((output_size + tblocks.x - 1) / tblocks.x, (KDim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_naive_interpol_unstructured<<<grid, tblocks>>>(output_size,
        KDim,
        v2e_gt_ctv,
        p_e_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_u_out_gt_tv,
        p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename S>
inline void interpolate_unstructured<S>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};

template <typename S>
inline void interpolate_unstructured<S>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
