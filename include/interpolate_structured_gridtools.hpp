#pragma once

#include "mo_intp_rbf_rbf_vec_interpol_vertex.hpp"

#if defined(__CUDACC__)
template <typename S>
constexpr block_dims get_block_dims_structured_interpol() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_interpol<std::size_t>() {
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_structured_interpol<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_interpol<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_interpol<int>() {
    return {32, 9, 1, 288};
};

constexpr block_dims block_dims_structured_interpol = get_block_dims_structured_interpol<index_type>();
#endif

template <typename S>
class interpolate_structured : public mo_intp_rbf_rbf_vec_interpol_vertex<S> {
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::KDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::VertexDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::EdgeDim;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::p_e_in_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::ptr_coeff_1_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::ptr_coeff_2_gt_ctv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::p_u_out_gt_tv;
    using mo_intp_rbf_rbf_vec_interpol_vertex<S>::p_v_out_gt_tv;

    const index_type y_dim;
    const index_type x_dim;
    const index_type halo;

  public:
    interpolate_structured(std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : y_dim(y_dim), x_dim(x_dim),
          halo(halo), mo_intp_rbf_rbf_vec_interpol_vertex<S>(
                          VertexDim, EdgeDim, KDim, (y_dim - 2 * halo) * (x_dim - 2 * halo)){};

    interpolate_structured(std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        std::vector<std::vector<WP_TYPE>> &p_e_in,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : y_dim(y_dim), x_dim(x_dim), halo(halo),
          mo_intp_rbf_rbf_vec_interpol_vertex<S>(
              VertexDim, EdgeDim, KDim, (y_dim - 2 * halo) * (x_dim - 2 * halo), p_e_in, ptr_coeff_1, ptr_coeff_2){};

  private:
    const std::array<index_type, 6> get_v2e(const int i, const int j, const index_type x_dim, const index_type y_dim) {
        const index_type i_j = i + j * x_dim;
        const index_type i_jm1 = i + (j - 1) * x_dim;
        return {(x_dim * y_dim) + i_j - 1,
            (x_dim * y_dim) + i_j,
            i_jm1,
            i_j,
            2 * (x_dim * y_dim) + i_j,
            2 * (x_dim * y_dim) + i_j + x_dim - 1};
    };
    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < KDim; ++k_index) {
            for (index_type j = halo; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type i = halo; i < x_dim - halo; ++i) {
                    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
                    std::array<WP_TYPE, 6> u;
                    std::array<WP_TYPE, 6> v;
                    const std::array<index_type, 6> v2e{get_v2e(i, j, x_dim, y_dim)};
#ifdef __clang__
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                    for (auto i{0}; i < 6; ++i) {
                        u[i] = p_e_in_gt_ctv(v2e[i], k_index) * ptr_coeff_1_gt_ctv(vertex_index_internal, i);
                        v[i] = p_e_in_gt_ctv(v2e[i], k_index) * ptr_coeff_2_gt_ctv(vertex_index_internal, i);
                    }
                    p_u_out_gt_tv(vertex_index_internal, k_index) = std::accumulate(u.begin(), u.end(), 0.0);
                    p_v_out_gt_tv(vertex_index_internal, k_index) = std::accumulate(v.begin(), v.end(), 0.0);
                };
            };
        };
    };
    void run_cpu_kfirst() {
        for (index_type j = halo; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type i = halo; i < x_dim - halo; ++i) {
                const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
                std::array<WP_TYPE, 6> u;
                std::array<WP_TYPE, 6> v;
                const std::array<index_type, 6> v2e{get_v2e(i, j, x_dim, y_dim)};
                std::array<WP_TYPE, 6> coeff1;
                std::array<WP_TYPE, 6> coeff2;
#ifdef __clang__
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                for (auto i{0}; i < 6; ++i) {
                    coeff1[i] = ptr_coeff_1_gt_ctv(vertex_index_internal, i);
                    coeff2[i] = ptr_coeff_2_gt_ctv(vertex_index_internal, i);
                }
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type k_index{}; k_index < KDim; ++k_index) {
#ifdef __clang__
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                    for (auto i{0}; i < 6; ++i) {
                        u[i] = p_e_in_gt_ctv(v2e[i], k_index) * coeff1[i];
                        v[i] = p_e_in_gt_ctv(v2e[i], k_index) * coeff2[i];
                    }
                    p_u_out_gt_tv(vertex_index_internal, k_index) = std::accumulate(u.begin(), u.end(), 0.0);
                    p_v_out_gt_tv(vertex_index_internal, k_index) = std::accumulate(v.begin(), v.end(), 0.0);
                };
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
__global__ void __launch_bounds__(block_dims_unstructured_interpol.size) run_gpu_interpol(index_type KDim,
    index_type x_dim,
    index_type y_dim,
    index_type halo,
    index_type inner_grid_size,
    interpolate_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_e_in_gt_ctv,
    interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
    interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
    interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
    interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;
    const auto k_index = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim)
        return;
    const std::array<index_type, 6> v2e{get_v2e(i, j, x_dim, y_dim)};
    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
#pragma unroll
    for (int i{0}; i < 6; ++i) {
        p_u_out_gt_tv(vertex_index_internal, k_index) +=
            p_e_in_gt_ctv(v2e_gt_ctv(vertex_index_internal, i), k_index) * ptr_coeff_1_gt_ctv(vertex_index_internal, i);
        p_v_out_gt_tv(vertex_index_internal, k_index) +=
            p_e_in_gt_ctv(v2e_gt_ctv(vertex_index_internal, i), k_index) * ptr_coeff_2_gt_ctv(vertex_index_internal, i);
    }
};

template <typename S>
inline void interpolate_structured<S>::run_gpu_helper() {
    dim3 tblocks(block_dims_structured_interpol.x, block_dims_structured_interpol.y, block_dims_structured_interpol.z);
    const index_type inner_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    dim3 grid((x_dim - 2 * halo + tblocks.x - 1) / tblocks.x,
        (y_dim - 2 * halo + tblocks.y - 1) / tblocks.y,
        (KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_interpol<<<grid, tblocks>>>(KDim,
        x_dim,
        y_dim,
        halo,
        inner_grid_size,
        p_e_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_u_out_gt_tv,
        p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename S>
inline void interpolate_structured<S>::run_gpu_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
