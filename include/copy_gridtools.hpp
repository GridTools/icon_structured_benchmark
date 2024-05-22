#pragma once
#include <array>
#include <iostream>
#include <vector>

#include "nabla4_gridtools.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_copy() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_copy<std::size_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_copy<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_copy<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_copy<int>() {
    return {32, 9, 1, 288};
};

constexpr block_dims block_dims_copy = get_block_dims_copy<index_type>();
#endif

template <typename T>
class copy_kernel : public nabla4_gt_data<T> {

    using nabla4_gt_data<T>::EdgeDim;
    using nabla4_gt_data<T>::KDim;
    using nabla4_gt_data<T>::z_nabla2_e_gt_tv;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt_tv;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    copy_kernel(index_type EdgeDim, index_type KDim) : nabla4_gt_data<T>(0, 0, EdgeDim, KDim, 0, EdgeDim){};

    inline __attribute__((always_inline)) void inner_kernel(index_type edge_index, index_type k_index) {
        z_nabla4_e2_wp_gt_tv(edge_index, k_index) = z_nabla2_e_gt_tv(edge_index, k_index);
    };

  private:
    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < KDim; ++k_index) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type edge_index = 0; edge_index < EdgeDim; ++edge_index) {
                inner_kernel(edge_index, k_index);
            };
        };
    };

    void run_cpu_kfirst() {
        for (index_type edge_index{}; edge_index < EdgeDim; ++edge_index) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type k_index = 0; k_index < KDim; ++k_index) {
                inner_kernel(edge_index, k_index);
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
__global__ void __launch_bounds__(block_dims_copy.size) run_gpu(index_type EdgeDim,
    index_type KDim,
    copy_kernel<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
    copy_kernel<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    const auto edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (edge_index >= EdgeDim || k_index >= KDim) {
        return;
    }
    z_nabla4_e2_wp_gt_tv(edge_index, k_index) = z_nabla2_e_gt_tv(edge_index, k_index);
};

template <typename T>
inline void copy_kernel<T>::run_gpu_helper() {
    // const auto u_vert_gt_chv = u_vert_gt->const_host_view();
    // std::cout << "u_vert_gt_chv strides[0] " << u_vert_gt_chv.strides()[0] << " strides[1] " <<
    // u_vert_gt_chv.strides()[1] << std::endl; std::cout << "u_vert_gt_chv lengths[0] " << u_vert_gt_chv.lengths()[0]
    // << " lengths[1] " << u_vert_gt_chv.lengths()[1] << std::endl; const auto e2c2v_gt_chv =
    // e2c2v_gt->const_host_view(); std::cout << "e2c2v_gt_chv strides[0] " << e2c2v_gt_chv.strides()[0] << " strides[1]
    // " << e2c2v_gt_chv.strides()[1] << std::endl; std::cout << "e2c2v_gt_chv lengths[0] " << e2c2v_gt_chv.lengths()[0]
    // << " lengths[1] " << e2c2v_gt_chv.lengths()[1] << std::endl;
    dim3 tblocks(block_dims_copy.x, block_dims_copy.y, block_dims_copy.z);
    dim3 grid((EdgeDim + tblocks.x - 1) / tblocks.x, (KDim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu<<<grid, tblocks>>>(EdgeDim, KDim, z_nabla2_e_gt_tv, z_nabla4_e2_wp_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void copy_kernel<T>::run_gpu_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
