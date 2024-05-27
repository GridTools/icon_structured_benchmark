#pragma once
#include <array>
#include <iostream>
#include <vector>

#include "nabla4_gridtools.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_copy_neighbor() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_copy_neighbor<std::size_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_copy_neighbor<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_copy_neighbor<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_copy_neighbor<std::int32_t>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_copy_neighbor = get_block_dims_copy_neighbor<index_type>();
#endif

template <typename T>
class copy_neighbor_kernel : public nabla4_gt_data<T> {

  public:
    using neighbors_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<index_type>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<index_type>().build()->const_target_view());

  private:
    using nabla4_gt_data<T>::EdgeDim;
    using nabla4_gt_data<T>::KDim;
    using nabla4_gt_data<T>::z_nabla2_e_gt_tv;
    using nabla4_gt_data<T>::dummy_field_gt_tv;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt_tv;

    neighbors_gt_t e2c2v_gt;
    neighbors_gt_ctv_t e2c2v_gt_tv;

  public:
    /// Constructor with all the necessary information for \c nabla4 compute
    /// kernel execution
    copy_neighbor_kernel(std::vector<std::array<index_type, 4>> e2c2v, index_type EdgeDim, index_type KDim) : e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(e2c2v.size(), 4_c).initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; }).build()), e2c2v_gt_tv(e2c2v_gt->const_target_view()), nabla4_gt_data<T>(0, 0, EdgeDim, KDim, 0, EdgeDim){};

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
    copy_neighbor_kernel<storage::gpu>::neighbors_gt_ctv_t e2c2v_gt_tv,
    copy_neighbor_kernel<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
    copy_neighbor_kernel<storage::gpu>::data_store_2d_ctv_WP_t dummy_field_gt_tv,
    copy_neighbor_kernel<storage::gpu>::data_store_2d_tv_VP_t z_nabla4_e2_wp_gt_tv) {
    const auto edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (edge_index >= EdgeDim || k_index >= KDim) {
        return;
    }
    // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //     printf("e2c2v_gt_tv strides[0]: %d strides[1]: %d\n", e2c2v_gt_tv.strides()[0], e2c2v_gt_tv.strides()[1]);
    //     printf("z_nabla2_e_gt_tv strides[0]: %d strides[1]: %d\n", z_nabla2_e_gt_tv.strides()[0],
    //     z_nabla2_e_gt_tv.strides()[1]); printf("z_nabla4_e2_wp_gt_tv strides[0]: %d strides[1]: %d\n",
    //     z_nabla4_e2_wp_gt_tv.strides()[0], z_nabla4_e2_wp_gt_tv.strides()[1]);
    // }
    // printf("threadIdx.x %d threadIdx.y %d blockIdx.x %d blockIdx.y %d edge_index %d k_index %d
    // z_nabla4_e2_wp_gt_tv(edge_index, k_index): %p, z_nabla2_e_gt_tv(e2c2v_gt_tv(edge_index, 0), k_index): %p\n",
    // threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, edge_index, k_index, &(z_nabla4_e2_wp_gt_tv(edge_index,
    // k_index)), &(z_nabla2_e_gt_tv(e2c2v_gt_tv(edge_index, 0), k_index)));
    double E2C2V_0 = z_nabla2_e_gt_tv(e2c2v_gt_tv(edge_index, 0), k_index);
    double E2C2V_1 = z_nabla2_e_gt_tv(e2c2v_gt_tv(edge_index, 1), k_index);
    double E2C2V_2 = z_nabla2_e_gt_tv(e2c2v_gt_tv(edge_index, 2), k_index);
    double E2C2V_3 = z_nabla2_e_gt_tv(e2c2v_gt_tv(edge_index, 3), k_index);
    double E2C2V_0_p42 = (E2C2V_0 + 42.0) * k_index;
    double E2C2V_1_m42 = (E2C2V_1 - 42.0) * k_index;
    double E2C2V_2_p42 = (E2C2V_2 + 42.0) * k_index;
    double E2C2V_3_m42 = (E2C2V_3 - 42.0) * k_index;
    double dummy_0 = dummy_field_gt_tv(e2c2v_gt_tv(edge_index, 0), k_index);
    double dummy_1 = dummy_field_gt_tv(e2c2v_gt_tv(edge_index, 1), k_index);
    double dummy_2 = dummy_field_gt_tv(e2c2v_gt_tv(edge_index, 2), k_index);
    double dummy_3 = dummy_field_gt_tv(e2c2v_gt_tv(edge_index, 3), k_index);
    z_nabla4_e2_wp_gt_tv(edge_index, k_index) =
        ((E2C2V_0_p42 + E2C2V_1_m42 + E2C2V_2_p42 + E2C2V_3_m42) / (4.0 * k_index) + dummy_0 + dummy_1 + dummy_2 +
            dummy_3) /
        5.0;
};

template <typename T>
inline void copy_neighbor_kernel<T>::run_gpu_helper() {
    // const auto u_vert_gt_chv = u_vert_gt->const_host_view();
    // std::cout << "u_vert_gt_chv strides[0] " << u_vert_gt_chv.strides()[0] << " strides[1] " <<
    // u_vert_gt_chv.strides()[1] << std::endl; std::cout << "u_vert_gt_chv lengths[0] " << u_vert_gt_chv.lengths()[0]
    // << " lengths[1] " << u_vert_gt_chv.lengths()[1] << std::endl;
    // const auto e2c2v_gt_chv = e2c2v_gt->const_target_view();
    // std::cout << "e2c2v_gt_chv strides[0] " << e2c2v_gt_chv.strides()[0] << " strides[1] " <<
    // e2c2v_gt_chv.strides()[1] << std::endl; std::cout << "e2c2v_gt_chv lengths[0] " << e2c2v_gt_chv.lengths()[0] << "
    // lengths[1] " << e2c2v_gt_chv.lengths()[1] << std::endl;
    dim3 tblocks(block_dims_copy_neighbor.x, block_dims_copy_neighbor.y, block_dims_copy_neighbor.z);
    dim3 grid((EdgeDim + tblocks.x - 1) / tblocks.x, (KDim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu<<<grid, tblocks>>>(EdgeDim, KDim, e2c2v_gt_tv, z_nabla2_e_gt_tv, dummy_field_gt_tv, z_nabla4_e2_wp_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void copy_neighbor_kernel<T>::run_gpu_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
