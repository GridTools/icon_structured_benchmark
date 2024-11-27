#pragma once

#include "verts2cells_vector_gridtools.hpp"

template <typename S>
class verts2cells_unstructured : public verts2cells_vector<S> {
  public:
    using verts2cells_vector<S>::KDim;
    using verts2cells_vector<S>::VertexDim;
    using verts2cells_vector<S>::CellDim;
    using verts2cells_vector<S>::output_size;
    using verts2cells_vector<S>::p_vert_u_in_gt_ctv;
    using verts2cells_vector<S>::p_vert_v_in_gt_ctv;
    using verts2cells_vector<S>::ptr_coeff_1_gt_ctv;
    using verts2cells_vector<S>::ptr_coeff_2_gt_ctv;
    using verts2cells_vector<S>::p_cell_out_gt_tv;
    using input_type = typename verts2cells_vector<S>::data_store_2d_WP_t;

    using neighbors_gt_t =
        decltype(gridtools::storage::builder<S>.dimensions(0, 3_c).template type<index_type>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<S>.dimensions(0, 3_c).template type<index_type>().build()->const_target_view());
    neighbors_gt_t c2v_gt;
    neighbors_gt_ctv_t c2v_gt_ctv;

    verts2cells_unstructured(const std::vector<std::array<index_type, 3>>& c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim)
        : verts2cells_vector<S>(VertexDim, CellDim, KDim, c2v.size()),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

    verts2cells_unstructured(const std::vector<std::array<index_type, 3>>& c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim, input_type p_vert_u_in_gt, input_type p_vert_v_in_gt)
        : verts2cells_vector<S>(VertexDim, CellDim, KDim, p_vert_u_in_gt, p_vert_v_in_gt, c2v.size()),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

    verts2cells_unstructured(const std::vector<std::array<index_type, 3>>& c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim,
        const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
        const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) : verts2cells_vector<S>(VertexDim, CellDim, KDim, c2v.size(), p_vert_u_in, p_vert_v_in, ptr_coeff_1, ptr_coeff_2),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

    verts2cells_unstructured(const std::vector<std::array<index_type, 3>>& c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim,
        const input_type &p_vert_u_in,
        const input_type &p_vert_v_in,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) : verts2cells_vector<S>(VertexDim, CellDim, KDim, c2v.size(), p_vert_u_in, p_vert_v_in, ptr_coeff_1, ptr_coeff_2),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

  private:
    void run_cpu_kfirst() {
        for (index_type cell_index = 0; cell_index < c2v_gt_ctv.lengths()[0]; ++cell_index) {
            const std::array<WP_TYPE, 3> coeff_1{ptr_coeff_1_gt_ctv(cell_index, 0),
                ptr_coeff_1_gt_ctv(cell_index, 1),
                ptr_coeff_1_gt_ctv(cell_index, 2)};
            const std::array<WP_TYPE, 3> coeff_2{ptr_coeff_2_gt_ctv(cell_index, 0),
                ptr_coeff_2_gt_ctv(cell_index, 1),
                ptr_coeff_2_gt_ctv(cell_index, 2)};
            const std::array<index_type, 3> c2v{
                c2v_gt_ctv(cell_index, 0), c2v_gt_ctv(cell_index, 1), c2v_gt_ctv(cell_index, 2)};
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type k_index{}; k_index < KDim; ++k_index) {
                p_cell_out_gt_tv(cell_index, k_index) = ((p_vert_u_in_gt_ctv(c2v[0], k_index) * coeff_1[0] +
                                                             p_vert_u_in_gt_ctv(c2v[1], k_index) * coeff_1[1] +
                                                             p_vert_u_in_gt_ctv(c2v[2], k_index) * coeff_1[2]) +
                                                            (p_vert_v_in_gt_ctv(c2v[0], k_index) * coeff_2[0] +
                                                                p_vert_v_in_gt_ctv(c2v[1], k_index) * coeff_2[1] +
                                                                p_vert_v_in_gt_ctv(c2v[2], k_index) * coeff_2[2])) /
                                                        2;
            };
        };
    };

    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_kfirst) {
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

template <typename S>
constexpr block_dims get_block_dims_unstructured_verts2cells_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_kloop<std::size_t>() {
    // not optimized
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_kloop<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_unstructured_verts2cells_kloop =
    get_block_dims_unstructured_verts2cells_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_unstructured_verts2cells_kloop.size)
    run_gpu_kloop_verts2cells_unstructured(index_type CellDim,
        index_type KDim,
        verts2cells_unstructured<storage::gpu>::neighbors_gt_ctv_t c2v_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_u_in_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_v_in_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell_index >= CellDim)
        return;
    const index_type c2v[3]{c2v_gt_ctv(cell_index, 0), c2v_gt_ctv(cell_index, 1), c2v_gt_ctv(cell_index, 2)};
    const WP_TYPE coeff_1[3]{
        ptr_coeff_1_gt_ctv(cell_index, 0), ptr_coeff_1_gt_ctv(cell_index, 1), ptr_coeff_1_gt_ctv(cell_index, 2)};
    const WP_TYPE coeff_2[3]{
        ptr_coeff_2_gt_ctv(cell_index, 0), ptr_coeff_2_gt_ctv(cell_index, 1), ptr_coeff_2_gt_ctv(cell_index, 2)};
    for (auto k_index{blockIdx.y * blockDim.y + threadIdx.y}; k_index < KDim; k_index += gridDim.y * blockDim.y) {
        p_cell_out_gt_tv(cell_index, k_index) =
            ((p_vert_u_in_gt_ctv(c2v[0], k_index) * coeff_1[0] + p_vert_u_in_gt_ctv(c2v[1], k_index) * coeff_1[1] +
                 p_vert_u_in_gt_ctv(c2v[2], k_index) * coeff_1[2]) +
                (p_vert_v_in_gt_ctv(c2v[0], k_index) * coeff_2[0] + p_vert_v_in_gt_ctv(c2v[1], k_index) * coeff_2[1] +
                    p_vert_v_in_gt_ctv(c2v[2], k_index) * coeff_2[2])) /
            2;
    }
};

template <typename S>
inline void verts2cells_unstructured<S>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_unstructured_verts2cells_kloop.x,
        block_dims_unstructured_verts2cells_kloop.y,
        block_dims_unstructured_verts2cells_kloop.z);
    dim3 grid((output_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu_kloop_verts2cells_unstructured<<<grid, tblocks>>>(output_size,
        KDim,
        c2v_gt_ctv,
        p_vert_u_in_gt_ctv,
        p_vert_v_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_cell_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

template <typename S>
constexpr block_dims get_block_dims_unstructured_verts2cells_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_naive<std::size_t>() {
    // not optimized
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_verts2cells_naive<int>() {
    return {32, 8, 1, 256};
};

constexpr block_dims block_dims_unstructured_verts2cells_naive =
    get_block_dims_unstructured_verts2cells_naive<index_type>();

__global__ void __launch_bounds__(block_dims_unstructured_verts2cells_naive.size)
    run_gpu_naive_verts2cells_unstructured(index_type CellDim,
        index_type KDim,
        verts2cells_unstructured<storage::gpu>::neighbors_gt_ctv_t c2v_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_u_in_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_v_in_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto cell_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (cell_index >= CellDim || k_index >= KDim)
        return;
    const index_type c2v[3]{c2v_gt_ctv(cell_index, 0), c2v_gt_ctv(cell_index, 1), c2v_gt_ctv(cell_index, 2)};
    p_cell_out_gt_tv(cell_index, k_index) =
        ((p_vert_u_in_gt_ctv(c2v[0], k_index) * ptr_coeff_1_gt_ctv(cell_index, 0) +
             p_vert_u_in_gt_ctv(c2v[1], k_index) * ptr_coeff_1_gt_ctv(cell_index, 1) +
             p_vert_u_in_gt_ctv(c2v[2], k_index) * ptr_coeff_1_gt_ctv(cell_index, 2)) +
            (p_vert_v_in_gt_ctv(c2v[0], k_index) * ptr_coeff_2_gt_ctv(cell_index, 0) +
                p_vert_v_in_gt_ctv(c2v[1], k_index) * ptr_coeff_2_gt_ctv(cell_index, 1) +
                p_vert_v_in_gt_ctv(c2v[2], k_index) * ptr_coeff_2_gt_ctv(cell_index, 2))) /
        2;
};

template <typename S>
inline void verts2cells_unstructured<S>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_unstructured_verts2cells_naive.x,
        block_dims_unstructured_verts2cells_naive.y,
        block_dims_unstructured_verts2cells_naive.z);
    dim3 grid((output_size + tblocks.x - 1) / tblocks.x, (KDim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_naive_verts2cells_unstructured<<<grid, tblocks>>>(output_size,
        KDim,
        c2v_gt_ctv,
        p_vert_u_in_gt_ctv,
        p_vert_v_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_cell_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

#else
template <typename S>
inline void verts2cells_unstructured<S>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};

template <typename S>
inline void verts2cells_unstructured<S>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
