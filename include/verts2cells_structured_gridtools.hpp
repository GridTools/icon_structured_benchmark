#pragma once

#include "verts2cells_vector_gridtools.hpp"

GT_FORCE_INLINE constexpr const std::array<index_type, 4> get_c2v_compressed(
    const index_type i, const index_type j, const index_type x_dim) {
    std::array<index_type, 4> c2v{};
    const index_type i_j = i + j * x_dim;
    const index_type ip1_j = i + 1 + j * x_dim;
    const index_type i_jp1 = i + (j + 1) * x_dim;
    const index_type ip1_jp1 = i + 1 + (j + 1) * x_dim;
    c2v[0] = i_j;
    c2v[1] = ip1_j;
    c2v[2] = i_jp1;
    c2v[3] = ip1_jp1;
    return c2v;
}

template <typename S>
class verts2cells_structured : public verts2cells_vector<S> {
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

    const index_type y_dim;
    const index_type x_dim;
    const index_type halo;

    verts2cells_structured(std::size_t VertexDim,
        std::size_t CellDim,
        std::size_t KDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : y_dim(y_dim), x_dim(x_dim),
          halo(halo), verts2cells_vector<S>(VertexDim, CellDim, KDim, 2 * (y_dim - halo) * (x_dim - halo)){};

    verts2cells_structured(std::size_t VertexDim,
        std::size_t CellDim,
        std::size_t KDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        input_type p_vert_u_in_gt,
        input_type p_vert_v_in_gt)
        : y_dim(y_dim), x_dim(x_dim), halo(halo),
          verts2cells_vector<S>(
              VertexDim, CellDim, KDim, p_vert_u_in_gt, p_vert_v_in_gt, 2 * (y_dim - halo) * (x_dim - halo)){};

    verts2cells_structured(std::size_t VertexDim,
        std::size_t CellDim,
        std::size_t KDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        const std::vector<std::vector<WP_TYPE>> &p_vert_u_in,
        const std::vector<std::vector<WP_TYPE>> &p_vert_v_in,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : y_dim(y_dim), x_dim(x_dim), halo(halo), verts2cells_vector<S>(VertexDim,
                                                      CellDim,
                                                      KDim,
                                                      2 * (y_dim - halo) * (x_dim - halo),
                                                      p_vert_u_in,
                                                      p_vert_v_in,
                                                      ptr_coeff_1,
                                                      ptr_coeff_2){};

    verts2cells_structured(std::size_t VertexDim,
        std::size_t CellDim,
        std::size_t KDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        const input_type &p_vert_u_in,
        const input_type &p_vert_v_in,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : y_dim(y_dim), x_dim(x_dim), halo(halo), verts2cells_vector<S>(VertexDim,
                                                      CellDim,
                                                      KDim,
                                                      2 * (y_dim - halo) * (x_dim - halo),
                                                      p_vert_u_in,
                                                      p_vert_v_in,
                                                      ptr_coeff_1,
                                                      ptr_coeff_2){};

  private:
    void run_cpu_kfirst() {
        for (index_type j = 0; j < y_dim - halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type i = 0; i < x_dim - halo; ++i) {
                const index_type index_internal = i + j * (x_dim - halo);
                const index_type cell_index_internal_upward{2 * index_internal};
                const index_type cell_index_internal_downward{2 * index_internal + 1};
                const std::array<index_type, 4> c2v_compressed{get_c2v_compressed(i, j, x_dim)};
                const std::array<WP_TYPE, 6> coeff_1{ptr_coeff_1_gt_ctv(cell_index_internal_upward, 0),
                    ptr_coeff_1_gt_ctv(cell_index_internal_upward, 1),
                    ptr_coeff_1_gt_ctv(cell_index_internal_upward, 2),
                    ptr_coeff_1_gt_ctv(cell_index_internal_downward, 0),
                    ptr_coeff_1_gt_ctv(cell_index_internal_downward, 1),
                    ptr_coeff_1_gt_ctv(cell_index_internal_downward, 2)};
                const std::array<WP_TYPE, 6> coeff_2{ptr_coeff_2_gt_ctv(cell_index_internal_upward, 0),
                    ptr_coeff_2_gt_ctv(cell_index_internal_upward, 1),
                    ptr_coeff_2_gt_ctv(cell_index_internal_upward, 2),
                    ptr_coeff_2_gt_ctv(cell_index_internal_downward, 0),
                    ptr_coeff_2_gt_ctv(cell_index_internal_downward, 1),
                    ptr_coeff_2_gt_ctv(cell_index_internal_downward, 2)};
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type k_index{}; k_index < KDim; ++k_index) {
                    const WP_TYPE p_vert_u_in_gt_ctv_0{p_vert_u_in_gt_ctv(c2v_compressed[0], k_index)};
                    const WP_TYPE p_vert_u_in_gt_ctv_1{p_vert_u_in_gt_ctv(c2v_compressed[1], k_index)};
                    const WP_TYPE p_vert_u_in_gt_ctv_2{p_vert_u_in_gt_ctv(c2v_compressed[2], k_index)};
                    const WP_TYPE p_vert_u_in_gt_ctv_3{p_vert_u_in_gt_ctv(c2v_compressed[3], k_index)};
                    const WP_TYPE p_vert_v_in_gt_ctv_0{p_vert_v_in_gt_ctv(c2v_compressed[0], k_index)};
                    const WP_TYPE p_vert_v_in_gt_ctv_1{p_vert_v_in_gt_ctv(c2v_compressed[1], k_index)};
                    const WP_TYPE p_vert_v_in_gt_ctv_2{p_vert_v_in_gt_ctv(c2v_compressed[2], k_index)};
                    const WP_TYPE p_vert_v_in_gt_ctv_3{p_vert_v_in_gt_ctv(c2v_compressed[3], k_index)};
                    p_cell_out_gt_tv(cell_index_internal_upward, k_index) =
                        ((p_vert_u_in_gt_ctv_0 * coeff_1[0] + p_vert_u_in_gt_ctv_2 * coeff_1[1] +
                             p_vert_u_in_gt_ctv_1 * coeff_1[2]) +
                            (p_vert_v_in_gt_ctv_0 * coeff_2[0] + p_vert_v_in_gt_ctv_2 * coeff_2[1] +
                                p_vert_v_in_gt_ctv_1 * coeff_2[2])) /
                        2;
                    p_cell_out_gt_tv(cell_index_internal_downward, k_index) =
                        ((p_vert_u_in_gt_ctv_1 * coeff_1[3] + p_vert_u_in_gt_ctv_2 * coeff_1[4] +
                             p_vert_u_in_gt_ctv_3 * coeff_1[5]) +
                            (p_vert_v_in_gt_ctv_1 * coeff_2[3] + p_vert_v_in_gt_ctv_2 * coeff_2[4] +
                                p_vert_v_in_gt_ctv_3 * coeff_2[5])) /
                        2;
                };
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
constexpr block_dims get_block_dims_structured_verts2cells_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_kloop<std::size_t>() {
    // not optimized
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_kloop<int>() {
    return {32, 3, 4, 384};
};

constexpr block_dims block_dims_structured_verts2cells_kloop =
    get_block_dims_structured_verts2cells_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_structured_verts2cells_kloop.size)
    run_gpu_kloop_verts2cells_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        verts2cells_structured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_u_in_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_v_in_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= x_dim - halo || j >= y_dim - halo)
        return;
    const index_type index_internal = i + j * (x_dim - halo);
    const index_type cell_index_internal_upward{2 * index_internal};
    const index_type cell_index_internal_downward{2 * index_internal + 1};
    const std::array<index_type, 4> c2v_compressed{get_c2v_compressed(i, j, x_dim)};
    const std::array<WP_TYPE, 6> coeff_1{ptr_coeff_1_gt_ctv(cell_index_internal_upward, 0),
        ptr_coeff_1_gt_ctv(cell_index_internal_upward, 1),
        ptr_coeff_1_gt_ctv(cell_index_internal_upward, 2),
        ptr_coeff_1_gt_ctv(cell_index_internal_downward, 0),
        ptr_coeff_1_gt_ctv(cell_index_internal_downward, 1),
        ptr_coeff_1_gt_ctv(cell_index_internal_downward, 2)};
    const std::array<WP_TYPE, 6> coeff_2{ptr_coeff_2_gt_ctv(cell_index_internal_upward, 0),
        ptr_coeff_2_gt_ctv(cell_index_internal_upward, 1),
        ptr_coeff_2_gt_ctv(cell_index_internal_upward, 2),
        ptr_coeff_2_gt_ctv(cell_index_internal_downward, 0),
        ptr_coeff_2_gt_ctv(cell_index_internal_downward, 1),
        ptr_coeff_2_gt_ctv(cell_index_internal_downward, 2)};
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
        const WP_TYPE p_vert_u_in_gt_ctv_0{p_vert_u_in_gt_ctv(c2v_compressed[0], k_index)};
        const WP_TYPE p_vert_u_in_gt_ctv_1{p_vert_u_in_gt_ctv(c2v_compressed[1], k_index)};
        const WP_TYPE p_vert_u_in_gt_ctv_2{p_vert_u_in_gt_ctv(c2v_compressed[2], k_index)};
        const WP_TYPE p_vert_u_in_gt_ctv_3{p_vert_u_in_gt_ctv(c2v_compressed[3], k_index)};
        const WP_TYPE p_vert_v_in_gt_ctv_0{p_vert_v_in_gt_ctv(c2v_compressed[0], k_index)};
        const WP_TYPE p_vert_v_in_gt_ctv_1{p_vert_v_in_gt_ctv(c2v_compressed[1], k_index)};
        const WP_TYPE p_vert_v_in_gt_ctv_2{p_vert_v_in_gt_ctv(c2v_compressed[2], k_index)};
        const WP_TYPE p_vert_v_in_gt_ctv_3{p_vert_v_in_gt_ctv(c2v_compressed[3], k_index)};
        p_cell_out_gt_tv(cell_index_internal_upward, k_index) =
            ((p_vert_u_in_gt_ctv_0 * coeff_1[0] + p_vert_u_in_gt_ctv_2 * coeff_1[1] +
                 p_vert_u_in_gt_ctv_1 * coeff_1[2]) +
                (p_vert_v_in_gt_ctv_0 * coeff_2[0] + p_vert_v_in_gt_ctv_2 * coeff_2[1] +
                    p_vert_v_in_gt_ctv_1 * coeff_2[2])) /
            2;
        p_cell_out_gt_tv(cell_index_internal_downward, k_index) =
            ((p_vert_u_in_gt_ctv_1 * coeff_1[3] + p_vert_u_in_gt_ctv_2 * coeff_1[4] +
                 p_vert_u_in_gt_ctv_3 * coeff_1[5]) +
                (p_vert_v_in_gt_ctv_1 * coeff_2[3] + p_vert_v_in_gt_ctv_2 * coeff_2[4] +
                    p_vert_v_in_gt_ctv_3 * coeff_2[5])) /
            2;
    }
};

template <typename S>
inline void verts2cells_structured<S>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_structured_verts2cells_kloop.x,
        block_dims_structured_verts2cells_kloop.y,
        block_dims_structured_verts2cells_kloop.z);
    dim3 grid((x_dim + tblocks.x - 1) / tblocks.x, (y_dim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_kloop_verts2cells_structured<<<grid, tblocks>>>(KDim,
        x_dim,
        y_dim,
        halo,
        p_vert_u_in_gt_ctv,
        p_vert_v_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_cell_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

template <typename S>
constexpr block_dims get_block_dims_structured_verts2cells_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + S::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_naive<std::size_t>() {
    // not optimized
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_verts2cells_naive<int>() {
    return {32, 1, 8, 256};
};

constexpr block_dims block_dims_structured_verts2cells_naive =
    get_block_dims_structured_verts2cells_naive<index_type>();

__global__ void __launch_bounds__(block_dims_structured_verts2cells_naive.size)
    run_gpu_naive_verts2cells_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        verts2cells_structured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_u_in_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_ctv_WP_t p_vert_v_in_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        verts2cells_structured<storage::gpu>::data_store_2d_tv_WP_t p_cell_out_gt_tv) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto j = blockIdx.y * blockDim.y + threadIdx.y;
    const auto k_index = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim)
        return;
    const index_type index_internal = i + j * (x_dim - halo);
    const index_type cell_index_internal_upward{2 * index_internal};
    const index_type cell_index_internal_downward{2 * index_internal + 1};
    const std::array<index_type, 4> c2v_compressed{get_c2v_compressed(i, j, x_dim)};
    const std::array<WP_TYPE, 6> coeff_1{ptr_coeff_1_gt_ctv(cell_index_internal_upward, 0),
        ptr_coeff_1_gt_ctv(cell_index_internal_upward, 1),
        ptr_coeff_1_gt_ctv(cell_index_internal_upward, 2),
        ptr_coeff_1_gt_ctv(cell_index_internal_downward, 0),
        ptr_coeff_1_gt_ctv(cell_index_internal_downward, 1),
        ptr_coeff_1_gt_ctv(cell_index_internal_downward, 2)};
    const std::array<WP_TYPE, 6> coeff_2{ptr_coeff_2_gt_ctv(cell_index_internal_upward, 0),
        ptr_coeff_2_gt_ctv(cell_index_internal_upward, 1),
        ptr_coeff_2_gt_ctv(cell_index_internal_upward, 2),
        ptr_coeff_2_gt_ctv(cell_index_internal_downward, 0),
        ptr_coeff_2_gt_ctv(cell_index_internal_downward, 1),
        ptr_coeff_2_gt_ctv(cell_index_internal_downward, 2)};
    const WP_TYPE p_vert_u_in_gt_ctv_0{p_vert_u_in_gt_ctv(c2v_compressed[0], k_index)};
    const WP_TYPE p_vert_u_in_gt_ctv_1{p_vert_u_in_gt_ctv(c2v_compressed[1], k_index)};
    const WP_TYPE p_vert_u_in_gt_ctv_2{p_vert_u_in_gt_ctv(c2v_compressed[2], k_index)};
    const WP_TYPE p_vert_u_in_gt_ctv_3{p_vert_u_in_gt_ctv(c2v_compressed[3], k_index)};
    const WP_TYPE p_vert_v_in_gt_ctv_0{p_vert_v_in_gt_ctv(c2v_compressed[0], k_index)};
    const WP_TYPE p_vert_v_in_gt_ctv_1{p_vert_v_in_gt_ctv(c2v_compressed[1], k_index)};
    const WP_TYPE p_vert_v_in_gt_ctv_2{p_vert_v_in_gt_ctv(c2v_compressed[2], k_index)};
    const WP_TYPE p_vert_v_in_gt_ctv_3{p_vert_v_in_gt_ctv(c2v_compressed[3], k_index)};
    p_cell_out_gt_tv(cell_index_internal_upward, k_index) =
        ((p_vert_u_in_gt_ctv_0 * coeff_1[0] + p_vert_u_in_gt_ctv_2 * coeff_1[1] + p_vert_u_in_gt_ctv_1 * coeff_1[2]) +
            (p_vert_v_in_gt_ctv_0 * coeff_2[0] + p_vert_v_in_gt_ctv_2 * coeff_2[1] +
                p_vert_v_in_gt_ctv_1 * coeff_2[2])) /
        2;
    p_cell_out_gt_tv(cell_index_internal_downward, k_index) =
        ((p_vert_u_in_gt_ctv_1 * coeff_1[3] + p_vert_u_in_gt_ctv_2 * coeff_1[4] + p_vert_u_in_gt_ctv_3 * coeff_1[5]) +
            (p_vert_v_in_gt_ctv_1 * coeff_2[3] + p_vert_v_in_gt_ctv_2 * coeff_2[4] +
                p_vert_v_in_gt_ctv_3 * coeff_2[5])) /
        2;
};

template <typename S>
inline void verts2cells_structured<S>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_structured_verts2cells_naive.x,
        block_dims_structured_verts2cells_naive.y,
        block_dims_structured_verts2cells_naive.z);
    dim3 grid(
        (x_dim + tblocks.x - 1) / tblocks.x, (y_dim + tblocks.y - 1) / tblocks.y, (KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_verts2cells_structured<<<grid, tblocks>>>(KDim,
        x_dim,
        y_dim,
        halo,
        p_vert_u_in_gt_ctv,
        p_vert_v_in_gt_ctv,
        ptr_coeff_1_gt_ctv,
        ptr_coeff_2_gt_ctv,
        p_cell_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

#else
template <typename S>
inline void verts2cells_structured<S>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};

template <typename S>
inline void verts2cells_structured<S>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
