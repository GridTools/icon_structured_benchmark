#pragma once

#include "verts2cells_scalar_gridtools.hpp"

template <typename S>
class verts2cells_unstructured : public verts2cells_scalar<S> {
  public:
    using verts2cells_scalar<S>::KDim;
    using verts2cells_scalar<S>::VertexDim;
    using verts2cells_scalar<S>::CellDim;
    using verts2cells_scalar<S>::output_size;
    using verts2cells_scalar<S>::p_vert_in_gt_ctv;
    using verts2cells_scalar<S>::ptr_coeff_gt_ctv;
    using verts2cells_scalar<S>::p_cell_out_gt_tv;
    using input_type = typename verts2cells_scalar<S>::data_store_2d_WP_t;

    using neighbors_gt_t =
        decltype(gridtools::storage::builder<S>.dimensions(0, 3_c).template type<index_type>().build());
    using neighbors_gt_ctv_t =
        decltype(gridtools::storage::builder<S>.dimensions(0, 3_c).template type<index_type>().build()->const_target_view());
    neighbors_gt_t c2v_gt;
    neighbors_gt_ctv_t c2v_gt_ctv;

    verts2cells_unstructured(std::vector<std::array<index_type, 3>> c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim)
        : verts2cells_scalar<S>(VertexDim, CellDim, KDim, c2v.size()),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

    verts2cells_unstructured(std::vector<std::array<index_type, 3>> c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim, input_type p_vert_in_gt)
        : verts2cells_scalar<S>(VertexDim, CellDim, KDim, p_vert_in_gt, c2v.size()),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

    verts2cells_unstructured(std::vector<std::array<index_type, 3>> c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim,
        std::vector<std::vector<WP_TYPE>> &p_vert_in,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff) : verts2cells_scalar<S>(VertexDim, CellDim, KDim, c2v.size(), p_vert_in, ptr_coeff),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

    verts2cells_unstructured(std::vector<std::array<index_type, 3>> c2v,
        std::size_t VertexDim, std::size_t CellDim, std::size_t KDim,
        const input_type &p_vert_in,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff) : verts2cells_scalar<S>(VertexDim, CellDim, KDim, c2v.size(), p_vert_in, ptr_coeff),
        c2v_gt(storage::builder<S>.template type<index_type>().dimensions(c2v.size(), 3_c).initializer([&c2v](int i, int j) { return c2v[i][j]; }).build()),
        c2v_gt_ctv(c2v_gt->const_target_view())
    {};

  private:

    void run_cpu_kfirst() {
        for (index_type cell_index = 0; cell_index < c2v_gt_ctv.lengths()[0]; ++cell_index) {
            const std::array<WP_TYPE, 3> coeff{ptr_coeff_gt_ctv(cell_index, 0),
                ptr_coeff_gt_ctv(cell_index, 1),
                ptr_coeff_gt_ctv(cell_index, 2)};
            const std::array<index_type, 3> c2v{c2v_gt_ctv(cell_index, 0),
                c2v_gt_ctv(cell_index, 1),
                c2v_gt_ctv(cell_index, 2)};
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type k_index{}; k_index < KDim; ++k_index) {
                p_cell_out_gt_tv(cell_index, k_index) = p_vert_in_gt_ctv(c2v[0], k_index) * coeff[0] +
                    p_vert_in_gt_ctv(c2v[1], k_index) * coeff[1] +
                    p_vert_in_gt_ctv(c2v[2], k_index) * coeff[2];
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
