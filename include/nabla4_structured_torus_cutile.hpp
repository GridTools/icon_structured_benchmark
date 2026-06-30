#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "nabla4_gridtools.hpp"

// TODO: Add cuTile includes here when the implementation is written.
// #include <cutile/cutile.hpp>

/// Placeholder structured-torus nabla4 benchmark class backed by cuTile.
///
/// The interface mirrors \c nabla4_structured_torus_halo_gt so that it can be
/// driven by the same wrapper functions and Python bindings.  The GPU kernel
/// implementation is intentionally left empty for future cuTile development.
template <typename T>
class nabla4_structured_torus_cutile_halo : public nabla4_gt_data<T> {
  public:
    using nabla4_gt_data<T>::CellDim;
    using nabla4_gt_data<T>::EdgeDim;
    using nabla4_gt_data<T>::VertexDim;
    using nabla4_gt_data<T>::KDim;
    using nabla4_gt_data<T>::ECVDim;
    using nabla4_gt_data<T>::u_vert_gt_tv;
    using nabla4_gt_data<T>::v_vert_gt_tv;
    using nabla4_gt_data<T>::primal_normal_vert_v1_gt_tv;
    using nabla4_gt_data<T>::primal_normal_vert_v2_gt_tv;
    using nabla4_gt_data<T>::z_nabla2_e_gt_tv;
    using nabla4_gt_data<T>::inv_vert_vert_length_gt_tv;
    using nabla4_gt_data<T>::inv_primal_edge_length_gt_tv;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt_tv;

    const index_type y_dim;
    const index_type x_dim;
    const index_type halo;

    /// Constructor for benchmarking (initializes data with random numbers).
    nabla4_structured_torus_cutile_halo(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : y_dim(y_dim), x_dim(x_dim), halo(halo),
          nabla4_gt_data<T>(CellDim,
              VertexDim,
              EdgeDim,
              KDim,
              ECVDim,
              (x_dim - 2 * halo) * (y_dim - halo * 2) * 3){};

    /// Constructor for validation (uses the provided input fields).
    nabla4_structured_torus_cutile_halo(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        const std::vector<std::vector<VP_TYPE>> &u_vert,
        const std::vector<std::vector<VP_TYPE>> &v_vert,
        const std::vector<WP_TYPE> &primal_normal_vert_v1,
        const std::vector<WP_TYPE> &primal_normal_vert_v2,
        const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        const std::vector<WP_TYPE> &inv_vert_vert_length,
        const std::vector<WP_TYPE> &inv_primal_edge_length)
        : y_dim(y_dim), x_dim(x_dim), halo(halo), nabla4_gt_data<T>(CellDim,
                                                        VertexDim,
                                                        EdgeDim,
                                                        KDim,
                                                        ECVDim,
                                                        (x_dim - 2 * halo) * (y_dim - halo * 2) * 3,
                                                        u_vert,
                                                        v_vert,
                                                        primal_normal_vert_v1,
                                                        primal_normal_vert_v2,
                                                        z_nabla2_e,
                                                        inv_vert_vert_length,
                                                        inv_primal_edge_length){};

  private:
    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

  public:
    /// Compute function timed for benchmarking.
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_kloop) {
            run_gpu_kloop_helper();
        } else if constexpr (I == backend_impl::gpu_naive) {
            run_gpu_naive_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation for cuTile nabla4");
        }
    };
};

#if defined(__CUDACC__)
template <typename T>
inline void nabla4_structured_torus_cutile_halo<T>::run_gpu_kloop_helper() {
    // TODO: implement the cuTile nabla4 kernel here.
    // The data stores are accessible through the inherited target views, e.g.:
    //   u_vert_gt_tv, v_vert_gt_tv, z_nabla2_e_gt_tv, z_nabla4_e2_wp_gt_tv, ...
}

template <typename T>
inline void nabla4_structured_torus_cutile_halo<T>::run_gpu_naive_helper() {
    // TODO: implement the cuTile nabla4 kernel here.
    // The data stores are accessible through the inherited target views, e.g.:
    //   u_vert_gt_tv, v_vert_gt_tv, z_nabla2_e_gt_tv, z_nabla4_e2_wp_gt_tv, ...
}
#else
template <typename T>
inline void nabla4_structured_torus_cutile_halo<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
}
template <typename T>
inline void nabla4_structured_torus_cutile_halo<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
}
#endif
