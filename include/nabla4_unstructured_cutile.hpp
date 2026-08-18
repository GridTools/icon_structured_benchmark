#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "nabla4_gridtools.hpp"

extern void run_cutile_nabla4_unstructured_launcher(index_type EdgeDim,
    index_type KDim,
    index_type u_vert_k_stride,
    index_type v_vert_k_stride,
    index_type z_nabla2_e_k_stride,
    index_type z_nabla4_e2_wp_k_stride,
    index_type e2c2v_slot_stride,
    index_type e2ecv_slot_stride,
    const index_type* __restrict__ e2c2v_ptr,
    const index_type* __restrict__ e2ecv_ptr,
    const VP_TYPE* __restrict__ u_vert_ptr,
    const VP_TYPE* __restrict__ v_vert_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v1_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v2_ptr,
    const WP_TYPE* __restrict__ z_nabla2_e_ptr,
    const WP_TYPE* __restrict__ inv_vert_vert_length_ptr,
    const WP_TYPE* __restrict__ inv_primal_edge_length_ptr,
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr);

extern void run_cutile_nabla4_unstructured_kloop_launcher(index_type EdgeDim,
    index_type KDim,
    index_type u_vert_k_stride,
    index_type v_vert_k_stride,
    index_type z_nabla2_e_k_stride,
    index_type z_nabla4_e2_wp_k_stride,
    index_type e2c2v_slot_stride,
    index_type e2ecv_slot_stride,
    const index_type* __restrict__ e2c2v_ptr,
    const index_type* __restrict__ e2ecv_ptr,
    const VP_TYPE* __restrict__ u_vert_ptr,
    const VP_TYPE* __restrict__ v_vert_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v1_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v2_ptr,
    const WP_TYPE* __restrict__ z_nabla2_e_ptr,
    const WP_TYPE* __restrict__ inv_vert_vert_length_ptr,
    const WP_TYPE* __restrict__ inv_primal_edge_length_ptr,
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr);

/// Unstructured nabla4 benchmark class backed by cuTile.
///
/// Mirrors \c nabla4_unstructured_gt's interface (same constructors, same
/// e2c2v/e2ecv GridTools connectivity stores) so it can be driven by the
/// same \c benchmark_gridtools / \c nabla4_validate_gridtools templates and
/// Python bindings that already drive \c nabla4_unstructured_gt and
/// \c nabla4_structured_torus_cutile_halo.
///
/// Unlike the structured torus, edge neighbors are not affine in the
/// storage layout, so E2C2V/E2ECV are gathered as scalar reads and only
/// used as *dynamic* base offsets for tile-loads along K (see
/// src/nabla4_unstructured_cutile.cu).
template <typename T>
class nabla4_unstructured_cutile : public nabla4_gt_data<T> {
  public:
    using nabla4_gt_data<T>::CellDim;
    using nabla4_gt_data<T>::EdgeDim;
    using nabla4_gt_data<T>::VertexDim;
    using nabla4_gt_data<T>::KDim;
    using nabla4_gt_data<T>::ECVDim;
    using nabla4_gt_data<T>::output_size;
    using nabla4_gt_data<T>::u_vert_gt;
    using nabla4_gt_data<T>::v_vert_gt;
    using nabla4_gt_data<T>::primal_normal_vert_v1_gt;
    using nabla4_gt_data<T>::primal_normal_vert_v2_gt;
    using nabla4_gt_data<T>::z_nabla2_e_gt;
    using nabla4_gt_data<T>::inv_vert_vert_length_gt;
    using nabla4_gt_data<T>::inv_primal_edge_length_gt;
    using nabla4_gt_data<T>::z_nabla4_e2_wp_gt;

    using neighbors_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 4_c).template type<index_type>().build());
    neighbors_gt_t e2c2v_gt;
    neighbors_gt_t e2ecv_gt;

    /// Constructor for benchmarking (initializes data with random numbers).
    nabla4_unstructured_cutile(const std::vector<std::array<index_type, 4>> &e2c2v,
        const std::vector<std::array<index_type, 4>> &e2ecv,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : e2c2v_gt(storage::builder<T>
                       .template type<index_type>()
                       .dimensions(e2c2v.size(), 4_c)
                       .initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; })
                       .build()),
          e2ecv_gt(storage::builder<T>
                       .template type<index_type>()
                       .dimensions(e2ecv.size(), 4_c)
                       .initializer([&e2ecv](int i, int j) { return e2ecv[i][j]; })
                       .build()),
          nabla4_gt_data<T>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, e2c2v.size()){};

    /// Constructor for validation (uses the provided input fields).
    nabla4_unstructured_cutile(const std::vector<std::array<index_type, 4>> &e2c2v,
        const std::vector<std::array<index_type, 4>> &e2ecv,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        const std::vector<std::vector<VP_TYPE>> &u_vert,
        const std::vector<std::vector<VP_TYPE>> &v_vert,
        const std::vector<WP_TYPE> &primal_normal_vert_v1,
        const std::vector<WP_TYPE> &primal_normal_vert_v2,
        const std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        const std::vector<WP_TYPE> &inv_vert_vert_length,
        const std::vector<WP_TYPE> &inv_primal_edge_length)
        : e2c2v_gt(storage::builder<T>
                       .template type<index_type>()
                       .dimensions(e2c2v.size(), 4_c)
                       .initializer([&e2c2v](int i, int j) { return e2c2v[i][j]; })
                       .build()),
          e2ecv_gt(storage::builder<T>
                       .template type<index_type>()
                       .dimensions(e2ecv.size(), 4_c)
                       .initializer([&e2ecv](int i, int j) { return e2ecv[i][j]; })
                       .build()),
          nabla4_gt_data<T>(CellDim,
              VertexDim,
              EdgeDim,
              KDim,
              ECVDim,
              e2c2v.size(),
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
            throw std::runtime_error("Undefined backend implementation for cuTile unstructured nabla4");
        }
    };
};

#if defined(__CUDACC__)

template <typename T>
inline void nabla4_unstructured_cutile<T>::run_gpu_naive_helper() {
    const auto* e2c2v_ptr = e2c2v_gt->get_const_target_ptr();
    const auto* e2ecv_ptr = e2ecv_gt->get_const_target_ptr();
    const auto* u_vert_ptr = u_vert_gt->get_const_target_ptr();
    const auto* v_vert_ptr = v_vert_gt->get_const_target_ptr();
    const auto* primal_normal_vert_v1_ptr = primal_normal_vert_v1_gt->get_const_target_ptr();
    const auto* primal_normal_vert_v2_ptr = primal_normal_vert_v2_gt->get_const_target_ptr();
    const auto* z_nabla2_e_ptr = z_nabla2_e_gt->get_const_target_ptr();
    const auto* inv_vert_vert_length_ptr = inv_vert_vert_length_gt->get_const_target_ptr();
    const auto* inv_primal_edge_length_ptr = inv_primal_edge_length_gt->get_const_target_ptr();
    auto* z_nabla4_e2_wp_ptr = z_nabla4_e2_wp_gt->get_target_ptr();

    const index_type e2c2v_slot_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(e2c2v_gt->native_strides()));
    const index_type e2ecv_slot_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(e2ecv_gt->native_strides()));
    const index_type u_vert_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(u_vert_gt->native_strides()));
    const index_type v_vert_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(v_vert_gt->native_strides()));
    const index_type z_nabla2_e_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(z_nabla2_e_gt->native_strides()));
    const index_type z_nabla4_e2_wp_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(z_nabla4_e2_wp_gt->native_strides()));

    run_cutile_nabla4_unstructured_launcher(output_size,
        KDim,
        u_vert_k_stride,
        v_vert_k_stride,
        z_nabla2_e_k_stride,
        z_nabla4_e2_wp_k_stride,
        e2c2v_slot_stride,
        e2ecv_slot_stride,
        e2c2v_ptr,
        e2ecv_ptr,
        u_vert_ptr,
        v_vert_ptr,
        primal_normal_vert_v1_ptr,
        primal_normal_vert_v2_ptr,
        z_nabla2_e_ptr,
        inv_vert_vert_length_ptr,
        inv_primal_edge_length_ptr,
        z_nabla4_e2_wp_ptr);
}

template <typename T>
inline void nabla4_unstructured_cutile<T>::run_gpu_kloop_helper() {
    const auto* e2c2v_ptr = e2c2v_gt->get_const_target_ptr();
    const auto* e2ecv_ptr = e2ecv_gt->get_const_target_ptr();
    const auto* u_vert_ptr = u_vert_gt->get_const_target_ptr();
    const auto* v_vert_ptr = v_vert_gt->get_const_target_ptr();
    const auto* primal_normal_vert_v1_ptr = primal_normal_vert_v1_gt->get_const_target_ptr();
    const auto* primal_normal_vert_v2_ptr = primal_normal_vert_v2_gt->get_const_target_ptr();
    const auto* z_nabla2_e_ptr = z_nabla2_e_gt->get_const_target_ptr();
    const auto* inv_vert_vert_length_ptr = inv_vert_vert_length_gt->get_const_target_ptr();
    const auto* inv_primal_edge_length_ptr = inv_primal_edge_length_gt->get_const_target_ptr();
    auto* z_nabla4_e2_wp_ptr = z_nabla4_e2_wp_gt->get_target_ptr();

    const index_type e2c2v_slot_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(e2c2v_gt->native_strides()));
    const index_type e2ecv_slot_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(e2ecv_gt->native_strides()));
    const index_type u_vert_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(u_vert_gt->native_strides()));
    const index_type v_vert_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(v_vert_gt->native_strides()));
    const index_type z_nabla2_e_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(z_nabla2_e_gt->native_strides()));
    const index_type z_nabla4_e2_wp_k_stride =
        static_cast<index_type>(tuple_util::host_device::get<1>(z_nabla4_e2_wp_gt->native_strides()));

    run_cutile_nabla4_unstructured_kloop_launcher(output_size,
        KDim,
        u_vert_k_stride,
        v_vert_k_stride,
        z_nabla2_e_k_stride,
        z_nabla4_e2_wp_k_stride,
        e2c2v_slot_stride,
        e2ecv_slot_stride,
        e2c2v_ptr,
        e2ecv_ptr,
        u_vert_ptr,
        v_vert_ptr,
        primal_normal_vert_v1_ptr,
        primal_normal_vert_v2_ptr,
        z_nabla2_e_ptr,
        inv_vert_vert_length_ptr,
        inv_primal_edge_length_ptr,
        z_nabla4_e2_wp_ptr);
}
#else
template <typename T>
inline void nabla4_unstructured_cutile<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
}
template <typename T>
inline void nabla4_unstructured_cutile<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
}
#endif
