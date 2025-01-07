#pragma once

#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>

GT_FORCE_INLINE constexpr index_type get_index(int i, int j, int x_dim) { return i + j * x_dim; }

GT_FORCE_INLINE constexpr const std::array<index_type, 7> get_v2e2c2v(int i, int j, int x_dim) {
    std::array<index_type, 7> v2e2c2v{};
    v2e2c2v[0] = get_index(i - 1, j, x_dim);
    v2e2c2v[1] = get_index(i, j, x_dim);
    v2e2c2v[2] = get_index(i - 1, j + 1, x_dim);
    v2e2c2v[3] = get_index(i, j - 1, x_dim);
    v2e2c2v[4] = get_index(i + 1, j, x_dim);
    v2e2c2v[5] = get_index(i, j + 1, x_dim);
    v2e2c2v[6] = get_index(i + 1, j - 1, x_dim);
    return v2e2c2v;
}

template <typename T>
struct nabla4_interpolate_structured_inlined {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;

  public:
    nabla4_interpolate_structured_inlined(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0),
          interpolate_data(VertexDim, EdgeDim, KDim, y_dim, x_dim, halo + 1, nabla4_data.get_output_gt()){};

    nabla4_interpolate_structured_inlined(index_type CellDim,
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
        const std::vector<WP_TYPE> &inv_primal_edge_length,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : nabla4_data(CellDim,
              VertexDim,
              EdgeDim,
              KDim,
              ECVDim,
              y_dim,
              x_dim,
              0,
              u_vert,
              v_vert,
              primal_normal_vert_v1,
              primal_normal_vert_v2,
              z_nabla2_e,
              inv_vert_vert_length,
              inv_primal_edge_length),
          interpolate_data(VertexDim,
              EdgeDim,
              KDim,
              y_dim,
              x_dim,
              halo + 1,
              nabla4_data.get_output_gt(),
              ptr_coeff_1,
              ptr_coeff_2){};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    auto get_output_gt() -> decltype(interpolate_data.get_output_gt()) { return interpolate_data.get_output_gt(); }

    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < interpolate_data.KDim; ++k_index) {
            for (index_type j = interpolate_data.halo; j < interpolate_data.y_dim - interpolate_data.halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type i = interpolate_data.halo; i < interpolate_data.x_dim - interpolate_data.halo; ++i) {
                    const std::array<index_type, 7> v2e2c2v_compressed{get_v2e2c2v(i, j, interpolate_data.x_dim)};
                    const std::array<index_type, 6> v2e{
                        get_v2e_per_vertex(i, j, interpolate_data.x_dim, interpolate_data.y_dim)};
                    const std::array<index_type, 24> v2e2ecv{v2e[0] * 4,
                        v2e[0] * 4 + 1,
                        v2e[0] * 4 + 2,
                        v2e[0] * 4 + 3,
                        v2e[1] * 4,
                        v2e[1] * 4 + 1,
                        v2e[1] * 4 + 2,
                        v2e[1] * 4 + 3,
                        v2e[2] * 4,
                        v2e[2] * 4 + 1,
                        v2e[2] * 4 + 2,
                        v2e[2] * 4 + 3,
                        v2e[3] * 4,
                        v2e[3] * 4 + 1,
                        v2e[3] * 4 + 2,
                        v2e[3] * 4 + 3,
                        v2e[4] * 4,
                        v2e[4] * 4 + 1,
                        v2e[4] * 4 + 2,
                        v2e[4] * 4 + 3,
                        v2e[5] * 4,
                        v2e[5] * 4 + 1,
                        v2e[5] * 4 + 2,
                        v2e[5] * 4 + 3};
                    const std::array<index_type, 24> v2e2c2v{v2e2c2v_compressed[0],
                        v2e2c2v_compressed[1],
                        v2e2c2v_compressed[2],
                        v2e2c2v_compressed[3],
                        v2e2c2v_compressed[1],
                        v2e2c2v_compressed[4],
                        v2e2c2v_compressed[5],
                        v2e2c2v_compressed[6],
                        v2e2c2v_compressed[3],
                        v2e2c2v_compressed[1],
                        v2e2c2v_compressed[0],
                        v2e2c2v_compressed[6],
                        v2e2c2v_compressed[1],
                        v2e2c2v_compressed[5],
                        v2e2c2v_compressed[2],
                        v2e2c2v_compressed[4],
                        v2e2c2v_compressed[1],
                        v2e2c2v_compressed[6],
                        v2e2c2v_compressed[4],
                        v2e2c2v_compressed[3],
                        v2e2c2v_compressed[2],
                        v2e2c2v_compressed[1],
                        v2e2c2v_compressed[5],
                        v2e2c2v_compressed[0]};
                    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
                    const index_type vertex_index_internal =
                        i - interpolate_data.halo +
                        (j - interpolate_data.halo) * (interpolate_data.x_dim - 2 * interpolate_data.halo);
#ifdef __clang__
#pragma clang loop unroll(full)
#elif defined(__NVCC__)
#pragma unroll 6
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                    for (auto i{0}; i < 6; ++i) {
                        const auto edge_index = v2e[i];
                        const auto E2C2V_0 = v2e2c2v[i * 4];
                        const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
                        const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
                        const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
                        const auto E2ECV_0 = v2e2ecv[i * 4];
                        const auto E2ECV_1 = v2e2ecv[i * 4 + 1];
                        const auto E2ECV_2 = v2e2ecv[i * 4 + 2];
                        const auto E2ECV_3 = v2e2ecv[i * 4 + 3];
                        const double nabv_tang_wp = nabla4_data.u_vert_gt_tv(E2C2V_0, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_0, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0) +
                                                    nabla4_data.u_vert_gt_tv(E2C2V_1, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_1, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1);
                        const double nabv_norm_wp = nabla4_data.u_vert_gt_tv(E2C2V_2, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_2, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2) +
                                                    nabla4_data.u_vert_gt_tv(E2C2V_3, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_3, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3);
                        z_nabla4_e2_wp[i] =
                            4.0 * ((nabv_norm_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                                          (nabla4_data.inv_vert_vert_length_gt_tv(edge_index) *
                                              nabla4_data.inv_vert_vert_length_gt_tv(edge_index)) +
                                      (nabv_tang_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                                          (nabla4_data.inv_primal_edge_length_gt_tv(edge_index) *
                                              nabla4_data.inv_primal_edge_length_gt_tv(edge_index)));
                    }
                    interpolate_data.p_u_out_gt_tv(vertex_index_internal, k_index) =
                        z_nabla4_e2_wp[0] * interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 0) +
                        z_nabla4_e2_wp[1] * interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 1) +
                        z_nabla4_e2_wp[2] * interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 2) +
                        z_nabla4_e2_wp[3] * interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 3) +
                        z_nabla4_e2_wp[4] * interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 4) +
                        z_nabla4_e2_wp[5] * interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 5);
                    interpolate_data.p_v_out_gt_tv(vertex_index_internal, k_index) =
                        z_nabla4_e2_wp[0] * interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 0) +
                        z_nabla4_e2_wp[1] * interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 1) +
                        z_nabla4_e2_wp[2] * interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 2) +
                        z_nabla4_e2_wp[3] * interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 3) +
                        z_nabla4_e2_wp[4] * interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 4) +
                        z_nabla4_e2_wp[5] * interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 5);
                };
            };
        };
    }

    void run_cpu_kfirst() {
        for (index_type j = interpolate_data.halo; j < interpolate_data.y_dim - interpolate_data.halo; ++j) {
            for (index_type i = interpolate_data.halo; i < interpolate_data.x_dim - interpolate_data.halo; ++i) {
                const std::array<index_type, 7> v2e2c2v_compressed{get_v2e2c2v(i, j, nabla4_data.x_dim)};
                const std::array<index_type, 6> v2e{get_v2e_per_vertex(i, j, nabla4_data.x_dim, nabla4_data.y_dim)};
                const std::array<index_type, 24> v2e2ecv{v2e[0] * 4,
                    v2e[0] * 4 + 1,
                    v2e[0] * 4 + 2,
                    v2e[0] * 4 + 3,
                    v2e[1] * 4,
                    v2e[1] * 4 + 1,
                    v2e[1] * 4 + 2,
                    v2e[1] * 4 + 3,
                    v2e[2] * 4,
                    v2e[2] * 4 + 1,
                    v2e[2] * 4 + 2,
                    v2e[2] * 4 + 3,
                    v2e[3] * 4,
                    v2e[3] * 4 + 1,
                    v2e[3] * 4 + 2,
                    v2e[3] * 4 + 3,
                    v2e[4] * 4,
                    v2e[4] * 4 + 1,
                    v2e[4] * 4 + 2,
                    v2e[4] * 4 + 3,
                    v2e[5] * 4,
                    v2e[5] * 4 + 1,
                    v2e[5] * 4 + 2,
                    v2e[5] * 4 + 3};
                const std::array<index_type, 24> v2e2c2v{v2e2c2v_compressed[0],
                    v2e2c2v_compressed[1],
                    v2e2c2v_compressed[2],
                    v2e2c2v_compressed[3],
                    v2e2c2v_compressed[1],
                    v2e2c2v_compressed[4],
                    v2e2c2v_compressed[5],
                    v2e2c2v_compressed[6],
                    v2e2c2v_compressed[3],
                    v2e2c2v_compressed[1],
                    v2e2c2v_compressed[0],
                    v2e2c2v_compressed[6],
                    v2e2c2v_compressed[1],
                    v2e2c2v_compressed[5],
                    v2e2c2v_compressed[2],
                    v2e2c2v_compressed[4],
                    v2e2c2v_compressed[1],
                    v2e2c2v_compressed[6],
                    v2e2c2v_compressed[4],
                    v2e2c2v_compressed[3],
                    v2e2c2v_compressed[2],
                    v2e2c2v_compressed[1],
                    v2e2c2v_compressed[5],
                    v2e2c2v_compressed[0]};
                const index_type vertex_index_internal =
                    i - interpolate_data.halo +
                    (j - interpolate_data.halo) * (interpolate_data.x_dim - 2 * interpolate_data.halo);
                const std::array<WP_TYPE, 6> inv_vert_vert_length{nabla4_data.inv_vert_vert_length_gt_tv(v2e[0]),
                    nabla4_data.inv_vert_vert_length_gt_tv(v2e[1]),
                    nabla4_data.inv_vert_vert_length_gt_tv(v2e[2]),
                    nabla4_data.inv_vert_vert_length_gt_tv(v2e[3]),
                    nabla4_data.inv_vert_vert_length_gt_tv(v2e[4]),
                    nabla4_data.inv_vert_vert_length_gt_tv(v2e[5])};
                const std::array<WP_TYPE, 6> inv_primal_edge_length{nabla4_data.inv_primal_edge_length_gt_tv(v2e[0]),
                    nabla4_data.inv_primal_edge_length_gt_tv(v2e[1]),
                    nabla4_data.inv_primal_edge_length_gt_tv(v2e[2]),
                    nabla4_data.inv_primal_edge_length_gt_tv(v2e[3]),
                    nabla4_data.inv_primal_edge_length_gt_tv(v2e[4]),
                    nabla4_data.inv_primal_edge_length_gt_tv(v2e[5])};
                const std::array<WP_TYPE, 6> coef1{interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 0),
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 1),
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 2),
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 3),
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 4),
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index_internal, 5)};
                const std::array<WP_TYPE, 6> coef2{interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 0),
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 1),
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 2),
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 3),
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 4),
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index_internal, 5)};
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type k_index{}; k_index < interpolate_data.KDim; ++k_index) {

                    std::array<WP_TYPE, 6> z_nabla4_e2_wp;

#ifdef __clang__
#pragma clang loop unroll(full)
#elif defined(__NVCC__)
#pragma unroll 6
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                    for (auto i{0}; i < 6; ++i) {
                        const auto edge_index = v2e[i];
                        const auto E2C2V_0 = v2e2c2v[i * 4];
                        const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
                        const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
                        const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
                        const auto E2ECV_0 = v2e2ecv[i * 4];
                        const auto E2ECV_1 = v2e2ecv[i * 4 + 1];
                        const auto E2ECV_2 = v2e2ecv[i * 4 + 2];
                        const auto E2ECV_3 = v2e2ecv[i * 4 + 3];
                        const double nabv_tang_wp = nabla4_data.u_vert_gt_tv(E2C2V_0, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_0, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0) +
                                                    nabla4_data.u_vert_gt_tv(E2C2V_1, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_1, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1);
                        const double nabv_norm_wp = nabla4_data.u_vert_gt_tv(E2C2V_2, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_2, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2) +
                                                    nabla4_data.u_vert_gt_tv(E2C2V_3, k_index) *
                                                        nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3) +
                                                    nabla4_data.v_vert_gt_tv(E2C2V_3, k_index) *
                                                        nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3);
                        z_nabla4_e2_wp[i] =
                            4.0 * ((nabv_norm_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                                          (inv_vert_vert_length[i] * inv_vert_vert_length[i]) +
                                      (nabv_tang_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                                          (inv_primal_edge_length[i] * inv_primal_edge_length[i]));
                    }
                    interpolate_data.p_u_out_gt_tv(vertex_index_internal, k_index) =
                        z_nabla4_e2_wp[0] * coef1[0] + z_nabla4_e2_wp[1] * coef1[1] + z_nabla4_e2_wp[2] * coef1[2] +
                        z_nabla4_e2_wp[3] * coef1[3] + z_nabla4_e2_wp[4] * coef1[4] + z_nabla4_e2_wp[5] * coef1[5];
                    interpolate_data.p_v_out_gt_tv(vertex_index_internal, k_index) =
                        z_nabla4_e2_wp[0] * coef2[0] + z_nabla4_e2_wp[1] * coef2[1] + z_nabla4_e2_wp[2] * coef2[2] +
                        z_nabla4_e2_wp[3] * coef2[3] + z_nabla4_e2_wp[4] * coef2[4] + z_nabla4_e2_wp[5] * coef2[5];
                };
            };
        };
    }

    void run_gpu_kloop_helper();
    void run_gpu_naive_helper();

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
template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_kloop<int>() {
    return {32, 4, 2, 256};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_kloop =
    get_block_dims_structured_nabla_interpol_inlined_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_structured_nabla_interpol_inlined_kloop.size)
    run_gpu_kloop_nabla4_interpolate_inlined_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        index_type inner_domain_size,
        index_type outer_domain_size,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim) {
        return;
    }
    const std::array<index_type, 7> v2e2c2v_compressed{get_v2e2c2v(i, j, x_dim)};
    const std::array<index_type, 6> v2e{get_v2e_per_orientation(i, j, x_dim, y_dim)};
    const std::array<index_type, 24> v2e2ecv{v2e[0],
        v2e[0] + outer_domain_size * 3,
        v2e[0] + outer_domain_size * 6,
        v2e[0] + outer_domain_size * 9,
        v2e[1],
        v2e[1] + outer_domain_size * 3,
        v2e[1] + outer_domain_size * 6,
        v2e[1] + outer_domain_size * 9,
        v2e[2],
        v2e[2] + outer_domain_size * 3,
        v2e[2] + outer_domain_size * 6,
        v2e[2] + outer_domain_size * 9,
        v2e[3],
        v2e[3] + outer_domain_size * 3,
        v2e[3] + outer_domain_size * 6,
        v2e[3] + outer_domain_size * 9,
        v2e[4],
        v2e[4] + outer_domain_size * 3,
        v2e[4] + outer_domain_size * 6,
        v2e[4] + outer_domain_size * 9,
        v2e[5],
        v2e[5] + outer_domain_size * 3,
        v2e[5] + outer_domain_size * 6,
        v2e[5] + outer_domain_size * 9};
    const std::array<index_type, 24> v2e2c2v{v2e2c2v_compressed[0],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[0],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[0]};
    const std::array<WP_TYPE, 24> primal_normal_vert_v1{primal_normal_vert_v1_gt_tv(v2e2ecv[0]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[1]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[2]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[3]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[4]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[5]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[6]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[7]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[8]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[9]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[10]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[11]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[12]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[13]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[14]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[15]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[16]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[17]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[18]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[19]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[20]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[21]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[22]),
        primal_normal_vert_v1_gt_tv(v2e2ecv[23])};
    const std::array<WP_TYPE, 24> primal_normal_vert_v2{primal_normal_vert_v2_gt_tv(v2e2ecv[0]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[1]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[2]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[3]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[4]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[5]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[6]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[7]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[8]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[9]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[10]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[11]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[12]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[13]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[14]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[15]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[16]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[17]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[18]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[19]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[20]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[21]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[22]),
        primal_normal_vert_v2_gt_tv(v2e2ecv[23])};
    const std::array<WP_TYPE, 6> inv_vert_vert_length{inv_vert_vert_length_gt_tv(v2e[0]),
        inv_vert_vert_length_gt_tv(v2e[1]),
        inv_vert_vert_length_gt_tv(v2e[2]),
        inv_vert_vert_length_gt_tv(v2e[3]),
        inv_vert_vert_length_gt_tv(v2e[4]),
        inv_vert_vert_length_gt_tv(v2e[5])};
    const std::array<WP_TYPE, 6> inv_primal_edge_length{inv_primal_edge_length_gt_tv(v2e[0]),
        inv_primal_edge_length_gt_tv(v2e[1]),
        inv_primal_edge_length_gt_tv(v2e[2]),
        inv_primal_edge_length_gt_tv(v2e[3]),
        inv_primal_edge_length_gt_tv(v2e[4]),
        inv_primal_edge_length_gt_tv(v2e[5])};
    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
    const std::array<WP_TYPE, 6> ptr_coeff_1{ptr_coeff_1_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_1_gt_ctv(vertex_index_internal, 5)};
    const std::array<WP_TYPE, 6> ptr_coeff_2{ptr_coeff_2_gt_ctv(vertex_index_internal, 0),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 1),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 2),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 3),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 4),
        ptr_coeff_2_gt_ctv(vertex_index_internal, 5)};
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
#pragma unroll 6
        for (auto color{0}; color < 6; ++color) {
            const auto edge_index = v2e[color];
            const auto E2C2V_0 = v2e2c2v[color * 4];
            const auto E2C2V_1 = v2e2c2v[color * 4 + 1];
            const auto E2C2V_2 = v2e2c2v[color * 4 + 2];
            const auto E2C2V_3 = v2e2c2v[color * 4 + 3];
            const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1[4 * color] +
                                        v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2[4 * color] +
                                        u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1[4 * color + 1] +
                                        v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2[4 * color + 1];
            const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1[4 * color + 2] +
                                        v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2[4 * color + 2] +
                                        u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1[4 * color + 3] +
                                        v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2[4 * color + 3];
            const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
            z_nabla4_e2_wp[color] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * (inv_vert_vert_length[color] * inv_vert_vert_length[color]) +
                          (nabv_tang_wp - 2.0 * z_nabla2_e) *
                              (inv_primal_edge_length[color] * inv_primal_edge_length[color]));
        }
        p_u_out_gt_tv(vertex_index_internal, k_index) =
            z_nabla4_e2_wp[0] * ptr_coeff_1[0] + z_nabla4_e2_wp[1] * ptr_coeff_1[1] +
            z_nabla4_e2_wp[2] * ptr_coeff_1[2] + z_nabla4_e2_wp[3] * ptr_coeff_1[3] +
            z_nabla4_e2_wp[4] * ptr_coeff_1[4] + z_nabla4_e2_wp[5] * ptr_coeff_1[5];
        p_v_out_gt_tv(vertex_index_internal, k_index) =
            z_nabla4_e2_wp[0] * ptr_coeff_2[0] + z_nabla4_e2_wp[1] * ptr_coeff_2[1] +
            z_nabla4_e2_wp[2] * ptr_coeff_2[2] + z_nabla4_e2_wp[3] * ptr_coeff_2[3] +
            z_nabla4_e2_wp[4] * ptr_coeff_2[4] + z_nabla4_e2_wp[5] * ptr_coeff_2[5];
    }
};

template <typename T>
inline void nabla4_interpolate_structured_inlined<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_structured_nabla_interpol_inlined_kloop.x,
        block_dims_structured_nabla_interpol_inlined_kloop.y,
        block_dims_structured_nabla_interpol_inlined_kloop.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x, (inner_y_dim + tblocks.y - 1) / tblocks.y, 1);
    run_gpu_kloop_nabla4_interpolate_inlined_structured<<<grid, tblocks>>>(interpolate_data.KDim,
        interpolate_data.x_dim,
        interpolate_data.y_dim,
        interpolate_data.halo,
        inner_domain_size,
        outer_domain_size,
        nabla4_data.u_vert_gt_tv,
        nabla4_data.v_vert_gt_tv,
        nabla4_data.primal_normal_vert_v1_gt_tv,
        nabla4_data.primal_normal_vert_v2_gt_tv,
        nabla4_data.z_nabla2_e_gt_tv,
        nabla4_data.inv_vert_vert_length_gt_tv,
        nabla4_data.inv_primal_edge_length_gt_tv,
        interpolate_data.ptr_coeff_1_gt_ctv,
        interpolate_data.ptr_coeff_2_gt_ctv,
        interpolate_data.p_u_out_gt_tv,
        interpolate_data.p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};

template <typename T>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_nabla_interpol_inlined_naive<int>() {
    return {32, 8, 2, 512};
};

constexpr block_dims block_dims_structured_nabla_interpol_inlined_naive =
    get_block_dims_structured_nabla_interpol_inlined_naive<index_type>();

__global__ void
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 5)
__launch_bounds__(block_dims_structured_nabla_interpol_inlined_naive.size)
#else
__maxnreg__(62)
#endif
    run_gpu_naive_nabla4_interpolate_inlined_structured(index_type KDim,
        index_type x_dim,
        index_type y_dim,
        index_type halo,
        index_type inner_domain_size,
        index_type outer_domain_size,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_structured_torus_halo_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_structured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto i{blockIdx.x * blockDim.x + threadIdx.x + halo};
    const auto j{blockIdx.y * blockDim.y + threadIdx.y + halo};
    const auto k_index{blockIdx.z * blockDim.z + threadIdx.z};
    if (i >= x_dim - halo || j >= y_dim - halo || k_index >= KDim) {
        return;
    }
    const std::array<index_type, 7> v2e2c2v_compressed{get_v2e2c2v(i, j, x_dim)};
    const std::array<index_type, 6> v2e{get_v2e_per_orientation(i, j, x_dim, y_dim)};
    const std::array<index_type, 24> v2e2ecv{v2e[0],
        v2e[0] + outer_domain_size * 3,
        v2e[0] + outer_domain_size * 6,
        v2e[0] + outer_domain_size * 9,
        v2e[1],
        v2e[1] + outer_domain_size * 3,
        v2e[1] + outer_domain_size * 6,
        v2e[1] + outer_domain_size * 9,
        v2e[2],
        v2e[2] + outer_domain_size * 3,
        v2e[2] + outer_domain_size * 6,
        v2e[2] + outer_domain_size * 9,
        v2e[3],
        v2e[3] + outer_domain_size * 3,
        v2e[3] + outer_domain_size * 6,
        v2e[3] + outer_domain_size * 9,
        v2e[4],
        v2e[4] + outer_domain_size * 3,
        v2e[4] + outer_domain_size * 6,
        v2e[4] + outer_domain_size * 9,
        v2e[5],
        v2e[5] + outer_domain_size * 3,
        v2e[5] + outer_domain_size * 6,
        v2e[5] + outer_domain_size * 9};
    const std::array<index_type, 24> v2e2c2v{v2e2c2v_compressed[0],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[0],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[6],
        v2e2c2v_compressed[4],
        v2e2c2v_compressed[3],
        v2e2c2v_compressed[2],
        v2e2c2v_compressed[1],
        v2e2c2v_compressed[5],
        v2e2c2v_compressed[0]};
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    const index_type vertex_index_internal = i - halo + (j - halo) * (x_dim - 2 * halo);
#pragma unroll 6
    for (auto i{0}; i < 6; ++i) {
        const auto edge_index = v2e[i];
        const auto E2C2V_0 = v2e2c2v[i * 4];
        const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
        const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
        const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
        const auto E2ECV_0 = v2e2ecv[i * 4];
        const auto E2ECV_1 = v2e2ecv[i * 4 + 1];
        const auto E2ECV_2 = v2e2ecv[i * 4 + 2];
        const auto E2ECV_3 = v2e2ecv[i * 4 + 3];
        const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0) +
                                    v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0) +
                                    u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1) +
                                    v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1);
        const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2) +
                                    v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2) +
                                    u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3) +
                                    v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3);
        const WP_TYPE z_nabla2_e = z_nabla2_e_gt_tv(edge_index, k_index);
        const WP_TYPE inv_vert_vert_length = inv_vert_vert_length_gt_tv(edge_index);
        const WP_TYPE inv_vert_vert_length_sqr = inv_vert_vert_length * inv_vert_vert_length;
        const WP_TYPE inv_primal_edge_length = inv_primal_edge_length_gt_tv(edge_index);
        const WP_TYPE inv_primal_edge_length_sqr = inv_primal_edge_length * inv_primal_edge_length;
        z_nabla4_e2_wp[i] = 4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e) * inv_vert_vert_length_sqr +
                                      (nabv_tang_wp - 2.0 * z_nabla2_e) * inv_primal_edge_length_sqr);
    }
    p_u_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_1_gt_ctv(vertex_index_internal, 0) +
                                                    z_nabla4_e2_wp[1] * ptr_coeff_1_gt_ctv(vertex_index_internal, 1) +
                                                    z_nabla4_e2_wp[2] * ptr_coeff_1_gt_ctv(vertex_index_internal, 2) +
                                                    z_nabla4_e2_wp[3] * ptr_coeff_1_gt_ctv(vertex_index_internal, 3) +
                                                    z_nabla4_e2_wp[4] * ptr_coeff_1_gt_ctv(vertex_index_internal, 4) +
                                                    z_nabla4_e2_wp[5] * ptr_coeff_1_gt_ctv(vertex_index_internal, 5);
    p_v_out_gt_tv(vertex_index_internal, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_2_gt_ctv(vertex_index_internal, 0) +
                                                    z_nabla4_e2_wp[1] * ptr_coeff_2_gt_ctv(vertex_index_internal, 1) +
                                                    z_nabla4_e2_wp[2] * ptr_coeff_2_gt_ctv(vertex_index_internal, 2) +
                                                    z_nabla4_e2_wp[3] * ptr_coeff_2_gt_ctv(vertex_index_internal, 3) +
                                                    z_nabla4_e2_wp[4] * ptr_coeff_2_gt_ctv(vertex_index_internal, 4) +
                                                    z_nabla4_e2_wp[5] * ptr_coeff_2_gt_ctv(vertex_index_internal, 5);
};

template <typename T>
inline void nabla4_interpolate_structured_inlined<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_structured_nabla_interpol_inlined_naive.x,
        block_dims_structured_nabla_interpol_inlined_naive.y,
        block_dims_structured_nabla_interpol_inlined_naive.z);
    const index_type inner_domain_size =
        (interpolate_data.x_dim - 2 * interpolate_data.halo) * (interpolate_data.y_dim - 2 * interpolate_data.halo);
    const index_type outer_domain_size = interpolate_data.x_dim * interpolate_data.y_dim;
    const index_type inner_x_dim = interpolate_data.x_dim - 2 * interpolate_data.halo;
    const index_type inner_y_dim = interpolate_data.y_dim - 2 * interpolate_data.halo;
    dim3 grid((inner_x_dim + tblocks.x - 1) / tblocks.x,
        (inner_y_dim + tblocks.y - 1) / tblocks.y,
        (interpolate_data.KDim + tblocks.z - 1) / tblocks.z);
    run_gpu_naive_nabla4_interpolate_inlined_structured<<<grid, tblocks>>>(interpolate_data.KDim,
        interpolate_data.x_dim,
        interpolate_data.y_dim,
        interpolate_data.halo,
        inner_domain_size,
        outer_domain_size,
        nabla4_data.u_vert_gt_tv,
        nabla4_data.v_vert_gt_tv,
        nabla4_data.primal_normal_vert_v1_gt_tv,
        nabla4_data.primal_normal_vert_v2_gt_tv,
        nabla4_data.z_nabla2_e_gt_tv,
        nabla4_data.inv_vert_vert_length_gt_tv,
        nabla4_data.inv_primal_edge_length_gt_tv,
        interpolate_data.ptr_coeff_1_gt_ctv,
        interpolate_data.ptr_coeff_2_gt_ctv,
        interpolate_data.p_u_out_gt_tv,
        interpolate_data.p_v_out_gt_tv);
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void nabla4_interpolate_structured_inlined<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_structured_inlined<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
