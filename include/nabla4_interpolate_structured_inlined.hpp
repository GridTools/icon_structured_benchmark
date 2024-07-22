#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>

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
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, 0), interpolate_data(VertexDim,
                                                                                       EdgeDim,
                                                                                       KDim,
                                                                                       y_dim - 2 * (halo + 1),
                                                                                       x_dim - 2 * (halo + 1),
                                                                                       halo + 1,
                                                                                       nabla4_data.get_output_gt()){};

    nabla4_interpolate_structured_inlined(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
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
              y_dim - 2 * (halo + 1),
              x_dim - 2 * (halo + 1),
              halo + 1,
              nabla4_data.get_output_gt(),
              ptr_coeff_1,
              ptr_coeff_2){};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    GT_FORCE_INLINE constexpr index_type get_index(int i, int j, int x_dim) const { return i + j * x_dim; }

    GT_FORCE_INLINE constexpr const std::array<index_type, 7> get_v2e2c2v(int i, int j, int x_dim) const {
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

    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < interpolate_data.KDim; ++k_index) {
            for (index_type j = interpolate_data.halo; j < nabla4_data.y_dim - interpolate_data.halo; ++j) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
                for (index_type i = interpolate_data.halo; i < nabla4_data.x_dim - interpolate_data.halo; ++i) {
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
                    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
                    const index_type vertex_index_internal =
                        i - interpolate_data.halo +
                        (j - interpolate_data.halo) * (nabla4_data.x_dim - 2 * interpolate_data.halo);
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

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};
