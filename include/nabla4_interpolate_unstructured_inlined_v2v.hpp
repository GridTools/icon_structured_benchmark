#include <interpolate_unstructured_gridtools.hpp>
#include <nabla4_unstructured_gridtools.hpp>

template <typename T>
struct nabla4_interpolate_unstructured_inlined_v2v {
    nabla4_unstructured_gt<T> nabla4_data;
    interpolate_unstructured<T> interpolate_data;

    using neighbors_e2c2v_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 7_c).template type<index_type>().build());
    using neighbors_e2c2v_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 7_c).template type<index_type>().build()->const_target_view());
    using neighbors_e2ecv_gt_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 24_c).template type<index_type>().build());
    using neighbors_e2ecv_gt_ctv_t =
        decltype(gridtools::storage::builder<T>.dimensions(0, 24_c).template type<index_type>().build()->const_target_view());
    neighbors_e2c2v_gt_t v2e2c2v_gt;
    neighbors_e2c2v_gt_ctv_t v2e2c2v_gt_ctv;
    neighbors_e2ecv_gt_t v2e2ecv_gt;
    neighbors_e2ecv_gt_ctv_t v2e2ecv_gt_ctv;

    nabla4_interpolate_unstructured_inlined_v2v(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        std::vector<std::array<index_type, 6>> v2e,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : v2e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 7_c).initializer([&e2c2v, &v2e](int i, int j) {
                switch (j) {
                    case 0:
                        return e2c2v[v2e[i][0]][0]; 
                    case 1:
                        return e2c2v[v2e[i][0]][1];
                    case 2:
                        return e2c2v[v2e[i][0]][2];
                    case 3:
                        return e2c2v[v2e[i][0]][3];
                    case 4:
                        return e2c2v[v2e[i][1]][1];
                    case 5:
                        return e2c2v[v2e[i][1]][2];
                    case 6:
                        return e2c2v[v2e[i][1]][3];
                    default:
                        __builtin_unreachable();
                }
            }).build()),
            v2e2c2v_gt_ctv(v2e2c2v_gt->const_target_view()),
            v2e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 24_c).initializer([&e2ecv, &v2e](int i, int j) { return e2ecv[v2e[i][j/4]][j%4]; }).build()),
            v2e2ecv_gt_ctv(v2e2ecv_gt->const_target_view()),
            nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim),
            interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt()) {};

    nabla4_interpolate_unstructured_inlined_v2v(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        std::vector<std::array<index_type, 6>> v2e,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        std::vector<std::vector<VP_TYPE>> &u_vert,
        std::vector<std::vector<VP_TYPE>> &v_vert,
        std::vector<WP_TYPE> &primal_normal_vert_v1,
        std::vector<WP_TYPE> &primal_normal_vert_v2,
        std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
        std::vector<WP_TYPE> &inv_vert_vert_length,
        std::vector<WP_TYPE> &inv_primal_edge_length,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : v2e2c2v_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 7_c).initializer([&e2c2v, &v2e](int i, int j) {
                switch (j) {
                    case 0:
                        return e2c2v[v2e[i][0]][0]; 
                    case 1:
                        return e2c2v[v2e[i][0]][1];
                    case 2:
                        return e2c2v[v2e[i][0]][2];
                    case 3:
                        return e2c2v[v2e[i][0]][3];
                    case 4:
                        return e2c2v[v2e[i][1]][1];
                    case 5:
                        return e2c2v[v2e[i][1]][2];
                    case 6:
                        return e2c2v[v2e[i][1]][3];
                    default:
                        __builtin_unreachable();
                }
            }).build()),
            v2e2c2v_gt_ctv(v2e2c2v_gt->const_target_view()),
            v2e2ecv_gt(storage::builder<T>.template type<index_type>().dimensions(v2e.size(), 24_c).initializer([&e2ecv, &v2e](int i, int j) { return e2ecv[v2e[i][j/4]][j%4]; }).build()),
            v2e2ecv_gt_ctv(v2e2ecv_gt->const_target_view()),
            nabla4_data(e2c2v,
              e2ecv,
              CellDim,
              VertexDim,
              EdgeDim,
              KDim,
              ECVDim,
              u_vert,
              v_vert,
              primal_normal_vert_v1,
              primal_normal_vert_v2,
              z_nabla2_e,
              inv_vert_vert_length,
              inv_primal_edge_length),
          interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.get_output_gt(), ptr_coeff_1, ptr_coeff_2) {};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    void run_cpu_ifirst() {
        for (index_type k_index{}; k_index < interpolate_data.KDim; ++k_index) {
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type vertex_index = 0; vertex_index < interpolate_data.output_size; ++vertex_index) {
                std::array<WP_TYPE, 6> z_nabla4_e2_wp;
                const std::array<index_type, 24> e2c2v{v2e2c2v_gt_ctv(vertex_index, 0),
                    v2e2c2v_gt_ctv(vertex_index, 1),
                    v2e2c2v_gt_ctv(vertex_index, 2),
                    v2e2c2v_gt_ctv(vertex_index, 3),
                    v2e2c2v_gt_ctv(vertex_index, 1),
                    v2e2c2v_gt_ctv(vertex_index, 4),
                    v2e2c2v_gt_ctv(vertex_index, 5),
                    v2e2c2v_gt_ctv(vertex_index, 6),
                    v2e2c2v_gt_ctv(vertex_index, 3),
                    v2e2c2v_gt_ctv(vertex_index, 1),
                    v2e2c2v_gt_ctv(vertex_index, 0),
                    v2e2c2v_gt_ctv(vertex_index, 6),
                    v2e2c2v_gt_ctv(vertex_index, 1),
                    v2e2c2v_gt_ctv(vertex_index, 5),
                    v2e2c2v_gt_ctv(vertex_index, 2),
                    v2e2c2v_gt_ctv(vertex_index, 4),
                    v2e2c2v_gt_ctv(vertex_index, 1),
                    v2e2c2v_gt_ctv(vertex_index, 6),
                    v2e2c2v_gt_ctv(vertex_index, 4),
                    v2e2c2v_gt_ctv(vertex_index, 3),
                    v2e2c2v_gt_ctv(vertex_index, 2),
                    v2e2c2v_gt_ctv(vertex_index, 1),
                    v2e2c2v_gt_ctv(vertex_index, 5),
                    v2e2c2v_gt_ctv(vertex_index, 0)};
#ifdef __clang__
#pragma clang loop unroll(enable) interleave(enable)
#elif defined(__NVCC__)
#pragma unroll 6
#elif defined(__GNUC__)
#pragma GCC unroll 6
#endif
                for (int i{0}; i < 6; ++i) {
                    const auto E2C2V_0 = e2c2v[i * 4];
                    const auto E2C2V_1 = e2c2v[i * 4 + 1];
                    const auto E2C2V_2 = e2c2v[i * 4 + 2];
                    const auto E2C2V_3 = e2c2v[i * 4 + 3];
                    const auto E2ECV_0 = v2e2ecv_gt_ctv(vertex_index, i * 4);
                    const auto E2ECV_1 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 1);
                    const auto E2ECV_2 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 2);
                    const auto E2ECV_3 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 3);
                    const double nabv_tang_wp =
                        nabla4_data.u_vert_gt_tv(E2C2V_0, k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0) +
                        nabla4_data.v_vert_gt_tv(E2C2V_0, k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0) +
                        nabla4_data.u_vert_gt_tv(E2C2V_1, k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1) +
                        nabla4_data.v_vert_gt_tv(E2C2V_1, k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1);
                    const double nabv_norm_wp =
                        nabla4_data.u_vert_gt_tv(E2C2V_2, k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2) +
                        nabla4_data.v_vert_gt_tv(E2C2V_2, k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2) +
                        nabla4_data.u_vert_gt_tv(E2C2V_3, k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3) +
                        nabla4_data.v_vert_gt_tv(E2C2V_3, k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3);
                    const auto edge_index = interpolate_data.v2e_gt_ctv(vertex_index, i);
                    z_nabla4_e2_wp[i] =
                        4.0 * ((nabv_norm_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                                      (nabla4_data.inv_vert_vert_length_gt_tv(edge_index) *
                                          nabla4_data.inv_vert_vert_length_gt_tv(edge_index)) +
                                  (nabv_tang_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                                      (nabla4_data.inv_primal_edge_length_gt_tv(edge_index) *
                                          nabla4_data.inv_primal_edge_length_gt_tv(edge_index)));
                }
                interpolate_data.p_u_out_gt_tv(vertex_index, k_index) =
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp[0] +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp[1] +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp[2] +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp[3] +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp[4] +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp[5];
                interpolate_data.p_v_out_gt_tv(vertex_index, k_index) =
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp[0] +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp[1] +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp[2] +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp[3] +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp[4] +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp[5];
            }
        }
    }

    void run_cpu_kfirst() {
        for (index_type vertex_index = 0; vertex_index < interpolate_data.output_size; ++vertex_index) {
            std::array<index_type, 6> edge_indexes;
            const std::array<index_type, 6> E2ECV_0{v2e2ecv_gt_ctv(vertex_index, 0),
                v2e2ecv_gt_ctv(vertex_index, 4),
                v2e2ecv_gt_ctv(vertex_index, 8),
                v2e2ecv_gt_ctv(vertex_index, 12),
                v2e2ecv_gt_ctv(vertex_index, 16),
                v2e2ecv_gt_ctv(vertex_index, 20)};
            const std::array<index_type, 6> E2ECV_1{v2e2ecv_gt_ctv(vertex_index, 1),
                v2e2ecv_gt_ctv(vertex_index, 5),
                v2e2ecv_gt_ctv(vertex_index, 9),
                v2e2ecv_gt_ctv(vertex_index, 13),
                v2e2ecv_gt_ctv(vertex_index, 17),
                v2e2ecv_gt_ctv(vertex_index, 21)};
            const std::array<index_type, 6> E2ECV_2{v2e2ecv_gt_ctv(vertex_index, 2),
                v2e2ecv_gt_ctv(vertex_index, 6),
                v2e2ecv_gt_ctv(vertex_index, 10),
                v2e2ecv_gt_ctv(vertex_index, 14),
                v2e2ecv_gt_ctv(vertex_index, 18),
                v2e2ecv_gt_ctv(vertex_index, 22)};
            const std::array<index_type, 6> E2ECV_3{v2e2ecv_gt_ctv(vertex_index, 3),
                v2e2ecv_gt_ctv(vertex_index, 7),
                v2e2ecv_gt_ctv(vertex_index, 11),
                v2e2ecv_gt_ctv(vertex_index, 15),
                v2e2ecv_gt_ctv(vertex_index, 19),
                v2e2ecv_gt_ctv(vertex_index, 23)};
            const std::array<index_type, 24> e2c2v{v2e2c2v_gt_ctv(vertex_index, 0),
                v2e2c2v_gt_ctv(vertex_index, 1),
                v2e2c2v_gt_ctv(vertex_index, 2),
                v2e2c2v_gt_ctv(vertex_index, 3),
                v2e2c2v_gt_ctv(vertex_index, 1),
                v2e2c2v_gt_ctv(vertex_index, 4),
                v2e2c2v_gt_ctv(vertex_index, 5),
                v2e2c2v_gt_ctv(vertex_index, 6),
                v2e2c2v_gt_ctv(vertex_index, 3),
                v2e2c2v_gt_ctv(vertex_index, 1),
                v2e2c2v_gt_ctv(vertex_index, 0),
                v2e2c2v_gt_ctv(vertex_index, 6),
                v2e2c2v_gt_ctv(vertex_index, 1),
                v2e2c2v_gt_ctv(vertex_index, 5),
                v2e2c2v_gt_ctv(vertex_index, 2),
                v2e2c2v_gt_ctv(vertex_index, 4),
                v2e2c2v_gt_ctv(vertex_index, 1),
                v2e2c2v_gt_ctv(vertex_index, 6),
                v2e2c2v_gt_ctv(vertex_index, 4),
                v2e2c2v_gt_ctv(vertex_index, 3),
                v2e2c2v_gt_ctv(vertex_index, 2),
                v2e2c2v_gt_ctv(vertex_index, 1),
                v2e2c2v_gt_ctv(vertex_index, 5),
                v2e2c2v_gt_ctv(vertex_index, 0)};
#ifdef __clang__
#pragma clang loop unroll(enable) vectorize(assume_safety) interleave(enable)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
            for (index_type k_index{}; k_index < interpolate_data.KDim; ++k_index) {
                // Version 1
                // Slightly faster than the inlined version with Apple Clang + GCC 13 on my M3
                //                 std::array<WP_TYPE, 6> z_nabla4_e2_wp;
                // #ifdef __clang__
                // #pragma clang loop unroll(enable) interleave(enable)
                // #elif defined(__GNUC__)
                // #pragma GCC unroll 6
                // #endif
                //                 for (int i{0}; i < 6; ++i) {
                //                     const auto edge_index = interpolate_data.v2e_gt_ctv(vertex_index, i);
                //                     const double nabv_tang_wp = nabla4_data.u_vert_gt_tv(e2c2v[i*4], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[i])
                //                                                     +
                //                                                 nabla4_data.v_vert_gt_tv(e2c2v[i*4], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[i])
                //                                                     +
                //                                                 nabla4_data.u_vert_gt_tv(e2c2v[i*4+1], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[i])
                //                                                     +
                //                                                 nabla4_data.v_vert_gt_tv(e2c2v[i*4+1], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[i]);
                //                     const double nabv_norm_wp = nabla4_data.u_vert_gt_tv(e2c2v[i*4+2], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[i])
                //                                                     +
                //                                                 nabla4_data.v_vert_gt_tv(e2c2v[i*4+2], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[i])
                //                                                     +
                //                                                 nabla4_data.u_vert_gt_tv(e2c2v[i*4+3], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[i])
                //                                                     +
                //                                                 nabla4_data.v_vert_gt_tv(e2c2v[i*4+3], k_index) *
                //                                                     nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[i]);
                //                     z_nabla4_e2_wp[i] =
                //                         4.0 * ((nabv_norm_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index,
                //                         k_index)) *
                //                                         (nabla4_data.inv_vert_vert_length_gt_tv(edge_index) *
                //                                             nabla4_data.inv_vert_vert_length_gt_tv(edge_index)) +
                //                                     (nabv_tang_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index,
                //                                     k_index)) *
                //                                         (nabla4_data.inv_primal_edge_length_gt_tv(edge_index) *
                //                                             nabla4_data.inv_primal_edge_length_gt_tv(edge_index)));
                //                 }
                // Version 2
                // Much slower with GCC 13, with Apple Clang slower than inlined version
                //                 const std::array<WP_TYPE, 6> z_nabla4_e2_wp = [&]() {
                //                     std::array<WP_TYPE, 6> z_nabla4_e2_wp;
                // #ifdef __clang__
                // #pragma clang loop unroll(enable) interleave(enable)
                // #elif defined(__GNUC__)
                // #pragma GCC unroll 6
                // #endif
                //                     for (int i{0}; i < 6; ++i) {
                //                         const auto edge_index = interpolate_data.v2e_gt_ctv(vertex_index, i);
                //                         const double nabv_tang_wp = nabla4_data.u_vert_gt_tv(e2c2v[i*4], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[i])
                //                                                         +
                //                                                     nabla4_data.v_vert_gt_tv(e2c2v[i*4], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[i])
                //                                                         +
                //                                                     nabla4_data.u_vert_gt_tv(e2c2v[i*4+1], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[i])
                //                                                         +
                //                                                     nabla4_data.v_vert_gt_tv(e2c2v[i*4+1], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[i]);
                //                         const double nabv_norm_wp = nabla4_data.u_vert_gt_tv(e2c2v[i*4+2], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[i])
                //                                                         +
                //                                                     nabla4_data.v_vert_gt_tv(e2c2v[i*4+2], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[i])
                //                                                         +
                //                                                     nabla4_data.u_vert_gt_tv(e2c2v[i*4+3], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[i])
                //                                                         +
                //                                                     nabla4_data.v_vert_gt_tv(e2c2v[i*4+3], k_index) *
                //                                                         nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[i]);
                //                         z_nabla4_e2_wp[i] =
                //                             4.0 * ((nabv_norm_wp - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index,
                //                             k_index)) *
                //                                             (nabla4_data.inv_vert_vert_length_gt_tv(edge_index) *
                //                                                 nabla4_data.inv_vert_vert_length_gt_tv(edge_index)) +
                //                                         (nabv_tang_wp - 2.0 *
                //                                         nabla4_data.z_nabla2_e_gt_tv(edge_index, k_index)) *
                //                                             (nabla4_data.inv_primal_edge_length_gt_tv(edge_index) *
                //                                                 nabla4_data.inv_primal_edge_length_gt_tv(edge_index)));
                //                     }
                //                     return z_nabla4_e2_wp;
                //                 }();
                // Version 3
                // Much slower with GCC 13 compared to the inlined version. Much faster with Apple Clang
                const auto edge_index_0 = interpolate_data.v2e_gt_ctv(vertex_index, 0);
                const double nabv_tang_wp_0 =
                    nabla4_data.u_vert_gt_tv(e2c2v[0], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[0]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[0], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[0]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[1], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[0]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[1], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[0]);
                const double nabv_norm_wp_0 =
                    nabla4_data.u_vert_gt_tv(e2c2v[2], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[0]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[2], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[0]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[3], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[0]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[3], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[0]);
                const auto z_nabla4_e2_wp_0 =
                    4.0 * ((nabv_norm_wp_0 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_0, k_index)) *
                                  (nabla4_data.inv_vert_vert_length_gt_tv(edge_index_0) *
                                      nabla4_data.inv_vert_vert_length_gt_tv(edge_index_0)) +
                              (nabv_tang_wp_0 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_0, k_index)) *
                                  (nabla4_data.inv_primal_edge_length_gt_tv(edge_index_0) *
                                      nabla4_data.inv_primal_edge_length_gt_tv(edge_index_0)));
                const auto edge_index_1 = interpolate_data.v2e_gt_ctv(vertex_index, 1);
                const double nabv_tang_wp_1 =
                    nabla4_data.u_vert_gt_tv(e2c2v[4], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[1]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[4], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[1]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[5], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[1]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[5], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[1]);
                const double nabv_norm_wp_1 =
                    nabla4_data.u_vert_gt_tv(e2c2v[6], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[1]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[6], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[1]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[7], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[1]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[7], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[1]);
                const auto z_nabla4_e2_wp_1 =
                    4.0 * ((nabv_norm_wp_1 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_1, k_index)) *
                                  (nabla4_data.inv_vert_vert_length_gt_tv(edge_index_1) *
                                      nabla4_data.inv_vert_vert_length_gt_tv(edge_index_1)) +
                              (nabv_tang_wp_1 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_1, k_index)) *
                                  (nabla4_data.inv_primal_edge_length_gt_tv(edge_index_1) *
                                      nabla4_data.inv_primal_edge_length_gt_tv(edge_index_1)));
                const auto edge_index_2 = interpolate_data.v2e_gt_ctv(vertex_index, 2);
                const double nabv_tang_wp_2 =
                    nabla4_data.u_vert_gt_tv(e2c2v[8], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[2]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[8], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[2]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[9], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[2]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[9], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[2]);
                const double nabv_norm_wp_2 =
                    nabla4_data.u_vert_gt_tv(e2c2v[10], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[2]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[10], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[2]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[11], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[2]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[11], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[2]);
                const auto z_nabla4_e2_wp_2 =
                    4.0 * ((nabv_norm_wp_2 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_2, k_index)) *
                                  (nabla4_data.inv_vert_vert_length_gt_tv(edge_index_2) *
                                      nabla4_data.inv_vert_vert_length_gt_tv(edge_index_2)) +
                              (nabv_tang_wp_2 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_2, k_index)) *
                                  (nabla4_data.inv_primal_edge_length_gt_tv(edge_index_2) *
                                      nabla4_data.inv_primal_edge_length_gt_tv(edge_index_2)));
                const auto edge_index_3 = interpolate_data.v2e_gt_ctv(vertex_index, 3);
                const double nabv_tang_wp_3 =
                    nabla4_data.u_vert_gt_tv(e2c2v[12], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[3]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[12], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[3]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[13], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[3]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[13], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[3]);
                const double nabv_norm_wp_3 =
                    nabla4_data.u_vert_gt_tv(e2c2v[14], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[3]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[14], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[3]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[15], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[3]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[15], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[3]);
                const auto z_nabla4_e2_wp_3 =
                    4.0 * ((nabv_norm_wp_3 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_3, k_index)) *
                                  (nabla4_data.inv_vert_vert_length_gt_tv(edge_index_3) *
                                      nabla4_data.inv_vert_vert_length_gt_tv(edge_index_3)) +
                              (nabv_tang_wp_3 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_3, k_index)) *
                                  (nabla4_data.inv_primal_edge_length_gt_tv(edge_index_3) *
                                      nabla4_data.inv_primal_edge_length_gt_tv(edge_index_3)));
                const auto edge_index_4 = interpolate_data.v2e_gt_ctv(vertex_index, 4);
                const double nabv_tang_wp_4 =
                    nabla4_data.u_vert_gt_tv(e2c2v[16], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[4]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[16], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[4]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[17], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[4]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[17], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[4]);
                const double nabv_norm_wp_4 =
                    nabla4_data.u_vert_gt_tv(e2c2v[18], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[4]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[18], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[4]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[19], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[4]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[19], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[4]);
                const auto z_nabla4_e2_wp_4 =
                    4.0 * ((nabv_norm_wp_4 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_4, k_index)) *
                                  (nabla4_data.inv_vert_vert_length_gt_tv(edge_index_4) *
                                      nabla4_data.inv_vert_vert_length_gt_tv(edge_index_4)) +
                              (nabv_tang_wp_4 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_4, k_index)) *
                                  (nabla4_data.inv_primal_edge_length_gt_tv(edge_index_4) *
                                      nabla4_data.inv_primal_edge_length_gt_tv(edge_index_4)));
                const auto edge_index_5 = interpolate_data.v2e_gt_ctv(vertex_index, 5);
                const double nabv_tang_wp_5 =
                    nabla4_data.u_vert_gt_tv(e2c2v[20], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_0[5]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[20], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_0[5]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[21], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_1[5]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[21], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_1[5]);
                const double nabv_norm_wp_5 =
                    nabla4_data.u_vert_gt_tv(e2c2v[22], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_2[5]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[22], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_2[5]) +
                    nabla4_data.u_vert_gt_tv(e2c2v[23], k_index) * nabla4_data.primal_normal_vert_v1_gt_tv(E2ECV_3[5]) +
                    nabla4_data.v_vert_gt_tv(e2c2v[23], k_index) * nabla4_data.primal_normal_vert_v2_gt_tv(E2ECV_3[5]);
                const auto z_nabla4_e2_wp_5 =
                    4.0 * ((nabv_norm_wp_5 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_5, k_index)) *
                                  (nabla4_data.inv_vert_vert_length_gt_tv(edge_index_5) *
                                      nabla4_data.inv_vert_vert_length_gt_tv(edge_index_5)) +
                              (nabv_tang_wp_5 - 2.0 * nabla4_data.z_nabla2_e_gt_tv(edge_index_5, k_index)) *
                                  (nabla4_data.inv_primal_edge_length_gt_tv(edge_index_5) *
                                      nabla4_data.inv_primal_edge_length_gt_tv(edge_index_5)));
                interpolate_data.p_u_out_gt_tv(vertex_index, k_index) =
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp_0 +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp_1 +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp_2 +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp_3 +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp_4 +
                    interpolate_data.ptr_coeff_1_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp_5;
                interpolate_data.p_v_out_gt_tv(vertex_index, k_index) =
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp_0 +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp_1 +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp_2 +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp_3 +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp_4 +
                    interpolate_data.ptr_coeff_2_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp_5;
            }
        }
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
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_kloop() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_kloop<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_kloop<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_kloop<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_kloop<int>() {
    return {256, 1, 1, 256};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_inlined_v2v_kloop =
    get_block_dims_unstructured_nabla_interpol_inlined_v2v_kloop<index_type>();

__global__ void __launch_bounds__(block_dims_unstructured_nabla_interpol_inlined_v2v_kloop.size)
    run_gpu_kloop_nabla4_interpolate_inlined_v2v_unstructured(index_type nabla4_output_size,
        index_type interpolate_output_size,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        nabla4_interpolate_unstructured_inlined_v2v<storage::gpu>::neighbors_e2c2v_gt_ctv_t v2e2c2v_gt_ctv,
        nabla4_interpolate_unstructured_inlined_v2v<storage::gpu>::neighbors_e2ecv_gt_ctv_t v2e2ecv_gt_ctv,
        interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_ctv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
        nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
        interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_index >= interpolate_output_size)
        return;
    const std::array<index_type, 24> v2e2c2v{v2e2c2v_gt_ctv(vertex_index, 0),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 6),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 0),
        v2e2c2v_gt_ctv(vertex_index, 6),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 6),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 0)};
    const std::array<index_type, 24> v2e2ecv{v2e2ecv_gt_ctv(vertex_index, 0),
        v2e2ecv_gt_ctv(vertex_index, 1),
        v2e2ecv_gt_ctv(vertex_index, 2),
        v2e2ecv_gt_ctv(vertex_index, 3),
        v2e2ecv_gt_ctv(vertex_index, 4),
        v2e2ecv_gt_ctv(vertex_index, 5),
        v2e2ecv_gt_ctv(vertex_index, 6),
        v2e2ecv_gt_ctv(vertex_index, 7),
        v2e2ecv_gt_ctv(vertex_index, 8),
        v2e2ecv_gt_ctv(vertex_index, 9),
        v2e2ecv_gt_ctv(vertex_index, 10),
        v2e2ecv_gt_ctv(vertex_index, 11),
        v2e2ecv_gt_ctv(vertex_index, 12),
        v2e2ecv_gt_ctv(vertex_index, 13),
        v2e2ecv_gt_ctv(vertex_index, 14),
        v2e2ecv_gt_ctv(vertex_index, 15),
        v2e2ecv_gt_ctv(vertex_index, 16),
        v2e2ecv_gt_ctv(vertex_index, 17),
        v2e2ecv_gt_ctv(vertex_index, 18),
        v2e2ecv_gt_ctv(vertex_index, 19),
        v2e2ecv_gt_ctv(vertex_index, 20),
        v2e2ecv_gt_ctv(vertex_index, 21),
        v2e2ecv_gt_ctv(vertex_index, 22),
        v2e2ecv_gt_ctv(vertex_index, 23)};
    const std::array<index_type, 6> v2e{v2e_gt_ctv(vertex_index, 0),
        v2e_gt_ctv(vertex_index, 1),
        v2e_gt_ctv(vertex_index, 2),
        v2e_gt_ctv(vertex_index, 3),
        v2e_gt_ctv(vertex_index, 4),
        v2e_gt_ctv(vertex_index, 5)};
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
    const WP_TYPE ptr_coeff_1[6] = {ptr_coeff_1_gt_ctv(vertex_index, 0),
        ptr_coeff_1_gt_ctv(vertex_index, 1),
        ptr_coeff_1_gt_ctv(vertex_index, 2),
        ptr_coeff_1_gt_ctv(vertex_index, 3),
        ptr_coeff_1_gt_ctv(vertex_index, 4),
        ptr_coeff_1_gt_ctv(vertex_index, 5)};
    const WP_TYPE ptr_coeff_2[6] = {ptr_coeff_2_gt_ctv(vertex_index, 0),
        ptr_coeff_2_gt_ctv(vertex_index, 1),
        ptr_coeff_2_gt_ctv(vertex_index, 2),
        ptr_coeff_2_gt_ctv(vertex_index, 3),
        ptr_coeff_2_gt_ctv(vertex_index, 4),
        ptr_coeff_2_gt_ctv(vertex_index, 5)};
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    for (auto k_index{blockIdx.z * blockDim.z + threadIdx.z}; k_index < KDim; k_index += gridDim.z * blockDim.z) {
#pragma unroll
        for (auto i{0}; i < 6; ++i) {
            const auto edge_index = v2e[i];
            const auto E2C2V_0 = v2e2c2v[i * 4];
            const auto E2C2V_1 = v2e2c2v[i * 4 + 1];
            const auto E2C2V_2 = v2e2c2v[i * 4 + 2];
            const auto E2C2V_3 = v2e2c2v[i * 4 + 3];
            const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1[4 * i] +
                                        v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2[4 * i] +
                                        u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1[4 * i + 1] +
                                        v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2[4 * i + 1];
            const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1[4 * i + 2] +
                                        v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2[4 * i + 2] +
                                        u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1[4 * i + 3] +
                                        v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2[4 * i + 3];
            const auto z_nabla4_e = z_nabla2_e_gt_tv(edge_index, k_index);
            z_nabla4_e2_wp[i] =
                4.0 * ((nabv_norm_wp - 2.0 * z_nabla4_e) * (inv_vert_vert_length[i] * inv_vert_vert_length[i]) +
                          (nabv_tang_wp - 2.0 * z_nabla4_e) * (inv_primal_edge_length[i] * inv_primal_edge_length[i]));
        }
        p_u_out_gt_tv(vertex_index, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_1[0] + z_nabla4_e2_wp[1] * ptr_coeff_1[1] +
                                               z_nabla4_e2_wp[2] * ptr_coeff_1[2] + z_nabla4_e2_wp[3] * ptr_coeff_1[3] +
                                               z_nabla4_e2_wp[4] * ptr_coeff_1[4] + z_nabla4_e2_wp[5] * ptr_coeff_1[5];
        p_v_out_gt_tv(vertex_index, k_index) = z_nabla4_e2_wp[0] * ptr_coeff_2[0] + z_nabla4_e2_wp[1] * ptr_coeff_2[1] +
                                               z_nabla4_e2_wp[2] * ptr_coeff_2[2] + z_nabla4_e2_wp[3] * ptr_coeff_2[3] +
                                               z_nabla4_e2_wp[4] * ptr_coeff_2[4] + z_nabla4_e2_wp[5] * ptr_coeff_2[5];
    }
};

template <typename T>
inline void nabla4_interpolate_unstructured_inlined_v2v<T>::run_gpu_kloop_helper() {
    dim3 tblocks(block_dims_unstructured_nabla_interpol_inlined_v2v_kloop.x,
        block_dims_unstructured_nabla_interpol_inlined_v2v_kloop.y,
        block_dims_unstructured_nabla_interpol_inlined_v2v_kloop.z);
    dim3 grid((interpolate_data.output_size + tblocks.x - 1) / tblocks.x, 1, 1);
    run_gpu_kloop_nabla4_interpolate_inlined_v2v_unstructured<<<grid, tblocks>>>(nabla4_data.output_size,
        interpolate_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        v2e2c2v_gt_ctv,
        v2e2ecv_gt_ctv,
        interpolate_data.v2e_gt_ctv,
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
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_naive() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_naive<std::size_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_naive<std::int64_t>() {
    // not optimized
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_naive<std::uint32_t>() {
    // not optimized
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_unstructured_nabla_interpol_inlined_v2v_naive<int>() {
    return {32, 16, 1, 512};
};

constexpr block_dims block_dims_unstructured_nabla_interpol_inlined_v2v_naive =
    get_block_dims_unstructured_nabla_interpol_inlined_v2v_naive<index_type>();

__global__ void __maxnreg__(80) run_gpu_naive_nabla4_interpolate_inlined_v2v_unstructured(index_type nabla4_output_size,
    index_type interpolate_output_size,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    nabla4_interpolate_unstructured_inlined_v2v<storage::gpu>::neighbors_e2c2v_gt_ctv_t v2e2c2v_gt_ctv,
    nabla4_interpolate_unstructured_inlined_v2v<storage::gpu>::neighbors_e2ecv_gt_ctv_t v2e2ecv_gt_ctv,
    interpolate_unstructured<storage::gpu>::neighbors_gt_ctv_t v2e_gt_ctv,
    nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t u_vert_gt_tv,
    nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_VP_t v_vert_gt_tv,
    nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v1_gt_tv,
    nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t primal_normal_vert_v2_gt_tv,
    nabla4_unstructured_gt<storage::gpu>::data_store_2d_ctv_WP_t z_nabla2_e_gt_tv,
    nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_vert_vert_length_gt_tv,
    nabla4_unstructured_gt<storage::gpu>::data_store_1d_ctv_WP_t inv_primal_edge_length_gt_tv,
    interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_1_gt_ctv,
    interpolate_unstructured<storage::gpu>::data_store_2d_coef_ctv_WP_t ptr_coeff_2_gt_ctv,
    interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_u_out_gt_tv,
    interpolate_unstructured<storage::gpu>::data_store_2d_tv_WP_t p_v_out_gt_tv) {
    const auto vertex_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto k_index = blockIdx.y * blockDim.y + threadIdx.y;
    if (vertex_index >= interpolate_output_size || k_index >= KDim)
        return;
    std::array<WP_TYPE, 6> z_nabla4_e2_wp;
    const std::array<index_type, 24> e2c2v{v2e2c2v_gt_ctv(vertex_index, 0),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 6),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 0),
        v2e2c2v_gt_ctv(vertex_index, 6),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 6),
        v2e2c2v_gt_ctv(vertex_index, 4),
        v2e2c2v_gt_ctv(vertex_index, 3),
        v2e2c2v_gt_ctv(vertex_index, 2),
        v2e2c2v_gt_ctv(vertex_index, 1),
        v2e2c2v_gt_ctv(vertex_index, 5),
        v2e2c2v_gt_ctv(vertex_index, 0)};
#pragma unroll 6
    for (int i{0}; i < 6; ++i) {
        const auto E2C2V_0 = e2c2v[i * 4];
        const auto E2C2V_1 = e2c2v[i * 4 + 1];
        const auto E2C2V_2 = e2c2v[i * 4 + 2];
        const auto E2C2V_3 = e2c2v[i * 4 + 3];
        const auto E2ECV_0 = v2e2ecv_gt_ctv(vertex_index, i * 4);
        const auto E2ECV_1 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 1);
        const auto E2ECV_2 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 2);
        const auto E2ECV_3 = v2e2ecv_gt_ctv(vertex_index, i * 4 + 3);
        const double nabv_tang_wp = u_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_0) +
                                    v_vert_gt_tv(E2C2V_0, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_0) +
                                    u_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_1) +
                                    v_vert_gt_tv(E2C2V_1, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_1);
        const double nabv_norm_wp = u_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_2) +
                                    v_vert_gt_tv(E2C2V_2, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_2) +
                                    u_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v1_gt_tv(E2ECV_3) +
                                    v_vert_gt_tv(E2C2V_3, k_index) * primal_normal_vert_v2_gt_tv(E2ECV_3);
        const auto edge_index = v2e_gt_ctv(vertex_index, i);
        z_nabla4_e2_wp[i] =
            4.0 * ((nabv_norm_wp - 2.0 * z_nabla2_e_gt_tv(edge_index, k_index)) *
                          (inv_vert_vert_length_gt_tv(edge_index) * inv_vert_vert_length_gt_tv(edge_index)) +
                      (nabv_tang_wp - 2.0 * z_nabla2_e_gt_tv(edge_index, k_index)) *
                          (inv_primal_edge_length_gt_tv(edge_index) * inv_primal_edge_length_gt_tv(edge_index)));
    }
    p_u_out_gt_tv(vertex_index, k_index) = ptr_coeff_1_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp[0] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp[1] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp[2] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp[3] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp[4] +
                                           ptr_coeff_1_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp[5];
    p_v_out_gt_tv(vertex_index, k_index) = ptr_coeff_2_gt_ctv(vertex_index, 0) * z_nabla4_e2_wp[0] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 1) * z_nabla4_e2_wp[1] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 2) * z_nabla4_e2_wp[2] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 3) * z_nabla4_e2_wp[3] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 4) * z_nabla4_e2_wp[4] +
                                           ptr_coeff_2_gt_ctv(vertex_index, 5) * z_nabla4_e2_wp[5];
};

template <typename T>
inline void nabla4_interpolate_unstructured_inlined_v2v<T>::run_gpu_naive_helper() {
    dim3 tblocks(block_dims_unstructured_nabla_interpol_inlined_v2v_naive.x,
        block_dims_unstructured_nabla_interpol_inlined_v2v_naive.y,
        block_dims_unstructured_nabla_interpol_inlined_v2v_naive.z);
    dim3 grid((interpolate_data.output_size + tblocks.x - 1) / tblocks.x,
        (interpolate_data.KDim + tblocks.y - 1) / tblocks.y,
        1);
    run_gpu_naive_nabla4_interpolate_inlined_v2v_unstructured<<<grid, tblocks>>>(nabla4_data.output_size,
        interpolate_data.output_size,
        nabla4_data.CellDim,
        interpolate_data.VertexDim,
        interpolate_data.EdgeDim,
        interpolate_data.KDim,
        v2e2c2v_gt_ctv,
        v2e2ecv_gt_ctv,
        interpolate_data.v2e_gt_ctv,
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
inline void nabla4_interpolate_unstructured_inlined_v2v<T>::run_gpu_kloop_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
template <typename T>
inline void nabla4_interpolate_unstructured_inlined_v2v<T>::run_gpu_naive_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
