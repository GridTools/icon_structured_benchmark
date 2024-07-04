#include <nabla4_gridtools.hpp>
#include <mo_intp_rbf_rbf_vec_interpol_vertex.hpp>

template <typename T>
struct nabla4_interpolate_unstructured_naive_separate {
    nabla4_unstructured_gt_naive<T> nabla4_data;
    interpolate_unstructured_naive<T> interpolate_data;
  public:
    nabla4_interpolate_unstructured_naive_separate(std::vector<std::array<index_type, 4>> e2c2v,
        std::vector<std::array<index_type, 4>> e2ecv,
        std::vector<std::array<index_type, 6>> v2e,
        index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim)
        : nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim),
        interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.z_nabla4_e2_wp_gt)
    {};

    nabla4_interpolate_unstructured_naive_separate(std::vector<std::array<index_type, 4>> e2c2v,
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
        std::vector<std::vector<WP_TYPE>> &p_e_in,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : nabla4_data(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2, z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length),
        interpolate_data(v2e, VertexDim, EdgeDim, KDim, nabla4_data.z_nabla4_e2_wp_gt, ptr_coeff_1, ptr_coeff_2)
    {};

    auto get_output() -> decltype(interpolate_data.get_output()) {
        return interpolate_data.get_output();
    }

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            std::cout << "Running CPU IFirst" << std::endl;
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            std::cout << "Running CPU KFirst" << std::endl;
        } else if constexpr (I == backend_impl::gpu) {
            std::cout << "Running GPU" << std::endl;
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
}