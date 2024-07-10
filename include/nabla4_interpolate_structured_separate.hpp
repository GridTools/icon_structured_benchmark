#include <interpolate_structured_gridtools.hpp>
#include <nabla4_structured_torus_gridtools_halo.hpp>

template <typename T>
struct nabla4_interpolate_structured_separate {
    nabla4_structured_torus_halo_gt<T> nabla4_data;
    interpolate_structured<T> interpolate_data;

  public:
    nabla4_interpolate_structured_separate(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, halo),
          interpolate_data(VertexDim, EdgeDim, KDim, y_dim - 2 * halo, x_dim - 2 * halo, 1){};

    nabla4_interpolate_structured_separate(index_type CellDim,
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
              halo,
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
              y_dim - 2 * halo,
              x_dim - 2 * halo,
              1,
              nabla4_data.get_output_gt(),
              ptr_coeff_1,
              ptr_coeff_2){};

    auto get_output() -> decltype(interpolate_data.get_output()) { return interpolate_data.get_output(); }

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst || I == backend_impl::cpu_kfirst || I == backend_impl::gpu_naive ||
                      I == backend_impl::gpu_kloop) {
            nabla4_data.template run<I>();
            interpolate_data.template run<I>();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};
