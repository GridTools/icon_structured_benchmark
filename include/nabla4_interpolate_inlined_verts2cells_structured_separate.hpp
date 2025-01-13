#pragma once

#include <utility>

#include <nabla4_interpolate_structured_inlined.hpp>
#include <verts2cells_structured_gridtools.hpp>

template <typename T>
struct nabla4_interpolate_inlined_verts2cells_structured_separate {
    nabla4_interpolate_structured_inlined<T> nabla4_interpolate_data;
    verts2cells_structured<T> verts2cells_data;

  public:
    nabla4_interpolate_inlined_verts2cells_structured_separate(index_type CellDim,
        index_type VertexDim,
        index_type EdgeDim,
        index_type KDim,
        index_type ECVDim,
        index_type y_dim,
        index_type x_dim,
        index_type halo)
        : nabla4_interpolate_data(CellDim, VertexDim, EdgeDim, KDim, ECVDim, y_dim, x_dim, halo),
          verts2cells_data(VertexDim,
              CellDim,
              KDim,
              y_dim - 2 * (halo + 1),
              x_dim - 2 * (halo + 1),
              1,
              std::get<0>(nabla4_interpolate_data.get_output_gt()),
              std::get<1>(nabla4_interpolate_data.get_output_gt())){};

    nabla4_interpolate_inlined_verts2cells_structured_separate(index_type CellDim,
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
        const std::vector<std::vector<WP_TYPE>> &ptr_coeff_2,
        const std::vector<std::vector<WP_TYPE>> &ptr_c_coeff_1,
        const std::vector<std::vector<WP_TYPE>> &ptr_c_coeff_2)
        : nabla4_interpolate_data(CellDim,
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
              inv_primal_edge_length,
              ptr_coeff_1,
              ptr_coeff_2),
          verts2cells_data(VertexDim,
              CellDim,
              KDim,
              y_dim - 2 * (halo + 1),
              x_dim - 2 * (halo + 1),
              1,
              std::get<0>(nabla4_interpolate_data.get_output_gt()),
              std::get<1>(nabla4_interpolate_data.get_output_gt()),
              ptr_c_coeff_1,
              ptr_c_coeff_2){};

    auto get_output() -> decltype(verts2cells_data.get_output()) { return verts2cells_data.get_output(); }

    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::gpu_kloop) {
            nabla4_interpolate_data.template run<I>();
            verts2cells_data.template run<I>();
        } else if constexpr (I == backend_impl::gpu_naive) {
            nabla4_interpolate_data.template run<I>();
            verts2cells_data.template run<I>();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};
