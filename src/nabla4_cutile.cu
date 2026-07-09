#include "cuda_tile.h"
#include "nabla4_gridtools.hpp"

namespace ct = cuda::tiles;
using namespace ct::literals;

#define TILE_I 16
#define TILE_J 16

// Token-paste TILE_I/TILE_J into _ic literals so the tile size is defined
// once and stays in sync between the NTTP and brace-init forms below.
#define CT_IC_(x) x##_ic
#define CT_IC(x) CT_IC_(x)

using VTile = ct::tile<VP_TYPE, ct::shape<TILE_J, TILE_I>>;
using WTile = ct::tile<WP_TYPE, ct::shape<TILE_J, TILE_I>>;

// One color's worth of the 4 E2C2V vertex "slots". Slot 0 is always the
// cell center for every color; slots 1-3 are selected by the caller
// per-color (see the three call sites at the bottom of the kernel).
struct ColorNeighbors {
    VTile u0, v0, u1, v1, u2, v2, u3, v3;
};

template <class OutView>
__tile__ void process_color(
    ColorNeighbors const& nb,
    WTile const& pn1_0, WTile const& pn1_1, WTile const& pn1_2, WTile const& pn1_3,
    WTile const& pn2_0, WTile const& pn2_1, WTile const& pn2_2, WTile const& pn2_3,
    WTile const& z_nabla2_e, WTile const& inv_vv_len, WTile const& inv_pe_len,
    OutView& out_view, int by, int bx)
{
    auto nabv_tang = nb.u0 * pn1_0 + nb.v0 * pn2_0 + nb.u1 * pn1_1 + nb.v1 * pn2_1;
    auto nabv_norm = nb.u2 * pn1_2 + nb.v2 * pn2_2 + nb.u3 * pn1_3 + nb.v3 * pn2_3;
    auto inv_vv_sqr = inv_vv_len * inv_vv_len;
    auto inv_pe_sqr = inv_pe_len * inv_pe_len;
    auto z_nabla4_e2 = 4.0 * ((nabv_norm - 2.0 * z_nabla2_e) * inv_vv_sqr +
                              (nabv_tang - 2.0 * z_nabla2_e) * inv_pe_sqr);
    out_view.store_masked(z_nabla4_e2, by, bx);  // only if types already match
}

__tile_global__ void run_cutile_nabla4_structured(
    index_type KDim,
    index_type x_dim,
    index_type x_dim_inner,
    index_type y_dim,
    index_type y_dim_inner,
    index_type halo,
    index_type total_grid_size,
    index_type inner_grid_size,
    index_type global_edges_per_orientation,
    const VP_TYPE* __restrict__ u_vert_ptr,
    const VP_TYPE* __restrict__ v_vert_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v1_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v2_ptr,
    const WP_TYPE* __restrict__ z_nabla2_e_ptr,
    const WP_TYPE* __restrict__ inv_vert_vert_length_ptr,
    const WP_TYPE* __restrict__ inv_primal_edge_length_ptr,
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {

    u_vert_ptr                 = ct::assume_aligned(u_vert_ptr, 16_ic);
    v_vert_ptr                 = ct::assume_aligned(v_vert_ptr, 16_ic);
    primal_normal_vert_v1_ptr  = ct::assume_aligned(primal_normal_vert_v1_ptr, 16_ic);
    primal_normal_vert_v2_ptr  = ct::assume_aligned(primal_normal_vert_v2_ptr, 16_ic);
    z_nabla2_e_ptr              = ct::assume_aligned(z_nabla2_e_ptr, 16_ic);
    inv_vert_vert_length_ptr   = ct::assume_aligned(inv_vert_vert_length_ptr, 16_ic);
    inv_primal_edge_length_ptr = ct::assume_aligned(inv_primal_edge_length_ptr, 16_ic);
    z_nabla4_e2_wp_ptr          = ct::assume_aligned(z_nabla4_e2_wp_ptr, 16_ic);

    auto [bx, by, bz] = ct::bid();
    const int k_index = bz; // launch grid.z == KDim: one block per k-level

    const index_type vertex_plane_size = x_dim * y_dim;
    const index_type edge_plane_size = 3 * inner_grid_size;
    const index_type base_off = halo * x_dim + halo; // aligns tile (0,0) with first active vertex

    auto vertex_extents = ct::extents{y_dim_inner, x_dim_inner};
    auto inner_extents  = ct::extents{y_dim_inner, x_dim_inner};
    auto tile_shape      = ct::shape{CT_IC(TILE_J), CT_IC(TILE_I)};

    const VP_TYPE* u_k = u_vert_ptr + k_index * vertex_plane_size;
    const VP_TYPE* v_k = v_vert_ptr + k_index * vertex_plane_size;

    auto load_vview = [&](const VP_TYPE* p, index_type extra_off) {
        return ct::partition_view{ct::tensor_span{p + base_off + extra_off, vertex_extents}, tile_shape}
            .load_masked(by, bx);
    };

    // --- the 6 structured vertex neighbors (replaces the 12+ scattered
    // global loads in the SIMT kernel with 12 structured tile loads) ------
    auto u_center = load_vview(u_k, 0);
    auto v_center = load_vview(v_k, 0);
    auto u_n  = load_vview(u_k,  x_dim);
    auto v_n  = load_vview(v_k,  x_dim);
    auto u_s  = load_vview(u_k, -x_dim);
    auto v_s  = load_vview(v_k, -x_dim);
    auto u_e  = load_vview(u_k,  1);
    auto v_e  = load_vview(v_k,  1);
    auto u_nw = load_vview(u_k,  x_dim - 1);
    auto v_nw = load_vview(v_k,  x_dim - 1);
    auto u_se = load_vview(u_k, -x_dim + 1);
    auto v_se = load_vview(v_k, -x_dim + 1);

    auto load_pn = [&](const WP_TYPE* p, int color, int slot) {
        index_type off = base_off + color * total_grid_size + slot * global_edges_per_orientation;
        return ct::partition_view{ct::tensor_span{p + off, inner_extents}, tile_shape}.load_masked(by, bx);
    };
    auto load_len = [&](const WP_TYPE* p, int color) {
        return ct::partition_view{ct::tensor_span{p + color * inner_grid_size, inner_extents}, tile_shape}
            .load_masked(by, bx);
    };
    auto load_edge2d = [&](const WP_TYPE* p, int color) {
        const WP_TYPE* p_k = p + k_index * edge_plane_size;
        return ct::partition_view{ct::tensor_span{p_k + color * inner_grid_size, inner_extents}, tile_shape}
            .load_masked(by, bx);
    };
    auto out_view = [&](int color) {
        VP_TYPE* p_k = z_nabla4_e2_wp_ptr + k_index * edge_plane_size;
        return ct::partition_view{ct::tensor_span{p_k + color * inner_grid_size, inner_extents}, tile_shape};
    };

    // color 0: slots 1,2,3 = N, NW, E  (matches E2C2V_1[0]=i_jp1, E2C2V_2[0]=im1_jp1, E2C2V_3[0]=ip1_j)
    {
        ColorNeighbors nb{u_center, v_center, u_n, v_n, u_nw, v_nw, u_e, v_e};
        auto ov = out_view(0);
        process_color(nb,
            load_pn(primal_normal_vert_v1_ptr, 0, 0), load_pn(primal_normal_vert_v1_ptr, 0, 1),
            load_pn(primal_normal_vert_v1_ptr, 0, 2), load_pn(primal_normal_vert_v1_ptr, 0, 3),
            load_pn(primal_normal_vert_v2_ptr, 0, 0), load_pn(primal_normal_vert_v2_ptr, 0, 1),
            load_pn(primal_normal_vert_v2_ptr, 0, 2), load_pn(primal_normal_vert_v2_ptr, 0, 3),
            load_edge2d(z_nabla2_e_ptr, 0), load_len(inv_vert_vert_length_ptr, 0),
            load_len(inv_primal_edge_length_ptr, 0), ov, by, bx);
    }
    // color 1: slots 1,2,3 = E, N, SE  (E2C2V_1[1]=ip1_j, E2C2V_2[1]=i_jp1, E2C2V_3[1]=ip1_jm1)
    {
        ColorNeighbors nb{u_center, v_center, u_e, v_e, u_n, v_n, u_se, v_se};
        auto ov = out_view(1);
        process_color(nb,
            load_pn(primal_normal_vert_v1_ptr, 1, 0), load_pn(primal_normal_vert_v1_ptr, 1, 1),
            load_pn(primal_normal_vert_v1_ptr, 1, 2), load_pn(primal_normal_vert_v1_ptr, 1, 3),
            load_pn(primal_normal_vert_v2_ptr, 1, 0), load_pn(primal_normal_vert_v2_ptr, 1, 1),
            load_pn(primal_normal_vert_v2_ptr, 1, 2), load_pn(primal_normal_vert_v2_ptr, 1, 3),
            load_edge2d(z_nabla2_e_ptr, 1), load_len(inv_vert_vert_length_ptr, 1),
            load_len(inv_primal_edge_length_ptr, 1), ov, by, bx);
    }
    // color 2: slots 1,2,3 = SE, E, S  (E2C2V_1[2]=ip1_jm1, E2C2V_2[2]=ip1_j, E2C2V_3[2]=i_jm1)
    {
        ColorNeighbors nb{u_center, v_center, u_se, v_se, u_e, v_e, u_s, v_s};
        auto ov = out_view(2);
        process_color(nb,
            load_pn(primal_normal_vert_v1_ptr, 2, 0), load_pn(primal_normal_vert_v1_ptr, 2, 1),
            load_pn(primal_normal_vert_v1_ptr, 2, 2), load_pn(primal_normal_vert_v1_ptr, 2, 3),
            load_pn(primal_normal_vert_v2_ptr, 2, 0), load_pn(primal_normal_vert_v2_ptr, 2, 1),
            load_pn(primal_normal_vert_v2_ptr, 2, 2), load_pn(primal_normal_vert_v2_ptr, 2, 3),
            load_edge2d(z_nabla2_e_ptr, 2), load_len(inv_vert_vert_length_ptr, 2),
            load_len(inv_primal_edge_length_ptr, 2), ov, by, bx);
    }
}

void run_cutile_nabla4_structured_launcher(
    index_type KDim,
    index_type x_dim,
    index_type y_dim,
    index_type halo,
    const VP_TYPE* __restrict__ u_vert_ptr,
    const VP_TYPE* __restrict__ v_vert_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v1_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v2_ptr,
    const WP_TYPE* __restrict__ z_nabla2_e_ptr,
    const WP_TYPE* __restrict__ inv_vert_vert_length_ptr,
    const WP_TYPE* __restrict__ inv_primal_edge_length_ptr,
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {
    const index_type x_dim_inner = x_dim - 2 * halo;
    const index_type y_dim_inner = y_dim - 2 * halo;
    dim3 grid((x_dim_inner + TILE_I - 1) / TILE_I,
        (y_dim_inner + TILE_J - 1) / TILE_J,
        KDim);
    const index_type inner_grid_size = x_dim_inner * y_dim_inner;
    run_cutile_nabla4_structured<<<grid, 1>>>(KDim,
        x_dim,
        x_dim_inner,
        y_dim,
        y_dim_inner,
        halo,
        x_dim * y_dim,
        inner_grid_size,
        x_dim * y_dim * 3,
        u_vert_ptr,
        v_vert_ptr,
        primal_normal_vert_v1_ptr,
        primal_normal_vert_v2_ptr,
        z_nabla2_e_ptr,
        inv_vert_vert_length_ptr,
        inv_primal_edge_length_ptr,
        z_nabla4_e2_wp_ptr);
    GT_CUDA_CHECK(cudaGetLastError());
    return;
}