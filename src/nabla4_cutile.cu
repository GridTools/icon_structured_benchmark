#include "cuda_tile.h"
#include "nabla4_gridtools.hpp"

namespace ct = cuda::tiles;
using namespace ct::literals;

// ---------------------------------------------------------------------------
// Tile shape for the (j, i) plane. TILE_J is the row (y) extent, TILE_I the
// column (x) extent -- kept consistent with the tile-space index order used
// throughout (partition views are built over extents{y, x}, so loads/stores
// are indexed load_masked(bj, bi)).
//
// These are the SAME values the launcher below uses to size the grid, so
// keep them in sync if you retune them.
// ---------------------------------------------------------------------------
constexpr int TILE_I = 32;
constexpr int TILE_J = 2;
constexpr int TILE_K = 2;

// K_LOOP_LEN is orthogonal to TILE_K above: it belongs to the kloop kernel
// further down, where the K axis is NOT part of the tile shape (TILE_K == 1
// there, in the sense that every tile stays 2D and each single load/store
// touches exactly one k level). Instead, one block walks K_LOOP_LEN
// consecutive k levels via a runtime for loop, amortizing the K-independent
// pnv / edge-length loads across all of them. grid.z == ceil(KDim /
// K_LOOP_LEN) for that kernel, vs. grid.z == ceil(KDim / TILE_K) for the
// rank-3 naive kernel below. Tune independently against occupancy.
constexpr index_type K_LOOP_LEN = 80;

using vp3D_tile_t = ct::tile<VP_TYPE, ct::shape<TILE_K, TILE_J, TILE_I>>;
using wp3D_tile_t = ct::tile<WP_TYPE, ct::shape<TILE_K, TILE_J, TILE_I>>;
using vp2D_tile_t = ct::tile<VP_TYPE, ct::shape<TILE_J, TILE_I>>;
using wp2D_tile_t = ct::tile<WP_TYPE, ct::shape<TILE_J, TILE_I>>;

// ---------------------------------------------------------------------------
// All six E2C2V neighbor offsets used by the reference kernel reduce to a
// constant (dj, di) pointer shift, because the torus storage is laid out
// row-major as vertex_flat = j * x_dim + i. That means every neighbor tile
// can be produced with an ordinary structured tile-space load: build a
// tensor_span whose origin has been shifted by the neighbor's constant
// offset, then load the SAME tile-space index (bj, bi) that the block
// already owns. No gather/scatter is needed anywhere in this kernel.
//
// This helper uses runtime k-stride from GridTools storage info:
// element(vertex_flat, k) = ptr[k * k_stride + vertex_flat].
//
// IMPORTANT: the underlying buffer's true row pitch is x_dim (the full,
// halo-included grid width), NOT valid_i (= x_dim - halo - di). Those two
// only coincide when there is no halo. tensor_span must therefore be given
// an explicit stride of {x_dim, 1}; letting the masking extents double as
// the stride (the original bug) reads every row after the first from the
// wrong offset.
// ---------------------------------------------------------------------------
__tile__ vp3D_tile_t load_vertex_tile(const VP_TYPE* __restrict__ base_ptr,
    index_type y_dim,
    index_type x_dim,
    index_type KDim,
    index_type halo,
    index_type k_stride,
    index_type k,
    index_type dj,
    index_type di,
    int bi,
    int bj) {
    // const VP_TYPE* k_slice = base_ptr + k * k_stride;
    const VP_TYPE* shifted = base_ptr + (halo + dj) * x_dim + (halo + di);
    const index_type valid_j = y_dim - halo - dj;
    const index_type valid_i = x_dim - halo - di;

    // Fix: pass the true row pitch (x_dim) as an explicit stride via
    // layout_strided_mapping, instead of letting {valid_j, valid_i} double
    // as the (densely-packed) layout, as ct::layout_right would assume.
    ct::layout_strided_mapping mapping{ct::extents{KDim, valid_j, valid_i}, ct::extents{k_stride, x_dim, index_type{1}}};
    auto view = ct::partition_view{ct::tensor_span{shifted, mapping}, ct::shape<TILE_K, TILE_J, TILE_I>{}};
    vp3D_tile_t ret;
    [[ cutile::hint(0, latency=8) ]]
    ret = view.load_masked(k, bj, bi);
    return ret;
}

// primal_normal_vert_v1 / v2 are 1-D (no K dependence) fields flattened as
// [slot][color][j][i] (slot outer, see the index derivation: E2ECV index =
// slot * global_edges_per_orientation + color * total_grid_size + i_j).
// Every slot/color combination reads from the *unshifted* i_j position, so
// this never needs a neighbor shift -- only a constant slot/color offset.
//
// Same pitch issue as load_vertex_tile: the buffer's true row pitch is
// x_dim, not (x_dim - halo). Explicit stride required.
__tile__ wp2D_tile_t load_pnv_tile(const WP_TYPE* __restrict__ base_ptr,
    index_type y_dim,
    index_type x_dim,
    index_type halo,
    index_type total_grid_size,
    index_type global_edges_per_orientation,
    index_type slot,
    index_type color,
    int bi,
    int bj) {
    const WP_TYPE* slice = base_ptr + slot * global_edges_per_orientation + color * total_grid_size + halo * x_dim + halo;
    const index_type valid_j = y_dim - halo;
    const index_type valid_i = x_dim - halo;

    // Fix: explicit stride {x_dim, 1} via layout_strided_mapping -- valid_i
    // != x_dim whenever halo > 0, so it cannot double as the row pitch.
    ct::layout_strided_mapping mapping{ct::extents{valid_j, valid_i}, ct::extents{x_dim, index_type{1}}};
    auto view = ct::partition_view{ct::tensor_span{slice, mapping}, ct::shape<TILE_J, TILE_I>{}};
    wp2D_tile_t ret;
    [[ cutile::hint(0, latency=5) ]]
    ret = view.load_masked(bj, bi);
    return ret;
}

// inv_vert_vert_length / inv_primal_edge_length: 1-D, interior-only (no
// halo), flattened as [color][j_inner][i_inner]. Here the true row pitch
// genuinely IS x_dim_inner, matching the extent passed in, so the
// dense-layout (implicit-stride) form is correct as-is -- no change needed.
__tile__ wp2D_tile_t load_edge_len_tile(const WP_TYPE* __restrict__ base_ptr,
    index_type y_dim_inner,
    index_type x_dim_inner,
    index_type inner_grid_size,
    index_type color,
    int bi,
    int bj) {
    const WP_TYPE* slice = base_ptr + color * inner_grid_size;
    auto view = ct::partition_view{ct::tensor_span{slice, ct::extents{y_dim_inner, x_dim_inner}}, ct::shape<TILE_J, TILE_I>{}};
    wp2D_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(bj, bi);
    return ret;
}

// z_nabla2_e: interior-only, K-dependent, flattened as [k][color][j_inner][i_inner].
// Pitch == extent here too -- unaffected by the bug.
__tile__ wp3D_tile_t load_z_nabla2e_tile(const WP_TYPE* __restrict__ base_ptr,
    index_type y_dim_inner,
    index_type x_dim_inner,
    index_type KDim,
    index_type inner_grid_size,
    index_type k_stride,
    index_type k,
    index_type color,
    int bi,
    int bj) {
    const WP_TYPE* slice = base_ptr + color * inner_grid_size;
    ct::layout_strided_mapping mapping{ct::extents{KDim, y_dim_inner, x_dim_inner}, ct::extents{k_stride, x_dim_inner, index_type{1}}};
    auto view = ct::partition_view{ct::tensor_span{slice, mapping}, ct::shape<TILE_K, TILE_J, TILE_I>{}};
    wp3D_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(k, bj, bi);
    return ret;
}

// z_nabla4_e2: same [k][color][j_inner][i_inner] layout as z_nabla2_e, VP_TYPE output.
// Pitch == extent here too -- unaffected by the bug.
__tile__ void store_z_nabla4e2_tile(VP_TYPE* __restrict__ base_ptr,
    index_type KDim,
    index_type y_dim_inner,
    index_type x_dim_inner,
    index_type inner_grid_size,
    index_type k_stride,
    index_type k,
    index_type color,
    const vp3D_tile_t& value,
    int bi,
    int bj) {
    VP_TYPE* slice = base_ptr + color * inner_grid_size;
    ct::layout_strided_mapping mapping{ct::extents{KDim, y_dim_inner, x_dim_inner}, ct::extents{k_stride, x_dim_inner, index_type{1}}};
    auto view = ct::partition_view{ct::tensor_span{slice, mapping}, ct::shape<TILE_K, TILE_J, TILE_I>{}};
    [[ cutile::hint(0, latency=1) ]]
    view.store_masked(value, k, bj, bi);
}

[[ cutile::hint(900,  occupancy=8) ]]
__tile_global__ void run_cutile_nabla4_structured(index_type KDim,
    index_type x_dim,
    index_type x_dim_inner,
    index_type y_dim,
    index_type y_dim_inner,
    index_type halo,
    index_type total_grid_size,
    index_type inner_grid_size,
    index_type global_edges_per_orientation,
    index_type u_vert_k_stride,
    index_type v_vert_k_stride,
    index_type z_nabla2_e_k_stride,
    index_type z_nabla4_e2_wp_k_stride,
    const VP_TYPE* __restrict__ u_vert_ptr,
    const VP_TYPE* __restrict__ v_vert_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v1_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v2_ptr,
    const WP_TYPE* __restrict__ z_nabla2_e_ptr,
    const WP_TYPE* __restrict__ inv_vert_vert_length_ptr,
    const WP_TYPE* __restrict__ inv_primal_edge_length_ptr,
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {

    u_vert_ptr = ct::assume_aligned(u_vert_ptr, 16_ic);
    v_vert_ptr = ct::assume_aligned(v_vert_ptr, 16_ic);
    primal_normal_vert_v1_ptr = ct::assume_aligned(primal_normal_vert_v1_ptr, 16_ic);
    primal_normal_vert_v2_ptr = ct::assume_aligned(primal_normal_vert_v2_ptr, 16_ic);
    z_nabla2_e_ptr = ct::assume_aligned(z_nabla2_e_ptr, 16_ic);
    inv_vert_vert_length_ptr = ct::assume_aligned(inv_vert_vert_length_ptr, 16_ic);
    inv_primal_edge_length_ptr = ct::assume_aligned(inv_primal_edge_length_ptr, 16_ic);
    z_nabla4_e2_wp_ptr = ct::assume_aligned(z_nabla4_e2_wp_ptr, 16_ic);

    // Grid is launched (tiles_x, tiles_y, KDim): one block per (i-tile, j-tile, k).
    const auto bid3 = ct::bid();
    const int bi = static_cast<int>(bid3.x);
    const int bj = static_cast<int>(bid3.y);
    const index_type k = static_cast<index_type>(bid3.z);

    // --- The six structurally-shifted vertex tiles, loaded once and reused
    // across all three colors (see the offset table in the header comment). ---
    auto u_i_j = ct::element_cast<double>(load_vertex_tile(u_vert_ptr, y_dim, x_dim, KDim, halo, u_vert_k_stride, k, 0, 0, bi, bj));
    auto u_i_jp1 = ct::element_cast<double>(load_vertex_tile(u_vert_ptr, y_dim, x_dim, KDim, halo, u_vert_k_stride, k, 1, 0, bi, bj));
    auto u_im1_jp1 = ct::element_cast<double>(load_vertex_tile(u_vert_ptr, y_dim, x_dim, KDim, halo, u_vert_k_stride, k, 1, -1, bi, bj));
    auto u_ip1_j = ct::element_cast<double>(load_vertex_tile(u_vert_ptr, y_dim, x_dim, KDim, halo, u_vert_k_stride, k, 0, 1, bi, bj));
    auto u_ip1_jm1 = ct::element_cast<double>(load_vertex_tile(u_vert_ptr, y_dim, x_dim, KDim, halo, u_vert_k_stride, k, -1, 1, bi, bj));
    auto u_i_jm1 = ct::element_cast<double>(load_vertex_tile(u_vert_ptr, y_dim, x_dim, KDim, halo, u_vert_k_stride, k, -1, 0, bi, bj));

    auto v_i_j = ct::element_cast<double>(load_vertex_tile(v_vert_ptr, y_dim, x_dim, KDim, halo, v_vert_k_stride, k, 0, 0, bi, bj));
    auto v_i_jp1 = ct::element_cast<double>(load_vertex_tile(v_vert_ptr, y_dim, x_dim, KDim, halo, v_vert_k_stride, k, 1, 0, bi, bj));
    auto v_im1_jp1 = ct::element_cast<double>(load_vertex_tile(v_vert_ptr, y_dim, x_dim, KDim, halo, v_vert_k_stride, k, 1, -1, bi, bj));
    auto v_ip1_j = ct::element_cast<double>(load_vertex_tile(v_vert_ptr, y_dim, x_dim, KDim, halo, v_vert_k_stride, k, 0, 1, bi, bj));
    auto v_ip1_jm1 = ct::element_cast<double>(load_vertex_tile(v_vert_ptr, y_dim, x_dim, KDim, halo, v_vert_k_stride, k, -1, 1, bi, bj));
    auto v_i_jm1 = ct::element_cast<double>(load_vertex_tile(v_vert_ptr, y_dim, x_dim, KDim, halo, v_vert_k_stride, k, -1, 0, bi, bj));

    // --- inv_vert_vert_length^2 / inv_primal_edge_length^2, one tile per color. ---
    auto ivv0 = load_edge_len_tile(inv_vert_vert_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 0, bi, bj);
    auto ivv1 = load_edge_len_tile(inv_vert_vert_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 1, bi, bj);
    auto ivv2 = load_edge_len_tile(inv_vert_vert_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 2, bi, bj);
    ivv0 = ivv0 * ivv0;
    ivv1 = ivv1 * ivv1;
    ivv2 = ivv2 * ivv2;

    auto ipe0 = load_edge_len_tile(inv_primal_edge_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 0, bi, bj);
    auto ipe1 = load_edge_len_tile(inv_primal_edge_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 1, bi, bj);
    auto ipe2 = load_edge_len_tile(inv_primal_edge_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 2, bi, bj);
    ipe0 = ipe0 * ipe0;
    ipe1 = ipe1 * ipe1;
    ipe2 = ipe2 * ipe2;

    // z_nabla2_e, one tile per color.
    auto z2e0 = ct::element_cast<double>(load_z_nabla2e_tile(z_nabla2_e_ptr, y_dim_inner, x_dim_inner, KDim, inner_grid_size, z_nabla2_e_k_stride, k, 0, bi, bj));
    auto z2e1 = ct::element_cast<double>(load_z_nabla2e_tile(z_nabla2_e_ptr, y_dim_inner, x_dim_inner, KDim, inner_grid_size, z_nabla2_e_k_stride, k, 1, bi, bj));
    auto z2e2 = ct::element_cast<double>(load_z_nabla2e_tile(z_nabla2_e_ptr, y_dim_inner, x_dim_inner, KDim, inner_grid_size, z_nabla2_e_k_stride, k, 2, bi, bj));

    // =====================================================================
    // color 0: E2C2V = {i_j, i_jp1, im1_jp1, ip1_j}
    // =====================================================================
    {
        constexpr index_type color = 0;
        auto pnv1_0 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        auto pnv2_0 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        auto pnv1_1 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        auto pnv2_1 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        auto pnv1_2 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        auto pnv2_2 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        auto pnv1_3 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
        auto pnv2_3 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);

        auto nabv_tang = u_i_j * pnv1_0 + v_i_j * pnv2_0 + u_i_jp1 * pnv1_1 + v_i_jp1 * pnv2_1;
        auto nabv_norm = u_im1_jp1 * pnv1_2 + v_im1_jp1 * pnv2_2 + u_ip1_j * pnv1_3 + v_ip1_j * pnv2_3;
        auto result = 4.0 * ((nabv_norm - 2.0 * z2e0) * ivv0 + (nabv_tang - 2.0 * z2e0) * ipe0);
        store_z_nabla4e2_tile(z_nabla4_e2_wp_ptr, KDim, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla4_e2_wp_k_stride, k, color, ct::element_cast<VP_TYPE>(result), bi, bj);
    }

    // =====================================================================
    // color 1: E2C2V = {i_j, ip1_j, i_jp1, ip1_jm1}
    // =====================================================================
    {
        constexpr index_type color = 1;
        auto pnv1_0 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        auto pnv2_0 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        auto pnv1_1 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        auto pnv2_1 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        auto pnv1_2 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        auto pnv2_2 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        auto pnv1_3 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
        auto pnv2_3 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);

        auto nabv_tang = u_i_j * pnv1_0 + v_i_j * pnv2_0 + u_ip1_j * pnv1_1 + v_ip1_j * pnv2_1;
        auto nabv_norm = u_i_jp1 * pnv1_2 + v_i_jp1 * pnv2_2 + u_ip1_jm1 * pnv1_3 + v_ip1_jm1 * pnv2_3;

        auto result = 4.0 * ((nabv_norm - 2.0 * z2e1) * ivv1 + (nabv_tang - 2.0 * z2e1) * ipe1);
        store_z_nabla4e2_tile(z_nabla4_e2_wp_ptr, KDim, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla4_e2_wp_k_stride, k, color, ct::element_cast<VP_TYPE>(result), bi, bj);
    }

    // =====================================================================
    // color 2: E2C2V = {i_j, ip1_jm1, ip1_j, i_jm1}
    // =====================================================================
    {
        constexpr index_type color = 2;
        auto pnv1_0 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        auto pnv2_0 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        auto pnv1_1 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        auto pnv2_1 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        auto pnv1_2 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        auto pnv2_2 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        auto pnv1_3 = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
        auto pnv2_3 = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);

        auto nabv_tang = u_i_j * pnv1_0 + v_i_j * pnv2_0 + u_ip1_jm1 * pnv1_1 + v_ip1_jm1 * pnv2_1;
        auto nabv_norm = u_ip1_j * pnv1_2 + v_ip1_j * pnv2_2 + u_i_jm1 * pnv1_3 + v_i_jm1 * pnv2_3;

        auto result = 4.0 * ((nabv_norm - 2.0 * z2e2) * ivv2 + (nabv_tang - 2.0 * z2e2) * ipe2);
        store_z_nabla4e2_tile(z_nabla4_e2_wp_ptr, KDim, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla4_e2_wp_k_stride, k, color, ct::element_cast<VP_TYPE>(result), bi, bj);
    }
}

void run_cutile_nabla4_structured_launcher(index_type KDim,
    index_type x_dim,
    index_type y_dim,
    index_type halo,
    index_type u_vert_k_stride,
    index_type v_vert_k_stride,
    index_type z_nabla2_e_k_stride,
    index_type z_nabla4_e2_wp_k_stride,
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
        (KDim + TILE_K - 1) / TILE_K);
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
        u_vert_k_stride,
        v_vert_k_stride,
        z_nabla2_e_k_stride,
        z_nabla4_e2_wp_k_stride,
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

// =============================================================================
// K-loop variant. TILE_K == 1 here in the sense that no K axis is baked into
// the tile shape at all -- every tile is the plain 2D vp2D_tile_t/wp2D_tile_t
// type (same as the K-independent loads above), and each single load/store
// touches exactly one k level. Instead of the naive kernel's rank-3 tile (one
// block handles TILE_K levels "for free" via the tile shape, no loop) or
// spreading K across the whole grid, this variant gives each block a
// K_LOOP_LEN-sized chunk of consecutive k levels and walks it with an
// explicit runtime for loop, hoisting every K-independent load (the 24
// primal_normal_vert tiles, the 6 squared edge-length tiles) above the loop
// so they're computed once per block instead of once per k.
//
// grid.z == ceil(KDim / K_LOOP_LEN); bid3.z selects which chunk this block
// owns. The vertex / z_nabla2_e / store helpers below use plain pointer
// arithmetic (k * k_stride) rather than a rank-3 tensor_span, since K isn't
// part of the tile shape here.
// =============================================================================
__tile__ vp2D_tile_t load_vertex_tile_kloop(const VP_TYPE* __restrict__ base_ptr,
    index_type y_dim,
    index_type x_dim,
    index_type halo,
    index_type k_stride,
    index_type k,
    index_type dj,
    index_type di,
    int bi,
    int bj) {
    const VP_TYPE* k_slice = base_ptr + k * k_stride;
    const VP_TYPE* shifted = k_slice + (halo + dj) * x_dim + (halo + di);
    const index_type valid_j = y_dim - halo - dj;
    const index_type valid_i = x_dim - halo - di;

    ct::layout_strided_mapping mapping{ct::extents{valid_j, valid_i}, ct::extents{x_dim, index_type{1}}};
    auto view = ct::partition_view{ct::tensor_span{shifted, mapping}, ct::shape<TILE_J, TILE_I>{}};
    vp2D_tile_t ret;
    [[ cutile::hint(0, latency=8) ]]
    ret = view.load_masked(bj, bi);
    return ret;
}

__tile__ wp2D_tile_t load_z_nabla2e_tile_kloop(const WP_TYPE* __restrict__ base_ptr,
    index_type y_dim_inner,
    index_type x_dim_inner,
    index_type inner_grid_size,
    index_type k_stride,
    index_type k,
    index_type color,
    int bi,
    int bj) {
    const WP_TYPE* slice = base_ptr + k * k_stride + color * inner_grid_size;
    auto view = ct::partition_view{ct::tensor_span{slice, ct::extents{y_dim_inner, x_dim_inner}}, ct::shape<TILE_J, TILE_I>{}};
    wp2D_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(bj, bi);
    return ret;
}

__tile__ void store_z_nabla4e2_tile_kloop(VP_TYPE* __restrict__ base_ptr,
    index_type y_dim_inner,
    index_type x_dim_inner,
    index_type inner_grid_size,
    index_type k_stride,
    index_type k,
    index_type color,
    const vp2D_tile_t& value,
    int bi,
    int bj) {
    VP_TYPE* slice = base_ptr + k * k_stride + color * inner_grid_size;
    auto view = ct::partition_view{ct::tensor_span{slice, ct::extents{y_dim_inner, x_dim_inner}}, ct::shape<TILE_J, TILE_I>{}};
    [[ cutile::hint(0, latency=1) ]]
    view.store_masked(value, bj, bi);
}

[[ cutile::hint(900, occupancy=8, num_cta_in_cga=1) ]]
__tile_global__ void run_cutile_nabla4_structured_kloop(index_type KDim,
    index_type x_dim,
    index_type x_dim_inner,
    index_type y_dim,
    index_type y_dim_inner,
    index_type halo,
    index_type total_grid_size,
    index_type inner_grid_size,
    index_type global_edges_per_orientation,
    index_type u_vert_k_stride,
    index_type v_vert_k_stride,
    index_type z_nabla2_e_k_stride,
    index_type z_nabla4_e2_wp_k_stride,
    const VP_TYPE* __restrict__ u_vert_ptr,
    const VP_TYPE* __restrict__ v_vert_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v1_ptr,
    const WP_TYPE* __restrict__ primal_normal_vert_v2_ptr,
    const WP_TYPE* __restrict__ z_nabla2_e_ptr,
    const WP_TYPE* __restrict__ inv_vert_vert_length_ptr,
    const WP_TYPE* __restrict__ inv_primal_edge_length_ptr,
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {

    u_vert_ptr = ct::assume_aligned(u_vert_ptr, 16_ic);
    v_vert_ptr = ct::assume_aligned(v_vert_ptr, 16_ic);
    primal_normal_vert_v1_ptr = ct::assume_aligned(primal_normal_vert_v1_ptr, 16_ic);
    primal_normal_vert_v2_ptr = ct::assume_aligned(primal_normal_vert_v2_ptr, 16_ic);
    z_nabla2_e_ptr = ct::assume_aligned(z_nabla2_e_ptr, 16_ic);
    inv_vert_vert_length_ptr = ct::assume_aligned(inv_vert_vert_length_ptr, 16_ic);
    inv_primal_edge_length_ptr = ct::assume_aligned(inv_primal_edge_length_ptr, 16_ic);
    z_nabla4_e2_wp_ptr = ct::assume_aligned(z_nabla4_e2_wp_ptr, 16_ic);

    // Grid is launched (tiles_x, tiles_y, ceil(KDim / K_LOOP_LEN)): one block
    // per (i-tile, j-tile, k-chunk). bid3.z selects WHICH chunk of up to
    // K_LOOP_LEN consecutive k levels this block owns; the levels within
    // that chunk are walked by the runtime loop further down.
    const auto bid3 = ct::bid();
    const int bi = static_cast<int>(bid3.x);
    const int bj = static_cast<int>(bid3.y);
    const index_type k_base = static_cast<index_type>(bid3.z) * K_LOOP_LEN;
    // Clamp the tail chunk when KDim isn't a multiple of K_LOOP_LEN.
    const index_type k_count = (KDim - k_base < K_LOOP_LEN) ? (KDim - k_base) : K_LOOP_LEN;

    // --- inv_vert_vert_length^2 / inv_primal_edge_length^2, one tile per
    // color. K-independent -- hoisted above the loop. ---
    auto ivv0 = load_edge_len_tile(inv_vert_vert_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 0, bi, bj);
    auto ivv1 = load_edge_len_tile(inv_vert_vert_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 1, bi, bj);
    auto ivv2 = load_edge_len_tile(inv_vert_vert_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 2, bi, bj);
    ivv0 = ivv0 * ivv0;
    ivv1 = ivv1 * ivv1;
    ivv2 = ivv2 * ivv2;

    auto ipe0 = load_edge_len_tile(inv_primal_edge_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 0, bi, bj);
    auto ipe1 = load_edge_len_tile(inv_primal_edge_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 1, bi, bj);
    auto ipe2 = load_edge_len_tile(inv_primal_edge_length_ptr, y_dim_inner, x_dim_inner, inner_grid_size, 2, bi, bj);
    ipe0 = ipe0 * ipe0;
    ipe1 = ipe1 * ipe1;
    ipe2 = ipe2 * ipe2;

    // --- primal_normal_vert_v1/v2 tiles: 4 slots x 3 colors x 2 components
    // = 24 tiles total, all K-independent -- hoisted above the loop and
    // indexed by color inside it. ---
    wp2D_tile_t pnv1_0[3], pnv2_0[3], pnv1_1[3], pnv2_1[3], pnv1_2[3], pnv2_2[3], pnv1_3[3], pnv2_3[3];
    {
        constexpr index_type color = 0;
        pnv1_0[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        pnv2_0[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        pnv1_1[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        pnv2_1[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        pnv1_2[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        pnv2_2[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        pnv1_3[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
        pnv2_3[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
    }
    {
        constexpr index_type color = 1;
        pnv1_0[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        pnv2_0[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        pnv1_1[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        pnv2_1[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        pnv1_2[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        pnv2_2[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        pnv1_3[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
        pnv2_3[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
    }
    {
        constexpr index_type color = 2;
        pnv1_0[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        pnv2_0[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 0, color, bi, bj);
        pnv1_1[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        pnv2_1[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 1, color, bi, bj);
        pnv1_2[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        pnv2_2[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 2, color, bi, bj);
        pnv1_3[color] = load_pnv_tile(primal_normal_vert_v1_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
        pnv2_3[color] = load_pnv_tile(primal_normal_vert_v2_ptr, y_dim, x_dim, halo, total_grid_size, global_edges_per_orientation, 3, color, bi, bj);
    }

    // --- Walk this block's K_LOOP_LEN-sized chunk. Only the genuinely
    // K-dependent loads (the six u_vert / six v_vert vertex tiles, and the
    // three z_nabla2_e tiles) are reissued on every iteration; the pnv /
    // edge-length tiles hoisted above are reused across all of them. ---
    for (auto kk : ct::irange(index_type{0}, k_count)) {
        const index_type k = k_base + kk;

        auto u_i_j = ct::element_cast<double>(load_vertex_tile_kloop(u_vert_ptr, y_dim, x_dim, halo, u_vert_k_stride, k, 0, 0, bi, bj));
        auto u_i_jp1 = ct::element_cast<double>(load_vertex_tile_kloop(u_vert_ptr, y_dim, x_dim, halo, u_vert_k_stride, k, 1, 0, bi, bj));
        auto u_im1_jp1 = ct::element_cast<double>(load_vertex_tile_kloop(u_vert_ptr, y_dim, x_dim, halo, u_vert_k_stride, k, 1, -1, bi, bj));
        auto u_ip1_j = ct::element_cast<double>(load_vertex_tile_kloop(u_vert_ptr, y_dim, x_dim, halo, u_vert_k_stride, k, 0, 1, bi, bj));
        auto u_ip1_jm1 = ct::element_cast<double>(load_vertex_tile_kloop(u_vert_ptr, y_dim, x_dim, halo, u_vert_k_stride, k, -1, 1, bi, bj));
        auto u_i_jm1 = ct::element_cast<double>(load_vertex_tile_kloop(u_vert_ptr, y_dim, x_dim, halo, u_vert_k_stride, k, -1, 0, bi, bj));

        auto v_i_j = ct::element_cast<double>(load_vertex_tile_kloop(v_vert_ptr, y_dim, x_dim, halo, v_vert_k_stride, k, 0, 0, bi, bj));
        auto v_i_jp1 = ct::element_cast<double>(load_vertex_tile_kloop(v_vert_ptr, y_dim, x_dim, halo, v_vert_k_stride, k, 1, 0, bi, bj));
        auto v_im1_jp1 = ct::element_cast<double>(load_vertex_tile_kloop(v_vert_ptr, y_dim, x_dim, halo, v_vert_k_stride, k, 1, -1, bi, bj));
        auto v_ip1_j = ct::element_cast<double>(load_vertex_tile_kloop(v_vert_ptr, y_dim, x_dim, halo, v_vert_k_stride, k, 0, 1, bi, bj));
        auto v_ip1_jm1 = ct::element_cast<double>(load_vertex_tile_kloop(v_vert_ptr, y_dim, x_dim, halo, v_vert_k_stride, k, -1, 1, bi, bj));
        auto v_i_jm1 = ct::element_cast<double>(load_vertex_tile_kloop(v_vert_ptr, y_dim, x_dim, halo, v_vert_k_stride, k, -1, 0, bi, bj));

        auto z2e0 = ct::element_cast<double>(load_z_nabla2e_tile_kloop(z_nabla2_e_ptr, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla2_e_k_stride, k, 0, bi, bj));
        auto z2e1 = ct::element_cast<double>(load_z_nabla2e_tile_kloop(z_nabla2_e_ptr, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla2_e_k_stride, k, 1, bi, bj));
        auto z2e2 = ct::element_cast<double>(load_z_nabla2e_tile_kloop(z_nabla2_e_ptr, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla2_e_k_stride, k, 2, bi, bj));

        // =================================================================
        // color 0: E2C2V = {i_j, i_jp1, im1_jp1, ip1_j}
        // =================================================================
        {
            constexpr index_type color = 0;
            auto nabv_tang = u_i_j * pnv1_0[color] + v_i_j * pnv2_0[color] + u_i_jp1 * pnv1_1[color] + v_i_jp1 * pnv2_1[color];
            auto nabv_norm = u_im1_jp1 * pnv1_2[color] + v_im1_jp1 * pnv2_2[color] + u_ip1_j * pnv1_3[color] + v_ip1_j * pnv2_3[color];
            auto result = 4.0 * ((nabv_norm - 2.0 * z2e0) * ivv0 + (nabv_tang - 2.0 * z2e0) * ipe0);
            store_z_nabla4e2_tile_kloop(z_nabla4_e2_wp_ptr, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla4_e2_wp_k_stride, k, color, ct::element_cast<VP_TYPE>(result), bi, bj);
        }

        // =================================================================
        // color 1: E2C2V = {i_j, ip1_j, i_jp1, ip1_jm1}
        // =================================================================
        {
            constexpr index_type color = 1;
            auto nabv_tang = u_i_j * pnv1_0[color] + v_i_j * pnv2_0[color] + u_ip1_j * pnv1_1[color] + v_ip1_j * pnv2_1[color];
            auto nabv_norm = u_i_jp1 * pnv1_2[color] + v_i_jp1 * pnv2_2[color] + u_ip1_jm1 * pnv1_3[color] + v_ip1_jm1 * pnv2_3[color];
            auto result = 4.0 * ((nabv_norm - 2.0 * z2e1) * ivv1 + (nabv_tang - 2.0 * z2e1) * ipe1);
            store_z_nabla4e2_tile_kloop(z_nabla4_e2_wp_ptr, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla4_e2_wp_k_stride, k, color, ct::element_cast<VP_TYPE>(result), bi, bj);
        }

        // =================================================================
        // color 2: E2C2V = {i_j, ip1_jm1, ip1_j, i_jm1}
        // =================================================================
        {
            constexpr index_type color = 2;
            auto nabv_tang = u_i_j * pnv1_0[color] + v_i_j * pnv2_0[color] + u_ip1_jm1 * pnv1_1[color] + v_ip1_jm1 * pnv2_1[color];
            auto nabv_norm = u_ip1_j * pnv1_2[color] + v_ip1_j * pnv2_2[color] + u_i_jm1 * pnv1_3[color] + v_i_jm1 * pnv2_3[color];
            auto result = 4.0 * ((nabv_norm - 2.0 * z2e2) * ivv2 + (nabv_tang - 2.0 * z2e2) * ipe2);
            store_z_nabla4e2_tile_kloop(z_nabla4_e2_wp_ptr, y_dim_inner, x_dim_inner, inner_grid_size, z_nabla4_e2_wp_k_stride, k, color, ct::element_cast<VP_TYPE>(result), bi, bj);
        }
    }
}

void run_cutile_nabla4_structured_kloop_launcher(index_type KDim,
    index_type x_dim,
    index_type y_dim,
    index_type halo,
    index_type u_vert_k_stride,
    index_type v_vert_k_stride,
    index_type z_nabla2_e_k_stride,
    index_type z_nabla4_e2_wp_k_stride,
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
    // Same (x, y) tiling as the naive launcher; z now walks K in chunks of
    // K_LOOP_LEN via the runtime loop inside the kernel (TILE_K == 1 here, no
    // K axis in the tile shape), instead of baking TILE_K == 2 levels into
    // the tile shape the way the rank-3 naive kernel above does.
    dim3 grid((x_dim_inner + TILE_I - 1) / TILE_I,
        (y_dim_inner + TILE_J - 1) / TILE_J,
        (KDim + K_LOOP_LEN - 1) / K_LOOP_LEN);
    const index_type inner_grid_size = x_dim_inner * y_dim_inner;
    run_cutile_nabla4_structured_kloop<<<grid, 1>>>(KDim,
        x_dim,
        x_dim_inner,
        y_dim,
        y_dim_inner,
        halo,
        x_dim * y_dim,
        inner_grid_size,
        x_dim * y_dim * 3,
        u_vert_k_stride,
        v_vert_k_stride,
        z_nabla2_e_k_stride,
        z_nabla4_e2_wp_k_stride,
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
