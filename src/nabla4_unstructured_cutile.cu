#include "cuda_tile.h"
#include "nabla4_gridtools.hpp"

namespace ct = cuda::tiles;
using namespace ct::literals;

// ---------------------------------------------------------------------------
// Unlike the structured torus, edge neighbors here are not affine in the
// storage layout -- E2C2V/E2ECV are arbitrary per-edge indices coming from
// ICON's unstructured connectivity, so there is no compile-time (dj, di)
// pointer-shift trick to reuse. The only axis that IS structured/tileable is
// K: for a fixed (possibly gathered) vertex/edge index, K is contiguous with
// a runtime stride, exactly like the structured kernel's k-loop variant
// already relies on (base_ptr + k * k_stride, where k is itself a *runtime*
// value, not a compile-time constant).
//
// So each block here owns exactly one edge (no tiling across edges is
// possible), gathers its 4 E2C2V + 4 E2ECV connectivity indices as plain
// scalar reads, and then uses those gathered indices as dynamic base
// offsets for genuine tile-loads along K via ct::tensor_span/partition_view
// -- the same "runtime-computed base pointer" mechanism the structured
// kloop kernel already uses for its k index, just gathered instead of
// computed from (bi, bj).
// ---------------------------------------------------------------------------
constexpr int TILE_K = 16;
// K_LOOP_LEN must be a multiple of TILE_K: the kloop kernel below walks one
// edge's K_LOOP_LEN-sized K chunk as (K_LOOP_LEN / TILE_K) inner TILE_K-wide
// tile-loads, reusing the gathered connectivity + primal_normal_vert reads
// (hoisted once per block) across all of them.
constexpr index_type K_LOOP_LEN = 80;

using vp1D_tile_t = ct::tile<VP_TYPE, ct::shape<TILE_K>>;
using wp1D_tile_t = ct::tile<WP_TYPE, ct::shape<TILE_K>>;

// Loads a TILE_K-wide slice of K levels for `index` (either the edge's own
// index, for z_nabla2_e, or a gathered E2C2V vertex index, for u_vert/v_vert)
// starting at tile k_tile (i.e. K levels [k_tile*TILE_K, k_tile*TILE_K+TILE_K)).
// element(index, k) = ptr[k * k_stride + index], matching the unit-stride
// convention every 2D data store in this codebase uses for its non-K
// dimension (see nabla4_cutile.cu's load_vertex_tile_kloop for the structured
// precedent of this exact stride convention).
__tile__ vp1D_tile_t load_ktile_vp(const VP_TYPE* __restrict__ base_ptr,
    index_type KDim,
    index_type k_stride,
    index_type k_tile,
    index_type index) {
    const VP_TYPE* shifted = base_ptr + index;
    ct::layout_strided_mapping mapping{ct::extents{KDim}, ct::extents{k_stride}};
    auto view = ct::partition_view{ct::tensor_span{shifted, mapping}, ct::shape<TILE_K>{}};
    vp1D_tile_t ret;
    [[ cutile::hint(0, latency=8) ]]
    ret = view.load_masked(k_tile);
    return ret;
}

__tile__ wp1D_tile_t load_ktile_wp(const WP_TYPE* __restrict__ base_ptr,
    index_type KDim,
    index_type k_stride,
    index_type k_tile,
    index_type index) {
    const WP_TYPE* shifted = base_ptr + index;
    ct::layout_strided_mapping mapping{ct::extents{KDim}, ct::extents{k_stride}};
    auto view = ct::partition_view{ct::tensor_span{shifted, mapping}, ct::shape<TILE_K>{}};
    wp1D_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(k_tile);
    return ret;
}

__tile__ void store_ktile_vp(VP_TYPE* __restrict__ base_ptr,
    index_type KDim,
    index_type k_stride,
    index_type k_tile,
    index_type index,
    const vp1D_tile_t& value) {
    VP_TYPE* shifted = base_ptr + index;
    ct::layout_strided_mapping mapping{ct::extents{KDim}, ct::extents{k_stride}};
    auto view = ct::partition_view{ct::tensor_span{shifted, mapping}, ct::shape<TILE_K>{}};
    [[ cutile::hint(0, latency=1) ]]
    view.store_masked(value, k_tile);
}

// ---------------------------------------------------------------------------
// Naive variant: grid = (EdgeDim, ceil(KDim / TILE_K)). One block per
// (edge, K-tile) pair -- connectivity is re-gathered every K-tile (no reuse
// across K), mirroring the structured naive kernel's "no hoisting" design.
// ---------------------------------------------------------------------------
[[ cutile::hint(900, occupancy=8) ]]
__tile_global__ void run_cutile_nabla4_unstructured(index_type EdgeDim,
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
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {

    u_vert_ptr = ct::assume_aligned(u_vert_ptr, 16_ic);
    v_vert_ptr = ct::assume_aligned(v_vert_ptr, 16_ic);
    z_nabla2_e_ptr = ct::assume_aligned(z_nabla2_e_ptr, 16_ic);
    z_nabla4_e2_wp_ptr = ct::assume_aligned(z_nabla4_e2_wp_ptr, 16_ic);

    // Grid is launched (EdgeDim, ceil(KDim / TILE_K)): one block per
    // (edge, k-tile). No mask on edge_index needed -- grid.x == EdgeDim
    // exactly, so every block owns a valid edge.
    const auto bid2 = ct::bid();
    const index_type edge_index = static_cast<index_type>(bid2.x);
    const index_type k_tile = static_cast<index_type>(bid2.y);

    // Scalar gather -- E2C2V/E2ECV are arbitrary per-edge indices, not
    // affine, so this is a plain (non-tile) read, same as the reference
    // plain-CUDA kernel in nabla4_unstructured_gridtools.hpp.
    const index_type E2C2V_0 = e2c2v_ptr[edge_index + 0 * e2c2v_slot_stride];
    const index_type E2C2V_1 = e2c2v_ptr[edge_index + 1 * e2c2v_slot_stride];
    const index_type E2C2V_2 = e2c2v_ptr[edge_index + 2 * e2c2v_slot_stride];
    const index_type E2C2V_3 = e2c2v_ptr[edge_index + 3 * e2c2v_slot_stride];
    const index_type E2ECV_0 = e2ecv_ptr[edge_index + 0 * e2ecv_slot_stride];
    const index_type E2ECV_1 = e2ecv_ptr[edge_index + 1 * e2ecv_slot_stride];
    const index_type E2ECV_2 = e2ecv_ptr[edge_index + 2 * e2ecv_slot_stride];
    const index_type E2ECV_3 = e2ecv_ptr[edge_index + 3 * e2ecv_slot_stride];

    // primal_normal_vert_v1/v2 and the two edge lengths are K-independent
    // scalars -- plain reads, no tile machinery needed.
    const WP_TYPE pnv1_0 = primal_normal_vert_v1_ptr[E2ECV_0];
    const WP_TYPE pnv2_0 = primal_normal_vert_v2_ptr[E2ECV_0];
    const WP_TYPE pnv1_1 = primal_normal_vert_v1_ptr[E2ECV_1];
    const WP_TYPE pnv2_1 = primal_normal_vert_v2_ptr[E2ECV_1];
    const WP_TYPE pnv1_2 = primal_normal_vert_v1_ptr[E2ECV_2];
    const WP_TYPE pnv2_2 = primal_normal_vert_v2_ptr[E2ECV_2];
    const WP_TYPE pnv1_3 = primal_normal_vert_v1_ptr[E2ECV_3];
    const WP_TYPE pnv2_3 = primal_normal_vert_v2_ptr[E2ECV_3];

    const WP_TYPE ivv = inv_vert_vert_length_ptr[edge_index];
    const WP_TYPE ivv_sqr = ivv * ivv;
    const WP_TYPE ipe = inv_primal_edge_length_ptr[edge_index];
    const WP_TYPE ipe_sqr = ipe * ipe;

    // Genuine tile-loads along K: the base offset (E2C2V_i) is a *runtime
    // gathered* value, but the K axis itself is dense/strided, so this is a
    // real vectorized TILE_K-wide transaction, not a scalar gather.
    auto u0 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_0));
    auto u1 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_1));
    auto u2 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_2));
    auto u3 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_3));
    auto v0 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_0));
    auto v1 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_1));
    auto v2 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_2));
    auto v3 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_3));

    // z_nabla2_e is affine on the block's own edge_index -- a direct tile
    // load, no gather.
    auto z2e = ct::element_cast<double>(load_ktile_wp(z_nabla2_e_ptr, KDim, z_nabla2_e_k_stride, k_tile, edge_index));

    auto nabv_tang = u0 * pnv1_0 + v0 * pnv2_0 + u1 * pnv1_1 + v1 * pnv2_1;
    auto nabv_norm = u2 * pnv1_2 + v2 * pnv2_2 + u3 * pnv1_3 + v3 * pnv2_3;
    auto result = 4.0 * ((nabv_norm - 2.0 * z2e) * ivv_sqr + (nabv_tang - 2.0 * z2e) * ipe_sqr);

    store_ktile_vp(z_nabla4_e2_wp_ptr, KDim, z_nabla4_e2_wp_k_stride, k_tile, edge_index, ct::element_cast<VP_TYPE>(result));
}

void run_cutile_nabla4_unstructured_launcher(index_type EdgeDim,
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
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {
    dim3 grid(EdgeDim, (KDim + TILE_K - 1) / TILE_K, 1);
    run_cutile_nabla4_unstructured<<<grid, 1>>>(EdgeDim,
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
    GT_CUDA_CHECK(cudaGetLastError());
    return;
}

// ---------------------------------------------------------------------------
// K-loop variant: grid = (EdgeDim, ceil(KDim / K_LOOP_LEN)). One block per
// edge owns a K_LOOP_LEN-sized chunk of K levels; connectivity + pnv +
// edge-length reads are hoisted once per block (gathered a single time) and
// reused across (K_LOOP_LEN / TILE_K) inner TILE_K-wide tile-loads, instead
// of every K-tile re-gathering as the naive kernel does.
// ---------------------------------------------------------------------------
[[ cutile::hint(900, occupancy=8, num_cta_in_cga=1) ]]
__tile_global__ void run_cutile_nabla4_unstructured_kloop(index_type EdgeDim,
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
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {

    u_vert_ptr = ct::assume_aligned(u_vert_ptr, 16_ic);
    v_vert_ptr = ct::assume_aligned(v_vert_ptr, 16_ic);
    z_nabla2_e_ptr = ct::assume_aligned(z_nabla2_e_ptr, 16_ic);
    z_nabla4_e2_wp_ptr = ct::assume_aligned(z_nabla4_e2_wp_ptr, 16_ic);

    // Grid is launched (EdgeDim, ceil(KDim / K_LOOP_LEN)): one block per
    // (edge, k-chunk). bid2.y selects which K_LOOP_LEN-sized chunk of K
    // levels this block owns.
    const auto bid2 = ct::bid();
    const index_type edge_index = static_cast<index_type>(bid2.x);
    const index_type k_chunk_base = static_cast<index_type>(bid2.y) * K_LOOP_LEN;

    const index_type E2C2V_0 = e2c2v_ptr[edge_index + 0 * e2c2v_slot_stride];
    const index_type E2C2V_1 = e2c2v_ptr[edge_index + 1 * e2c2v_slot_stride];
    const index_type E2C2V_2 = e2c2v_ptr[edge_index + 2 * e2c2v_slot_stride];
    const index_type E2C2V_3 = e2c2v_ptr[edge_index + 3 * e2c2v_slot_stride];
    const index_type E2ECV_0 = e2ecv_ptr[edge_index + 0 * e2ecv_slot_stride];
    const index_type E2ECV_1 = e2ecv_ptr[edge_index + 1 * e2ecv_slot_stride];
    const index_type E2ECV_2 = e2ecv_ptr[edge_index + 2 * e2ecv_slot_stride];
    const index_type E2ECV_3 = e2ecv_ptr[edge_index + 3 * e2ecv_slot_stride];

    const WP_TYPE pnv1_0 = primal_normal_vert_v1_ptr[E2ECV_0];
    const WP_TYPE pnv2_0 = primal_normal_vert_v2_ptr[E2ECV_0];
    const WP_TYPE pnv1_1 = primal_normal_vert_v1_ptr[E2ECV_1];
    const WP_TYPE pnv2_1 = primal_normal_vert_v2_ptr[E2ECV_1];
    const WP_TYPE pnv1_2 = primal_normal_vert_v1_ptr[E2ECV_2];
    const WP_TYPE pnv2_2 = primal_normal_vert_v2_ptr[E2ECV_2];
    const WP_TYPE pnv1_3 = primal_normal_vert_v1_ptr[E2ECV_3];
    const WP_TYPE pnv2_3 = primal_normal_vert_v2_ptr[E2ECV_3];

    const WP_TYPE ivv = inv_vert_vert_length_ptr[edge_index];
    const WP_TYPE ivv_sqr = ivv * ivv;
    const WP_TYPE ipe = inv_primal_edge_length_ptr[edge_index];
    const WP_TYPE ipe_sqr = ipe * ipe;

    // Walk this block's K_LOOP_LEN-sized chunk as TILE_K-wide inner tiles.
    // Clamp on the tail chunk (KDim not a multiple of K_LOOP_LEN) so we
    // never issue an inner tile that is wholly beyond KDim -- load_masked /
    // store_masked already handle a *partially* out-of-range tile (same as
    // the naive kernel's last K-tile), but a wholly out-of-range one is
    // avoided here rather than relied upon.
    const index_type k_tile_base = k_chunk_base / TILE_K;
    const index_type k_remaining = KDim - k_chunk_base;
    const index_type num_inner_tiles =
        (k_remaining < K_LOOP_LEN) ? ((k_remaining + TILE_K - 1) / TILE_K) : (K_LOOP_LEN / TILE_K);

    for (index_type t = 0; t < num_inner_tiles; ++t) {
        const index_type k_tile = k_tile_base + t;

        auto u0 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_0));
        auto u1 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_1));
        auto u2 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_2));
        auto u3 = ct::element_cast<double>(load_ktile_vp(u_vert_ptr, KDim, u_vert_k_stride, k_tile, E2C2V_3));
        auto v0 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_0));
        auto v1 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_1));
        auto v2 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_2));
        auto v3 = ct::element_cast<double>(load_ktile_vp(v_vert_ptr, KDim, v_vert_k_stride, k_tile, E2C2V_3));

        auto z2e = ct::element_cast<double>(load_ktile_wp(z_nabla2_e_ptr, KDim, z_nabla2_e_k_stride, k_tile, edge_index));

        auto nabv_tang = u0 * pnv1_0 + v0 * pnv2_0 + u1 * pnv1_1 + v1 * pnv2_1;
        auto nabv_norm = u2 * pnv1_2 + v2 * pnv2_2 + u3 * pnv1_3 + v3 * pnv2_3;
        auto result = 4.0 * ((nabv_norm - 2.0 * z2e) * ivv_sqr + (nabv_tang - 2.0 * z2e) * ipe_sqr);

        store_ktile_vp(z_nabla4_e2_wp_ptr, KDim, z_nabla4_e2_wp_k_stride, k_tile, edge_index, ct::element_cast<VP_TYPE>(result));
    }
}

void run_cutile_nabla4_unstructured_kloop_launcher(index_type EdgeDim,
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
    VP_TYPE* __restrict__ z_nabla4_e2_wp_ptr) {
    dim3 grid(EdgeDim, (KDim + K_LOOP_LEN - 1) / K_LOOP_LEN, 1);
    run_cutile_nabla4_unstructured_kloop<<<grid, 1>>>(EdgeDim,
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
    GT_CUDA_CHECK(cudaGetLastError());
    return;
}
