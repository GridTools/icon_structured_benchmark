#include "cuda_tile.h"
#include "nabla4_gridtools.hpp"

namespace ct = cuda::tiles;
using namespace ct::literals;

// ---------------------------------------------------------------------------
// Redesign rationale (ncu-driven, torus_100000_100000_256.nc): the original
// design tiled along K for a single edge per block. Storage here is k-major
// ([k][edge]), so for one fixed edge, walking K strides by k_stride every
// element -- never contiguous. ncu confirmed this: ~98% "Memory Throughput"
// but only ~2.6% DRAM Throughput (i.e. L1TEX-pipe-bound, not bandwidth
// bound), 8 of 32 bytes used per sector on every load/store, and 89% of
// stall cycles were L1TEX scoreboard stalls.
//
// e2c2v_ptr/e2ecv_ptr and every array indexed directly by edge_index
// (z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length, the output
// store) ARE edge-major/contiguous, so tiling across EDGES instead of K
// turns those into genuine coalesced tile loads/stores. The u_vert/v_vert
// (via E2C2V) and primal_normal_vert (via E2ECV) reads are still inherently
// random gathers -- unstructured connectivity can't be made contiguous --
// but batching TILE_E edges' worth of gathers into one warp-wide gather
// instruction (ct::load on a tile of pointers) replaces what used to be a
// single edge's scalar reads repeated once per (edge, k) block.
// ---------------------------------------------------------------------------
constexpr int TILE_E = 32;
constexpr index_type K_LOOP_LEN = 80;

using vpE_tile_t = ct::tile<VP_TYPE, ct::shape<TILE_E>>;
using wpE_tile_t = ct::tile<WP_TYPE, ct::shape<TILE_E>>;
using idxE_tile_t = ct::tile<index_type, ct::shape<TILE_E>>;

// Connectivity (E2C2V/E2ECV) is edge-major/contiguous: e2c2v_ptr[edge_index +
// slot*slot_stride]. A TILE_E-wide tile over the edge dimension is therefore
// a genuine coalesced tile load, not a gather.
__tile__ idxE_tile_t load_conn_tile(const index_type* __restrict__ base_ptr,
    index_type EdgeDim,
    index_type slot_stride,
    index_type slot,
    index_type edge_tile) {
    const index_type* slice = base_ptr + slot * slot_stride;
    auto view = ct::partition_view{ct::tensor_span{slice, ct::extents{EdgeDim}}, ct::shape<TILE_E>{}};
    idxE_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(edge_tile);
    return ret;
}

// K-independent, edge-major array (inv_vert_vert_length / inv_primal_edge_length).
__tile__ wpE_tile_t load_edgeE_tile(const WP_TYPE* __restrict__ base_ptr, index_type EdgeDim, index_type edge_tile) {
    auto view = ct::partition_view{ct::tensor_span{base_ptr, ct::extents{EdgeDim}}, ct::shape<TILE_E>{}};
    wpE_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(edge_tile);
    return ret;
}

// K-dependent, edge-major array (z_nabla2_e): element(edge,k) = ptr[k*k_stride+edge].
// Shift by k first (scalar), then tile over the contiguous edge dimension.
__tile__ wpE_tile_t load_edgeE_ktile(const WP_TYPE* __restrict__ base_ptr,
    index_type EdgeDim,
    index_type k_stride,
    index_type k,
    index_type edge_tile) {
    const WP_TYPE* shifted = base_ptr + k * k_stride;
    auto view = ct::partition_view{ct::tensor_span{shifted, ct::extents{EdgeDim}}, ct::shape<TILE_E>{}};
    wpE_tile_t ret;
    [[ cutile::hint(0, latency=1) ]]
    ret = view.load_masked(edge_tile);
    return ret;
}

__tile__ void store_edgeE_ktile(VP_TYPE* __restrict__ base_ptr,
    index_type EdgeDim,
    index_type k_stride,
    index_type k,
    index_type edge_tile,
    const vpE_tile_t& value) {
    VP_TYPE* shifted = base_ptr + k * k_stride;
    auto view = ct::partition_view{ct::tensor_span{shifted, ct::extents{EdgeDim}}, ct::shape<TILE_E>{}};
    [[ cutile::hint(0, latency=1) ]]
    view.store_masked(value, edge_tile);
}

// Genuine gather: gathered_idx is a tile of TILE_E per-lane vertex indices
// (arbitrary/unstructured -- this is the one part of the computation that
// cannot be made contiguous). Unmasked ct::load is safe here because
// load_conn_tile above zero-pads out-of-range lanes to vertex index 0
// (always a valid address); the final store is what's masked against
// EdgeDim, so garbage results from padded lanes are simply discarded there.
__tile__ vpE_tile_t gather_vp_ktile(const VP_TYPE* __restrict__ base_ptr,
    index_type k_stride,
    index_type k,
    const idxE_tile_t& gathered_idx) {
    const VP_TYPE* k_slice = base_ptr + k * k_stride;
    auto ptrs = k_slice + gathered_idx;
    vpE_tile_t ret;
    [[ cutile::hint(0, latency=8) ]]
    ret = ct::load(ptrs);
    return ret;
}

__tile__ wpE_tile_t gather_wp(const WP_TYPE* __restrict__ base_ptr, const idxE_tile_t& gathered_idx) {
    auto ptrs = base_ptr + gathered_idx;
    wpE_tile_t ret;
    [[ cutile::hint(0, latency=5) ]]
    ret = ct::load(ptrs);
    return ret;
}

// ---------------------------------------------------------------------------
// Naive variant: grid = (ceil(EdgeDim/TILE_E), KDim). One block per
// (edge-tile, k) -- connectivity/pnv/ivv/ipe are re-read every k, mirroring
// the original naive kernel's "no hoisting" design (just tiled over edges
// now instead of K).
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

    // Grid is launched (ceil(EdgeDim/TILE_E), KDim): one block per (edge-tile, k).
    const auto bid2 = ct::bid();
    const index_type edge_tile = static_cast<index_type>(bid2.x);
    const index_type k = static_cast<index_type>(bid2.y);

    const idxE_tile_t E2C2V_0 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 0, edge_tile);
    const idxE_tile_t E2C2V_1 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 1, edge_tile);
    const idxE_tile_t E2C2V_2 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 2, edge_tile);
    const idxE_tile_t E2C2V_3 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 3, edge_tile);
    const idxE_tile_t E2ECV_0 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 0, edge_tile);
    const idxE_tile_t E2ECV_1 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 1, edge_tile);
    const idxE_tile_t E2ECV_2 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 2, edge_tile);
    const idxE_tile_t E2ECV_3 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 3, edge_tile);

    auto pnv1_0 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_0);
    auto pnv2_0 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_0);
    auto pnv1_1 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_1);
    auto pnv2_1 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_1);
    auto pnv1_2 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_2);
    auto pnv2_2 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_2);
    auto pnv1_3 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_3);
    auto pnv2_3 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_3);

    auto ivv = load_edgeE_tile(inv_vert_vert_length_ptr, EdgeDim, edge_tile);
    auto ivv_sqr = ivv * ivv;
    auto ipe = load_edgeE_tile(inv_primal_edge_length_ptr, EdgeDim, edge_tile);
    auto ipe_sqr = ipe * ipe;

    auto u0 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_0));
    auto u1 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_1));
    auto u2 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_2));
    auto u3 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_3));
    auto v0 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_0));
    auto v1 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_1));
    auto v2 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_2));
    auto v3 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_3));

    auto z2e = ct::element_cast<double>(load_edgeE_ktile(z_nabla2_e_ptr, EdgeDim, z_nabla2_e_k_stride, k, edge_tile));

    auto nabv_tang = u0 * pnv1_0 + v0 * pnv2_0 + u1 * pnv1_1 + v1 * pnv2_1;
    auto nabv_norm = u2 * pnv1_2 + v2 * pnv2_2 + u3 * pnv1_3 + v3 * pnv2_3;
    auto result = 4.0 * ((nabv_norm - 2.0 * z2e) * ivv_sqr + (nabv_tang - 2.0 * z2e) * ipe_sqr);

    store_edgeE_ktile(z_nabla4_e2_wp_ptr, EdgeDim, z_nabla4_e2_wp_k_stride, k, edge_tile, ct::element_cast<VP_TYPE>(result));
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
    dim3 grid((EdgeDim + TILE_E - 1) / TILE_E, KDim, 1);
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
// K-loop variant: grid = (ceil(EdgeDim/TILE_E), ceil(KDim/K_LOOP_LEN)). One
// block per edge-tile owns a K_LOOP_LEN-sized chunk of K levels;
// connectivity + pnv + ivv/ipe are hoisted once per block and reused across
// the runtime k loop, same amortization idea as the original kloop kernel,
// just on the edge axis instead of K.
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

    // Grid is launched (ceil(EdgeDim/TILE_E), ceil(KDim/K_LOOP_LEN)): one
    // block per (edge-tile, k-chunk). bid2.y selects which K_LOOP_LEN-sized
    // chunk of K levels this block owns.
    const auto bid2 = ct::bid();
    const index_type edge_tile = static_cast<index_type>(bid2.x);
    const index_type k_chunk_base = static_cast<index_type>(bid2.y) * K_LOOP_LEN;

    const idxE_tile_t E2C2V_0 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 0, edge_tile);
    const idxE_tile_t E2C2V_1 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 1, edge_tile);
    const idxE_tile_t E2C2V_2 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 2, edge_tile);
    const idxE_tile_t E2C2V_3 = load_conn_tile(e2c2v_ptr, EdgeDim, e2c2v_slot_stride, 3, edge_tile);
    const idxE_tile_t E2ECV_0 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 0, edge_tile);
    const idxE_tile_t E2ECV_1 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 1, edge_tile);
    const idxE_tile_t E2ECV_2 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 2, edge_tile);
    const idxE_tile_t E2ECV_3 = load_conn_tile(e2ecv_ptr, EdgeDim, e2ecv_slot_stride, 3, edge_tile);

    auto pnv1_0 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_0);
    auto pnv2_0 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_0);
    auto pnv1_1 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_1);
    auto pnv2_1 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_1);
    auto pnv1_2 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_2);
    auto pnv2_2 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_2);
    auto pnv1_3 = gather_wp(primal_normal_vert_v1_ptr, E2ECV_3);
    auto pnv2_3 = gather_wp(primal_normal_vert_v2_ptr, E2ECV_3);

    auto ivv = load_edgeE_tile(inv_vert_vert_length_ptr, EdgeDim, edge_tile);
    auto ivv_sqr = ivv * ivv;
    auto ipe = load_edgeE_tile(inv_primal_edge_length_ptr, EdgeDim, edge_tile);
    auto ipe_sqr = ipe * ipe;

    // Clamp on the tail chunk (KDim not a multiple of K_LOOP_LEN).
    const index_type k_remaining = KDim - k_chunk_base;
    const index_type num_k = (k_remaining < K_LOOP_LEN) ? k_remaining : K_LOOP_LEN;

    for (index_type t = 0; t < num_k; ++t) {
        const index_type k = k_chunk_base + t;

        auto u0 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_0));
        auto u1 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_1));
        auto u2 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_2));
        auto u3 = ct::element_cast<double>(gather_vp_ktile(u_vert_ptr, u_vert_k_stride, k, E2C2V_3));
        auto v0 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_0));
        auto v1 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_1));
        auto v2 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_2));
        auto v3 = ct::element_cast<double>(gather_vp_ktile(v_vert_ptr, v_vert_k_stride, k, E2C2V_3));

        auto z2e = ct::element_cast<double>(load_edgeE_ktile(z_nabla2_e_ptr, EdgeDim, z_nabla2_e_k_stride, k, edge_tile));

        auto nabv_tang = u0 * pnv1_0 + v0 * pnv2_0 + u1 * pnv1_1 + v1 * pnv2_1;
        auto nabv_norm = u2 * pnv1_2 + v2 * pnv2_2 + u3 * pnv1_3 + v3 * pnv2_3;
        auto result = 4.0 * ((nabv_norm - 2.0 * z2e) * ivv_sqr + (nabv_tang - 2.0 * z2e) * ipe_sqr);

        store_edgeE_ktile(z_nabla4_e2_wp_ptr, EdgeDim, z_nabla4_e2_wp_k_stride, k, edge_tile, ct::element_cast<VP_TYPE>(result));
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
    dim3 grid((EdgeDim + TILE_E - 1) / TILE_E, (KDim + K_LOOP_LEN - 1) / K_LOOP_LEN, 1);
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
