#pragma once

#include "mo_intp_rbf_rbf_vec_interpol_vertex.hpp"

#if defined(__CUDACC__)
template <typename T>
constexpr block_dims get_block_dims_structured_interpol() {
    throw std::runtime_error("Undefined block dimensions for type " + T::name + " in GPU backend");
};

template <>
constexpr block_dims get_block_dims_structured_interpol<std::size_t>() {
    return {32, 4, 1, 128};
};

template <>
constexpr block_dims get_block_dims_structured_interpol<std::int64_t>() {
    return {32, 8, 1, 256};
};

template <>
constexpr block_dims get_block_dims_structured_interpol<std::uint32_t>() {
    return {32, 9, 1, 288};
};

template <>
constexpr block_dims get_block_dims_structured_interpol<int>() {
    return {32, 9, 1, 288};
};

constexpr block_dims block_dims_structured_interpol = get_block_dims_structured_interpol<index_type>();
#endif

template <typename T>
class interpolate_structured : public mo_intp_rbf_rbf_vec_interpol_vertex<T> {
  public:
    interpolate_structured(std::size_t VertexDim, std::size_t EdgeDim, std::size_t KDim)
        : mo_intp_rbf_rbf_vec_interpol_vertex<T>(VertexDim, EdgeDim, KDim){};

    interpolate_structured(std::size_t VertexDim,
        std::size_t EdgeDim,
        std::size_t KDim,
        std::vector<std::vector<WP_TYPE>> &p_e_in,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
        std::vector<std::vector<WP_TYPE>> &ptr_coeff_2)
        : mo_intp_rbf_rbf_vec_interpol_vertex<T>(VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2){};

  private:
    void run_cpu_ifirst(){};
    void run_cpu_kfirst(){};
    void run_gpu_helper();

  public:
    /// Compute function timed for benchmarking
    template <backend_impl I>
    inline void run() {
        if constexpr (I == backend_impl::cpu_ifirst) {
            run_cpu_ifirst();
        } else if constexpr (I == backend_impl::cpu_kfirst) {
            run_cpu_kfirst();
        } else if constexpr (I == backend_impl::gpu) {
            run_gpu_helper();
        } else {
            throw std::runtime_error("Undefined backend implementation");
        }
    }
};

#if defined(__CUDACC__)
__global__ void __launch_bounds__(block_dims_structured_interpol.size, 2) run_gpu_interpol(){};

template <typename T>
inline void interpolate_structured<T>::run_gpu_helper() {
    // dim3 tblocks(block_dims_structured_interpol.x, block_dims_structured_interpol.y, block_dims_structured_interpol.z);
    // const index_type inner_grid_size = (x_dim - 2 * halo) * (y_dim - halo * 2);
    // dim3 grid((x_dim - 2 * halo + tblocks.x - 1) / tblocks.x,
    //     (y_dim - 2 * halo + tblocks.y - 1) / tblocks.y,
    //     (KDim + tblocks.z - 1) / tblocks.z);
    // run_gpu_interpol<<<grid, tblocks>>>();
    GT_CUDA_CHECK(cudaGetLastError());
};
#else
template <typename T>
inline void interpolate_structured<T>::run_gpu_helper() {
    throw std::runtime_error("GPU backend not enabled");
};
#endif
