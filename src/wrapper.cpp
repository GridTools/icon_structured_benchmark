#include <tuple>

#include "wrapper.hpp"

#include "validation.hpp"

/// TODO: See if we can avoid declaring nabla4_benchmark_* and use templates.
/// Problem with templates are python bindings

nabla4_data<Data::ifirst> get_nabla4_benchmark_validation_data(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim) {
    nabla4_unstructured<Data::ifirst> nabla4_benchmark_object{e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
    run_benchmark<nabla4_unstructured<Data::ifirst>, naive>(nabla4_benchmark_object, 1, 0);
    return nabla4_benchmark_object.get_validation_data();
}

template <backend_impl I, template <Data> typename T, typename... Args>
std::vector<double> nabla4_benchmark_vector(std::tuple<Args...> &&args, int repetitions, int dry_runs) {
    if constexpr (I == backend_impl::naive) {
        return run_benchmark<T<Data::ifirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<T<Data::ifirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<T<Data::kfirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else {
        throw std::runtime_error("Undefined backend implementation");
    }
}

template <backend_impl I, template <Data> typename T, typename... Args>
std::vector<std::vector<VP_TYPE>> nabla4_validate_vector(std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::naive) {
        T<Data::ifirst> benchmark_object{
            std::apply([](auto &&...args) { return T<Data::ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<Data::ifirst>, naive>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        T<Data::ifirst> benchmark_object{
            std::apply([](auto &&...args) { return T<Data::ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<Data::ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<Data::kfirst> benchmark_object{
            std::apply([](auto &&...args) { return T<Data::kfirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<Data::kfirst>, cpu_kfirst>(benchmark_object);
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::vector<double> benchmark_gridtools(std::tuple<Args...> &&args, int repetitions, int dry_runs) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<T<storage::cpu_ifirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<T<storage::cpu_kfirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu) {
        return run_benchmark<T<storage::gpu>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::vector<std::vector<VP_TYPE>> nabla4_validate_gridtools(std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        T<storage::cpu_ifirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<storage::cpu_kfirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_kfirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_kfirst>, cpu_kfirst>(benchmark_object);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu) {
        T<storage::gpu> benchmark_object{
            std::apply([](auto &&...args) { return T<storage::gpu>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::gpu>, gpu>(benchmark_object);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

std::vector<double> nabla4_benchmark_unstructured_naive(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<naive, nabla4_unstructured>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_ifirst(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_ifirst, nabla4_unstructured>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_ifirst_gridtools(std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_kfirst(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_kfirst, nabla4_unstructured>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_cpu_kfirst_gridtools(std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu_gridtools(std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, nabla4_unstructured_gt>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu_gridtools_naive(std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, nabla4_unstructured_gt_naive>(
        std::make_tuple(e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_naive(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<naive, nabla4_unstructured>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_ifirst(
    std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_ifirst, nabla4_unstructured>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_ifirst_gridtools(
    std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
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
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_ifirst, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_kfirst(
    std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_kfirst, nabla4_unstructured>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_cpu_kfirst_gridtools(
    std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
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
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_kfirst, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu_gridtools(
    std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
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
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu, nabla4_unstructured_gt>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu_gridtools_naive(
    std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
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
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu, nabla4_unstructured_gt_naive>(std::make_tuple(e2c2v,
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
        inv_primal_edge_length));
}

std::vector<double> nabla4_benchmark_structured_simple_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<naive, nabla4_structured_simple>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_ifirst, nabla4_structured_simple>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_kfirst, nabla4_structured_simple>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<naive, nabla4_structured_simple>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_ifirst, nabla4_structured_simple>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_kfirst, nabla4_structured_simple>(std::make_tuple(CellDim,
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
        inv_primal_edge_length));
}

std::vector<double> nabla4_benchmark_structured_torus_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<naive, nabla4_structured_torus>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_ifirst, nabla4_structured_torus>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_ifirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_structured_torus_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_vector<cpu_kfirst, nabla4_structured_torus>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_kfirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_structured_torus_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim), repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, nabla4_structured_torus_halo_gt>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu_gridtools_halo_naive(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, nabla4_structured_torus_halo_gt_naive>(
        std::make_tuple(CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, halo),
        repetitions,
        dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<naive, nabla4_structured_torus>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_ifirst, nabla4_structured_torus>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_ifirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_ifirst, nabla4_structured_torus_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_ifirst_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_ifirst, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        halo,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_vector<cpu_kfirst, nabla4_structured_torus>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_kfirst_gridtools(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_kfirst, nabla4_structured_torus_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_cpu_kfirst_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<cpu_kfirst, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        halo,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu_gridtools_halo(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu, nabla4_structured_torus_halo_gt>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        halo,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu_gridtools_halo_naive(index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<WP_TYPE> &primal_normal_vert_v1,
    std::vector<WP_TYPE> &primal_normal_vert_v2,
    std::vector<std::vector<WP_TYPE>> &z_nabla2_e,
    std::vector<WP_TYPE> &inv_vert_vert_length,
    std::vector<WP_TYPE> &inv_primal_edge_length) {
    return nabla4_validate_gridtools<gpu, nabla4_structured_torus_halo_gt_naive>(std::make_tuple(CellDim,
        VertexDim,
        EdgeDim,
        KDim,
        ECVDim,
        longitude_dim,
        latitude_dim,
        halo,
        u_vert,
        v_vert,
        primal_normal_vert_v1,
        primal_normal_vert_v2,
        z_nabla2_e,
        inv_vert_vert_length,
        inv_primal_edge_length));
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>> interpolate_validate_gridtools(
    std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        T<storage::cpu_ifirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<storage::cpu_kfirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_kfirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_kfirst>, cpu_kfirst>(benchmark_object);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu) {
        T<storage::gpu> benchmark_object{
            std::apply([](auto &&...args) { return T<storage::gpu>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::gpu>, gpu>(benchmark_object);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_cpu_ifirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::vector<std::array<index_type, 6>> &v2e,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_ifirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_cpu_kfirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::vector<std::array<index_type, 6>> &v2e,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_kfirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>> interpolate_validate_unstructured_gpu(
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::vector<std::array<index_type, 6>> &v2e,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_unstructured_gpu_naive(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::vector<std::array<index_type, 6>> &v2e,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu, interpolate_unstructured_naive>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::vector<double> interpolate_benchmark_unstructured_cpu_ifirst(std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_unstructured_cpu_kfirst(std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_unstructured_gpu(std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, interpolate_unstructured>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_unstructured_gpu_naive(std::vector<std::array<index_type, 6>> &v2e,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, interpolate_unstructured_naive>(
        std::make_tuple(v2e, VertexDim, EdgeDim, KDim), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_cpu_ifirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_ifirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_cpu_kfirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<cpu_kfirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_gpu(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::vector<double> interpolate_benchmark_structured_gpu_naive(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, interpolate_structured_naive>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo), repetitions, dry_runs);
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_cpu_ifirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_ifirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_cpu_kfirst(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<cpu_kfirst, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>> interpolate_validate_structured_gpu(
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu, interpolate_structured>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>>
interpolate_validate_structured_gpu_naive(std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    index_type longitude_dim,
    index_type latitude_dim,
    index_type halo,
    std::vector<std::vector<WP_TYPE>> &p_e_in,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_1,
    std::vector<std::vector<WP_TYPE>> &ptr_coeff_2) {
    return interpolate_validate_gridtools<gpu, interpolate_structured_naive>(
        std::make_tuple(VertexDim, EdgeDim, KDim, longitude_dim, latitude_dim, halo, p_e_in, ptr_coeff_1, ptr_coeff_2));
}

std::vector<double> nabla4_interpolate_benchmark_unstructured_gpu_naive_separate(
    std::vector<std::array<index_type, 4>> &e2c2v,
    std::vector<std::array<index_type, 4>> &e2ecv,
    std::vector<std::array<index_type, 6>> &v2e,
    index_type CellDim,
    index_type VertexDim,
    index_type EdgeDim,
    index_type KDim,
    index_type ECVDim,
    int repetitions,
    int dry_runs) {
    return benchmark_gridtools<gpu, interpolate_unstructured_naive_separate>(
        std::make_tuple(e2c2v, e2ecv, v2e, CellDim, VertexDim, EdgeDim, KDim, ECVDim), repetitions, dry_runs);
}
