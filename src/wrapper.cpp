#include "wrapper.hpp"

#include "validation.hpp"

/// TODO: See if we can avoid declaring nabla4_benchmark_* and use templates.
/// Problem with templates are python bindings

nabla4_validation_data get_nabla4_benchmark_validation_data(std::vector<std::array<std::size_t, 4>> &e2c2v,
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

template <backend_impl I>
std::vector<double> nabla4_benchmark_unstructured(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    if constexpr (I == backend_impl::naive) {
        return run_benchmark<nabla4_unstructured<Data::ifirst>, I>(
            e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<nabla4_unstructured<Data::ifirst>, I>(
            e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<nabla4_unstructured<Data::kfirst>, I>(
            e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::gpu) {
        return run_benchmark<nabla4_unstructured<Data::kfirst>, I>(
            e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else {
        throw std::runtime_error("Undefined backend implementation");
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
    return nabla4_benchmark_unstructured<naive>(
        e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
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
    return nabla4_benchmark_unstructured<cpu_ifirst>(
        e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
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
    return nabla4_benchmark_unstructured<cpu_kfirst>(
        e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_gpu(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_unstructured<gpu>(
        e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
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
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured<Data::ifirst> nabla4_benchmark_object{e2c2v,
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
        inv_primal_edge_length};
    return run_validation<nabla4_unstructured<Data::ifirst>, naive>(nabla4_benchmark_object);
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
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured<Data::ifirst> nabla4_benchmark_object{e2c2v,
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
        inv_primal_edge_length};
    return run_validation<nabla4_unstructured<Data::ifirst>, cpu_ifirst>(nabla4_benchmark_object);
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
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured<Data::kfirst> nabla4_benchmark_object{e2c2v,
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
        inv_primal_edge_length};
    return run_validation<nabla4_unstructured<Data::kfirst>, cpu_kfirst>(nabla4_benchmark_object);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_unstructured_gpu(std::vector<std::array<std::size_t, 4>> &e2c2v,
    std::vector<std::array<std::size_t, 4>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured<Data::kfirst> nabla4_benchmark_object{e2c2v,
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
        inv_primal_edge_length};
    return run_validation<nabla4_unstructured<Data::kfirst>, gpu>(nabla4_benchmark_object);
}

template <backend_impl I>
std::vector<double> nabla4_benchmark_structured(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    if constexpr (I == backend_impl::naive) {
        return run_benchmark<nabla4_structured_simple<Data::ifirst>, I>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<nabla4_structured_simple<Data::ifirst>, I>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<nabla4_structured_simple<Data::kfirst>, I>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::gpu) {
        return run_benchmark<nabla4_structured_simple<Data::kfirst>, I>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
    } else {
        throw std::runtime_error("Undefined backend implementation");
    }
}

std::vector<double> nabla4_benchmark_structured_simple_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<naive>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<cpu_ifirst>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<cpu_kfirst>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_simple_gpu(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<gpu>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_simple<Data::ifirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_simple<Data::ifirst>, naive>(nabla4_benchmark_object);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_simple<Data::ifirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_simple<Data::ifirst>, cpu_ifirst>(nabla4_benchmark_object);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_simple<Data::kfirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_simple<Data::kfirst>, cpu_kfirst>(nabla4_benchmark_object);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_simple_gpu(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_simple<Data::ifirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_simple<Data::ifirst>, gpu>(nabla4_benchmark_object);
}

template <backend_impl I>
std::vector<double> nabla4_benchmark_structured_torus(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    if constexpr (I == backend_impl::naive) {
        return run_benchmark<nabla4_structured_torus<Data::ifirst>, backend_impl::cpu_ifirst>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<nabla4_structured_torus<Data::ifirst>, backend_impl::cpu_ifirst>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<nabla4_structured_torus<Data::kfirst>, backend_impl::cpu_kfirst>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
    } else if constexpr (I == backend_impl::gpu) {
        return run_benchmark<nabla4_structured_torus<Data::kfirst>, I>(
            CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
    } else {
        throw std::runtime_error("Undefined backend implementation");
    }
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
    return nabla4_benchmark_structured_torus<naive>(
        CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
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
    return nabla4_benchmark_structured_torus<cpu_ifirst>(
        CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
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
    return nabla4_benchmark_structured_torus<cpu_kfirst>(
        CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_torus_gpu(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured_torus<gpu>(
        CellDim, VertexDim, EdgeDim, KDim, ECVDim, longitude_dim, latitude_dim, repetitions, dry_runs);
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
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_torus<Data::ifirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_torus<Data::ifirst>, naive>(nabla4_benchmark_object);
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
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_torus<Data::ifirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_torus<Data::ifirst>, cpu_ifirst>(nabla4_benchmark_object);
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
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_torus<Data::kfirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_torus<Data::kfirst>, cpu_kfirst>(nabla4_benchmark_object);
}

std::vector<std::vector<VP_TYPE>> nabla4_validate_structured_torus_gpu(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::size_t longitude_dim,
    std::size_t latitude_dim,
    std::vector<std::vector<VP_TYPE>> &u_vert,
    std::vector<std::vector<VP_TYPE>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured_torus<Data::kfirst> nabla4_benchmark_object{CellDim,
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
        inv_primal_edge_length};
    return run_validation<nabla4_structured_torus<Data::kfirst>, gpu>(nabla4_benchmark_object);
}
