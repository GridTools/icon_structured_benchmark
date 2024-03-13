#include "wrapper.hpp"

#include "validation.hpp"

/// TODO: See if we can avoid declaring nabla4_benchmark_* and use templates.
/// Problem with templates are python bindings

template <backend_impl I>
std::vector<double> nabla4_benchmark_unstructured(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v, e2ecv, CellDim, VertexDim, EdgeDim, KDim, ECVDim};
    return run_benchmark<nabla4_unstructured, I>(nabla4_benchmark_object, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_unstructured_naive(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
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

std::vector<double> nabla4_benchmark_unstructured_cpu_ifirst(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
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

std::vector<double> nabla4_benchmark_unstructured_cpu_kfirst(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
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

std::vector<double> nabla4_benchmark_unstructured_gpu(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
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

std::vector<std::vector<float>> nabla4_validate_unstructured_naive(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v,
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
    return run_validation<nabla4_unstructured, naive>(nabla4_benchmark_object);
}

std::vector<std::vector<float>> nabla4_validate_unstructured_cpu_ifirst(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v,
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
    return run_validation<nabla4_unstructured, cpu_ifirst>(nabla4_benchmark_object);
}

std::vector<std::vector<float>> nabla4_validate_unstructured_cpu_kfirst(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v,
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
    return run_validation<nabla4_unstructured, cpu_kfirst>(nabla4_benchmark_object);
}

std::vector<std::vector<float>> nabla4_validate_unstructured_gpu(std::vector<std::vector<std::size_t>> &e2c2v,
    std::vector<std::vector<std::size_t>> &e2ecv,
    std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_unstructured nabla4_benchmark_object{e2c2v,
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
    return run_validation<nabla4_unstructured, gpu>(nabla4_benchmark_object);
}

template <backend_impl I>
std::vector<double> nabla4_benchmark_structured(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    nabla4_structured nabla4_benchmark_object{CellDim, VertexDim, EdgeDim, KDim, ECVDim};
    return run_benchmark<nabla4_structured, I>(nabla4_benchmark_object, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<naive>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<cpu_ifirst>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<cpu_kfirst>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<double> nabla4_benchmark_structured_gpu(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    int repetitions,
    int dry_runs) {
    return nabla4_benchmark_structured<gpu>(CellDim, VertexDim, EdgeDim, KDim, ECVDim, repetitions, dry_runs);
}

std::vector<std::vector<float>> nabla4_validate_structured_naive(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured nabla4_benchmark_object{CellDim,
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
    return run_validation<nabla4_structured, naive>(nabla4_benchmark_object);
}

std::vector<std::vector<float>> nabla4_validate_structured_cpu_ifirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured nabla4_benchmark_object{CellDim,
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
    return run_validation<nabla4_structured, cpu_ifirst>(nabla4_benchmark_object);
}

std::vector<std::vector<float>> nabla4_validate_structured_cpu_kfirst(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured nabla4_benchmark_object{CellDim,
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
    return run_validation<nabla4_structured, cpu_kfirst>(nabla4_benchmark_object);
}

std::vector<std::vector<float>> nabla4_validate_structured_gpu(std::size_t CellDim,
    std::size_t VertexDim,
    std::size_t EdgeDim,
    std::size_t KDim,
    std::size_t ECVDim,
    std::vector<std::vector<float>> &u_vert,
    std::vector<std::vector<float>> &v_vert,
    std::vector<double> &primal_normal_vert_v1,
    std::vector<double> &primal_normal_vert_v2,
    std::vector<std::vector<double>> &z_nabla2_e,
    std::vector<double> &inv_vert_vert_length,
    std::vector<double> &inv_primal_edge_length) {
    nabla4_structured nabla4_benchmark_object{CellDim,
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
    return run_validation<nabla4_structured, gpu>(nabla4_benchmark_object);
}