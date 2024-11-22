#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

#include "wrapper.hpp"

NB_MODULE(icon_benchmark, m) {
    nanobind::class_<nabla4_data<Data::ifirst>>(m, "nabla4_data")
        .def_ro("CellDim", &nabla4_data<Data::ifirst>::CellDim)
        .def_ro("EdgeDim", &nabla4_data<Data::ifirst>::EdgeDim)
        .def_ro("VertexDim", &nabla4_data<Data::ifirst>::VertexDim)
        .def_ro("KDim", &nabla4_data<Data::ifirst>::KDim)
        .def_ro("ECVDim", &nabla4_data<Data::ifirst>::ECVDim)
        .def_ro("u_vert", &nabla4_data<Data::ifirst>::u_vert)
        .def_ro("v_vert", &nabla4_data<Data::ifirst>::v_vert)
        .def_ro("primal_normal_vert_v1", &nabla4_data<Data::ifirst>::primal_normal_vert_v1)
        .def_ro("primal_normal_vert_v2", &nabla4_data<Data::ifirst>::primal_normal_vert_v2)
        .def_ro("z_nabla2_e", &nabla4_data<Data::ifirst>::z_nabla2_e)
        .def_ro("inv_vert_vert_length", &nabla4_data<Data::ifirst>::inv_vert_vert_length)
        .def_ro("inv_primal_edge_length", &nabla4_data<Data::ifirst>::inv_primal_edge_length)
        .def_ro("z_nabla4_e2_wp", &nabla4_data<Data::ifirst>::z_nabla4_e2_wp);
    m.def("get_nabla4_benchmark_validation_data", &get_nabla4_benchmark_validation_data);
    /// Unstructured
    m.def("nabla4_benchmark_unstructured_naive", &nabla4_benchmark_unstructured_naive);
    m.def("nabla4_benchmark_unstructured_cpu_ifirst", &nabla4_benchmark_unstructured_cpu_ifirst);
    m.def("nabla4_benchmark_unstructured_cpu_ifirst_gridtools", &nabla4_benchmark_unstructured_cpu_ifirst_gridtools);
    m.def("nabla4_benchmark_unstructured_cpu_kfirst", &nabla4_benchmark_unstructured_cpu_kfirst);
    m.def("nabla4_benchmark_unstructured_cpu_kfirst_gridtools", &nabla4_benchmark_unstructured_cpu_kfirst_gridtools);
    m.def("nabla4_benchmark_unstructured_gpu_kloop_gridtools", &nabla4_benchmark_unstructured_gpu_kloop_gridtools);
    m.def("nabla4_vertical_benchmark_unstructured_gpu_kloop_gridtools",
        &nabla4_vertical_benchmark_unstructured_gpu_kloop_gridtools);
    m.def("nabla4_benchmark_unstructured_gpu_naive_gridtools", &nabla4_benchmark_unstructured_gpu_naive_gridtools);
    m.def("nabla4_vertical_benchmark_unstructured_gpu_naive_gridtools",
        &nabla4_vertical_benchmark_unstructured_gpu_naive_gridtools);
    m.def("nabla4_validate_unstructured_naive", &nabla4_validate_unstructured_naive);
    m.def("nabla4_validate_unstructured_cpu_ifirst", &nabla4_validate_unstructured_cpu_ifirst);
    m.def("nabla4_validate_unstructured_cpu_ifirst_gridtools", &nabla4_validate_unstructured_cpu_ifirst_gridtools);
    m.def("nabla4_validate_unstructured_cpu_kfirst", &nabla4_validate_unstructured_cpu_kfirst);
    m.def("nabla4_validate_unstructured_cpu_kfirst_gridtools", &nabla4_validate_unstructured_cpu_kfirst_gridtools);
    m.def("nabla4_validate_unstructured_gpu_kloop_gridtools", &nabla4_validate_unstructured_gpu_kloop_gridtools);
    m.def("nabla4_vertical_validate_unstructured_gpu_kloop_gridtools",
        &nabla4_vertical_validate_unstructured_gpu_kloop_gridtools);
    m.def("nabla4_validate_unstructured_gpu_naive_gridtools", &nabla4_validate_unstructured_gpu_naive_gridtools);
    m.def("nabla4_vertical_validate_unstructured_gpu_naive_gridtools",
        &nabla4_vertical_validate_unstructured_gpu_naive_gridtools);
    /// Structured
    m.def("nabla4_benchmark_structured_simple_naive", &nabla4_benchmark_structured_simple_naive);
    m.def("nabla4_benchmark_structured_simple_cpu_ifirst", &nabla4_benchmark_structured_simple_cpu_ifirst);
    m.def("nabla4_benchmark_structured_simple_cpu_kfirst", &nabla4_benchmark_structured_simple_cpu_kfirst);
    m.def("nabla4_validate_structured_simple_naive", &nabla4_validate_structured_simple_naive);
    m.def("nabla4_validate_structured_simple_cpu_ifirst", &nabla4_validate_structured_simple_cpu_ifirst);
    m.def("nabla4_validate_structured_simple_cpu_kfirst", &nabla4_validate_structured_simple_cpu_kfirst);
    /// Structured torus
    m.def("nabla4_benchmark_structured_torus_naive", &nabla4_benchmark_structured_torus_naive);
    m.def("nabla4_benchmark_structured_torus_cpu_ifirst", &nabla4_benchmark_structured_torus_cpu_ifirst);
    m.def("nabla4_benchmark_structured_torus_cpu_ifirst_gridtools",
        &nabla4_benchmark_structured_torus_cpu_ifirst_gridtools);
    m.def("nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo",
        &nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo);
    m.def("nabla4_benchmark_structured_torus_cpu_kfirst", &nabla4_benchmark_structured_torus_cpu_kfirst);
    m.def("nabla4_benchmark_structured_torus_cpu_kfirst_gridtools",
        &nabla4_benchmark_structured_torus_cpu_kfirst_gridtools);
    m.def("nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo",
        &nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo);
    m.def("nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo",
        &nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo);
    m.def("nabla4_vertical_benchmark_structured_torus_gpu_kloop_gridtools_halo",
        &nabla4_vertical_benchmark_structured_torus_gpu_kloop_gridtools_halo);
    m.def("nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo",
        &nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo);
    m.def("nabla4_vertical_benchmark_structured_torus_gpu_naive_gridtools_halo",
        &nabla4_vertical_benchmark_structured_torus_gpu_naive_gridtools_halo);
    m.def("nabla4_validate_structured_torus_naive", &nabla4_validate_structured_torus_naive);
    m.def("nabla4_validate_structured_torus_cpu_ifirst", &nabla4_validate_structured_torus_cpu_ifirst);
    m.def("nabla4_validate_structured_torus_cpu_ifirst_gridtools",
        &nabla4_validate_structured_torus_cpu_ifirst_gridtools);
    m.def("nabla4_validate_structured_torus_cpu_ifirst_gridtools_halo",
        &nabla4_validate_structured_torus_cpu_ifirst_gridtools_halo);
    m.def("nabla4_validate_structured_torus_cpu_kfirst", &nabla4_validate_structured_torus_cpu_kfirst);
    m.def("nabla4_validate_structured_torus_cpu_kfirst_gridtools",
        &nabla4_validate_structured_torus_cpu_kfirst_gridtools);
    m.def("nabla4_validate_structured_torus_cpu_kfirst_gridtools_halo",
        &nabla4_validate_structured_torus_cpu_kfirst_gridtools_halo);
    m.def("nabla4_validate_structured_torus_gpu_kloop_gridtools_halo",
        &nabla4_validate_structured_torus_gpu_kloop_gridtools_halo);
    m.def("nabla4_vertical_validate_structured_torus_gpu_kloop_gridtools_halo",
        &nabla4_vertical_validate_structured_torus_gpu_kloop_gridtools_halo);
    m.def("nabla4_validate_structured_torus_gpu_naive_gridtools_halo",
        &nabla4_validate_structured_torus_gpu_naive_gridtools_halo);
    m.def("nabla4_vertical_validate_structured_torus_gpu_naive_gridtools_halo",
        &nabla4_vertical_validate_structured_torus_gpu_naive_gridtools_halo);
    /// Interpolate
    m.def("interpolate_validate_unstructured_cpu_ifirst", &interpolate_validate_unstructured_cpu_ifirst);
    m.def("interpolate_validate_unstructured_cpu_kfirst", &interpolate_validate_unstructured_cpu_kfirst);
    m.def("interpolate_validate_unstructured_gpu_kloop", &interpolate_validate_unstructured_gpu_kloop);
    m.def("interpolate_validate_unstructured_gpu_naive", &interpolate_validate_unstructured_gpu_naive);
    m.def("interpolate_benchmark_unstructured_cpu_ifirst", &interpolate_benchmark_unstructured_cpu_ifirst);
    m.def("interpolate_benchmark_unstructured_cpu_kfirst", &interpolate_benchmark_unstructured_cpu_kfirst);
    m.def("interpolate_benchmark_unstructured_gpu_kloop", &interpolate_benchmark_unstructured_gpu_kloop);
    m.def("interpolate_benchmark_unstructured_gpu_naive", &interpolate_benchmark_unstructured_gpu_naive);
    m.def("interpolate_validate_structured_cpu_ifirst", &interpolate_validate_structured_cpu_ifirst);
    m.def("interpolate_validate_structured_cpu_kfirst", &interpolate_validate_structured_cpu_kfirst);
    m.def("interpolate_validate_structured_gpu_kloop", &interpolate_validate_structured_gpu_kloop);
    m.def("interpolate_validate_structured_gpu_naive", &interpolate_validate_structured_gpu_naive);
    m.def("interpolate_benchmark_structured_cpu_ifirst", &interpolate_benchmark_structured_cpu_ifirst);
    m.def("interpolate_benchmark_structured_cpu_kfirst", &interpolate_benchmark_structured_cpu_kfirst);
    m.def("interpolate_benchmark_structured_gpu_kloop", &interpolate_benchmark_structured_gpu_kloop);
    m.def("interpolate_benchmark_structured_gpu_naive", &interpolate_benchmark_structured_gpu_naive);
    // Nabla4 & Interpolate separate
    m.def("nabla4_interpolate_validate_unstructured_cpu_ifirst_separate",
        &nabla4_interpolate_validate_unstructured_cpu_ifirst_separate);
    m.def("nabla4_interpolate_validate_structured_cpu_ifirst_separate",
        &nabla4_interpolate_validate_structured_cpu_ifirst_separate);
    m.def("nabla4_interpolate_validate_unstructured_cpu_kfirst_separate",
        &nabla4_interpolate_validate_unstructured_cpu_kfirst_separate);
    m.def("nabla4_interpolate_validate_structured_cpu_kfirst_separate",
        &nabla4_interpolate_validate_structured_cpu_kfirst_separate);
    m.def("nabla4_interpolate_validate_unstructured_gpu_naive_separate",
        &nabla4_interpolate_validate_unstructured_gpu_naive_separate);
    m.def("nabla4_vertical_interpolate_validate_unstructured_gpu_naive_separate",
        &nabla4_vertical_interpolate_validate_unstructured_gpu_naive_separate);
    m.def("nabla4_interpolate_validate_unstructured_gpu_kloop_separate",
        &nabla4_interpolate_validate_unstructured_gpu_kloop_separate);
    m.def("nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_separate",
        &nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_separate);
    m.def("nabla4_interpolate_validate_structured_gpu_naive_separate",
        &nabla4_interpolate_validate_structured_gpu_naive_separate);
    m.def("nabla4_vertical_interpolate_validate_structured_gpu_naive_separate",
        &nabla4_vertical_interpolate_validate_structured_gpu_naive_separate);
    m.def("nabla4_interpolate_validate_structured_gpu_kloop_separate",
        &nabla4_interpolate_validate_structured_gpu_kloop_separate);
    m.def("nabla4_vertical_interpolate_validate_structured_gpu_kloop_separate",
        &nabla4_vertical_interpolate_validate_structured_gpu_kloop_separate);
    m.def("nabla4_interpolate_benchmark_unstructured_cpu_ifirst_separate",
        &nabla4_interpolate_benchmark_unstructured_cpu_ifirst_separate);
    m.def("nabla4_interpolate_benchmark_structured_cpu_ifirst_separate",
        &nabla4_interpolate_benchmark_structured_cpu_ifirst_separate);
    m.def("nabla4_interpolate_benchmark_unstructured_cpu_kfirst_separate",
        &nabla4_interpolate_benchmark_unstructured_cpu_kfirst_separate);
    m.def("nabla4_interpolate_benchmark_structured_cpu_kfirst_separate",
        &nabla4_interpolate_benchmark_structured_cpu_kfirst_separate);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_naive_separate",
        &nabla4_interpolate_benchmark_unstructured_gpu_naive_separate);
    m.def("nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_separate",
        &nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_separate);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate",
        &nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate);
    m.def("nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_separate",
        &nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_separate);
    m.def("nabla4_interpolate_benchmark_structured_gpu_naive_separate",
        &nabla4_interpolate_benchmark_structured_gpu_naive_separate);
    m.def("nabla4_vertical_interpolate_benchmark_structured_gpu_naive_separate",
        &nabla4_vertical_interpolate_benchmark_structured_gpu_naive_separate);
    m.def("nabla4_interpolate_benchmark_structured_gpu_kloop_separate",
        &nabla4_interpolate_benchmark_structured_gpu_kloop_separate);
    m.def("nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_separate",
        &nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_separate);
    // Nabla4 & Interpolate inlined
    m.def("nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined",
        &nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined);
    m.def("nabla4_interpolate_validate_structured_cpu_ifirst_inlined",
        &nabla4_interpolate_validate_structured_cpu_ifirst_inlined);
    m.def("nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined",
        &nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined);
    m.def("nabla4_interpolate_validate_structured_cpu_kfirst_inlined",
        &nabla4_interpolate_validate_structured_cpu_kfirst_inlined);
    m.def("nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined",
        &nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined);
    m.def("nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined",
        &nabla4_interpolate_benchmark_structured_cpu_ifirst_inlined);
    m.def("nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined",
        &nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined);
    m.def("nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined",
        &nabla4_interpolate_benchmark_structured_cpu_kfirst_inlined);
    m.def("nabla4_interpolate_validate_unstructured_gpu_naive_inlined",
        &nabla4_interpolate_validate_unstructured_gpu_naive_inlined);
    m.def("nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined",
        &nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined);
    m.def("nabla4_interpolate_validate_unstructured_gpu_kloop_inlined",
        &nabla4_interpolate_validate_unstructured_gpu_kloop_inlined);
    m.def("nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined",
        &nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined);
    m.def("nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_cached",
        &nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_cached);
    m.def("nabla4_interpolate_validate_structured_gpu_naive_inlined",
        &nabla4_interpolate_validate_structured_gpu_naive_inlined);
    m.def("nabla4_vertical_interpolate_validate_structured_gpu_naive_inlined",
        &nabla4_vertical_interpolate_validate_structured_gpu_naive_inlined);
    m.def("nabla4_interpolate_validate_structured_gpu_naive_inlined_cached",
        &nabla4_interpolate_validate_structured_gpu_naive_inlined_cached);
    m.def("nabla4_vertical_interpolate_validate_structured_gpu_naive_inlined_cached",
        &nabla4_vertical_interpolate_validate_structured_gpu_naive_inlined_cached);
    m.def("nabla4_interpolate_validate_structured_gpu_kloop_inlined",
        &nabla4_interpolate_validate_structured_gpu_kloop_inlined);
    m.def("nabla4_vertical_interpolate_validate_structured_gpu_kloop_inlined",
        &nabla4_vertical_interpolate_validate_structured_gpu_kloop_inlined);
    m.def("nabla4_interpolate_validate_structured_gpu_kloop_inlined_cached",
        &nabla4_interpolate_validate_structured_gpu_kloop_inlined_cached);
    m.def("nabla4_vertical_interpolate_validate_structured_gpu_kloop_inlined_cached",
        &nabla4_vertical_interpolate_validate_structured_gpu_kloop_inlined_cached);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined",
        &nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined);
    m.def("nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined",
        &nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined",
        &nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined);
    m.def("nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined",
        &nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_cached",
        &nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_cached);
    m.def("nabla4_interpolate_benchmark_structured_gpu_naive_inlined",
        &nabla4_interpolate_benchmark_structured_gpu_naive_inlined);
    m.def("nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined",
        &nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined);
    m.def("nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached",
        &nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached);
    m.def("nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined_cached",
        &nabla4_vertical_interpolate_benchmark_structured_gpu_naive_inlined_cached);
    m.def("nabla4_interpolate_benchmark_gpu_kloop_roofline",
        &nabla4_interpolate_benchmark_gpu_kloop_roofline);
    m.def("nabla4_interpolate_benchmark_structured_gpu_kloop_inlined",
        &nabla4_interpolate_benchmark_structured_gpu_kloop_inlined);
    m.def("nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline",
        &nabla4_vertical_interpolate_benchmark_gpu_kloop_roofline);
    m.def("nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined",
        &nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined);
    m.def("nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached",
        &nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached);
    m.def("nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined_cached",
        &nabla4_vertical_interpolate_benchmark_structured_gpu_kloop_inlined_cached);
    // Nabla4 & Interpolate inlined v2v
    m.def("nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined_v2v",
        &nabla4_interpolate_validate_unstructured_cpu_ifirst_inlined_v2v);
    m.def("nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined_v2v",
        &nabla4_interpolate_validate_unstructured_cpu_kfirst_inlined_v2v);
    m.def("nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v",
        &nabla4_interpolate_benchmark_unstructured_cpu_ifirst_inlined_v2v);
    m.def("nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v",
        &nabla4_interpolate_benchmark_unstructured_cpu_kfirst_inlined_v2v);
    m.def("nabla4_interpolate_validate_unstructured_gpu_naive_inlined_v2v",
        &nabla4_interpolate_validate_unstructured_gpu_naive_inlined_v2v);
    m.def("nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined_v2v",
        &nabla4_vertical_interpolate_validate_unstructured_gpu_naive_inlined_v2v);
    m.def("nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_v2v",
        &nabla4_interpolate_validate_unstructured_gpu_kloop_inlined_v2v);
    m.def("nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined_v2v",
        &nabla4_vertical_interpolate_validate_unstructured_gpu_kloop_inlined_v2v);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v",
        &nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v);
    m.def("nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v",
        &nabla4_vertical_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v);
    m.def("nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v",
        &nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v);
    m.def("nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v",
        &nabla4_vertical_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v);
}
