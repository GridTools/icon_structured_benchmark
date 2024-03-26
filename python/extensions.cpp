#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

#include "wrapper.hpp"

NB_MODULE(icon_benchmark, m) {
    nanobind::class_<nabla4_validation_data>(m, "nabla4_validation_data")
        .def_rw("CellDim", &nabla4_validation_data::CellDim)
        .def_rw("EdgeDim", &nabla4_validation_data::EdgeDim)
        .def_rw("VertexDim", &nabla4_validation_data::VertexDim)
        .def_rw("KDim", &nabla4_validation_data::KDim)
        .def_rw("ECVDim", &nabla4_validation_data::ECVDim)
        .def_rw("u_vert", &nabla4_validation_data::u_vert)
        .def_rw("v_vert", &nabla4_validation_data::v_vert)
        .def_rw("primal_normal_vert_v1", &nabla4_validation_data::primal_normal_vert_v1)
        .def_rw("primal_normal_vert_v2", &nabla4_validation_data::primal_normal_vert_v2)
        .def_rw("z_nabla2_e", &nabla4_validation_data::z_nabla2_e)
        .def_rw("inv_vert_vert_length", &nabla4_validation_data::inv_vert_vert_length)
        .def_rw("inv_primal_edge_length", &nabla4_validation_data::inv_primal_edge_length)
        .def_rw("z_nabla4_e2_wp", &nabla4_validation_data::z_nabla4_e2_wp);
    m.def("get_nabla4_benchmark_validation_data", &get_nabla4_benchmark_validation_data);
    /// Unstructured
    m.def("nabla4_benchmark_unstructured_naive", &nabla4_benchmark_unstructured_naive);
    m.def("nabla4_benchmark_unstructured_cpu_ifirst", &nabla4_benchmark_unstructured_cpu_ifirst);
    m.def("nabla4_benchmark_unstructured_cpu_kfirst", &nabla4_benchmark_unstructured_cpu_kfirst);
    m.def("nabla4_benchmark_unstructured_gpu", &nabla4_benchmark_unstructured_gpu);
    m.def("nabla4_validate_unstructured_naive", &nabla4_validate_unstructured_naive);
    m.def("nabla4_validate_unstructured_cpu_ifirst", &nabla4_validate_unstructured_cpu_ifirst);
    m.def("nabla4_validate_unstructured_cpu_kfirst", &nabla4_validate_unstructured_cpu_kfirst);
    m.def("nabla4_validate_unstructured_gpu", &nabla4_validate_unstructured_gpu);
    /// Structured
    m.def("nabla4_benchmark_structured_simple_naive", &nabla4_benchmark_structured_simple_naive);
    m.def("nabla4_benchmark_structured_simple_cpu_ifirst", &nabla4_benchmark_structured_simple_cpu_ifirst);
    m.def("nabla4_benchmark_structured_simple_cpu_kfirst", &nabla4_benchmark_structured_simple_cpu_kfirst);
    m.def("nabla4_benchmark_structured_simple_gpu", &nabla4_benchmark_structured_simple_gpu);
    m.def("nabla4_validate_structured_simple_naive", &nabla4_validate_structured_simple_naive);
    m.def("nabla4_validate_structured_simple_cpu_ifirst", &nabla4_validate_structured_simple_cpu_ifirst);
    m.def("nabla4_validate_structured_simple_cpu_kfirst", &nabla4_validate_structured_simple_cpu_kfirst);
    m.def("nabla4_validate_structured_simple_gpu", &nabla4_validate_structured_simple_gpu);
    /// Structured torus
    m.def("nabla4_benchmark_structured_torus_naive", &nabla4_benchmark_structured_torus_naive);
    m.def("nabla4_benchmark_structured_torus_cpu_ifirst", &nabla4_benchmark_structured_torus_cpu_ifirst);
    m.def("nabla4_benchmark_structured_torus_cpu_kfirst", &nabla4_benchmark_structured_torus_cpu_kfirst);
    m.def("nabla4_benchmark_structured_torus_gpu", &nabla4_benchmark_structured_torus_gpu);
    m.def("nabla4_validate_structured_torus_naive", &nabla4_validate_structured_torus_naive);
    m.def("nabla4_validate_structured_torus_cpu_ifirst", &nabla4_validate_structured_torus_cpu_ifirst);
    m.def("nabla4_validate_structured_torus_cpu_kfirst", &nabla4_validate_structured_torus_cpu_kfirst);
    m.def("nabla4_validate_structured_torus_gpu", &nabla4_validate_structured_torus_gpu);
}
