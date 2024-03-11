#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "wrapper.hpp"

NB_MODULE(icon_benchmark, m) {
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
    m.def("nabla4_benchmark_structured_naive", &nabla4_benchmark_structured_naive);
    m.def("nabla4_benchmark_structured_cpu_ifirst", &nabla4_benchmark_structured_cpu_ifirst);
    m.def("nabla4_benchmark_structured_cpu_kfirst", &nabla4_benchmark_structured_cpu_kfirst);
    m.def("nabla4_benchmark_structured_gpu", &nabla4_benchmark_structured_gpu);
}
