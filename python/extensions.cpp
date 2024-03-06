#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "wrapper.hpp"

NB_MODULE(icon_benchmark, m) {
    m.def("nabla4_benchmark_naive", &nabla4_benchmark_naive);
    m.def("nabla4_benchmark_cpu_ifirst", &nabla4_benchmark_cpu_ifirst);
    m.def("nabla4_benchmark_cpu_kfirst", &nabla4_benchmark_cpu_kfirst);
    m.def("nabla4_benchmark_gpu", &nabla4_benchmark_gpu);
    m.def("nabla4_validate_naive", &nabla4_validate_naive);
    m.def("nabla4_validate_cpu_ifirst", &nabla4_validate_cpu_ifirst);
    m.def("nabla4_validate_cpu_kfirst", &nabla4_validate_cpu_kfirst);
    m.def("nabla4_validate_gpu", &nabla4_validate_gpu);
}
