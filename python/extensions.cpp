#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "wrapper.hpp"

NB_MODULE(icon_benchmark, m) {
    m.def("nabla4_benchmark", &nabla4_benchmark);
}
