#include <vector>

#include "nabla4_unstructured.hpp"

template<typename T, backend_impl I>
std::vector<std::vector<float>> run_validation(T& benchmark_object) {
    benchmark_object.template run<I>();
    return benchmark_object.get_output();
}
