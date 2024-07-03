#include <vector>

#include "common.hpp"

template <typename T, backend_impl I>
auto run_validation(T &benchmark_object) -> decltype(benchmark_object.get_output()) {
    benchmark_object.template run<I>();
    return benchmark_object.get_output();
}
