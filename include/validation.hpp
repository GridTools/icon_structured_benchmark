#include <vector>

#include "common.hpp"

template <typename R, typename T, backend_impl I>
R run_validation(T &benchmark_object) {
    benchmark_object.template run<I>();
    return benchmark_object.get_output();
}
