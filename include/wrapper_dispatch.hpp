#pragma once

#include <tuple>

// Every src/wrapper_*.cpp already includes "wrapper.hpp" and "validation.hpp" before
// this header. Do not re-include them here: none of the headers they pull in
// (wrapper.hpp, benchmark.hpp, and the per-kernel headers below it) have include
// guards, so a second inclusion in the same translation unit redefines every
// class/function template in that tree.

template <backend_impl I, template <Data> typename T, typename... Args>
std::vector<double> nabla4_benchmark_vector(std::tuple<Args...> &&args, int repetitions, int dry_runs) {
    if constexpr (I == backend_impl::naive) {
        return run_benchmark<T<Data::ifirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<T<Data::ifirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<T<Data::kfirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else {
        throw std::runtime_error("Undefined backend implementation");
    }
}

template <backend_impl I, template <Data> typename T, typename... Args>
std::vector<std::vector<VP_TYPE>> nabla4_validate_vector(std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::naive) {
        T<Data::ifirst> benchmark_object{
            std::apply([](auto &&...args) { return T<Data::ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<Data::ifirst>, naive>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_ifirst) {
        T<Data::ifirst> benchmark_object{
            std::apply([](auto &&...args) { return T<Data::ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<Data::ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<Data::kfirst> benchmark_object{
            std::apply([](auto &&...args) { return T<Data::kfirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<Data::kfirst>, cpu_kfirst>(benchmark_object);
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::vector<double> benchmark_gridtools(std::tuple<Args...> &&args, int repetitions, int dry_runs) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        return run_benchmark<T<storage::cpu_ifirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        return run_benchmark<T<storage::cpu_kfirst>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu_kloop) {
        return run_benchmark<T<storage::gpu>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
    } else if constexpr (I == backend_impl::gpu_naive) {
        return run_benchmark<T<storage::gpu>, I>(std::forward<decltype(args)>(args), repetitions, dry_runs);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::vector<std::vector<VP_TYPE>> nabla4_validate_gridtools(std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        T<storage::cpu_ifirst> benchmark_object{
            std::make_from_tuple<T<storage::cpu_ifirst>>(std::forward<decltype(args)>(args))};
        return run_validation<T<storage::cpu_ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<storage::cpu_kfirst> benchmark_object{
            std::make_from_tuple<T<storage::cpu_kfirst>>(std::forward<decltype(args)>(args))};
        return run_validation<T<storage::cpu_kfirst>, cpu_kfirst>(benchmark_object);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu_kloop) {
        T<storage::gpu> benchmark_object{
            std::make_from_tuple<T<storage::gpu>>(std::forward<decltype(args)>(args))};
        return run_validation<T<storage::gpu>, gpu_kloop>(benchmark_object);
    } else if constexpr (I == backend_impl::gpu_naive) {
        T<storage::gpu> benchmark_object{
            std::make_from_tuple<T<storage::gpu>>(std::forward<decltype(args)>(args))};
        return run_validation<T<storage::gpu>, gpu_naive>(benchmark_object);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::pair<std::vector<std::vector<WP_TYPE>>, std::vector<std::vector<WP_TYPE>>> interpolate_validate_gridtools(
    std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        T<storage::cpu_ifirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<storage::cpu_kfirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_kfirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_kfirst>, cpu_kfirst>(benchmark_object);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu_kloop) {
        T<storage::gpu> benchmark_object{
            std::apply([](auto &&...args) { return T<storage::gpu>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::gpu>, gpu_kloop>(benchmark_object);
    } else if constexpr (I == backend_impl::gpu_naive) {
        T<storage::gpu> benchmark_object{
            std::apply([](auto &&...args) { return T<storage::gpu>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::gpu>, gpu_naive>(benchmark_object);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}

template <backend_impl I, template <typename> typename T, typename... Args>
std::vector<std::vector<WP_TYPE>> verts2cells_validate_gridtools(std::tuple<Args...> &&args) {
    if constexpr (I == backend_impl::cpu_ifirst) {
        T<storage::cpu_ifirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_ifirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_ifirst>, cpu_ifirst>(benchmark_object);
    } else if constexpr (I == backend_impl::cpu_kfirst) {
        T<storage::cpu_kfirst> benchmark_object{std::apply(
            [](auto &&...args) { return T<storage::cpu_kfirst>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::cpu_kfirst>, cpu_kfirst>(benchmark_object);
#ifdef __CUDACC__
    } else if constexpr (I == backend_impl::gpu_kloop) {
        T<storage::gpu> benchmark_object{
            std::apply([](auto &&...args) { return T<storage::gpu>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::gpu>, gpu_kloop>(benchmark_object);
    } else if constexpr (I == backend_impl::gpu_naive) {
        T<storage::gpu> benchmark_object{
            std::apply([](auto &&...args) { return T<storage::gpu>{std::forward<decltype(args)>(args)...}; }, args)};
        return run_validation<T<storage::gpu>, gpu_naive>(benchmark_object);
#endif
    } else {
        throw std::runtime_error("[wrapper] Undefined backend implementation");
    }
}
