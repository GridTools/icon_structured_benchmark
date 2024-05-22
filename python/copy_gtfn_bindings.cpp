#include "copy_gtfn.hpp"

#include <gridtools/common/defs.hpp>
#include <gridtools/common/tuple_util.hpp>
#include <gridtools/fn/cartesian.hpp>
#include <gridtools/fn/unstructured.hpp>
#if defined(IS_GPU)
#include <gridtools/fn/backend/gpu.hpp>
#endif
#include <gridtools/sid/composite.hpp>
#include <gridtools/sid/rename_dimensions.hpp>
#include <gridtools/sid/unknown_kind.hpp>
#include <gridtools/storage/adapter/nanobind_adapter.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

decltype(auto) calculate_copy_wrapper_cpu(
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> z_nabla2_e,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> z_nabla4_e2,
    std::int32_t horizontal_start,
    std::int32_t horizontal_end,
    std::int32_t vertical_start,
    std::int32_t vertical_end) {
    return calculate_copy(
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::K_t>( // strides: (1, 72)
            gridtools::sid::shift_sid_origin(gridtools::nanobind::as_sid(z_nabla2_e.first), z_nabla2_e.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::K_t>( // strides: (1, 72)
            gridtools::sid::shift_sid_origin(gridtools::nanobind::as_sid(z_nabla4_e2.first), z_nabla4_e2.second)),
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        gridtools::fn::backend::naive{});
}

#if defined(IS_GPU)
decltype(auto) calculate_copy_wrapper_gpu(
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> z_nabla2_e,
    std::pair<nanobind::ndarray<double, nanobind::shape<nanobind::any, nanobind::any>>,
        std::tuple<ptrdiff_t, ptrdiff_t>> z_nabla4_e2,
    std::int32_t horizontal_start,
    std::int32_t horizontal_end,
    std::int32_t vertical_start,
    std::int32_t vertical_end) {
    return calculate_copy(
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::K_t>(
            gridtools::sid::shift_sid_origin(gridtools::nanobind::as_sid(z_nabla2_e.first), z_nabla2_e.second)),
        gridtools::sid::rename_numbered_dimensions<generated::Edge_t, generated::K_t>(
            gridtools::sid::shift_sid_origin(gridtools::nanobind::as_sid(z_nabla4_e2.first), z_nabla4_e2.second)),
        horizontal_start,
        horizontal_end,
        vertical_start,
        vertical_end,
        gridtools::fn::backend::gpu<generated::block_sizes_t>{});
}
#endif

NB_MODULE(copy_gtfn, module) {
    module.def("calculate_copy_cpu", &calculate_copy_wrapper_cpu);
#if defined(IS_GPU)
    module.def("calculate_copy_gpu", &calculate_copy_wrapper_gpu);
#endif
}
