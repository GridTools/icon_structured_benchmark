#include <gridtools/fn/backend/naive.hpp>
#if defined(IS_GPU)
#include <gridtools/fn/backend/gpu.hpp>
#endif
#include <gridtools/sid/dimension_to_tuple_like.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <gridtools/fn/sid_neighbor_table.hpp>
#include <gridtools/fn/unstructured.hpp>

#include <iostream>

#include "timer.hpp"

namespace generated {

    namespace gtfn = ::gridtools::fn;

    namespace {
        using namespace ::gridtools::literals;

        using Edge_t = gtfn::unstructured::dim::horizontal;
        constexpr inline Edge_t Edge{};

        using K_t = gtfn::unstructured::dim::vertical;
        constexpr inline K_t K{};

        using Koff_t = K_t;
        constexpr inline Koff_t Koff{};

        struct _fun_1 {
            constexpr auto operator()() const {
                return [](auto const &__stencil_arg0) {
                    return [=]() { return gtfn::deref(gtfn::shift(__stencil_arg0)); }();
                };
            }
        };

        using block_sizes_t = gridtools::meta::list<
            gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, 32>>,
            gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, 8>>>;

        inline auto calculate_copy_kernel = [](auto... connectivities__) {
            return [connectivities__...](auto backend,
                       auto &&z_nabla2_e,
                       auto &&z_nabla4_e2,
                       auto &&horizontal_start,
                       auto &&horizontal_end,
                       auto &&vertical_start,
                       auto &&vertical_end) {
                auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);
                auto gtfn_backend = make_backend(backend,
                    gtfn::unstructured_domain(
                        ::gridtools::tuple((horizontal_end - horizontal_start), (vertical_end - vertical_start)),
                        ::gridtools::tuple(horizontal_start, vertical_start),
                        connectivities__...));
#if defined(IS_GPU)
                if constexpr (std::is_same_v<decltype(backend), gridtools::fn::backend::gpu<block_sizes_t>>) {
                    timer<backend_impl::gpu> t;
                    t.start();
                    gtfn_backend.stencil_executor()()
                        .arg(z_nabla4_e2)
                        .arg(z_nabla2_e)
                        .assign(0_c, _fun_1(), 1_c)
                        .execute();
                    t.stop();
                    return t.elapsed();
                } else {
#endif
                    timer<backend_impl::cpu_ifirst> t;
                    t.start();
                    gtfn_backend.stencil_executor()()
                        .arg(z_nabla4_e2)
                        .arg(z_nabla2_e)
                        .assign(0_c, _fun_1(), 1_c)
                        .execute();
                    t.stop();
                    return t.elapsed();
#if defined(IS_GPU)
                }
#endif
            };
        };
    } // namespace
} // namespace generated

template <class BufferT0, class BufferT1, class backend>
decltype(auto) calculate_copy(BufferT0 &&z_nabla2_e,
    BufferT1 &&z_nabla4_e2,
    std::int32_t horizontal_start,
    std::int32_t horizontal_end,
    std::int32_t vertical_start,
    std::int32_t vertical_end,
    backend &&backend_instance) {
    return generated::calculate_copy_kernel()(backend_instance,
        std::forward<decltype(z_nabla2_e)>(z_nabla2_e),
        std::forward<decltype(z_nabla4_e2)>(z_nabla4_e2),
        std::forward<decltype(horizontal_start)>(horizontal_start),
        std::forward<decltype(horizontal_end)>(horizontal_end),
        std::forward<decltype(vertical_start)>(vertical_start),
        std::forward<decltype(vertical_end)>(vertical_end));
}
