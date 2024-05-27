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

        struct E2C2V_t {};
        constexpr inline E2C2V_t E2C2V{};

        using K_t = gtfn::unstructured::dim::vertical;
        constexpr inline K_t K{};

        using Koff_t = K_t;
        constexpr inline Koff_t Koff{};

        struct _fun_1 {
            constexpr auto operator()() const {
                return [](auto const &__stencil_arg0, auto const &__stencil_arg1) {
                    return [=]() {
                        double E2C2V_0 = gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 0_c));
                        double E2C2V_1 = gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 1_c));
                        double E2C2V_2 = gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 2_c));
                        double E2C2V_3 = gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 3_c));
                        double E2C2V_0_p1 = (E2C2V_0 + 42.0) * Koff;
                        double E2C2V_1_m1 = (E2C2V_1 - 42.0) * Koff;
                        double E2C2V_2_p1 = (E2C2V_2 + 42.0) * Koff;
                        double E2C2V_3_m1 = (E2C2V_3 - 42.0) * Koff;
                        double dummy_0 = gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 0_c));
                        double dummy_1 = gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 1_c));
                        double dummy_2 = gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 2_c));
                        double dummy_3 = gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 3_c));
                        double avg = ((E2C2V_0_p1 + E2C2V_1_m1 + E2C2V_2_p1 + E2C2V_3_m1) / (4.0 * Koff) + dummy_0 +
                                         dummy_1 + dummy_2 + dummy_3) /
                                     5.0;
                        return avg;
                    }();
                };
            }
        };

        using block_sizes_t = gridtools::meta::list<
            gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, 32>>,
            gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, 8>>>;

        inline auto calculate_copy_neighbor_kernel = [](auto... connectivities__) {
            return [connectivities__...](int repetitions,
                       auto backend,
                       auto &&z_nabla2_e,
                       auto &&dummy_field,
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
                    std::vector<double> times;
                    for (int i{0}; i < repetitions; ++i) {
                        t.start();
                        gtfn_backend.stencil_executor()()
                            .arg(z_nabla4_e2)
                            .arg(z_nabla2_e)
                            .arg(dummy_field)
                            .assign(0_c, _fun_1(), 1_c, 2_c)
                            .execute();
                        t.stop();
                        times.push_back(t.elapsed());
                    };
                    return times;
                } else {
#endif
                    timer<backend_impl::cpu_ifirst> t;
                    std::vector<double> times;
                    for (int i{0}; i < repetitions; ++i) {
                        t.start();
                        gtfn_backend.stencil_executor()()
                            .arg(z_nabla4_e2)
                            .arg(z_nabla2_e)
                            .arg(dummy_field)
                            .assign(0_c, _fun_1(), 1_c, 2_c)
                            .execute();
                        t.stop();
                        times.push_back(t.elapsed());
                    };
                    return times;
#if defined(IS_GPU)
                }
#endif
            };
        };
    } // namespace
} // namespace generated

template <class BufferT0, class BufferT1, class BufferT2, class BufferT3, class backend>
decltype(auto) calculate_copy_neighbor(int repetitions,
    BufferT0 &&z_nabla2_e,
    BufferT1 &&dummy_field,
    BufferT2 &&z_nabla4_e2,
    std::int32_t horizontal_start,
    std::int32_t horizontal_end,
    std::int32_t vertical_start,
    std::int32_t vertical_end,
    BufferT3 &&gt_conn_e2c2v,
    backend &&backend_instance) {
    return generated::calculate_copy_neighbor_kernel(gridtools::hymap::keys<generated::E2C2V_t>::make_values(
        gridtools::fn::sid_neighbor_table::as_neighbor_table<generated::Edge_t, generated::E2C2V_t, 4>(
            std::forward<decltype(gt_conn_e2c2v)>(gt_conn_e2c2v))))(repetitions,
        backend_instance,
        std::forward<decltype(z_nabla2_e)>(z_nabla2_e),
        std::forward<decltype(dummy_field)>(dummy_field),
        std::forward<decltype(z_nabla4_e2)>(z_nabla4_e2),
        std::forward<decltype(horizontal_start)>(horizontal_start),
        std::forward<decltype(horizontal_end)>(horizontal_end),
        std::forward<decltype(vertical_start)>(vertical_start),
        std::forward<decltype(vertical_end)>(vertical_end));
}
