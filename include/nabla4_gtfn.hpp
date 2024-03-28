#include <gridtools/fn/backend/naive.hpp>
#include <gridtools/sid/dimension_to_tuple_like.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include <cmath>
#include <cstdint>
#include <functional>
#include <gridtools/fn/sid_neighbor_table.hpp>
#include <gridtools/fn/unstructured.hpp>

#include <chrono>
#include <iostream>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

namespace generated {

    namespace gtfn = ::gridtools::fn;

    namespace {
        using namespace ::gridtools::literals;

        using Edge_t = gtfn::unstructured::dim::horizontal;
        constexpr inline Edge_t Edge{};

        using K_t = gtfn::unstructured::dim::vertical;
        constexpr inline K_t K{};

        using Cell_t = gtfn::unstructured::dim::horizontal;
        constexpr inline Cell_t Cell{};

        using Vertex_t = gtfn::unstructured::dim::horizontal;
        constexpr inline Vertex_t Vertex{};

        struct E2C2V_t {};
        constexpr inline E2C2V_t E2C2V{};

        using ECV_t = gtfn::unstructured::dim::horizontal;
        constexpr inline ECV_t ECV{};

        struct E2ECV_t {};
        constexpr inline E2ECV_t E2ECV{};

        using Koff_t = K_t;
        constexpr inline Koff_t Koff{};

        struct _fun_1 {
            constexpr auto operator()() const {
                return [](auto const &__stencil_arg0,
                           auto const &__stencil_arg1,
                           auto const &__stencil_arg2,
                           auto const &__stencil_arg3,
                           auto const &__stencil_arg4,
                           auto const &__stencil_arg5,
                           auto const &__stencil_arg6) {
                    return [=](auto _cs_1, auto _cs_2, auto _cs_3) {
                        return static_cast<double>((
                            4.0 *
                            (((static_cast<double>(static_cast<double>((
                                   (((static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 2_c))) *
                                         gtfn::deref(gtfn::shift(__stencil_arg2, E2ECV, 2_c))) +
                                        (static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 2_c))) *
                                            gtfn::deref(gtfn::shift(__stencil_arg3, E2ECV, 2_c)))) +
                                       (static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 3_c))) *
                                           gtfn::deref(gtfn::shift(__stencil_arg2, E2ECV, 3_c)))) +
                                   (static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 3_c))) *
                                       gtfn::deref(gtfn::shift(__stencil_arg3, E2ECV, 3_c)))))) -
                                  _cs_1) *
                                 (_cs_2 * _cs_2)) +
                                ((static_cast<double>(static_cast<double>(
                                      ((((static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 0_c))) *
                                             gtfn::deref(gtfn::shift(__stencil_arg2, E2ECV, 0_c))) +
                                            (static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 0_c))) *
                                                gtfn::deref(gtfn::shift(__stencil_arg3, E2ECV, 0_c)))) +
                                           (static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg0, E2C2V, 1_c))) *
                                               gtfn::deref(gtfn::shift(__stencil_arg2, E2ECV, 1_c)))) +
                                          (static_cast<double>(gtfn::deref(gtfn::shift(__stencil_arg1, E2C2V, 1_c))) *
                                              gtfn::deref(gtfn::shift(__stencil_arg3, E2ECV, 1_c)))))) -
                                     _cs_1) *
                                    (_cs_3 * _cs_3)))));
                    }((2.0 * gtfn::deref(__stencil_arg4)), gtfn::deref(__stencil_arg5), gtfn::deref(__stencil_arg6));
                };
            }
        };

        using block_sizes_t = gridtools::meta::list<
            gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, 32>>,
            gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, 8>>>;

        inline auto calculate_nabla4 = [](auto... connectivities__) {
            return [connectivities__...](auto backend,
                       auto &&u_vert,
                       auto &&v_vert,
                       auto &&primal_normal_vert_v1,
                       auto &&primal_normal_vert_v2,
                       auto &&z_nabla2_e,
                       auto &&inv_vert_vert_length,
                       auto &&inv_primal_edge_length,
                       auto &&z_nabla4_e2,
                       auto &&horizontal_start,
                       auto &&horizontal_end,
                       auto &&vertical_start,
                       auto &&vertical_end,
                       int repetitions) {
                std::vector<double> runtimes;
                for (int i = 0; i < repetitions; ++i) {
                    auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);

                    const auto start = high_resolution_clock::now();
                    make_backend(backend,
                        gtfn::unstructured_domain(
                            ::gridtools::tuple((horizontal_end - horizontal_start), (vertical_end - vertical_start)),
                            ::gridtools::tuple(horizontal_start, vertical_start),
                            connectivities__...))
                        .stencil_executor()()
                        .arg(z_nabla4_e2)
                        .arg(u_vert)
                        .arg(v_vert)
                        .arg(primal_normal_vert_v1)
                        .arg(primal_normal_vert_v2)
                        .arg(z_nabla2_e)
                        .arg(inv_vert_vert_length)
                        .arg(inv_primal_edge_length)
                        .assign(0_c, _fun_1(), 1_c, 2_c, 3_c, 4_c, 5_c, 6_c, 7_c)
                        .execute();
                    const auto end = high_resolution_clock::now();
                    runtimes.push_back(duration<double>(end - start).count());
                }
                return runtimes;
            };
        };
    } // namespace
} // namespace generated

template <class BufferT0,
    class BufferT1,
    class BufferT2,
    class BufferT3,
    class BufferT4,
    class BufferT5,
    class BufferT6,
    class BufferT7,
    class BufferT12,
    class BufferT13>
decltype(auto) calculate_nabla4(BufferT0 &&u_vert,
    BufferT1 &&v_vert,
    BufferT2 &&primal_normal_vert_v1,
    BufferT3 &&primal_normal_vert_v2,
    BufferT4 &&z_nabla2_e,
    BufferT5 &&inv_vert_vert_length,
    BufferT6 &&inv_primal_edge_length,
    BufferT7 &&z_nabla4_e2,
    std::int32_t horizontal_start,
    std::int32_t horizontal_end,
    std::int32_t vertical_start,
    std::int32_t vertical_end,
    BufferT12 &&gt_conn_e2c2v,
    BufferT13 &&gt_conn_e2ecv,
    int repetitions) {
    return generated::calculate_nabla4(
        gridtools::hymap::keys<generated::E2C2V_t>::make_values(
            gridtools::fn::sid_neighbor_table::as_neighbor_table<generated::Edge_t, generated::E2C2V_t, 4>(
                std::forward<decltype(gt_conn_e2c2v)>(gt_conn_e2c2v))),
        gridtools::hymap::keys<generated::E2ECV_t>::make_values(
            gridtools::fn::sid_neighbor_table::as_neighbor_table<generated::Edge_t, generated::E2ECV_t, 4>(
                std::forward<decltype(gt_conn_e2ecv)>(gt_conn_e2ecv))))(gridtools::fn::backend::naive{},
        std::forward<decltype(u_vert)>(u_vert),
        std::forward<decltype(v_vert)>(v_vert),
        std::forward<decltype(primal_normal_vert_v1)>(primal_normal_vert_v1),
        std::forward<decltype(primal_normal_vert_v2)>(primal_normal_vert_v2),
        std::forward<decltype(z_nabla2_e)>(z_nabla2_e),
        std::forward<decltype(inv_vert_vert_length)>(inv_vert_vert_length),
        std::forward<decltype(inv_primal_edge_length)>(inv_primal_edge_length),
        std::forward<decltype(z_nabla4_e2)>(z_nabla4_e2),
        std::forward<decltype(horizontal_start)>(horizontal_start),
        std::forward<decltype(horizontal_end)>(horizontal_end),
        std::forward<decltype(vertical_start)>(vertical_start),
        std::forward<decltype(vertical_end)>(vertical_end),
        repetitions);
}
