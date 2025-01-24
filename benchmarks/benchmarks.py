# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from .run_filtered_torus_grid_int_nabla4_interpolate_c2v import *  # type: ignore

import numpy as np


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        class Args:
            def __init__(self, **entries):
                self.__dict__.update(entries)

        args = Args(
            grid="/Users/ioannmag/cscs_repos/cycle20/torus_100000_100000_1024.nc",
            transformation="gt4py",
            klevels=80,
            repetitions=101,
            dry_run=True,
            output="output",
            sanity_checks=False,
            backend="all_cpu",
            combination="all",
            e2c2v_ordering="per-vertex",
            halo=2,
            vertical=False,
        )
        self.args = args

        transformation = (
            ToZeroBasedIndexTransformation()
            if args.transformation == "gt4py"
            else IndexTransformation()
        )

        torus_grid = get_torus_grid(
            args.grid, args.klevels, transformation, args.e2c2v_ordering
        )

        repetitions = args.repetitions
        self.repetitions = repetitions
        dry_runs = 10 if args.dry_run else 0
        self.dry_runs = dry_runs

        grid_cartesian_dimensions = get_torus_cartesian_dimensions(args.grid)

        (
            self.filtered_e2c2v_separate,
            self.filtered_e2ecv_separate,
            self.filtered_v2e_separate,
        ) = filter_neighbors(
            args,
            grid_cartesian_dimensions,
            torus_grid.get_offset_provider("E2C2V").table,
            torus_grid.get_offset_provider("E2ECV").table,
            torus_grid.get_offset_provider("V2E").table,
            args.e2c2v_ordering,
            "separate",
        )
        (
            self.filtered_e2c2v_inlined,
            self.filtered_e2ecv_inlined,
            self.filtered_v2e_inlined,
        ) = filter_neighbors(
            torus_grid.get_offset_provider("E2C2V").table,
            torus_grid.get_offset_provider("E2ECV").table,
            torus_grid.get_offset_provider("V2E").table,
            args.e2c2v_ordering,
            "inlined",
        )

        original_c2v = torus_grid.get_offset_provider("C2V").table
        filtered_c2v_original_indexes = filter_c2v_vector(
            original_c2v, grid_cartesian_dimensions, args.halo + 2
        )

        filtered_c2v = []
        for cell, vertices in enumerate(filtered_c2v_original_indexes):
            new_vertices = []
            for vertex in vertices:
                i = vertex % grid_cartesian_dimensions[1]
                j = vertex // grid_cartesian_dimensions[1]
                new_vertices.append(
                    i - 3 + (j - 3) * (grid_cartesian_dimensions[1] - 2 * 3)
                )
            filtered_c2v.append(new_vertices)
        self.filtered_c2v = np.array(filtered_c2v)

    def time_nabla4_interpolate_verts2cells_benchmark_unstructured_cpu_kfirst_separate(
        self,
    ):
        icon_benchmark.nabla4_interpolate_verts2cells_benchmark_unstructured_cpu_kfirst_separate(
            self.filtered_e2c2v_separate,
            self.filtered_e2ecv_separate,
            self.filtered_v2e_separate,
            self.filtered_c2v,
            self.torus_grid.num_cells,
            self.torus_grid.num_vertices,
            self.torus_grid.num_edges,
            self.torus_grid.num_levels,
            self.torus_grid.size[E2C2VDim],
            self.repetitions,
            self.dry_runs,
        )


# class MemSuite:
#     def mem_list(self):
#         return [0] * 256
