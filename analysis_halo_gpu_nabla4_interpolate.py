from analysis import (
    read_torus_results_commit_backend_index_type,
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    print_median_acceleration_over_k,
    filter_runtime_data,
    generate_violin_plots,
)

if __name__ == "__main__":
    git_commit = "a31fdd3"

    backend = "gpu"

    index_type = "int"

    torus_files = [
        "torus_100000_100000_128",
        "torus_100000_100000_64",
    ]

    klevels = [1, 16, 65, 80]

    runtimes_output = read_torus_results_commit_backend_index_type(
        "results/nabla4_interpolate_output_{}_{}_{}".format(
            git_commit, backend, index_type
        ),
        torus_files,
        klevels,
        git_commit,
        backend,
        index_type,
    )

    output_directory = "results/plot_nabla4_interpolate_output_halo_{}_{}_{}".format(
        git_commit, backend, index_type
    )

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output, git_commit)

    print_confidence_interval(runtimes_output, 85, 10)

    # print_median_acceleration_over_k(
    #     filter_runtime_data(
    #         runtimes_output,
    #         "nabla4_interpolate_benchmark_unstructured_gpu_naive",
    #         "nabla4_interpolate_benchmark_unstructured_gpu_naive",
    #     ),
    #     "gpu",
    #     output_directory,
    # )

    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_interpolate_benchmark_unstructured_gpu_naive",
            "nabla4_interpolate_benchmark_structured_gpu_naive",
        ),
        "gpu_naive",
        output_directory,
    )

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            # sort runtime_data[k] in the following order: nabla4_benchmark_unstructured_gtfn_gpu, nabla4_benchmark_unstructured_gpu_gridtools_naive, nabla4_benchmark_structured_torus_gpu_gridtools_halo_naive, nabla4_benchmark_unstructured_gpu_gridtools, nabla4_benchmark_structured_torus_gpu_gridtools_halo
            runtime_data[k] = {
                "nabla4_interpolate_benchmark_unstructured_gpu_naive": runtime_data[k][
                    "nabla4_interpolate_benchmark_unstructured_gpu_naive"
                ],
                "nabla4_interpolate_benchmark_structured_gpu_naive": runtime_data[k][
                    "nabla4_interpolate_benchmark_structured_gpu_naive"
                ],
                # "nabla4_benchmark_unstructured_gpu_gridtools": runtime_data[k][
                #     "nabla4_benchmark_unstructured_gpu_gridtools"
                # ],
                # "nabla4_benchmark_structured_torus_gpu_gridtools_halo": runtime_data[k][
                #     "nabla4_benchmark_structured_torus_gpu_gridtools_halo"
                # ],
            }
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
