from analysis import (
    read_torus_results_commit_backend_index_type,
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    print_median_acceleration_over_k,
    filter_runtime_data,
    generate_violin_plots,
    generate_violin_plots_acceleration,
)

if __name__ == "__main__":
    git_commit = "pipeline"

    backend = "gpu"

    index_type = "int"

    torus_files = [
        "torus_100000_100000_256",
        # "torus_100000_100000_128",
        # "torus_100000_100000_64",
    ]

    klevels = [80]

    runtimes_output = read_torus_results_commit_backend_index_type(
        "results/nabla4_output_{}_{}_{}".format(git_commit, backend, index_type),
        torus_files,
        klevels,
        git_commit,
        backend,
        index_type,
    )

    output_directory = "results/plot_nabla4_output_halo_{}_{}_{}".format(
        git_commit, backend, index_type
    )

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output, git_commit)

    print_confidence_interval(runtimes_output, 85, 10)

    # print_median_acceleration_over_k(
    #     filter_runtime_data(
    #         runtimes_output,
    #         "nabla4_benchmark_unstructured_gpu_kloop_gridtools",
    #         "nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo",
    #     ),
    #     "gpu_kloop",
    #     output_directory,
    # )

    # print_median_acceleration_over_k(
    #     filter_runtime_data(
    #         runtimes_output,
    #         "nabla4_benchmark_unstructured_gpu_naive_gridtools",
    #         "nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo",
    #     ),
    #     "gpu_naive",
    #     output_directory,
    # )

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            # sort runtime_data[k] in the following order: nabla4_benchmark_unstructured_gtfn_gpu, nabla4_benchmark_unstructured_gpu_naive_gridtools, nabla4_benchmark_structured_torus_gpu_naivegridtools_halo, nabla4_benchmark_unstructured_gpu_kloop_gridtools, nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo
            runtime_data[k] = {
                # "nabla4_benchmark_unstructured_gtfn_gpu": runtime_data[k][
                #     "nabla4_benchmark_unstructured_gtfn_gpu"
                # ],
                # "nabla4_benchmark_unstructured_gpu_naive_gridtools": runtime_data[k][
                #     "nabla4_benchmark_unstructured_gpu_naive_gridtools"
                # ],
                # "nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo": runtime_data[
                #     k
                # ]["nabla4_benchmark_structured_torus_gpu_naive_gridtools_halo"],
                "nabla4_benchmark_unstructured_gpu_kloop": runtime_data[k][
                    "nabla4_benchmark_unstructured_gpu_kloop_gridtools"
                ],
                "nabla4_benchmark_structured_gpu_kloop": runtime_data[k][
                    "nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo"
                ],
                "nabla4_benchmark_structured_gpu_kloop_pipeline": runtime_data[k][
                    "nabla4_benchmark_structured_torus_gpu_kloop_gridtools_halo_pipeline"
                ],
            }
            generate_violin_plots_acceleration(
                runtime_data[k],
                k,
                torus_size,
                output_directory,
                "nabla4_benchmark_unstructured_gpu_kloop",
                "Nabla4 kernel",
            )
