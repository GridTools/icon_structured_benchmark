from analysis import (
    read_torus_results_commit_backend_index_type,
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    print_median_acceleration_over_k,
    filter_runtime_data,
    generate_violin_plots,
    torus_files,
    klevels,
)

if __name__ == "__main__":
    git_commit = "473bf2d"

    backend = "cpu"

    index_type = "int64"

    runtimes_output = read_torus_results_commit_backend_index_type(
        "results/output_{}_{}_{}".format(git_commit, backend, index_type),
        torus_files,
        klevels,
        git_commit,
        backend,
        index_type,
    )

    output_directory = "results/plot_output_halo_{}_{}_{}".format(
        git_commit, backend, index_type
    )

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output, git_commit)

    print_confidence_interval(runtimes_output, 85, 10)

    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_benchmark_unstructured_cpu_ifirst_gridtools",
            "nabla4_benchmark_structured_torus_cpu_ifirst_gridtools_halo",
        ),
        "cpu_ifirst",
        output_directory,
    )
    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_benchmark_unstructured_cpu_kfirst_gridtools",
            "nabla4_benchmark_structured_torus_cpu_kfirst_gridtools_halo",
        ),
        "cpu_kfirst",
        output_directory,
    )

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
