from analysis import (
    read_torus_results,
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
    git_commit = "b6bc6f"

    runtimes_output = read_torus_results(
        "results/output_halo_{}".format(git_commit), torus_files, klevels
    )

    output_directory = "results/plot_output_halo_{}".format(git_commit)

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
