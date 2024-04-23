from analysis import (
    read_torus_results_commit,
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    print_median_acceleration_over_k,
    filter_runtime_data,
    generate_violin_plots,
)

if __name__ == "__main__":
    git_commit = "a7c5fd9"

    torus_files = [
        "torus_100000_100000_512",
        "torus_100000_100000_256",
        "torus_100000_100000_128",
    ]

    klevels = [1, 16, 65, 180]

    runtimes_output = read_torus_results_commit(
        "results/output_{}".format(git_commit), torus_files, klevels, git_commit
    )

    output_directory = "results/plot_output_halo_{}".format(git_commit)

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output, git_commit)

    print_confidence_interval(runtimes_output, 85, 10)

    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_benchmark_unstructured_gpu_gridtools",
            "nabla4_benchmark_structured_torus_gpu_gridtools_halo",
        ),
        "gpu",
        output_directory,
    )

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
