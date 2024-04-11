from analysis import (
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    klevels,
)

from analysis_multi import (
    read_torus_results,
    generate_violin_plots,
    torus_files,
)

if __name__ == "__main__":
    git_commit = "890b981"

    runtimes_output_multi = read_torus_results(
        "results/output_halo_{}_multi".format(git_commit), torus_files, klevels
    )

    output_directory = "results/plot_output_halo_{}_multi".format(git_commit)

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output_multi, git_commit)

    print_confidence_interval(runtimes_output_multi, 85, 10)

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output_multi.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
