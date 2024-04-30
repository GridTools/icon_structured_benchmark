from analysis import (
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    klevels,
)

from analysis_multi import (
    generate_violin_plots,
)


def read_torus_results_halo_commit_backend_index_type(
    directory, torus_filenames, klevels, commit, backend, index_type
):
    import json
    import numpy as np

    runtimes = {}
    for filename in torus_filenames:
        runtimes[filename] = {}
        for k in klevels:
            runtimes[filename][k] = {}
            with open(
                "{}/{}_k{}_{}_{}_{}_0.json".format(
                    directory, filename, k, commit, backend, index_type
                )
            ) as f:
                runtimes_per_process = json.load(f)
                for backend_impl in runtimes_per_process.keys():
                    runtimes[filename][k][backend_impl] = []
            for process in range(0, 72):
                with open(
                    "{}/{}_k{}_{}_{}_{}_0.json".format(
                        directory, filename, k, commit, backend, index_type
                    )
                ) as f:
                    runtimes_per_process = json.load(f)
                for backend_impl in runtimes_per_process.keys():
                    runtimes[filename][k][backend_impl] = np.append(
                        runtimes[filename][k][backend_impl],
                        runtimes_per_process[backend_impl],
                    )
    return runtimes


if __name__ == "__main__":
    git_commit = "04348ac"

    backend = "cpu"

    index_type = "sizet"

    torus_files = [
        "torus_100000_100000_128",
    ]

    runtimes_output_multi = read_torus_results_halo_commit_backend_index_type(
        "results/output_{}_{}_{}_multi".format(git_commit, backend, index_type),
        torus_files,
        klevels,
        git_commit,
        backend,
        index_type,
    )

    output_directory = "results/plot_output_halo_{}_{}_{}_multi".format(
        git_commit, backend, index_type
    )

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output_multi, git_commit)

    print_confidence_interval(runtimes_output_multi, 85, 10)

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output_multi.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
