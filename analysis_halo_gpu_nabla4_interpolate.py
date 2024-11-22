from analysis import (
    read_torus_results_commit_backend_index_type,
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    generate_violin_plots,
    generate_violin_plots_acceleration,
)

if __name__ == "__main__":
    git_commit = "979c2b9"

    backend = "gpu"

    index_type = "int"

    suffix = "_all_gpu_101"

    torus_files = [
        "torus_100000_100000_256",
        "torus_100000_100000_128",
    ]

    klevels = [80]

    runtimes_output = read_torus_results_commit_backend_index_type(
        "results/nabla4_interpolate_output_{}_{}_{}{}".format(
            git_commit, backend, index_type, suffix
        ),
        torus_files,
        klevels,
        git_commit,
        backend,
        index_type,
    )

    output_directory = "results/plot_nabla4_interpolate_output_halo_{}_{}_{}{}".format(
        git_commit, backend, index_type, suffix
    )

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output, git_commit)

    print_confidence_interval(runtimes_output, 85, 10)

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)

    # Generate summary with violin plots and acceleration
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            runtime_data[k] = {
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_separate": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_unstructured_gpu_naive_separate"],
                "nabla4_interpolate_benchmark_structured_gpu_naive_separate": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_structured_gpu_naive_separate"],
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined"],
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_unstructured_gpu_naive_inlined_v2v"],
                "nabla4_interpolate_benchmark_structured_gpu_naive_inlined": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_structured_gpu_naive_inlined"],
                "nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_structured_gpu_naive_inlined_cached"],
                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_unstructured_gpu_kloop_separate"],
                "nabla4_interpolate_benchmark_structured_gpu_kloop_separate": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_structured_gpu_kloop_separate"],
                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined"],
                "nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_unstructured_gpu_kloop_inlined_v2v"],
                "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_structured_gpu_kloop_inlined"],
                "nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached": runtime_data[
                    k
                ]["nabla4_interpolate_benchmark_structured_gpu_kloop_inlined_cached"],
            }
            generate_violin_plots_acceleration(
                runtime_data[k],
                k,
                torus_size,
                output_directory,
                "nabla4_interpolate_benchmark_unstructured_gpu_naive_separate",
                "Nabla4 and interpolate kernels",
            )
