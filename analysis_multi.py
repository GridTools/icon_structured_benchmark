import json
import matplotlib.pyplot as plt  # type: ignore [import-not-found]
import numpy as np

from analysis import (
    create_output_directory,
    print_median_runtimes,
    print_confidence_interval,
    edges_size,
    klevels,
)


def read_torus_results(directory, torus_filenames, klevels):
    runtimes = {}
    for filename in torus_filenames:
        runtimes[filename] = {}
        for k in klevels:
            runtimes[filename][k] = {}
            with open("{}/{}_k{}_0.json".format(directory, filename, k)) as f:
                runtimes_per_process = json.load(f)
                for backend_impl in runtimes_per_process.keys():
                    runtimes[filename][k][backend_impl] = []
            for process in range(0, 72):
                with open(
                    "{}/{}_k{}_{}.json".format(directory, filename, k, process)
                ) as f:
                    runtimes_per_process = json.load(f)
                for backend_impl in runtimes_per_process.keys():
                    runtimes[filename][k][backend_impl] = np.append(
                        runtimes[filename][k][backend_impl],
                        runtimes_per_process[backend_impl],
                    )
    return runtimes


def calculate_median_95_quantile(data):
    lower_bound = np.sort(data)[int(len(data) * 0.025)]
    upper_bound = np.sort(data)[int(len(data) * 0.975)]
    return lower_bound, upper_bound


# Function to generate violin plots
def generate_violin_plots(runtime_data, k, torus_name, output_dir):
    torus_size = torus_name.split("_")[-1]
    plt.figure(figsize=(10, 10))
    plt.title(f"Runtime Distribution for {edges_size[torus_size]} Edges with k: {k}")
    plt.ylabel("Runtime (s)")
    plt.xlabel("Implementation")
    violin_data = []
    labels = []
    medians = []  # To store medians
    median_95_quantile = []  # To store median confidence intervals

    for implementation, runtimes in runtime_data.items():
        if (
            "cpu_ifirst" in implementation
            or "cpu_kfirst" in implementation
            or "gtfn" in implementation
        ):
            violin_data.append(runtimes)
            labels.append(
                "{}_{}".format(
                    "unstructured"
                    if "unstructured" in implementation
                    else "structured"
                    if "structured" in implementation
                    else "gtfn",
                    "cpu_ifirst"
                    if "cpu_ifirst" in implementation
                    else "cpu_kfirst"
                    if "cpu_kfirst" in implementation
                    else "gtfn",
                )
            )
            median_value = np.median(runtimes)
            medians.append(median_value)  # Calculating median for each set of runtimes

            # Calculate median confidence interval
            ci_low, ci_high = calculate_median_95_quantile(runtimes)
            median_95_quantile.append((ci_low, ci_high))

            plt.text(
                len(labels),
                median_value * 1.15,
                f"{median_value:.6f}",
                ha="center",
                va="bottom",
                fontsize=12,
                color="red",
            )

    plt.violinplot(violin_data)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")

    # Plotting medians with confidence intervals
    plt.scatter(
        np.arange(1, len(labels) + 1),
        medians,
        color="red",
        zorder=3,
        label="Median",
        s=20,
        marker="_",
    )

    for i, (ci_low, ci_high) in enumerate(median_95_quantile):
        plt.plot([i + 1, i + 1], [ci_low, ci_high], color="cornflowerblue", lw=6)

    plt.tight_layout()
    plt.legend()  # Show legend with median
    plt.savefig(
        "{}/runtimes_torus_{}_{}_multiproc.png".format(output_dir, torus_size, k),
        dpi=800,
    )


torus_files = [
    "torus_100000_100000_1024",
    "torus_100000_100000_512",
    "torus_100000_100000_256",
    "torus_100000_100000_128",
]

if __name__ == "__main__":
    git_commit = "9ced41e"

    runtimes_output_multi = read_torus_results(
        "results/output_{}_multi".format(git_commit), torus_files, klevels
    )

    output_directory = "results/plot_output_{}_multi".format(git_commit)

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output_multi, git_commit)

    print_confidence_interval(runtimes_output_multi, 85, 10)

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output_multi.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
