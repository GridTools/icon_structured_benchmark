import json
import matplotlib.pyplot as plt  # type: ignore [import-not-found]
import numpy as np
from os import path, makedirs
import pandas as pd  # type: ignore [import-untyped]
import seaborn as sns  # type: ignore [import-not-found]


def create_output_directory(output_dir):
    if not path.exists(output_dir):
        makedirs(output_dir)
    else:
        print(
            f"Directory '{output_dir}' already exists. Make sure the correct directory is selected. To proceed with the same directory name delete the folder and rerun script"
        )
        quit(1)


def read_torus_results(directory, torus_filenames, k_levels):
    runtimes = {}
    for filename in torus_filenames:
        runtimes[filename] = {}
        for k in k_levels:
            with open("{}/{}_k{}.json".format(directory, filename, k)) as f:
                runtimes[filename][k] = json.load(f)
    return runtimes


def read_torus_results_commit(directory, torus_filenames, k_levels, commit):
    runtimes = {}
    for filename in torus_filenames:
        runtimes[filename] = {}
        for k in k_levels:
            with open("{}/{}_k{}_{}.json".format(directory, filename, k, commit)) as f:
                runtimes[filename][k] = json.load(f)
    return runtimes


def read_torus_results_commit_backend_index_type(
    directory, torus_filenames, k_levels, commit, backend, index_type, suffix=""
):
    runtimes = {}
    for filename in torus_filenames:
        runtimes[filename] = {}
        for k in k_levels:
            with open(
                "{}/{}_k{}_{}_{}_{}{}.json".format(
                    directory, filename, k, commit, backend, index_type, suffix
                )
            ) as f:
                runtimes[filename][k] = json.load(f)
    return runtimes


def print_median_runtimes(runtimes_output, git_commit):
    for torus_file in runtimes_output.keys():
        print("=== Torus file: {} ===".format(torus_file))
        for k in runtimes_output[torus_file].keys():
            print("= k levels: {} =".format(k))
            for backend_impl in runtimes_output[torus_file][k].keys():
                print(
                    "{} median runtimes ({}): {}".format(
                        backend_impl,
                        git_commit,
                        np.median(runtimes_output[torus_file][k][backend_impl]),
                    )
                )


def test_ci_confidence(vector, ci_interval, mean_interval):
    sorted_vec = np.sort(vector)
    ci_interval_div_2 = (100 - ci_interval) / 100 / 2
    ci_interval = sorted_vec[
        int(len(sorted_vec) * ci_interval_div_2) : int(
            len(sorted_vec) * (1 - ci_interval_div_2)
        )
    ]
    mean = np.mean(vector)
    mean_interval_div_2 = mean_interval / 100 / 2
    return ci_interval[0] > mean * (1 - mean_interval_div_2) and ci_interval[
        -1
    ] < mean * (1 + mean_interval_div_2)


def print_confidence_interval(runtimes_output, ci, mean_interval):
    for torus_file in runtimes_output.keys():
        print("=== Torus file: {} ===".format(torus_file))
        for k in runtimes_output[torus_file].keys():
            print("= k levels: {} =".format(k))
            for backend_impl in runtimes_output[torus_file][k].keys():
                print(
                    "[{}] {}% CI within {}% of the mean: {}".format(
                        backend_impl,
                        ci,
                        mean_interval,
                        test_ci_confidence(
                            runtimes_output[torus_file][k][backend_impl],
                            ci,
                            mean_interval,
                        ),
                    )
                )


def print_median_acceleration_over_k(data, backend_title, output_dir):
    # Flatten the data
    flat_data = []
    for torus, torus_data in data.items():
        for k, k_data in torus_data.items():
            for structure, values in k_data.items():
                for val in values:
                    flat_data.append((torus, k, structure, val))

    # Create DataFrame
    df = pd.DataFrame(flat_data, columns=["Torus", "K", "Structure", "Value"])

    # Calculate median runtimes for structured and unstructured
    medians = df.groupby(["Torus", "K", "Structure"])["Value"].median().reset_index()
    structured_medians = medians[medians["Structure"] == "structured"].set_index(
        ["Torus", "K"]
    )
    unstructured_medians = medians[medians["Structure"] == "unstructured"].set_index(
        ["Torus", "K"]
    )

    # Calculate acceleration
    acceleration = unstructured_medians["Value"] / structured_medians["Value"]
    acceleration = acceleration.reset_index()

    # Sort by torus size
    acceleration["Edges"] = [edges_size[torus] for torus in acceleration["Torus"]]
    acceleration = acceleration.sort_values(by="Edges", ascending=False)

    # Plot
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(
        data=acceleration,
        x="Edges",
        y="Value",
        hue="K",
        palette=sns.color_palette("tab10")[0:4],
    )
    plt.title(
        "Acceleration of Structured over Unstructured for {} implementation (Median Runtimes)".format(
            backend_title
        )
    )
    plt.ylabel("Acceleration")
    plt.xlabel("Edges")
    plt.legend(title="K")

    # Add values on top of bars
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                format(p.get_height(), ".2f"),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 10),
                textcoords="offset points",
            )
    plt.savefig(
        "{}/k_sweep_acc_torus_{}.png".format(output_dir, backend_title), dpi=800
    )


# Function to generate violin plots
def generate_violin_plots(data, k, torus_name, output_dir):
    import matplotlib.pyplot as plt

    torus_size = torus_name.split("_")[-1]
    plt.figure(figsize=(10, 8))
    plt.title(
        f"Nabla4 & Interpolation kernels runtime for {edges_size[torus_size]} Edges with {k} K-levels"
    )
    plt.ylabel("Runtime (s)")
    plt.xlabel("Implementation")
    violin_data = []
    labels = []
    medians = []  # To store medians

    for implementation, runtimes in data.items():
        if (
            "cpu_ifirst" in implementation
            or "cpu_kfirst" in implementation
            or "gpu" in implementation
            or "gtfn" in implementation
        ):
            violin_data.append(runtimes)
            labels.append(implementation.split("benchmark_")[-1])
            median_value = np.median(runtimes)
            medians.append(median_value)  # Calculating median for each set of runtimes

            plt.text(
                len(labels),
                median_value + 0.000035,
                f"{median_value:.6f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="red",
            )

    plt.violinplot(violin_data)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")

    # Plotting medians
    plt.scatter(
        np.arange(1, len(labels) + 1),
        medians,
        color="red",
        zorder=3,
        label="Median",
        s=20,
        marker="_",
    )

    plt.tight_layout()
    plt.legend()  # Show legend with median
    plt.savefig(
        "{}/runtimes_torus_{}_{}.png".format(output_dir, torus_size, k), dpi=500
    )


# Function to generate violin plots
def generate_violin_plots_acceleration(
    data,
    k,
    torus_name,
    output_dir,
    baseline_name="nabla4_interpolate_benchmark_unstructured_gpu_naive_separate",
    kernel_name="nabla4",
):
    import matplotlib.pyplot as plt

    torus_size = torus_name.split("_")[-1]
    plt.figure(figsize=(10, 8))
    plt.title(
        f"{kernel_name} runtime for {edges_size[torus_size]} Edges with {k} K-levels"
    )
    plt.ylabel("Runtime (s)")
    plt.xlabel("Implementation")
    violin_data = []
    labels = []
    medians = []  # To store medians
    baseline_median = np.median(data[baseline_name])
    plt.axhline(baseline_median, linestyle="dotted", color="red")
    for implementation, runtimes in data.items():
        if (
            "cpu_ifirst" in implementation
            or "cpu_kfirst" in implementation
            or "gpu" in implementation
            or "gtfn" in implementation
        ):
            violin_data.append(runtimes)
            labels.append(implementation.split("benchmark_")[-1])
            median_value = np.median(runtimes)
            medians.append(median_value)  # Calculating median for each set of runtimes

            percentage_diff = (median_value - baseline_median) / baseline_median * 100
            plt.text(
                len(labels),
                median_value + 0.000025,
                f"{median_value:.6f}\n({percentage_diff:.2f}%)"
                if implementation != baseline_name
                else f"{median_value:.5f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
            )

    plt.violinplot(violin_data)
    plt.xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")

    # Plotting medians
    plt.scatter(
        np.arange(1, len(labels) + 1),
        medians,
        color="red",
        zorder=3,
        label="Median",
        s=20,
        marker="_",
    )

    plt.tight_layout()
    plt.legend()  # Show legend with median
    plt.savefig(
        "{}/runtimes_torus_accel_{}_{}.png".format(output_dir, torus_size, k), dpi=400
    )


def filter_runtime_data(runtimes_output, unstructured_key: str, structured_key: str):
    torus_klevels_runtimes = {}  # type: ignore [var-annotated]
    for torus_file in runtimes_output.keys():
        torus_name = torus_file.split("_")[-1]
        torus_klevels_runtimes[torus_name] = {}
        for k in runtimes_output[torus_file].keys():
            torus_klevels_runtimes[torus_name][k] = {}
            for backend_impl in [
                unstructured_key,
                structured_key,
            ]:
                name = (
                    "unstructured" if backend_impl == unstructured_key else "structured"
                )
                torus_klevels_runtimes[torus_name][k][name] = runtimes_output[
                    torus_file
                ][k][backend_impl]
    return torus_klevels_runtimes


torus_files = [
    "torus_100000_100000_1024",
    "torus_100000_100000_512",
    "torus_100000_100000_256",
    "torus_100000_100000_128",
    "torus_100000_100000_64",
]

klevels = [1, 16, 65]

edges_size = {"1024": 14208, "512": 58050, "256": 229758, "128": 915948, "64": 3663792}

if __name__ == "__main__":
    git_commit = "9ced41e"

    runtimes_output = read_torus_results(
        "results/output_{}".format(git_commit), torus_files, klevels
    )

    output_directory = "results/plot_output_{}".format(git_commit)

    create_output_directory(output_directory)

    print_median_runtimes(runtimes_output, git_commit)

    print_confidence_interval(runtimes_output, 85, 10)

    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_benchmark_unstructured_naive",
            "nabla4_benchmark_structured_torus_naive",
        ),
        "naive",
        output_directory,
    )
    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_benchmark_unstructured_cpu_ifirst_gridtools",
            "nabla4_benchmark_structured_torus_cpu_ifirst_gridtools",
        ),
        "cpu_ifirst",
        output_directory,
    )
    print_median_acceleration_over_k(
        filter_runtime_data(
            runtimes_output,
            "nabla4_benchmark_unstructured_cpu_kfirst_gridtools",
            "nabla4_benchmark_structured_torus_cpu_kfirst_gridtools",
        ),
        "cpu_kfirst",
        output_directory,
    )

    # Generate violin plots for each torus size
    for torus_size, runtime_data in runtimes_output.items():
        for k in runtime_data.keys():
            generate_violin_plots(runtime_data[k], k, torus_size, output_directory)
