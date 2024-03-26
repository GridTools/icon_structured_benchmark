import json
import matplotlib.pyplot as plt  # type: ignore [import-not-found]
import numpy as np
from os import path, makedirs


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


def calculate_median_95_quantile(data):
    lower_bound = np.sort(data)[int(len(data) * 0.025)]
    upper_bound = np.sort(data)[int(len(data) * 0.975)]
    return lower_bound, upper_bound


# Function to generate violin plots
def generate_violin_plots_v4(runtime_data, k, torus_name, output_dir):
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
        if "cpu_ifirst" in implementation or "cpu_kfirst" in implementation:
            violin_data.append(runtimes)
            labels.append(
                "{}_{}_{}".format(
                    "unstructured"
                    if "unstructured" in implementation
                    else "structured",
                    implementation.split("_")[-2],
                    implementation.split("_")[-1],
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

edges_size = {"1024": 14208, "512": 58050, "256": 229758, "128": 915948, "64": 3663792}

klevels = [1, 16, 65]

git_commit = "b6bc6f"

runtimes_output_multi = read_torus_results(
    "results/output_{}_multi".format(git_commit), torus_files, klevels
)

output_directory = "results/plot_output_{}_multi".format(git_commit)
if not path.exists(output_directory):
    makedirs(output_directory)
else:
    print(
        f"Directory '{output_directory}' already exists. Make sure the correct directory is selected. To proceed with the same directory name delete the folder and rerun script"
    )
    quit(1)

for torus_file in runtimes_output_multi.keys():
    print("=== Torus file: {} ===".format(torus_file))
    for k in klevels:
        print("= k levels: {} =".format(k))
        for backend_impl in runtimes_output_multi[torus_file][k].keys():
            print(
                "{} median runtimes ({}_multi): {}".format(
                    backend_impl,
                    git_commit,
                    np.median(runtimes_output_multi[torus_file][k][backend_impl]),
                )
            )

for torus_file in runtimes_output_multi.keys():
    print("=== Torus file: {} ===".format(torus_file))
    for k in klevels:
        print("= k levels: {} =".format(k))
        for backend_impl in runtimes_output_multi[torus_file][k].keys():
            print(
                "[{}] 85% CI within 10% of the mean: {}".format(
                    backend_impl,
                    test_ci_confidence(
                        runtimes_output_multi[torus_file][k][backend_impl], 85, 10
                    ),
                )
            )

# Generate violin plots for each torus size
for torus_size, runtime_data in runtimes_output_multi.items():
    for k in runtime_data.keys():
        generate_violin_plots_v4(runtime_data[k], k, torus_size, output_directory)
