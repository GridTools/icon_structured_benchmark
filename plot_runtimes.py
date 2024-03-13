import seaborn as sns  # type: ignore [import-not-found]
import pandas as pd  # type: ignore [import-untyped]
import matplotlib.pyplot as plt  # type: ignore [import-not-found]
import numpy as np


def plot_runtimes(data, output="output.pdf"):
    # Filter data for each algorithm to include only 25% to 75% of the runtimes
    filtered_data = {
        key: np.sort(val)[int(0.25 * len(val)) : int(0.75 * len(val))]
        for key, val in data.items()
    }

    # Calculate medians and quartiles
    medians = {key: np.median(val) for key, val in filtered_data.items()}

    # Convert data to long format for Seaborn
    df = pd.melt(
        pd.DataFrame(filtered_data), var_name="Algorithm", value_name="Runtime"
    )

    plt.figure(figsize=(12, 6))

    # Create violin plot
    sns.violinplot(x="Algorithm", y="Runtime", data=df, order=medians.keys())

    # Add median points
    for i, algorithm in enumerate(medians.keys()):
        plt.scatter(i, medians[algorithm], color="black", zorder=10)

    # Set labels and title
    plt.xlabel("Algorithm")
    plt.ylabel("Runtime")
    plt.title("Runtime Distribution (25% to 75%)")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output)
