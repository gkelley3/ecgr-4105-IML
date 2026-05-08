import sys
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "m", "L", "b", "u_max",
    "theta0", "theta_dot0",
    "kE", "k1", "k2",
    "cost",
    "success",
    "settling_time",
    "upright_error",
    "control_effort",
    "hit_grid_edge"
]


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def print_summary_metrics(df):
    print("\nDATASET SUMMARY\n")
    print(f"Number of samples: {len(df)}")

    print("\nCONTROLLER PARAMETER SUMMARY\n")
    print(df[["kE", "k1", "k2"]].describe())

    print("\nCOST SUMMARY\n")
    print(df["cost"].describe())

    print("\nCOST VARIANCE\n")
    print(f"Variance: {df['cost'].var():.6f}")
    print(f"Standard deviation: {df['cost'].std():.6f}")
    
    print("\nDATASET QUALITY\n")
    print(f"Success rate: {df['success'].mean():.3f}")
    print(f"Grid edge hit rate: {df['hit_grid_edge'].mean():.3f}")
    print(f"Average settling time: {df['settling_time'].mean():.3f}")
    print(f"Average upright error: {df['upright_error'].mean():.3f}")
    print(f"Average control effort: {df['control_effort'].mean():.3f}")


def plot_controller_distributions(df):
    for col in ["kE", "k1", "k2"]:
        plt.figure()
        plt.hist(df[col], bins=30)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"{col}_distribution.png", dpi=300)
        plt.show()


def plot_cost_distribution(df):
    plt.figure()
    plt.hist(df["cost"], bins=30)
    plt.xlabel("Cost")
    plt.ylabel("Count")
    plt.title("Distribution of Cost")
    plt.tight_layout()
    plt.savefig("cost_distribution.png", dpi=300)
    plt.show()


def plot_correlations(df):
    corr_cols = [
        "m", "L", "b", "u_max",
        "theta0", "theta_dot0",
        "kE", "k1", "k2",
        "cost"
    ]

    corr = df[corr_cols].corr()

    print("\nCorrelation Matrix\n")
    print(corr)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr)
    plt.colorbar(label="Correlation coefficient")

    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
    plt.yticks(range(len(corr_cols)), corr_cols)

    plt.title("Dataset Correlation Matrix")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300)
    plt.show()


def plot_cost_vs_controller_params(df):
    for col in ["kE", "k1", "k2"]:
        plt.figure()
        plt.scatter(df[col], df["cost"], s=10, alpha=0.5)
        plt.xlabel(col)
        plt.ylabel("Cost")
        plt.title(f"Cost vs {col}")
        plt.tight_layout()
        plt.savefig(f"cost_vs_{col}.png", dpi=300)
        plt.show()


def plot_controller_vs_inputs(df):
    input_cols = ["m", "L", "b", "u_max", "theta0", "theta_dot0"]
    output_cols = ["kE", "k1", "k2"]

    for output in output_cols:
        for input_col in input_cols:
            plt.figure()
            plt.scatter(df[input_col], df[output], s=10, alpha=0.5)
            plt.xlabel(input_col)
            plt.ylabel(output)
            plt.title(f"{output} vs {input_col}")
            plt.tight_layout()
            plt.savefig(f"{output}_vs_{input_col}.png", dpi=300)
            plt.show()


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python analyze_dataset.py pendulum_dataset.csv")
        sys.exit(1)

    csv_path = "/mnt/d/IML_Controller_Project/datasets/pendulum_final_dataset.csv"

    df = load_dataset(csv_path)

    print_summary_metrics(df)
    plot_controller_distributions(df)
    plot_cost_distribution(df)
    plot_correlations(df)
    plot_cost_vs_controller_params(df)
    plot_controller_vs_inputs(df)

    print("\nAnalysis complete. Figures saved as PNG files.")


if __name__ == "__main__":
    main()