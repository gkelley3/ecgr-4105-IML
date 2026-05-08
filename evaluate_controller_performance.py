from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from dataset_generator import (
    simulate_and_score,
    TorqueControlledPendulum,
    Hybrid_EShape_LQR_Controller,
)

from pydrake.all import DiagramBuilder, Simulator, LogVectorOutput

DATA_PATH = Path("/mnt/d/IML_Controller_Project/datasets/pendulum_final_dataset.csv")
MODEL_PATH = Path("/mnt/d/IML_Controller_Project/models/hybrid_gain_predictor.joblib")
OUT_DIR = Path("/mnt/d/IML_Controller_Project/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_TIME = 10.0
N_EVAL = 250


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_features(row):
    return pd.DataFrame([{
        "m": row["m"],
        "L": row["L"],
        "b": row["b"],
        "u_max": row["u_max"],
        "sin_theta0": np.sin(row["theta0"]),
        "cos_theta0": np.cos(row["theta0"]),
        "theta_dot0": row["theta_dot0"],
    }])


def get_dataset_gains(row):
    return {
        "kE": float(row["kE"]),
        "k1": float(row["k1"]),
        "k2": float(row["k2"]),
    }


def predict_ml_gains(row, model_bundle):
    X = make_features(row)

    kE_model = model_bundle["kE_model"]
    K_model = model_bundle["K_model"]

    kE = float(kE_model.predict(X).ravel()[0])
    k1, k2 = K_model.predict(X).ravel()

    return {
        "kE": kE,
        "k1": float(k1),
        "k2": float(k2),
    }


def simulate_with_logs(row, gains):
    m = row["m"]
    L = row["L"]
    b = row["b"]
    u_max = row["u_max"]

    x0 = [row["theta0"], row["theta_dot0"]]

    builder = DiagramBuilder()

    plant = builder.AddSystem(
        TorqueControlledPendulum(m=m, L=L, b=b)
    )

    controller = builder.AddSystem(
        Hybrid_EShape_LQR_Controller(
            m=m,
            L=L,
            g=9.81,
            k_EShape=gains["kE"],
            K_LQR=np.array([gains["k1"], gains["k2"]]),
            theta_switch_deg=20.0,
            theta_dot_switch=1.5,
            u_max=u_max,
        )
    )

    builder.Connect(plant.get_output_port(), controller.get_input_port())
    builder.Connect(controller.get_output_port(), plant.get_input_port())

    state_logger = LogVectorOutput(plant.get_output_port(), builder)
    control_logger = LogVectorOutput(controller.get_output_port(), builder)

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant_context.SetContinuousState(x0)

    simulator.AdvanceTo(SIM_TIME)

    state_log = state_logger.FindLog(context)
    control_log = control_logger.FindLog(context)

    t = state_log.sample_times()
    x = state_log.data()
    u = control_log.data()[0]

    theta = x[0]
    theta_dot = x[1]

    return {
        "t": t,
        "theta": theta,
        "theta_dot": theta_dot,
        "theta_error": wrap_to_pi(theta - np.pi),
        "u": u,
    }


def evaluate_one(row, gains):
    metrics = simulate_and_score(
        params=(row["m"], row["L"], row["b"], row["u_max"]),
        x0=[row["theta0"], row["theta_dot0"]],
        kE=gains["kE"],
        K=np.array([gains["k1"], gains["k2"]]),
    )

    rms_control = np.sqrt(metrics["control_effort"] / SIM_TIME)

    return {
        "success": metrics["success"],
        "settling_time": metrics["settling_time"],
        "cost": metrics["cost"],
        "upright_error": metrics["upright_error"],
        "control_effort": metrics["control_effort"],
        "rms_control": rms_control,
    }


def plot_example(row, dataset_gains, ml_gains):
    dataset_log = simulate_with_logs(row, dataset_gains)
    ml_log = simulate_with_logs(row, ml_gains)

    plt.figure(figsize=(8, 4.5))
    plt.plot(dataset_log["t"], dataset_log["theta"], label="Dataset grid-search gains")
    plt.plot(ml_log["t"], ml_log["theta"], label="ML-predicted gains")
    plt.axhline(np.pi, linestyle="--", label="Upright target")
    plt.xlabel("Time [s]")
    plt.ylabel("Pendulum angle θ [rad]")
    plt.title("Pendulum Angle vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "angle_vs_time_dataset_vs_ml.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(dataset_log["t"], dataset_log["u"], label="Dataset grid-search gains")
    plt.plot(ml_log["t"], ml_log["u"], label="ML-predicted gains")
    plt.xlabel("Time [s]")
    plt.ylabel("Control input u [N·m]")
    plt.title("Control Input vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "control_input_dataset_vs_ml.png", dpi=300)
    plt.close()


def main():
    df = pd.read_csv(DATA_PATH)

    df = df[np.isfinite(df["cost"])]
    df = df[df["cost"] < 1e6]
    df = df[df["success"] == 1]

    eval_df = df.sample(
        min(N_EVAL, len(df)),
        random_state=42,
    ).copy()

    model_bundle = joblib.load(MODEL_PATH)

    dataset_results = []
    ml_results = []
    gain_errors = []

    for _, row in eval_df.iterrows():
        dataset_gains = get_dataset_gains(row)
        ml_gains = predict_ml_gains(row, model_bundle)

        dataset_results.append(evaluate_one(row, dataset_gains))
        ml_results.append(evaluate_one(row, ml_gains))

        gain_errors.append({
            "kE_abs_error": abs(ml_gains["kE"] - dataset_gains["kE"]),
            "k1_abs_error": abs(ml_gains["k1"] - dataset_gains["k1"]),
            "k2_abs_error": abs(ml_gains["k2"] - dataset_gains["k2"]),
        })

    dataset_results = pd.DataFrame(dataset_results)
    ml_results = pd.DataFrame(ml_results)
    gain_errors = pd.DataFrame(gain_errors)

    summary = pd.DataFrame({
        "Controller": ["Dataset Grid-Search", "ML-Predicted"],
        "Success Rate": [
            dataset_results["success"].mean(),
            ml_results["success"].mean(),
        ],
        "Mean Settling Time [s]": [
            dataset_results["settling_time"].mean(),
            ml_results["settling_time"].mean(),
        ],
        "Mean RMS Control": [
            dataset_results["rms_control"].mean(),
            ml_results["rms_control"].mean(),
        ],
        "Mean Cost": [
            dataset_results["cost"].mean(),
            ml_results["cost"].mean(),
        ],
        "Mean Upright Error": [
            dataset_results["upright_error"].mean(),
            ml_results["upright_error"].mean(),
        ],
        "Mean Control Effort": [
            dataset_results["control_effort"].mean(),
            ml_results["control_effort"].mean(),
        ],
    })

    print("\nQUANTITATIVE RESULTS\n")
    print(summary.to_string(index=False))

    cost_ratio = ml_results["cost"].mean() / dataset_results["cost"].mean()
    success_drop = dataset_results["success"].mean() - ml_results["success"].mean()

    print("\nCONTROLLER COMPARISON\n")
    print(f"ML / dataset cost ratio = {cost_ratio:.4f}")
    print(f"Success rate drop       = {success_drop:.4f}")

    print("\nGAIN PREDICTION ERROR\n")
    print(f"kE MAE = {gain_errors['kE_abs_error'].mean():.4f}")
    print(f"k1 MAE = {gain_errors['k1_abs_error'].mean():.4f}")
    print(f"k2 MAE = {gain_errors['k2_abs_error'].mean():.4f}")

    summary.to_csv(OUT_DIR / "controller_performance_dataset_vs_ml.csv", index=False)
    gain_errors.to_csv(OUT_DIR / "gain_prediction_errors_dataset_vs_ml.csv", index=False)

    example_row = eval_df.iloc[0]
    example_dataset_gains = get_dataset_gains(example_row)
    example_ml_gains = predict_ml_gains(example_row, model_bundle)

    plot_example(example_row, example_dataset_gains, example_ml_gains)

    print(f"\nSaved results to: {OUT_DIR}")
    print("Saved plots:")
    print(f"  {OUT_DIR / 'angle_vs_time_dataset_vs_ml.png'}")
    print(f"  {OUT_DIR / 'control_input_dataset_vs_ml.png'}")


if __name__ == "__main__":
    main()