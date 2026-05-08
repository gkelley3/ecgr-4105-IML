import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pydrake.all import (
    LeafSystem,
    DiagramBuilder,
    Simulator,
    LogVectorOutput,
)

# user configuration

# 5,000 samples in final dataset
N_SAMPLES = 5000

# pilot samples - used only for automatic gain-bound tuning
N_PILOT_SAMPLES = 50

# num grid points per gain during both pilot and final dataset generation
N_GRID = 7

# num automatic bound-tuning rounds
N_TUNING_ROUNDS = 10

# expand gain bounds if min/max edge hit rate exceeds this value.
EDGE_THRESHOLD = 0.2

EXPAND_FACTOR = 1.25

SAVE_DIR = Path("/mnt/d/IML_Controller_Project/datasets")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PATH = SAVE_DIR / "pendulum_final_dataset.csv"
PILOT_SAVE_PATH = "/mnt/d/IML_Controller_Project/datasets/pilot_bounds_dataset.csv"

SAVE_EVERY_MINUTES = 10

N_WORKERS = max(cpu_count() - 1, 1)

SIM_TIME = 10.0

# Initial search bounds. These will be automatically adjusted by pilot tuning.
gain_bounds = {
    "kE": [0.1, 7.45],
    "k1": [5.0, 574.53125],
    "k2": [2.0, 194.375],
}


#########################

# utility functions

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_gain_grids(bounds, n_grid=N_GRID):
    kE_vals = np.linspace(bounds["kE"][0], bounds["kE"][1], n_grid)
    k1_vals = np.linspace(bounds["k1"][0], bounds["k1"][1], n_grid)
    k2_vals = np.linspace(bounds["k2"][0], bounds["k2"][1], n_grid)
    return kE_vals, k1_vals, k2_vals


def edge_flags(kE, k1, k2, kE_vals, k1_vals, k2_vals):
    flags = {
        "kE_hit_min": int(kE == kE_vals[0]),
        "kE_hit_max": int(kE == kE_vals[-1]),
        "k1_hit_min": int(k1 == k1_vals[0]),
        "k1_hit_max": int(k1 == k1_vals[-1]),
        "k2_hit_min": int(k2 == k2_vals[0]),
        "k2_hit_max": int(k2 == k2_vals[-1]),
    }
    flags["hit_grid_edge"] = int(any(flags.values()))
    return flags


#########################
# PLANT MODEL
#########################

class TorqueControlledPendulum(LeafSystem):
    def __init__(self, m=1.0, L=1.0, g=9.81, b=0.0):
        super().__init__()

        self.m = m
        self.L = L
        self.g = g
        self.b = b
        self.I = m * L**2

        self.DeclareVectorInputPort("u", 1)
        self.state_index = self.DeclareContinuousState(2)
        self.DeclareStateOutputPort("x", self.state_index)

    def _get_state(self, context):
        x = context.get_continuous_state_vector().CopyToVector()
        return x[0], x[1]

    def DoCalcTimeDerivatives(self, context, derivatives):
        u = self.get_input_port(0).Eval(context)[0]
        theta, theta_dot = self._get_state(context)

        theta_ddot = (u - self.b * theta_dot - self.m * self.g * self.L * np.sin(theta)) / self.I

        derivatives.get_mutable_vector().SetFromVector([
            theta_dot,
            theta_ddot,
        ])


#########################
# HYBRID ENERGY-SHAPING + LQR CONTROLLER
#########################

class Hybrid_EShape_LQR_Controller(LeafSystem):
    def __init__(
        self,
        m=1.0,
        L=1.0,
        g=9.81,
        k_EShape=2.0,
        K_LQR=np.array([20.0, 6.0]),
        theta_switch_deg=20.0,
        theta_dot_switch=1.5,
        u_max=np.inf,
    ):
        super().__init__()

        self.m = m
        self.L = L
        self.g = g
        self.I = m * L**2
        self.k_EShape = k_EShape
        self.K_LQR = np.asarray(K_LQR, dtype=float)
        self.theta_switch = np.deg2rad(theta_switch_deg)
        self.theta_dot_switch = theta_dot_switch
        self.u_max = u_max

        self.DeclareVectorInputPort("x", 2)
        self.DeclareVectorOutputPort("u", 1, self.DoOutput)

    def _get_state(self, context):
        x = self.get_input_port(0).Eval(context)
        return x[0], x[1]

    def eval_total_energy(self, theta, theta_dot):
        kinetic = 0.5 * self.I * theta_dot**2
        potential = self.m * self.g * self.L * (1 - np.cos(theta))
        return kinetic + potential

    def eval_desired_energy(self):
        return 2.0 * self.m * self.g * self.L

    def energy_shaping_control(self, theta, theta_dot):
        E = self.eval_total_energy(theta, theta_dot)
        E_d = self.eval_desired_energy()
        return self.k_EShape * theta_dot * (E_d - E)

    def lqr_control(self, theta, theta_dot):
        theta_err = wrap_to_pi(theta - np.pi)
        x_err = np.array([theta_err, theta_dot])
        return -self.K_LQR @ x_err

    def use_lqr(self, theta, theta_dot):
        theta_err = wrap_to_pi(theta - np.pi)
        return abs(theta_err) < self.theta_switch and abs(theta_dot) < self.theta_dot_switch

    def DoOutput(self, context, output):
        theta, theta_dot = self._get_state(context)

        if self.use_lqr(theta, theta_dot):
            u = self.lqr_control(theta, theta_dot)
        else:
            u = self.energy_shaping_control(theta, theta_dot)

        u = np.clip(u, -self.u_max, self.u_max)
        output.SetFromVector([u])


#########################
# METRICS AND SIMULATION
#########################

def compute_metrics(t, theta, theta_dot, u, sim_time=SIM_TIME):
    theta_error = wrap_to_pi(theta - np.pi)

    upright_error = np.trapezoid(theta_error**2, t)
    control_effort = np.trapezoid(u**2, t)

    final_theta_error = abs(theta_error[-1])
    final_theta_dot = abs(theta_dot[-1])

    success = final_theta_error < np.deg2rad(10.0) and final_theta_dot < 0.5

    angle_tol = np.deg2rad(10.0)
    velocity_tol = 0.5

    settled_mask = (np.abs(theta_error) < angle_tol) & (np.abs(theta_dot) < velocity_tol)

    settling_time = sim_time
    for i in range(len(t)):
        if np.all(settled_mask[i:]):
            settling_time = t[i]
            break

    if not success:
        settling_time = sim_time

    cost = (
        1.0 * settling_time
        + 10.0 * upright_error
        + 0.1 * control_effort
    )

    return {
        "cost": cost,
        "success": int(success),
        "settling_time": settling_time,
        "upright_error": upright_error,
        "control_effort": control_effort,
    }


def simulate_and_score(params, x0, kE, K):
    m, L, b, u_max = params

    builder = DiagramBuilder()

    plant = builder.AddSystem(
        TorqueControlledPendulum(m=m, L=L, b=b)
    )

    hybrid_ctrl = builder.AddSystem(
        Hybrid_EShape_LQR_Controller(
            m=m,
            L=L,
            g=9.81,
            k_EShape=kE,
            K_LQR=K,
            theta_switch_deg=20.0,
            theta_dot_switch=1.5,
            u_max=u_max,
        )
    )

    builder.Connect(plant.get_output_port(), hybrid_ctrl.get_input_port())
    builder.Connect(hybrid_ctrl.get_output_port(), plant.get_input_port())

    state_logger = LogVectorOutput(plant.get_output_port(), builder)
    control_logger = LogVectorOutput(hybrid_ctrl.get_output_port(), builder)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    plant_context.SetContinuousState(x0)

    try:
        simulator.AdvanceTo(SIM_TIME)
    except RuntimeError:
        return {
            "cost": 1e6,
            "success": 0,
            "settling_time": SIM_TIME,
            "upright_error": 1e6,
            "control_effort": 1e6,
        }

    state_log = state_logger.FindLog(context)
    control_log = control_logger.FindLog(context)

    t = state_log.sample_times()
    x = state_log.data()
    u = control_log.data()[0]

    theta = x[0]
    theta_dot = x[1]

    return compute_metrics(t, theta, theta_dot, u, SIM_TIME)

#########################
# ONE-SAMPLE GENERATION
#########################

def generate_one_sample(seed, kE_vals, k1_vals, k2_vals):
    rng = np.random.default_rng(seed)

    # System parameters.
    m = rng.uniform(0.5, 2.0)
    L = rng.uniform(0.5, 1.5)
    b = rng.uniform(0.0, 0.2)
    u_max = rng.uniform(2.0, 10.0)

    # Initial condition.
    theta0 = rng.uniform(-np.pi, np.pi)
    theta_dot0 = rng.uniform(-2.0, 2.0)

    params = (m, L, b, u_max)
    x0 = [theta0, theta_dot0]

    best_cost = np.inf
    best_params = None
    best_metrics = None

    for kE in kE_vals:
        for k1 in k1_vals:
            for k2 in k2_vals:
                K = np.array([k1, k2])

                metrics = simulate_and_score(
                    params=params,
                    x0=x0,
                    kE=kE,
                    K=K,
                )

                cost = metrics["cost"]

                if cost < best_cost:
                    best_cost = cost
                    best_params = (kE, k1, k2)
                    best_metrics = metrics

    kE, k1, k2 = best_params
    flags = edge_flags(kE, k1, k2, kE_vals, k1_vals, k2_vals)

    return {
        "m": m,
        "L": L,
        "b": b,
        "u_max": u_max,
        "theta0": theta0,
        "theta_dot0": theta_dot0,
        "kE": kE,
        "k1": k1,
        "k2": k2,
        "cost": best_metrics["cost"],
        "success": best_metrics["success"],
        "settling_time": best_metrics["settling_time"],
        "upright_error": best_metrics["upright_error"],
        "control_effort": best_metrics["control_effort"],
        **flags,
    }


def generate_one_sample_worker(args):
    seed, kE_vals, k1_vals, k2_vals = args
    return generate_one_sample(seed, kE_vals, k1_vals, k2_vals)


#########################
# AUTOMATIC BOUND TUNING
#########################

def tune_bounds_from_pilot(df, bounds, edge_threshold=EDGE_THRESHOLD, expand_factor=EXPAND_FACTOR):
    new_bounds = {key: value.copy() for key, value in bounds.items()}

    for gain in ["kE", "k1", "k2"]:
        lo, hi = bounds[gain]
        width = hi - lo

        hit_min_rate = df[f"{gain}_hit_min"].mean()
        hit_max_rate = df[f"{gain}_hit_max"].mean()

        if hit_min_rate > edge_threshold:
            new_bounds[gain][0] = max(1e-4, lo - width * (expand_factor - 1.0))

        if hit_max_rate > edge_threshold:
            new_bounds[gain][1] = hi + width * (expand_factor - 1.0)

    return new_bounds


def run_pilot_dataset(bounds, n_pilot=N_PILOT_SAMPLES):
    kE_vals, k1_vals, k2_vals = make_gain_grids(bounds, N_GRID)

    seeds = np.arange(n_pilot)

    worker_args = [
        (seed, kE_vals, k1_vals, k2_vals)
        for seed in seeds
    ]

    rows = []

    with Pool(processes=N_WORKERS) as pool:
        for row in tqdm(
            pool.imap_unordered(generate_one_sample_worker, worker_args),
            total=n_pilot,
            desc="Pilot tuning"
        ):
            rows.append(row)

    return pd.DataFrame(rows)


def auto_tune_bounds(initial_bounds):
    bounds = {key: value.copy() for key, value in initial_bounds.items()}

    for round_idx in range(N_TUNING_ROUNDS):
        pilot_df = run_pilot_dataset(bounds, N_PILOT_SAMPLES)
        pilot_df.to_csv(PILOT_SAVE_PATH, index=False)

        edge_rate = pilot_df["hit_grid_edge"].mean()

        print("\n")
        print(f"BOUND TUNING ROUND {round_idx + 1}")
        print("==============================")
        print(f"Bounds: {bounds}")
        print(f"Overall edge-hit rate: {edge_rate:.3f}")
        print("Per-edge rates:")
        print(
            pilot_df[
                [
                    "kE_hit_min", "kE_hit_max",
                    "k1_hit_min", "k1_hit_max",
                    "k2_hit_min", "k2_hit_max",
                ]
            ].mean()
        )

        edge_cols = [
            "kE_hit_min", "kE_hit_max",
            "k1_hit_min", "k1_hit_max",
            "k2_hit_min", "k2_hit_max",
        ]

        per_edge_rates = pilot_df[edge_cols].mean()
        max_edge_rate = per_edge_rates.max()

        if max_edge_rate < EDGE_THRESHOLD:
            print("Bounds accepted.")
            return bounds

        bounds = tune_bounds_from_pilot(pilot_df, bounds)

    print("Reached maximum bound-tuning rounds.")
    return bounds


#########################
# FINAL DATASET GENERATION
#########################

def generate_dataset(final_bounds):
    import gc
    import time

    kE_vals, k1_vals, k2_vals = make_gain_grids(final_bounds, N_GRID)

    if SAVE_PATH.exists():
        start_idx = len(pd.read_csv(SAVE_PATH))
        print(f"Resuming from sample {start_idx}")
    else:
        start_idx = 0

    remaining = N_SAMPLES - start_idx

    if remaining <= 0:
        print("Dataset already complete.")
        return pd.read_csv(SAVE_PATH)

    seeds = np.arange(start_idx, N_SAMPLES)
    worker_args = [(seed, kE_vals, k1_vals, k2_vals) for seed in seeds]

    print("\nFINAL DATASET GENERATION\n")
    print(f"Using {N_WORKERS} workers")
    print(f"Generating {remaining} remaining samples")
    print(f"Final bounds: {final_bounds}")

    buffer = []
    last_save_time = time.time()
    SAVE_EVERY_SECONDS = 10 * 60

    def flush_buffer():
        nonlocal buffer

        if not buffer:
            return

        file_exists = SAVE_PATH.exists()

        pd.DataFrame(buffer).to_csv(
            SAVE_PATH,
            mode="a",
            header=not file_exists,
            index=False,
        )

        print(f"\nSaved {len(buffer)} rows to {SAVE_PATH}")
        buffer.clear()
        gc.collect()

    with Pool(processes=N_WORKERS, maxtasksperchild=1) as pool:
        for idx, row in enumerate(
            tqdm(pool.imap_unordered(generate_one_sample_worker, worker_args), total=remaining),
            start=start_idx + 1,
        ):
            buffer.append(row)

            time_since_save = time.time() - last_save_time

            if time_since_save >= SAVE_EVERY_SECONDS:
                flush_buffer()
                last_save_time = time.time()

    flush_buffer()

    df = pd.read_csv(SAVE_PATH)

    print(f"Dataset saved to {SAVE_PATH}")
    print(f"Final edge-hit rate: {df['hit_grid_edge'].mean():.3f}")
    print(f"Success rate: {df['success'].mean():.3f}")

    return df

# entry point
if __name__ == "__main__":
    final_bounds = gain_bounds
    print("\nFinal tuned bounds:")
    print(final_bounds)

    df = generate_dataset(final_bounds)
