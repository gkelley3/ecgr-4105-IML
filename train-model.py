from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# imported from dataset generation script, datset_generator.py
from dataset_generator import simulate_and_score

DATA_PATH = Path("/mnt/d/IML_Controller_Project/datasets/pendulum_final_dataset.csv")
MODEL_PATH = Path("/mnt/d/IML_Controller_Project/models/hybrid_gain_predictor.joblib")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CLOSED_LOOP_EVAL = 250

df = pd.read_csv(DATA_PATH)

# df = df[np.isfinite(df["cost"])]
# df = df[df["cost"] < 1e6]
# df = df[df["success"] == 1]

print(f"Training samples after filtering: {len(df)}")

# angle periodicity features
df["sin_theta0"] = np.sin(df["theta0"])
df["cos_theta0"] = np.cos(df["theta0"])

feature_cols = [ "m", "L", "b", "u_max", 
                "sin_theta0", "cos_theta0", "theta_dot0",
            ]

target_cols = ["kE", "k1", "k2"]

X = df[feature_cols]
y = df[target_cols]

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[target_cols]
y_test = test_df[target_cols]

# model 1: MLP for kE
kE_model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("x_scaler", StandardScaler()),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(256, 256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=5e-4,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=RANDOM_STATE,
            verbose=True,
        )),
    ]),
    transformer=StandardScaler(),
)

# model 2: Random forest for LQR gains k1, k2
K_model = Pipeline([
    ("x_scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )),
])

print("\nTraining kE MLP model...")
kE_model.fit(X_train, y_train[["kE"]])

print("\nTraining K = [k1, k2] random forest model...")
K_model.fit(X_train, y_train[["k1", "k2"]])

# predict gains
kE_pred = kE_model.predict(X_test).reshape(-1, 1)
K_pred = K_model.predict(X_test)

y_pred = np.hstack([kE_pred, K_pred])
y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

# clip to observed gain ranges
for col in target_cols:
    lo = train_df[col].min()
    hi = train_df[col].max()
    y_pred_df[col] = y_pred_df[col].clip(lo, hi)


# Gain prediction metrics
mae = mean_absolute_error(y_test, y_pred_df, multioutput="raw_values")
rmse = np.sqrt(mean_squared_error(y_test, y_pred_df, multioutput="raw_values"))
r2 = r2_score(y_test, y_pred_df, multioutput="raw_values")

print("\nGain Prediction Performance\n")
for i, target in enumerate(target_cols):
    target_range = y_test[target].max() - y_test[target].min()
    normalized_mae = mae[i] / target_range if target_range != 0 else np.nan

    print(f"{target}:")
    print(f"  MAE            = {mae[i]:.4f}")
    print(f"  RMSE           = {rmse[i]:.4f}")
    print(f"  R^2            = {r2[i]:.4f}")
    print(f"  Normalized MAE = {normalized_mae:.4f}")


# Closed-loop controller evaluation
def evaluate_controller_row(row, gains):
    params = ( row["m"], row["L"], row["b"], row["u_max"])

    x0 = [row["theta0"], row["theta_dot0"]]

    kE = gains["kE"]
    K = np.array([gains["k1"], gains["k2"]])

    return simulate_and_score(params=params, x0=x0, kE=kE, K=K)


eval_df = test_df.copy()
eval_df[["kE_ml", "k1_ml", "k2_ml"]] = y_pred_df[["kE", "k1", "k2"]]

if len(eval_df) > N_CLOSED_LOOP_EVAL:
    eval_df = eval_df.sample(N_CLOSED_LOOP_EVAL, random_state=RANDOM_STATE)

dataset_results = []
ml_results = []

print("\nRunning closed-loop evaluation")
for _, row in eval_df.iterrows():
    dataset_gains = {
        "kE": row["kE"],
        "k1": row["k1"],
        "k2": row["k2"],
    }

    ml_gains = {
        "kE": row["kE_ml"],
        "k1": row["k1_ml"],
        "k2": row["k2_ml"],
    }

    dataset_metrics = evaluate_controller_row(row, dataset_gains)
    ml_metrics = evaluate_controller_row(row, ml_gains)

    dataset_results.append(dataset_metrics)
    ml_results.append(ml_metrics)

dataset_results = pd.DataFrame(dataset_results)
ml_results = pd.DataFrame(ml_results)

print("\nClosed-Loop Performance\n")

print("Dataset grid-search gains:")
print(f"  Success rate    = {dataset_results['success'].mean():.4f}")
print(f"  Mean cost       = {dataset_results['cost'].mean():.4f}")
print(f"  Mean settling   = {dataset_results['settling_time'].mean():.4f}")
print(f"  Mean error      = {dataset_results['upright_error'].mean():.4f}")
print(f"  Mean effort     = {dataset_results['control_effort'].mean():.4f}")

print("\nML-predicted gains:")
print(f"  Success rate    = {ml_results['success'].mean():.4f}")
print(f"  Mean cost       = {ml_results['cost'].mean():.4f}")
print(f"  Mean settling   = {ml_results['settling_time'].mean():.4f}")
print(f"  Mean error      = {ml_results['upright_error'].mean():.4f}")
print(f"  Mean effort     = {ml_results['control_effort'].mean():.4f}")

cost_ratio = ml_results["cost"].mean() / dataset_results["cost"].mean()
success_drop = dataset_results["success"].mean() - ml_results["success"].mean()

print("\nController comparison:")
print(f"  ML / dataset cost ratio = {cost_ratio:.4f}")
print(f"  Success rate drop       = {success_drop:.4f}")


# save trained model bundle
joblib.dump(
    {
        "kE_model": kE_model,
        "K_model": K_model,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
    },
    MODEL_PATH,
)

print(f"\nModel saved at: {MODEL_PATH}")