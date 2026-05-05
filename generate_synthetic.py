"""
generate_synthetic.py
=====================
Generate artificial highway-driving trajectories for human-driven and
self-driving (AV) cars, matching the feature schema of the i24 dataset
(as extracted by extract_features.py → extract_trajectory_features).

Usage
-----
python generate_synthetic.py \
    --n_human 500 \
    --n_av    200 \
    --output  synthetic_highway.csv \
    --seed    42

The output CSV has the same columns as the i24 trajectory-feature CSVs:
  vel_min/max/mean/std, acc_*/jerk_*/acc_dir_*/jerk_dir_*/curv_*,
  delta_x, delta_y, path, delta_t, frac_stop, frac_accel, frac_break,
  x_0..x_{P-1}, y_0..y_{P-1},
  object_id, intersection_id, classification, sub_classification,
  obj_length, obj_width, obj_height
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Physical / simulation constants ──────────────────────────────────────────
DT          = 0.1          # time step [s]  (matches i24 ~0.1 s)
ROAD_LENGTH = 300.0        # highway segment length [m] along y
LANE_WIDTH  = 3.7          # [m]
N_LANES     = 3
PATH_POINTS = 10           # number of path-encoding waypoints (matches i24 default)

VEL_THRESHOLD = 0.5        # same as extract_features.py
ACC_THRESHOLD = 1.0        # same as extract_features.py

# ── IDM default parameters ────────────────────────────────────────────────────
# Reference: Treiber et al. (2000)  https://arxiv.org/abs/cond-mat/0002177
IDM_DEFAULTS = dict(
    v0    = 27.0,   # desired speed [m/s] ≈ 97 km/h
    T     = 1.5,    # desired time headway [s]
    a     = 2.0,    # max acceleration [m/s²]
    b     = 3.0,    # comfortable deceleration [m/s²]
    s0    = 2.0,    # minimum gap [m]
    delta = 4.0,    # acceleration exponent
)

# ── Vehicle geometry distributions (from i24 stats) ─────────────────────────
# Classification = VEHICLE, sub_classification = car
CAR_LENGTH_MU,  CAR_LENGTH_SD  = 4.5,  0.4
CAR_WIDTH_MU,   CAR_WIDTH_SD   = 1.95, 0.10
CAR_HEIGHT_MU,  CAR_HEIGHT_SD  = 1.55, 0.12


# ─────────────────────────────────────────────────────────────────────────────
# IDM longitudinal model
# ─────────────────────────────────────────────────────────────────────────────

def idm_accel(v, delta_v, s, params):
    """
    Compute IDM acceleration for a single vehicle.

    Parameters
    ----------
    v       : current speed [m/s]
    delta_v : approach rate = ego_speed − leader_speed (positive ⇒ closing)
    s       : current gap to leader [m]
    params  : dict with IDM keys
    """
    v0, T, a, b, s0, delta = (
        params["v0"], params["T"], params["a"],
        params["b"], params["s0"], params["delta"],
    )
    s = max(s, 0.1)                        # avoid division by zero
    s_star = s0 + max(0.0, v * T + v * delta_v / (2 * np.sqrt(a * b)))
    acc = a * (1.0 - (v / v0) ** delta - (s_star / s) ** 2)
    return np.clip(acc, -b * 2, a)         # hard limits


# ─────────────────────────────────────────────────────────────────────────────
# Per-driver parameter sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_human_params(rng):
    """Human driver: higher variability in IDM parameters + reaction noise."""
    return dict(
        v0    = rng.normal(25.0, 3.0),      # ±3 m/s desired speed spread
        T     = rng.uniform(0.8, 2.5),      # wide time-headway range
        a     = rng.uniform(1.2, 3.5),
        b     = rng.uniform(2.0, 5.0),
        s0    = rng.uniform(1.5, 4.0),
        delta = 4.0,
        # Noise parameters (not IDM, used during simulation)
        long_noise_std  = rng.uniform(0.4, 1.2),   # longitudinal acc noise [m/s²]
        lat_noise_std   = rng.uniform(0.05, 0.25),  # lateral velocity noise [m/s]
        reaction_delay  = int(rng.integers(1, 4)),   # steps of delayed perception
    )


def sample_av_params(rng):
    """AV: tight IDM parameters, minimal noise."""
    return dict(
        v0    = rng.normal(26.0, 1.0),      # very consistent desired speed
        T     = rng.uniform(1.2, 1.8),
        a     = rng.uniform(1.5, 2.5),
        b     = rng.uniform(2.5, 3.5),
        s0    = rng.uniform(1.5, 2.5),
        delta = 4.0,
        long_noise_std  = rng.uniform(0.02, 0.10),
        lat_noise_std   = rng.uniform(0.005, 0.02),
        reaction_delay  = 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single trajectory simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_trajectory(params, rng, n_steps_range=(40, 400)):
    """
    Simulate one vehicle trajectory using IDM + Gaussian noise.

    The 'leader' is a virtual ghost car maintaining a random constant speed
    ahead at a random initial gap, so each trajectory is independent.

    Returns
    -------
    ts, obj_x, obj_y, vel_x, vel_y   — all as Python lists of floats
    """
    n_steps = int(rng.integers(*n_steps_range))

    # ── Initial conditions ────────────────────────────────────────────────────
    lane = rng.integers(0, N_LANES)
    x0   = (lane + 0.5) * LANE_WIDTH + rng.normal(0, 0.15)   # lateral position
    y0   = rng.uniform(-50, 0)                                  # longitudinal start

    # Random direction: ~70 % forward (positive y), 30 % backward
    direction = 1.0 if rng.random() < 0.7 else -1.0

    v_init    = rng.uniform(5.0, params["v0"])
    v_leader  = rng.uniform(max(0, params["v0"] - 8), params["v0"] + 3)
    gap_init  = rng.uniform(10.0, 60.0)

    delay = params.get("reaction_delay", 0)
    v_history = [v_init] * max(1, delay + 1)  # circular buffer for delay

    x, y   = x0, y0
    vx, vy = 0.0, v_init * direction
    ts_list,  x_list,  y_list  = [], [], []
    vx_list, vy_list = [], []

    gap  = gap_init
    t    = 0.0
    lane_center = x0

    long_ns = params["long_noise_std"]
    lat_ns  = params["lat_noise_std"]

    for _ in range(n_steps):
        ts_list.append(t)
        x_list.append(x)
        y_list.append(y)
        vx_list.append(vx)
        vy_list.append(vy)

        v_abs = abs(vy)

        # ── Longitudinal: IDM on delayed speed ───────────────────────────────
        v_delayed   = v_history[0]
        delta_v     = v_delayed - v_leader
        acc_idm     = idm_accel(v_delayed, delta_v, gap, params)
        acc_lon     = acc_idm + rng.normal(0, long_ns)

        new_vy = vy + direction * acc_lon * DT
        # Enforce non-negative speed magnitude
        if direction * new_vy < 0:
            new_vy = 0.0
        vy = new_vy

        # ── Lateral: spring toward lane center + noise ────────────────────────
        # Spring constant: humans weaker, AVs stronger (encoded via lat_noise_std)
        k_lat = 0.5 if lat_ns > 0.08 else 2.5
        acc_lat = -k_lat * (x - lane_center) + rng.normal(0, lat_ns)
        vx += acc_lat * DT
        vx = np.clip(vx, -2.0, 2.0)

        # ── Integrate position ────────────────────────────────────────────────
        x += vx * DT
        y += vy * DT

        # ── Update virtual leader gap ─────────────────────────────────────────
        gap += (v_leader - abs(vy)) * DT
        gap = max(gap, params["s0"])

        # ── Update delay buffer ───────────────────────────────────────────────
        v_history.append(abs(vy))
        if len(v_history) > max(1, delay + 1):
            v_history.pop(0)

        t += DT

    return ts_list, x_list, y_list, vx_list, vy_list


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (mirrors extract_features.py / process_trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def path_encoding(path_cumlen, x_arr, y_arr, n):
    """Interpolate n evenly-spaced path waypoints."""
    j = 0
    px, py = [], []
    for i in range(n):
        target = path_cumlen[0] + i * (path_cumlen[-1] - path_cumlen[0]) / max(n - 1, 1)
        j = j + int(np.argmax(path_cumlen[j:] >= target))
        if j == 0:
            px.append(x_arr[0]);  py.append(y_arr[0])
        else:
            coeff = (target - path_cumlen[j - 1]) / (path_cumlen[j] - path_cumlen[j - 1] + 1e-12)
            px.append(x_arr[j - 1] * (1 - coeff) + x_arr[j] * coeff)
            py.append(y_arr[j - 1] * (1 - coeff) + y_arr[j] * coeff)
    return px, py


def extract_features(ts, obj_x, obj_y, vel_x, vel_y, points=PATH_POINTS):
    """
    Extract trajectory-level features exactly as in extract_features.py.
    Returns a dict ready to append to a DataFrame row.
    """
    t      = np.array(ts,    dtype=float)
    x      = np.array(obj_x, dtype=float)
    y      = np.array(obj_y, dtype=float)
    vx     = np.array(vel_x, dtype=float)
    vy     = np.array(vel_y, dtype=float)
    vel    = np.sqrt(vx**2 + vy**2)

    t_deltas = np.diff(t)
    t_deltas = np.where(t_deltas == 0, 1e-9, t_deltas)

    acc_x  = np.diff(vx) / t_deltas
    acc_y  = np.diff(vy) / t_deltas
    acc    = np.sqrt(acc_x**2 + acc_y**2)

    jerk_x = np.diff(acc_x) / t_deltas[1:]
    jerk_y = np.diff(acc_y) / t_deltas[1:]
    jerk   = np.sqrt(jerk_x**2 + jerk_y**2)

    acc_dir  = np.diff(vel) / t_deltas
    jerk_dir = np.diff(acc_dir) / t_deltas[1:]

    # Curvature (only where |vel| > threshold)
    vel1 = vel[1:]
    curv_mask = vel1 > VEL_THRESHOLD
    curv_vals = (
        np.abs(vx[1:] * acc_y - vy[1:] * acc_x) / np.maximum(vel1, 1e-9) ** 3
    )[curv_mask]

    x_deltas   = np.diff(x)
    y_deltas   = np.diff(y)
    path_steps = np.sqrt(x_deltas**2 + y_deltas**2)
    path_total = float(np.sum(path_steps))
    path_cumlen = np.concatenate(([0.0], np.cumsum(path_steps)))

    delta_t = float(t[-1] - t[0])

    res = dict(
        vel_min  = float(np.min(vel)),
        vel_max  = float(np.max(vel)),
        vel_mean = float(np.mean(vel)),
        vel_std  = float(np.std(vel)),

        acc_min  = float(np.min(acc)),
        acc_max  = float(np.max(acc)),
        acc_mean = float(np.mean(acc)),
        acc_std  = float(np.std(acc)),

        jerk_min  = float(np.min(jerk)),
        jerk_max  = float(np.max(jerk)),
        jerk_mean = float(np.mean(jerk)),
        jerk_std  = float(np.std(jerk)),

        acc_dir_min  = float(np.min(acc_dir)),
        acc_dir_max  = float(np.max(acc_dir)),
        acc_dir_mean = float(np.mean(acc_dir)),
        acc_dir_std  = float(np.std(acc_dir)),

        jerk_dir_min  = float(np.min(jerk_dir)),
        jerk_dir_max  = float(np.max(jerk_dir)),
        jerk_dir_mean = float(np.mean(jerk_dir)),
        jerk_dir_std  = float(np.std(jerk_dir)),

        curv_min  = float(np.min(curv_vals))  if len(curv_vals) else float("nan"),
        curv_max  = float(np.max(curv_vals))  if len(curv_vals) else float("nan"),
        curv_mean = float(np.mean(curv_vals)) if len(curv_vals) else float("nan"),
        curv_std  = float(np.std(curv_vals))  if len(curv_vals) else float("nan"),

        delta_x = float(x[-1] - x[0]),
        delta_y = float(y[-1] - y[0]),
        path    = path_total,
        delta_t = delta_t,

        frac_stop   = float(np.sum(t_deltas[vel[1:] <= VEL_THRESHOLD]) / delta_t)
                      if delta_t > 0 else 0.0,
        frac_accel  = float(np.sum(t_deltas[acc_dir >  ACC_THRESHOLD]) / delta_t)
                      if delta_t > 0 else 0.0,
        frac_break  = float(np.sum(t_deltas[acc_dir < -ACC_THRESHOLD]) / delta_t)
                      if delta_t > 0 else 0.0,
    )

    # Path-encoding waypoints (relative to start)
    px, py = path_encoding(path_cumlen, x, y, points)
    for i in range(points):
        res[f"x_{i}"] = px[i] - x[0]
        res[f"y_{i}"] = py[i] - y[0]

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate(n_human: int, n_av: int, seed: int, n_steps_range=(40, 400)):
    rng = np.random.default_rng(seed)
    rows = []
    obj_id = 300_000          # start well above real i24 object_ids

    configs = [
        ("human", n_human, sample_human_params),
        ("av",    n_av,    sample_av_params),
    ]

    for driver_type, n, param_fn in configs:
        sub_cls = "car"
        cls     = "VEHICLE"
        print(f"\nSimulating {n} {driver_type} cars …")

        for _ in tqdm(range(n)):
            params = param_fn(rng)

            # Vehicle geometry
            length = float(np.clip(rng.normal(CAR_LENGTH_MU, CAR_LENGTH_SD), 3.0, 6.5))
            width  = float(np.clip(rng.normal(CAR_WIDTH_MU,  CAR_WIDTH_SD),  1.5, 2.5))
            height = float(np.clip(rng.normal(CAR_HEIGHT_MU, CAR_HEIGHT_SD), 1.1, 2.1))

            ts, ox, oy, vx, vy = simulate_trajectory(
                params, rng, n_steps_range=n_steps_range
            )

            # Need at least 3 points for acc and jerk calculations
            if len(ts) < 4:
                continue

            feat = extract_features(ts, ox, oy, vx, vy)
            feat.update(dict(
                object_id           = obj_id,
                intersection_id     = 7,           # matches i24 datasets
                classification      = cls,
                sub_classification  = sub_cls,
                obj_length          = length,
                obj_width           = width,
                obj_height          = height,
                driver_type         = driver_type,  # extra label column
            ))
            rows.append(feat)
            obj_id += 1

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic human / AV highway trajectories."
    )
    parser.add_argument("--n_human",    type=int,   default=500,
                        help="Number of human-driven car trajectories")
    parser.add_argument("--n_av",       type=int,   default=200,
                        help="Number of AV (self-driving) car trajectories")
    parser.add_argument("--output",     type=str,   default="synthetic_highway.csv",
                        help="Output CSV path")
    parser.add_argument("--seed",       type=int,   default=42,
                        help="Random seed")
    parser.add_argument("--min_steps",  type=int,   default=40,
                        help="Minimum trajectory length in time steps")
    parser.add_argument("--max_steps",  type=int,   default=400,
                        help="Maximum trajectory length in time steps")
    args = parser.parse_args()

    df = generate(
        n_human=args.n_human,
        n_av=args.n_av,
        seed=args.seed,
        n_steps_range=(args.min_steps, args.max_steps),
    )

    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df):,} trajectories → {args.output}")
    print(df[["driver_type", "vel_mean", "acc_std", "jerk_std",
              "frac_accel", "frac_break", "frac_stop"]].groupby("driver_type").mean().round(4))


if __name__ == "__main__":
    main()
