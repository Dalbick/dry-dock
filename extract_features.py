import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm


def extract_window_features(input_path, output_path, window, stride):
    with open(input_path, "rb") as f:
        raw_data = json.load(f)
    fields = [
        ("object_id", False, "int64"),
        ("intersection_id", False, "int64"),
        ("classification", False, "S16"),
        ("sub_classification", False, "S16"),
        ("obj_length", False, "float64"),
        ("obj_width", False, "float64"),
        ("obj_height", False, "float64"),
        ("ts", True, "float64"),
        ("obj_x", True, "float64"),
        ("obj_y", True, "float64"),
        ("vel_x", True, "float64"),
        ("vel_y", True, "float64"),
        ("heading", True, "float64"),
        # ("utm_x", True, "float64"),
        # ("utm_y", True, "float64"),
        # ("rot_heading", True, "float64"),
    ]
    df = pd.DataFrame(
        np.array(
            [
                tuple(
                    row[field][i] if indexed else row[field]
                    for field, indexed, _ in fields
                )
                for row in raw_data
                for i in range(len(row["ts"]))
            ],
            [(field, dtype) for field, _, dtype in fields],
        )
    )

    object_bounds = df["object_id"] != df["object_id"].shift(1)
    df["ts_delta"] = df["ts"].diff()
    df.loc[object_bounds, "ts_delta"] = pd.NA

    diff_fields = {
        "obj_x": ("disp_x", False),
        "obj_y": ("disp_y", False),
        "vel_x": ("acc_x", True),
        "vel_y": ("acc_y", True),
        "acc_x": ("jerk_x", True),
        "acc_y": ("jerk_y", True),
    }
    for in_field, (out_field, timed) in diff_fields.items():
        df[out_field] = df[in_field].diff()
        if timed:
            df[out_field] = df[out_field] / df["ts_delta"]
    df = df.dropna()

    xy_fields = ["disp", "vel", "acc", "jerk"]
    for field in xy_fields:
        df[field] = (df[field + "_x"].pow(2) + df[field + "_y"].pow(2)).pow(0.5)
    df = df.rename(columns={"disp": "path"})

    df["curv"] = (df["vel_x"] * df["acc_y"] - df["vel_y"] * df["acc_x"]).abs() / df[
        "vel"
    ].pow(3)

    agg_fields = [
        "vel_x",
        "vel_y",
        "vel",
        "acc_x",
        "acc_y",
        "acc",
        "jerk_x",
        "jerk_y",
        "jerk",
        "curv",
        "heading",
    ]
    sum_fields = ["disp_x", "disp_y", "path"]
    rolled = (
        df.groupby("object_id")[agg_fields + sum_fields]
        .rolling(window)
        .agg(
            {field: ["mean", "std", "min", "max"] for field in agg_fields}
            | {field: ["sum"] for field in sum_fields}
        )
    )
    rolled.columns = [
        f"{col}_{stat}" if stat != "sum" else col for col, stat in rolled.columns
    ]
    rolled = rolled.reset_index(level=0, drop=True)
    df = df.drop(columns=["disp_x", "disp_y", "path"])
    df = df.join(rolled)
    df["local_idx"] = df.groupby("object_id").cumcount()
    df = df[(df["local_idx"] % stride == stride - 1) & (df["local_idx"] >= window - 1)]
    df = df.drop(columns=["local_idx", "ts", "ts_delta", "obj_x", "obj_y"] + agg_fields)
    df["disp"] = (df["disp_x"].pow(2) + df["disp_y"].pow(2)).pow(0.5)

    df.to_csv(output_path, index=False)


def extract_trajectory_features(input_path, output_path, points):
    with open(input_path, "rb") as f:
        raw_data = json.load(f)
    preserved_fields = [
        "object_id",
        "intersection_id",
        "classification",
        "sub_classification",
        "obj_length",
        "obj_width",
        "obj_height",
    ]
    res = []
    for obj in tqdm(raw_data):
        data = process_trajectory(obj, points)
        data |= {field: obj[field] for field in preserved_fields}
        res.append(data)
    df = pd.DataFrame(res)
    df.to_csv(output_path, index=False)


VEL_THRESHOLD = 0.5
ACC_THRESHOLD = 1.0


def process_trajectory(data, points):
    res = {}
    t = np.array(data["ts"])
    vel_x = np.array(data["vel_x"])
    vel_y = np.array(data["vel_y"])
    vel = np.sqrt(np.square(vel_x) + np.square(vel_y))
    res["vel_min"] = np.min(vel)
    res["vel_max"] = np.max(vel)
    res["vel_mean"] = np.mean(vel)
    res["vel_std"] = np.std(vel)
    t_deltas = np.diff(t)
    acc_x = np.diff(vel_x) / t_deltas
    acc_y = np.diff(vel_y) / t_deltas
    acc = np.sqrt(np.square(acc_x) + np.square(acc_y))
    res["acc_min"] = np.min(acc)
    res["acc_max"] = np.max(acc)
    res["acc_mean"] = np.mean(acc)
    res["acc_std"] = np.std(acc)
    jerk_x = np.diff(acc_x) / t_deltas[1:]
    jerk_y = np.diff(acc_y) / t_deltas[1:]
    jerk = np.sqrt(np.square(jerk_x) + np.square(jerk_y))
    res["jerk_min"] = np.min(jerk)
    res["jerk_max"] = np.max(jerk)
    res["jerk_mean"] = np.mean(jerk)
    res["jerk_std"] = np.std(jerk)
    acc_dir = np.diff(vel) / t_deltas
    res["acc_dir_min"] = np.min(acc_dir)
    res["acc_dir_max"] = np.max(acc_dir)
    res["acc_dir_mean"] = np.mean(acc_dir)
    res["acc_dir_std"] = np.std(acc_dir)
    jerk_dir = np.diff(acc_dir) / t_deltas[1:]
    res["jerk_dir_min"] = np.min(jerk_dir)
    res["jerk_dir_max"] = np.max(jerk_dir)
    res["jerk_dir_mean"] = np.mean(jerk_dir)
    res["jerk_dir_std"] = np.std(jerk_dir)
    curv = (np.abs(vel_x[1:] * acc_y - vel_y[1:] * acc_x) / np.pow(vel[1:], 3))[
        vel[1:] > VEL_THRESHOLD
    ]
    res["curv_min"] = np.min(curv) if len(curv) else np.nan
    res["curv_max"] = np.max(curv) if len(curv) else np.nan
    res["curv_mean"] = np.mean(curv) if len(curv) else np.nan
    res["curv_std"] = np.std(curv) if len(curv) else np.nan
    x = np.array(data["obj_x"])
    y = np.array(data["obj_y"])
    x_deltas = np.diff(x)
    y_deltas = np.diff(y)
    path_deltas = np.sqrt(np.square(x_deltas) + np.square(y_deltas))
    res["delta_x"] = x[-1] - x[0]
    res["delta_y"] = y[-1] - y[0]
    res["path"] = np.sum(path_deltas)
    res["delta_t"] = t[-1] - t[0]
    res["frac_stop"] = np.sum(t_deltas[vel[1:] <= VEL_THRESHOLD]) / res["delta_t"]
    res["frac_accel"] = np.sum(t_deltas[acc_dir > ACC_THRESHOLD]) / res["delta_t"]
    res["frac_break"] = np.sum(t_deltas[acc_dir < -ACC_THRESHOLD]) / res["delta_t"]
    p_x, p_y = path_encoding(
        np.concatenate((np.array([0.0]), np.cumsum(path_deltas))), x, y, points
    )
    for i in range(points):
        res[f"x_{i}"] = p_x[i] - x[0]
        res[f"y_{i}"] = p_y[i] - y[0]
    return res


def path_encoding(path, x, y, n):
    j = 0
    p_x = []
    p_y = []
    for i in range(n):
        target = path[0] + i * (path[-1] - path[0]) / (n - 1)
        j = j + np.argmax(path[j:] >= target)
        if j == 0:
            p_x.append(x[0])
            p_y.append(y[0])
        else:
            coeff = (target - path[j - 1]) / (path[j] - path[j - 1])
            p_x.append(x[j - 1] * (1 - coeff) + x[j] * coeff)
            p_y.append(y[j - 1] * (1 - coeff) + y[j] * coeff)
    return p_x, p_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--type", "-t", choices=["window", "trajectory"], required=True)
    parser.add_argument("--window", "-w", type=int, default=20)
    parser.add_argument("--stride", "-s", type=int, default=10)
    parser.add_argument("--points", "-p", type=int, default=0)

    args = parser.parse_args()
    if args.type == "window":
        extract_window_features(
            args.input_path, args.output_path, args.window, args.stride
        )
    else:
        extract_trajectory_features(args.input_path, args.output_path, args.points)


if __name__ == "__main__":
    main()
