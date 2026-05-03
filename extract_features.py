import numpy as np
import pandas as pd
import json
import argparse


def extract_features(input_path, output_path, window, stride):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--window", "-w", type=int, default=20)
    parser.add_argument("--stride", "-s", type=int, default=10)

    args = parser.parse_args()
    extract_features(args.input_path, args.output_path, args.window, args.stride)


if __name__ == "__main__":
    main()
