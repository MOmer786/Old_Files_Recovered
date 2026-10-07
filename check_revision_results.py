#!/usr/bin/env python3
"""Read-only Rim comparison and F2B-Net three-seed results.

Save in ~/cfi_project/scripts, then run from ~/cfi_project:
    python scripts/check_revision_results.py

Requires pandas and numpy. Does not import training scripts or change files.
The embedded references are Rim's rounded Table 2 values in the uploaded
Updated_Manuscript_v3 (2).tex, reviewed October 7, 2026.
"""

import argparse
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd


# MAE mean, MAE sample SD, RMSE mean, RMSE sample SD, across three seeds.
MANUSCRIPT = {
    "Age": (3.41, .12, 4.87, .08),
    "Height": (5.88, .02, 7.34, .02),
    "Weight": (9.17, .13, 11.69, .11),
    "Waist": (6.18, .01, 8.02, .01),
    "BMI": (2.56, .00, 3.36, .00),
    "SBP": (10.89, .00, 13.51, .01),
    "DBP": (7.86, .28, 9.70, .35),
    "Glucose": (11.57, .09, 20.17, .29),
    "HbA1c": (.379, .007, .656, .005),
    "Creatinine": (.133, .002, .170, .002),
    "BUN": (2.70, .01, 3.51, .02),
    "TotalCholesterol": (29.06, .05, 37.83, .12),
    "Triglyceride": (53.98, .27, 77.38, .23),
    "UricAcid": (1.04, .01, 1.32, .01),
    "Albumin": (.209, .001, .262, .001),
    "eGFR": (11.27, .10, 14.64, .08),
}
SEEDS = {0, 1, 2}
CONFIGS = ["base", "pe", "rc", "f2b"]
TEST_PARTICIPANTS = 1047


def read_csv(path, **kwargs):
    """Briefly retry if the active training process is updating the CSV."""
    error = None
    for attempt in range(4):
        try:
            before = path.stat()
            frame = pd.read_csv(path, **kwargs)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
                return frame
            error = RuntimeError("file changed while it was being read")
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            error = exc
        if attempt < 3:
            time.sleep(0.2)
    raise RuntimeError("Cannot read {}: {}. Try again after a few seconds.".format(path, error))


def require(frame, columns, source):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("{} is missing columns: {}".format(source, ", ".join(missing)))


def decimals(target):
    return 3 if target in {"HbA1c", "Creatinine", "Albumin"} else 2


def mean_sd(values, target):
    k = decimals(target)
    sd = "N/A" if len(values) < 2 else format(values.std(ddof=1), ".{}f".format(k))
    return "{} +/- {}".format(format(values.mean(), ".{}f".format(k)), sd)


def report_rim(project):
    root = project / "runs/sota_faithful_reg/rim_v4"
    d = read_csv(root / "summary.csv")
    require(d, ["model", "biomarker", "seed", "n_patients", "MAE_patient", "RMSE_patient"], "Rim summary")
    d = d[(d.model == "rim") & d.biomarker.isin(MANUSCRIPT)].copy()
    if d.empty:
        raise ValueError("No completed Rim regression results in summary.csv yet.")
    if not d.seed.isin(SEEDS).all() or d.duplicated(["biomarker", "seed"]).any():
        raise ValueError("Unexpected seeds or duplicate biomarker/seed rows in Rim summary.")
    d["seed"] = d.seed.astype(int)
    if not d.n_patients.eq(TEST_PARTICIPANTS).all():
        raise ValueError("Rim participant count differs from the manuscript's 1,047.")
    if not np.isfinite(d[["MAE_patient", "RMSE_patient"]].to_numpy(dtype=float)).all():
        raise ValueError("Rim summary contains missing or non-finite metrics.")

    print("\nRIM: {} / 48 completed runs in summary.csv".format(len(d)))
    print("Manuscript values use three seeds; current results use the listed completed seeds.")
    print("Rows with fewer than three seeds are preliminary. SD is N/A for one seed.")
    print("Negative change = lower current error. Changes use rounded manuscript means.")
    for metric, j in [("MAE", 0), ("RMSE", 2)]:
        rows = []
        for target, ref in MANUSCRIPT.items():
            g = d[d.biomarker == target].sort_values("seed")
            k = decimals(target)
            previous = "{:.{k}f} +/- {:.{k}f}".format(ref[j], ref[j + 1], k=k)
            if g.empty:
                rows.append([target, "-", previous, "pending", "-"])
                continue
            values = g[metric + "_patient"]
            seeds = ",".join(map(str, g.seed))
            change = "{:+.1f}%".format(100 * (values.mean() / ref[j] - 1))
            rows.append([target, seeds, previous, mean_sd(values, target), change])
        print("\n{} per participant".format(metric))
        print(pd.DataFrame(rows, columns=["Biomarker", "Seeds", "Manuscript", "Current Rim", "Change"]).to_string(index=False))

    details = d[["biomarker", "seed", "MAE_patient", "RMSE_patient"]].copy()
    try:
        meta = read_csv(root / "run_meta.csv")
        require(meta, ["model", "biomarker", "seed"], "Rim metadata")
        meta = meta[meta.model == "rim"].copy()
        extra = [c for c in ["epochs", "best_epoch", "best_val", "val_spread"] if c in meta]
        details = details.merge(meta[["biomarker", "seed"] + extra], on=["biomarker", "seed"], how="left", validate="one_to_one")
    except (RuntimeError, ValueError, pd.errors.MergeError) as exc:
        print("\nMetadata unavailable; test metrics above are still shown: {}".format(exc))
    details["order"] = details.biomarker.map({t: i for i, t in enumerate(MANUSCRIPT)})
    details = details.sort_values(["order", "seed"]).drop(columns="order")
    print("\nINDIVIDUAL RIM SEEDS / VALIDATION DIAGNOSTICS")
    print(details.to_string(index=False, float_format=lambda v: "{:.5f}".format(v)))
    print("best_val: validation MAE in raw units (per eye).")
    print("val_spread: SD(validation predictions) / SD(validation targets); near zero means little variation.")


def f2b_metrics(project):
    frames = []
    keys = ["biomarker", "config", "seed", "subid", "eye"]
    for seed in sorted(SEEDS):
        path = project / "runs" / ("f2b_reg_v13_s{}".format(seed)) / "preds_long.csv"
        d = read_csv(path, dtype={"subid": str})
        require(d, keys + ["y_true", "y_pred"], str(path))
        d = d[d.config.isin(CONFIGS) & d.biomarker.isin(MANUSCRIPT)].copy()
        if d.empty or not d.seed.eq(seed).all():
            raise ValueError("Missing predictions or unexpected seed in {}".format(path))
        frames.append(d[keys + ["y_true", "y_pred"]])
    x = pd.concat(frames, ignore_index=True)
    if x[keys].isna().any().any() or x.duplicated(keys).any():
        raise ValueError("F2B-Net predictions have missing keys or duplicate image rows.")
    if not x.eye.isin([0, 1]).all():
        raise ValueError("Expected pooled-eye labels 0 and 1.")
    if not np.isfinite(x[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("F2B-Net predictions contain non-finite numbers.")
    coverage = x.groupby(["biomarker", "subid"]).size()
    if not coverage.eq(len(SEEDS) * len(CONFIGS) * 2).all():
        raise ValueError("F2B-Net participants are not aligned across all seeds, configurations and eyes.")
    counts = x.groupby("biomarker").subid.nunique()
    if set(counts.index) != set(MANUSCRIPT) or not counts.eq(TEST_PARTICIPANTS).all():
        raise ValueError("Expected 16 biomarkers and 1,047 test participants for each.")
    truth = x.groupby(["biomarker", "subid"]).y_true.agg(["min", "max"])
    if (truth["max"] - truth["min"]).abs().gt(1e-3).any():
        raise ValueError("Ground truth differs between eyes/configurations/seeds.")

    # Average the two eyes first; compute each seed's metrics separately.
    p = x.groupby(["biomarker", "config", "seed", "subid"], as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    p["ae"] = (p.y_pred - p.y_true).abs()
    p["se"] = (p.y_pred - p.y_true) ** 2
    metrics = p.groupby(["biomarker", "config", "seed"], as_index=False).agg(
        MAE=("ae", "mean"), MSE=("se", "mean"))
    metrics["RMSE"] = np.sqrt(metrics.pop("MSE"))
    return metrics


def report_f2b(project):
    metrics = f2b_metrics(project)
    print("\nF2B-NET POOLED EYES: all 16 biomarkers, seeds 0,1,2; 1,047 participants")
    print("Both eyes averaged per participant; metrics then summarized across seeds.")
    for metric in ["MAE", "RMSE"]:
        rows = []
        for target in MANUSCRIPT:
            cells = [target]
            for config in CONFIGS:
                values = metrics[(metrics.biomarker == target) & (metrics.config == config)][metric]
                cells.append(mean_sd(values, target))
            rows.append(cells)
        print("\n{}: three-seed mean +/- sample SD".format(metric))
        print(pd.DataFrame(rows, columns=["Biomarker", "MobileViT", "+PE", "+RC", "F2B-Net"]).to_string(index=False))
    print("\nThese are descriptive results; paired significance tests are not included.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=Path.cwd(), help="Project folder; default: current directory")
    parser.add_argument("--only", choices=["all", "rim", "f2b"], default="all")
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    if not (project / "runs").is_dir():
        parser.error("No runs folder under {}. Run from ~/cfi_project or use --project ~/cfi_project.".format(project))
    print("CHECKED: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("Read-only results check. Project: {}".format(project))
    failed = False
    for name, report in [("rim", report_rim), ("f2b", report_f2b)]:
        if args.only not in ["all", name]:
            continue
        try:
            report(project)
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            failed = True
            print("\n{} REPORT COULD NOT COMPLETE: {}".format(name.upper(), exc))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
