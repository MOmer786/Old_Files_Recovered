#!/usr/bin/env python3
"""
final_stats.py   [FINAL_STATS_V1]
=================================
The last numbers needed for the manuscript text. CPU only, 2-4 minutes, safe next to training.

  A  Table 4 (F2B-Net left eye / right eye / pooled, participant level, 3 seeds) in the style of
     the final Table 2                                      -> runs/final_stats/table4_final.tex
     plus, per configuration and biomarker, the label-permutation (chance-level) test: the
     predictions are re-paired with the true values of other participants 10,000 times and the
     p-value is the share of re-pairings with an MAE at least as low as the observed one
     (largest p over the three seeds is reported)           -> runs/final_stats/table4_chance.csv
  B  Paired comparisons for Table 4 from runs/table4_v13/table4_paired.csv (written by
     check_eyes.py) and where the pooled-eye gain comes from (training vs averaging).
  C  Sex classification: the same chance-level test on AUROC for the pooled, left-eye and
     right-eye F2B-Net classifiers                          -> runs/final_stats/sex_chance.csv
  D  Regression ablation, incremental: +PE vs MobileViT, F2B-Net vs +PE, F2B-Net vs MobileViT
     (paired sign-flip test on participant-level errors averaged over the seeds, bootstrap 95% CI,
     Benjamini-Hochberg within each comparison)             -> runs/final_stats/reg_ablation_incremental.csv

  python scripts/final_stats.py
"""
import os
import sys

import numpy as np
import pandas as pd

T = ["Age", "Height", "Weight", "Waist", "BMI", "SBP", "DBP", "Glucose", "HbA1c", "Creatinine", "BUN",
     "TotalCholesterol", "Triglyceride", "UricAcid", "Albumin", "eGFR"]
LABEL = {"Age": "Age (year)", "Height": "Height (cm)", "Weight": "Weight (kg)", "Waist": "Waist (cm)",
         "BMI": "BMI (kg/m$^2$)", "SBP": "SBP (mmHg)", "DBP": "DBP (mmHg)", "Glucose": "Glucose (mg/dL)",
         "HbA1c": "HbA1c (\\%)", "Creatinine": "Creatinine (mg/dL)", "BUN": "BUN (mg/dL)",
         "TotalCholesterol": "Total Cholesterol (mg/dL)", "Triglyceride": "Triglyceride (mg/dL)",
         "UricAcid": "Uric Acid (mg/dL)", "Albumin": "Albumin (g/dL)", "eGFR": "eGFR (mL/min/1.73m$^2$)"}
DEC3 = {"HbA1c", "Creatinine", "Albumin"}
SEEDS = (0, 1, 2)
EYES = ("left", "right", "both")
NAME = {"left": "Left-Eye", "right": "Right-Eye", "both": "Pooled-Eye"}
STORE = "runs/reg_v13_store"
PAIRED4 = "runs/table4_v13/table4_paired.csv"
ABL = ["runs/f2b_reg_v13_s0", "runs/f2b_reg_v13_s1", "runs/f2b_reg_v13_s2"]
CLS = {"Pooled-Eye": "runs/f2b_cls_v8", "Left-Eye": "runs/f2b_cls_v8_left", "Right-Eye": "runs/f2b_cls_v8_right"}
OUT = "runs/final_stats"
N_PERM, N_BOOT, CHUNK = 10000, 10000, 1000
rng = np.random.default_rng(2026)


def dec(t):
    return 3 if t in DEC3 else 2


# ---------------------------------------------------------------------------- statistics
def chance_p_mae(y, p):
    """Label-permutation test for one run: share of re-pairings with MAE <= observed."""
    obs = float(np.abs(y - p).mean())
    n, cnt = len(y), 0
    for s in range(0, N_PERM, CHUNK):
        k = min(CHUNK, N_PERM - s)
        idx = rng.permuted(np.tile(np.arange(n), (k, 1)), axis=1)
        cnt += int((np.abs(y[None, :] - p[idx]).mean(axis=1) <= obs + 1e-12).sum())
    return (cnt + 1) / (N_PERM + 1)


def chance_p_auc(y, p):
    """Label-permutation test on AUROC (rank based): share of permutations with AUROC >= observed."""
    r = pd.Series(p).rank(method="average").to_numpy()
    y = y.astype(float)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = lambda lab: (lab @ r - n1 * (n1 + 1) / 2) / (n1 * n0)
    obs = float(auc(y))
    cnt = 0
    for s in range(0, N_PERM, CHUNK):
        k = min(CHUNK, N_PERM - s)
        lab = rng.permuted(np.tile(y, (k, 1)), axis=1)
        cnt += int(((lab @ r - n1 * (n1 + 1) / 2) / (n1 * n0) >= obs - 1e-12).sum())
    return obs, (cnt + 1) / (N_PERM + 1)


def paired(ec, ep):
    """ec, ep: one error per participant (control, proposed). delta = ec - ep (> 0: proposed lower)."""
    d = ec - ep
    n, mean_d = len(d), float(d.mean())
    boot = np.empty(N_BOOT)
    for s in range(0, N_BOOT, CHUNK):
        k = min(CHUNK, N_BOOT - s)
        idx = rng.integers(0, n, size=(k, n))
        boot[s:s + k] = d[idx].mean(axis=1)
    ext = 0
    for s in range(0, N_PERM, CHUNK):
        k = min(CHUNK, N_PERM - s)
        ext += int((np.abs((rng.choice(np.array([-1.0, 1.0]), size=(k, n)) * d).mean(axis=1)) >= abs(mean_d) - 1e-12).sum())
    return dict(delta_MAE=mean_d, ci_low=float(np.quantile(boot, 0.025)), ci_high=float(np.quantile(boot, 0.975)),
                rel_reduction_pct=100 * mean_d / float(ec.mean()), p_signflip=(ext + 1) / (N_PERM + 1))


def bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(ranked, 1.0); return out


def colours(vals):
    distinct = sorted(set(vals))
    return ["red" if v == distinct[0] else ("blue" if len(distinct) > 1 and v == distinct[1] else None) for v in vals]


def fmt_p(p):
    return "$<$0.0001" if p < 1e-4 else f"{p:.4f}"


# ---------------------------------------------------------------------------- A + B: Table 4
def load_eye_runs():
    pipe = pd.read_csv("runs/reg_select/final_pipeline.csv").set_index("biomarker")
    pp = {t: "minimal" if str(pipe.loc[t, "preprocessing"]).strip().lower() == "none" else "paper" for t in T}
    res, missing = {}, []
    for e in EYES:
        for t in T:
            for s in SEEDS:
                f = os.path.join(STORE, f"{pp[t]}_s{s}_{e}", t, "preds_long.csv")
                if not os.path.isfile(f):
                    missing.append(f); continue
                x = pd.read_csv(f, dtype={"subid": str})
                x = x[(x.biomarker == t) & (x.config == "f2b") & (x.seed.astype(int) == s)]
                if x.empty:
                    missing.append(f + " (no f2b rows)"); continue
                x = x.assign(subid=x.subid.astype(str).str.strip())
                res[(e, t, s)] = x
    if missing:
        sys.exit("ERROR: missing single-eye / pooled runs:\n  " + "\n  ".join(missing[:10]))
    return res


def table4(res):
    rows, chance, err, img_mae = [], [], {}, {}
    for t in T:
        ref = None
        for e in EYES:
            per_seed = []
            for s in SEEDS:
                x = res[(e, t, s)]
                g = x.groupby("subid").agg(y=("y_true", "first"), p=("y_pred", "mean")).sort_index()
                if ref is None:
                    ref = g.y
                if set(g.index) != set(ref.index) or float((g.y - ref.reindex(g.index)).abs().max()) > 1e-3:
                    sys.exit(f"ERROR: {t} {NAME[e]} seed {s}: different participants or true values")
                y, p = g.y.to_numpy(float), g.p.to_numpy(float)
                per_seed.append(pd.Series(np.abs(p - y), index=g.index))
                rows.append(dict(biomarker=t, eye=e, seed=s, n=len(g), MAE=float(np.abs(p - y).mean()),
                                 RMSE=float(np.sqrt(((p - y) ** 2).mean())),
                                 MAE_image=float((x.y_pred - x.y_true).abs().mean())))
                chance.append(dict(biomarker=t, eye=NAME[e], seed=s, MAE=rows[-1]["MAE"], p_chance=chance_p_mae(y, p)))
            err[(t, e)] = pd.concat(per_seed, axis=1).mean(axis=1)
        n_part = len(ref)
    ps = pd.DataFrame(rows)
    sm = ps.groupby(["biomarker", "eye"]).agg(MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                                              RMSE_mean=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
                                              MAE_image_mean=("MAE_image", "mean")).reset_index()
    sm.to_csv(os.path.join(OUT, "table4_metrics.csv"), index=False)
    ch = pd.DataFrame(chance)
    ch.to_csv(os.path.join(OUT, "table4_chance.csv"), index=False)
    worst = ch.groupby(["biomarker", "eye"]).p_chance.max().reset_index()
    v = sm.set_index(["biomarker", "eye"])

    def head(e):
        return (r"\textbf{{\renewcommand{\arraystretch}{1.15}\begin{tabular}{@{}c@{}}\rule{0pt}{3.0ex}F2B-Net\\(" + NAME[e]
                + r")\rule[-1.4ex]{0pt}{0pt}\end{tabular}}}")

    pmax = float(worst.p_chance.max())
    chance_txt = (r"Every configuration predicted every biomarker better than chance (label-permutation test, 10{,}000 "
                  r"re-pairings of predictions and true values: $p<0.0001$ for all configurations, biomarkers and seeds). "
                  if pmax < 1e-4 else
                  r"All configurations were compared with the label-permutation null (10{,}000 re-pairings); the largest "
                  f"$p$-value over configurations, biomarkers and seeds was {pmax:.4f}. ")
    L = [r"\begin{table}[!t]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.5}",
         r"\begin{tabular}{|l|c|c|c|c|}", r"\hline",
         r"\textbf{Parameters (Units)} & \textbf{Metrics} & " + " & ".join(head(e) for e in EYES) + r" \\", r"\hline"]
    for i, t in enumerate(T, 1):
        k = dec(t)
        for j, met in enumerate(("MAE", "RMSE")):
            means = [round(float(v.loc[(t, e), f"{met}_mean"]), k) for e in EYES]
            cells = []
            for e, c in zip(EYES, colours(means)):
                txt = f"{v.loc[(t, e), f'{met}_mean']:.{k}f} $\\pm$ {v.loc[(t, e), f'{met}_sd']:.{k}f}"
                cells.append(f"\\textcolor{{{c}}}{{{txt}}}" if c else txt)
            L.append((f"{i}. {LABEL[t]}" if j == 0 else "") + f" & {met} & " + " & ".join(cells) + r" \\")
        L.append(r"\hline")
    L += [r"\end{tabular}",
          r"\caption{Regression performance of F2B-Net trained and evaluated on the left eye only, the right eye only, "
          r"or both eyes pooled, for the 16 biomarkers at the participant level. The single-eye configurations use one "
          r"image per participant; for the pooled-eye configuration the predictions of each participant's two eyes were "
          f"averaged. All configurations were evaluated on the same {n_part:,} test participants. Values are mean "
          r"$\pm$ SD over three independent training seeds. " + chance_txt +
          r"The lowest and second-lowest distinct mean errors are highlighted in red and blue, respectively, separately "
          r"for each metric; highlighting is based on the displayed precision, with ties assigned the same colour. "
          r"Sex classification results are reported in Table~\ref{tab:sex_classification}.}",
          r"\label{tab:comparison_left_right_both_eyes}", r"\end{table}"]
    open(os.path.join(OUT, "table4_final.tex"), "w").write("\n".join(L) + "\n")

    print(f"\nA. TABLE 4 (test, {n_part} participants, mean +/- SD over 3 seeds) and chance-level p (largest over seeds)")
    print(f"  {'biomarker':17s}" + "".join(f"{NAME[e]:>24s}" for e in EYES) + "   p_chance (L / R / pooled)")
    for t in T:
        k = dec(t)
        cells = "".join(f"{v.loc[(t, e), 'MAE_mean']:>13.{k}f} +/- {v.loc[(t, e), 'MAE_sd']:.{k}f}" for e in EYES)
        pc = " / ".join(f"{worst[(worst.biomarker == t) & (worst.eye == NAME[e])].p_chance.iloc[0]:.4f}" for e in EYES)
        print(f"  {t:17s}{cells}   {pc}")
    print(f"  largest chance-level p over all configurations, biomarkers and seeds: {pmax:.4f}")

    print("\nB. PAIRED COMPARISONS (from check_eyes.py) and pooled-eye gain over the better single eye")
    if os.path.exists(PAIRED4):
        pr = pd.read_csv(PAIRED4)
        for c, g in pr.groupby("comparison", sort=False):
            better = int(((g.q_bh < 0.05) & (g.delta_MAE > 0)).sum()); worse = int(((g.q_bh < 0.05) & (g.delta_MAE < 0)).sum())
            print(f"  {c:26s}: lower MAE on {int((g.delta_MAE > 0).sum())}/16, significantly lower on {better}, "
                  f"significantly higher on {worse} (BH q<0.05); median relative reduction {g.rel_reduction_pct.median():+.2f}%")
        q = {(r.comparison, r.biomarker): r for r in pr.itertuples()}
        print(f"  {'biomarker':17s} {'best single':>12s} {'pooled':>8s} {'reduction':>10s}   q (pooled vs L / vs R)   q (R vs L)")
        for t in T:
            k = dec(t)
            l, r_, b = (float(v.loc[(t, e), "MAE_mean"]) for e in EYES)
            best = min(l, r_)
            ql = q[("Pooled-Eye vs Left-Eye", t)].q_bh; qr = q[("Pooled-Eye vs Right-Eye", t)].q_bh
            qrl = q[("Right-Eye vs Left-Eye", t)].q_bh
            print(f"  {t:17s} {best:12.{k}f} {b:8.{k}f} {100 * (best - b) / best:+9.2f}%   {ql:.4f} / {qr:.4f}          {qrl:.4f}")
    else:
        print(f"  ({PAIRED4} not found; run python scripts/check_eyes.py first)")
    single = (v.xs("left", level="eye").MAE_mean + v.xs("right", level="eye").MAE_mean) / 2
    img = v.xs("both", level="eye").MAE_image_mean
    part = v.xs("both", level="eye").MAE_mean
    tr_gain, av_gain = 100 * (single - img) / single, 100 * (img - part) / single
    print(f"  pooled-eye gain, % of the mean single-eye MAE (median over biomarkers): training on both eyes "
          f"{np.median(tr_gain):+.2f}%, averaging the two eyes at prediction {np.median(av_gain):+.2f}%, "
          f"total {np.median(tr_gain + av_gain):+.2f}%")
    print(f"  per biomarker (training / averaging): " + ", ".join(f"{t} {tr_gain[t]:+.1f}/{av_gain[t]:+.1f}" for t in T))


# ---------------------------------------------------------------------------- C: sex chance test
def sex_chance():
    print("\nC. SEX CLASSIFICATION, chance-level test on participant-level AUROC (largest p over seeds)")
    rows = []
    for name, d in CLS.items():
        f = os.path.join(d, "preds_long.csv")
        if not os.path.isfile(f):
            print(f"  {name:10s} {f} not found; skipped"); continue
        x = pd.read_csv(f, dtype={"subid": str})
        x = x[x.config == "cls_cov"] if "config" in x.columns else x
        for s in sorted(x.seed.astype(int).unique()):
            g = x[x.seed.astype(int) == s].groupby("subid").agg(y=("y_true", "first"), p=("y_pred", "mean"))
            auc, p = chance_p_auc(g.y.to_numpy(), g.p.to_numpy(float))
            rows.append(dict(config=name, seed=s, n=len(g), AUROC=auc, p_chance=p))
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, "sex_chance.csv"), index=False)
        for name, g in df.groupby("config", sort=False):
            print(f"  {name:10s} n={int(g.n.iloc[0])}  AUROC per seed {', '.join(f'{a:.3f}' for a in g.AUROC)}  "
                  f"largest p {g.p_chance.max():.4f}")


# ---------------------------------------------------------------------------- D: ablation
def ablation():
    print("\nD. REGRESSION ABLATION, incremental (participant level, errors averaged over seeds; BH within comparison)")
    frames = []
    for d in ABL:
        f = os.path.join(d, "preds_long.csv")
        if not os.path.isfile(f):
            print(f"  {f} not found; skipped"); return
        x = pd.read_csv(f, dtype={"subid": str})
        frames.append(x[x.config.isin(["base", "pe", "f2b"])])
    x = pd.concat(frames)
    x["subid"] = x.subid.astype(str).str.strip()
    pat = x.groupby(["config", "biomarker", "seed", "subid"], as_index=False).agg(y=("y_true", "first"), p=("y_pred", "mean"))
    pat["ae"] = (pat.p - pat.y).abs()
    err = pat.groupby(["config", "biomarker", "subid"]).ae.mean().unstack("config")
    mae = pat.groupby(["config", "biomarker", "seed"]).ae.mean().groupby(["config", "biomarker"]).mean()
    rows = []
    for ctrl, prop, label in (("base", "pe", "+PE vs MobileViT"), ("pe", "f2b", "F2B-Net vs +PE"), ("base", "f2b", "F2B-Net vs MobileViT")):
        for t in T:
            e = err.xs(t, level="biomarker")
            st = paired(e[ctrl].to_numpy(float), e[prop].to_numpy(float))
            rows.append(dict(comparison=label, biomarker=t, MAE_control=float(mae[(ctrl, t)]), MAE_proposed=float(mae[(prop, t)]), **st))
    pr = pd.DataFrame(rows)
    pr["q_bh"] = np.nan
    for c, idx in pr.groupby("comparison").groups.items():
        pr.loc[idx, "q_bh"] = bh(pr.loc[idx, "p_signflip"])
    pr.to_csv(os.path.join(OUT, "reg_ablation_incremental.csv"), index=False)
    for c, g in pr.groupby("comparison", sort=False):
        better = int(((g.q_bh < 0.05) & (g.delta_MAE > 0)).sum()); worse = int(((g.q_bh < 0.05) & (g.delta_MAE < 0)).sum())
        print(f"  {c:22s}: lower MAE on {int((g.delta_MAE > 0).sum())}/16, significantly lower on {better}, "
              f"significantly higher on {worse}; median relative reduction {g.rel_reduction_pct.median():+.2f}%")
        top = g.sort_values("rel_reduction_pct", ascending=False).head(4)
        print("      largest: " + ", ".join(f"{r.biomarker} {r.rel_reduction_pct:+.2f}% (q={r.q_bh:.3f})" for r in top.itertuples()))
        print("      q<0.05 : " + (", ".join(g[g.q_bh < 0.05].biomarker) or "none"))


if __name__ == "__main__":
    if not os.path.isdir(STORE):
        sys.exit(f"run this from ~/cfi_project ({STORE} not found)")
    os.makedirs(OUT, exist_ok=True)
    res = load_eye_runs()
    table4(res)
    sex_chance()
    ablation()
    print(f"\nwritten to {OUT}: table4_final.tex, table4_metrics.csv, table4_chance.csv, sex_chance.csv, reg_ablation_incremental.csv")
