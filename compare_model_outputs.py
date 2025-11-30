import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style="whitegrid")


def load_metrics(path):
    df = pd.read_csv(path)
    # make sure numeric types are correct
    numeric_cols = [
        "inference_time_ms", "num_predictions", "num_labels",
        "matched_detections", "precision", "recall",
        "mean_iou", "mean_confidence", "max_confidence", "min_confidence",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize the image/id column to string so merges won't fail when dtypes differ
    if "image" in df.columns:
        # coerce to str, strip whitespace, and use the filename stem (remove path and extension)
        df["image"] = df["image"].astype(str).str.strip().apply(lambda p: Path(p).stem)

    return df


def summary_table(df, name):
    metrics = ["inference_time_ms", "precision", "recall", "mean_iou", "mean_confidence"]
    s = {}
    for m in metrics:
        if m in df.columns:
            s[m] = {
                "count": int(df[m].count()),
                "mean": float(df[m].mean()),
                "median": float(df[m].median()),
                "std": float(df[m].std()),
                "min": float(df[m].min()),
                "max": float(df[m].max()),
            }
    print(f"\nSummary for {name}:")
    print(pd.DataFrame(s).T.round(4))
    return s


def plot_distributions(df_a, df_b, label_a, label_b, outdir):
    os.makedirs(outdir, exist_ok=True)
    metrics = ["inference_time_ms", "precision", "recall", "mean_iou", "mean_confidence"]
    for m in metrics:
        if m not in df_a.columns and m not in df_b.columns:
            continue
        plt.figure(figsize=(8, 4))
        if m in df_a.columns:
            sns.kdeplot(df_a[m].dropna(), label=label_a, fill=True)
        if m in df_b.columns:
            sns.kdeplot(df_b[m].dropna(), label=label_b, fill=True)
        plt.title(f"Distribution: {m}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"dist_{m}.png"), dpi=150)
        plt.close()

    # Boxplots
    combined = []
    for df, label in [(df_a, label_a), (df_b, label_b)]:
        tmp = df[["image", "precision", "recall", "mean_iou", "mean_confidence"]].copy()
        tmp["model"] = label
        combined.append(tmp)
    comb = pd.concat(combined, ignore_index=True)
    for m in ["precision", "recall", "mean_iou", "mean_confidence"]:
        if m in comb.columns:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x="model", y=m, data=comb)
            plt.title(f"Boxplot {m}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"box_{m}.png"), dpi=150)
            plt.close()


def plot_pairwise(dfA, dfB, labelA, labelB, outdir):
    # merge on image for paired comparisons
    merged = pd.merge(dfA, dfB, on="image", suffixes=("_A", "_B"))
    os.makedirs(outdir, exist_ok=True)
    print(f"\nPaired images: {len(merged)} (matched on image column)")

    # if there are no paired images, avoid running numeric reductions on empty arrays
    if merged.empty:
        print("No paired images found — skipping pairwise plots and stats.")
        # show small diagnostic samples to help understand why there are no matches
        sample_a = dfA["image"].dropna().astype(str).head(10).tolist() if "image" in dfA.columns else []
        sample_b = dfB["image"].dropna().astype(str).head(10).tolist() if "image" in dfB.columns else []
        print("Sample image IDs from A:", sample_a)
        print("Sample image IDs from B:", sample_b)
        # still write an empty merged CSV and an empty stats summary for consistency
        merged.to_csv(os.path.join(outdir, "merged_per_image.csv"), index=False)
        pd.DataFrame().to_csv(os.path.join(outdir, "paired_stats_summary.csv"))
        return

    compare_metrics = [
        ("inference_time_ms", True),
        ("precision", False),
        ("recall", False),
        ("mean_iou", False),
        ("mean_confidence", False),
    ]

    stats = {}
    for m, logscale in compare_metrics:
        a_col = f"{m}_A"
        b_col = f"{m}_B"
        if a_col not in merged.columns or b_col not in merged.columns:
            continue
        x = merged[a_col].to_numpy(dtype=float)
        y = merged[b_col].to_numpy(dtype=float)
        diff = y - x  # B - A
        stats[m] = {
            "N": int(len(x)),
            "mean_A": float(np.nanmean(x)),
            "mean_B": float(np.nanmean(y)),
            "mean_diff_B_minus_A": float(np.nanmean(diff)),
            "std_diff": float(np.nanstd(diff)),
            "pearson_r": float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan,
        }

        # scatter
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, alpha=0.7, s=20)
        lims = [
            np.nanmin([np.nanmin(x), np.nanmin(y)]),
            np.nanmax([np.nanmax(x), np.nanmax(y)])
        ]
        # expand a little
        pad = (lims[1] - lims[0]) * 0.05 if lims[1] > lims[0] else 1
        plt.plot([lims[0]-pad, lims[1]+pad], [lims[0]-pad, lims[1]+pad], c="k", lw=0.8, linestyle="--")
        plt.xlabel(f"{labelA} {m}")
        plt.ylabel(f"{labelB} {m}")
        plt.title(f"{m}: {labelA} vs {labelB}\npearson r={stats[m]['pearson_r']:.3f}")
        if logscale and (np.nanmin(x) > 0 and np.nanmin(y) > 0):
            plt.xscale("log")
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"scatter_{m}.png"), dpi=150)
        plt.close()

        # histogram of differences
        plt.figure(figsize=(6, 3))
        sns.histplot(diff[~np.isnan(diff)], kde=True)
        plt.title(f"Difference (B - A) for {m}\nmean={stats[m]['mean_diff_B_minus_A']:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_diff_{m}.png"), dpi=150)
        plt.close()

    # save merged csv and stats
    merged.to_csv(os.path.join(outdir, "merged_per_image.csv"), index=False)
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(outdir, "paired_stats_summary.csv"))
    print("\nPaired stats summary:")
    print(stats_df.round(4))


def main():
    parser = argparse.ArgumentParser(description="Compare two inference-metrics CSV outputs")
    parser.add_argument("--a", required=True, help="CSV for model A (e.g. rfdetr)")
    parser.add_argument("--b", required=True, help="CSV for model B (e.g. yolo)")
    parser.add_argument("--label-a", default="Model_A", help="Label for model A")
    parser.add_argument("--label-b", default="Model_B", help="Label for model B")
    parser.add_argument("--outdir", default="compare_plots", help="Output directory for plots and summaries")
    args = parser.parse_args()

    dfA = load_metrics(args.a)
    dfB = load_metrics(args.b)

    print(f"Loaded {len(dfA)} rows from {args.a}")
    print(f"Loaded {len(dfB)} rows from {args.b}")

    summary_table(dfA, args.label_a)
    summary_table(dfB, args.label_b)

    # distributions and boxplots
    plot_distributions(dfA, dfB, args.label_a, args.label_b, args.outdir)

    # paired comparisons (only images present in both files)
    plot_pairwise(dfA, dfB, args.label_a, args.label_b, args.outdir)

    print(f"\nPlots and CSVs saved in: {args.outdir}")


if __name__ == "__main__":
    main()