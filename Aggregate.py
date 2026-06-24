"""
Aggregate the per-combo .npz files from the cluster run into one summary table.

Each output/run_*.npz holds N-seed results for one parameter combination.
This script reduces each to a single row of parameters + summary metrics
(averaged over seeds), ready for sensitivity analysis.

Usage:
    python Aggregate.py                      # reads output/run_*.npz -> results_summary.csv
    python Aggregate.py --tail 50 --glob "output/run_*.npz" --out results_summary.csv
"""
import glob
import csv
import argparse
import numpy as np


def summarise(npz_path: str, tail: int) -> dict:
    """Reduce one combo's N-seed npz to a single summary row."""
    d = np.load(npz_path)

    density, defector_frac, neighbourhood_count, activation_rate = d["params"]

    all_H = d["all_H"]                      # shape (n_seeds, max_steps)
    # tail-averaged H per seed (mean over last `tail` steps), then averaged over seeds
    tail_H_per_seed = all_H[:, -tail:].mean(axis=1)

    morans = d["final_morans_I"]           # (n_seeds,)
    nb_var = d["final_nb_variance"]        # (n_seeds,)
    steps  = d["convergence_steps"]        # (n_seeds,)

    return {
        "density":             round(float(density), 4),
        "defector_frac":       round(float(defector_frac), 4),
        "neighbourhood_count": int(neighbourhood_count),
        "activation_rate":     round(float(activation_rate), 4),
        # segregation metrics (mean over seeds) + seed-to-seed spread
        "H_tail_mean":         float(tail_H_per_seed.mean()),
        "H_tail_std":          float(tail_H_per_seed.std()),
        "morans_I_mean":       float(morans.mean()),
        "morans_I_std":        float(morans.std()),
        "nb_variance_mean":    float(nb_var.mean()),
        "nb_variance_std":     float(nb_var.std()),
        "steps_mean":          float(steps.mean()),
        "n_seeds":             int(all_H.shape[0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="results_saltelli/run_*.npz", help="glob for the npz files")
    p.add_argument("--tail", type=int, default=100, help="steps to average H over (tail window)")
    p.add_argument("--out", default="results_summary.csv")
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No files matched {args.glob!r} — run the sweep first.")
        return

    rows = []
    for f in files:
        try:
            rows.append(summarise(f, args.tail))
        except Exception as e:
            print(f"  skipped {f}: {e}")

    # sort by the parameter grid for a tidy, reproducible table
    rows.sort(key=lambda r: (r["density"], r["defector_frac"],
                             r["neighbourhood_count"], r["activation_rate"]))

    with open(args.out, "w", newline="") as out:
        w = csv.DictWriter(out, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"Aggregated {len(rows)} combos from {len(files)} files -> {args.out}")


if __name__ == "__main__":
    main()
