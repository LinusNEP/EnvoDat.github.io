#!/usr/bin/env python
import argparse
import sys

try:
    import copy
    import numpy as np
    from evo.core import metrics, sync
    from evo.tools import file_interface
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install evo numpy")


def load_pair(est_path, gt_path, max_diff=0.02):
    traj_est = file_interface.read_tum_trajectory_file(est_path)
    traj_gt = file_interface.read_tum_trajectory_file(gt_path)
    gt, est = sync.associate_trajectories(traj_gt, traj_est, max_diff=max_diff)
    return gt, est


def _aligned(gt, est, align_mode):
    est_a = copy.deepcopy(est)
    if align_mode in ("se3", "sim3"):
        est_a.align(gt, correct_scale=(align_mode == "sim3"))
    return est_a


def ate_mean_std(gt, est, align_mode):
    est_a = _aligned(gt, est, align_mode)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt, est_a))
    m = ape.get_statistic(metrics.StatisticsType.mean)
    s = ape.get_statistic(metrics.StatisticsType.std)
    return round(m, 2), round(s, 2)


def rpe_mean_std(gt, est, align_mode, delta=1):
    est_a = _aligned(gt, est, align_mode)
    rpe = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=metrics.Unit.frames,  
        all_pairs=False,
    )
    rpe.process_data((gt, est_a))
    m = rpe.get_statistic(metrics.StatisticsType.mean)
    s = rpe.get_statistic(metrics.StatisticsType.std)
    return round(m, 2), round(s, 2)


def sd_mean_std(gt, est, delta=1, form="ratio", eps=1e-6):
    e = est.positions_xyz
    g = gt.positions_xyz
    n = min(len(e), len(g))
    est_seg = np.linalg.norm(e[delta:n] - e[: n - delta], axis=1)
    gt_seg = np.linalg.norm(g[delta:n] - g[: n - delta], axis=1)
    valid = gt_seg > eps
    if not np.any(valid):
        return None, None
    ratio = est_seg[valid] / gt_seg[valid]
    sd = ratio if form == "ratio" else np.abs(ratio - 1.0)
    return round(float(np.mean(sd)), 2), round(float(np.std(sd)), 2)


def maybe_update_yaml(path, algorithm, scene, values):
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    for entry in data.get("entries", []):
        if entry["algorithm"].lower() == algorithm.lower():
            entry.setdefault("results", {}).setdefault(scene, {})
            entry["results"][scene].update(values)
            with open(path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
            print(f"Updated {path}: {algorithm} / {scene} -> {values}")
            return
    print(f"WARNING: no entry named '{algorithm}' in {path}; not updated.")


def main():
    ap = argparse.ArgumentParser(description="ATE/RPE/scale-drift (EnvoDat Eqs. 1-3) for a SLAM trajectory.")
    ap.add_argument("--est", required=True, help="Estimated trajectory (TUM format)")
    ap.add_argument("--gt", required=True, help="Ground-truth trajectory (TUM format)")
    ap.add_argument("--align", choices=["none", "se3", "sim3"], default="se3",
                    help="Alignment for ATE/RPE (default se3; SD never scale-aligns)")
    ap.add_argument("--delta", type=int, default=1, help="Segment offset Δ for RPE/SD (default 1)")
    ap.add_argument("--sd-form", choices=["ratio", "abs"], default="ratio",
                    help="SD convention: 'ratio' (optimum 1, paper table) or 'abs' (optimum 0)")
    ap.add_argument("--max-diff", type=float, default=0.02, help="Max timestamp diff for association [s]")

    ap.add_argument("--algorithm", help="Algorithm name (for --update)")
    ap.add_argument("--scene", help="Scene id, e.g. mu-hall-01 (for --update)")
    ap.add_argument("--update", help="Path to slam_results.yaml to update in place")
    args = ap.parse_args()

    gt, est = load_pair(args.est, args.gt, max_diff=args.max_diff)

    ate = ate_mean_std(gt, est, args.align)
    rpe = rpe_mean_std(gt, est, args.align, args.delta)
    sd = sd_mean_std(gt, est, args.delta, args.sd_form)

    out = {
        "ate_rmse": list(ate),
        "rpe_rmse": list(rpe),
        "sd": list(sd) if sd[0] is not None else None,
    }
    print(f"ATE (mean/std): {ate[0]} / {ate[1]} m")
    print(f"RPE (mean/std): {rpe[0]} / {rpe[1]} m")
    print(f"SD  (mean/std): {sd[0]} / {sd[1]}   (optimum {'1.0' if args.sd_form=='ratio' else '0.0'})")

    if args.update:
        if not (args.algorithm and args.scene):
            sys.exit("--update requires --algorithm and --scene")
        maybe_update_yaml(args.update, args.algorithm, args.scene, out)


if __name__ == "__main__":
    main()
