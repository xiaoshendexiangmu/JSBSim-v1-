#!/usr/bin/env python3
"""Generate an abstract, non-tactical JSBSim dataset.

The simulator trajectory is a fixed baseline. A separate latent functional
capacity variable represents a synthetic intervention outcome; it does not
alter the aircraft model or encode a real intervention mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


OUTCOMES = ("no_effect", "degraded", "disabled")
EVAL_LABELS = (*OUTCOMES, "indeterminate")


def load_jsbsim(model: str, jsbsim_root: str, aircraft_path: str, dt_s: float):
    try:
        import jsbsim
    except ImportError as exc:
        raise RuntimeError(
            "当前 Python 环境未找到 jsbsim。请先确认已有安装所在的解释器路径；"
            "本程序不会自动下载或安装依赖。"
        ) from exc

    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    fdm.set_dt(dt_s)
    if jsbsim_root:
        root = Path(jsbsim_root)
        required_dirs = (root / "aircraft", root / "engine", root / "systems")
        missing = [str(path) for path in required_dirs if not path.is_dir()]
        if missing:
            raise RuntimeError("JSBSim 根目录缺少模型目录：" + ", ".join(missing))
        fdm.set_aircraft_path(str(required_dirs[0]))
        fdm.set_engine_path(str(required_dirs[1]))
        fdm.set_systems_path(str(required_dirs[2]))
    elif aircraft_path:
        fdm.set_aircraft_path(aircraft_path)
    ok = fdm.load_model(model)
    if not ok:
        raise RuntimeError(
            f"JSBSim 无法加载模型 {model!r}。请确认模型数据已随安装提供，"
            "并通过 --aircraft-path 指向 aircraft 目录。"
        )
    if not fdm.run_ic():
        raise RuntimeError("JSBSim 初始条件初始化失败。")
    return fdm


def read_property(fdm: Any, key: str) -> float | None:
    try:
        value = float(fdm[key])
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def outcome_for_capacity(capacity: float, thresholds: dict[str, float]) -> str:
    if capacity >= thresholds["no_effect_min"]:
        return "no_effect"
    if capacity >= thresholds["degraded_min"]:
        return "degraded"
    return "disabled"


def capacity_for_episode(index: int, rng: random.Random) -> float:
    """Counterbalance three abstract outcomes, then randomize within each band."""
    band = index % len(OUTCOMES)
    if band == 0:
        return rng.uniform(0.82, 1.0)
    if band == 1:
        return rng.uniform(0.25, 0.75)
    return rng.uniform(0.0, 0.15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--episodes", type=int, help="覆盖配置中的 episode 数量")
    parser.add_argument("--model", help="覆盖配置中的 JSBSim 模型名")
    parser.add_argument("--jsbsim-root", help="JSBSim 源码/数据根目录，内含 aircraft、engine、systems")
    parser.add_argument("--aircraft-path", help="JSBSim aircraft 目录；留空使用包内默认模型目录")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    episodes = args.episodes if args.episodes is not None else int(config["episodes"])
    model = args.model or config["model"]
    jsbsim_root = args.jsbsim_root or config.get("jsbsim_root", "")
    aircraft_path = args.aircraft_path if args.aircraft_path is not None else config.get("aircraft_path", "")
    dt_s = float(config["dt_s"])
    steps = round(float(config["duration_s"]) / dt_s)
    if episodes <= 0 or dt_s <= 0 or steps <= 0:
        parser.error("episodes、dt_s 和 duration_s 必须为正数")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "observations.csv"
    oracle_path = output_dir / "oracle_labels.csv"
    rng = random.Random(int(config["seed"]))
    thresholds = config["class_thresholds"]

    observation_fields = [
        "episode_id", "time_s", "sim_altitude_ft", "sim_speed_fps",
        "sim_roll_rad", "sim_pitch_rad", "sim_heading_rad",
        "health_proxy_observed", "sensor_valid", "coverage_fraction",
        "max_gap_s", "observability_status", "evaluation_label",
    ]
    oracle_fields = [
        "episode_id", "ground_truth_outcome", "functional_capacity_truth",
        "synthetic_effect_family", "truth_definition",
    ]
    observations: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    episode_summaries: list[tuple[str, float, float, str, str]] = []

    for episode_index in range(episodes):
        episode_id = f"sim-{episode_index:05d}"
        capacity = capacity_for_episode(episode_index, rng)
        truth_outcome = outcome_for_capacity(capacity, thresholds)
        fdm = load_jsbsim(model, jsbsim_root, aircraft_path, dt_s)
        valid_flags: list[bool] = []
        episode_rows: list[dict[str, Any]] = []

        for step in range(steps + 1):
            if step > 0 and not fdm.run():
                raise RuntimeError(f"JSBSim 在 {episode_id} 的 step={step} 提前停止。")
            valid = rng.random() >= float(config["sensor_dropout_probability"])
            valid_flags.append(valid)
            observed_proxy = (
                max(0.0, min(1.0, capacity + rng.gauss(0.0, float(config["sensor_noise_sigma"]))))
                if valid else None
            )
            episode_rows.append({
                "episode_id": episode_id,
                "time_s": round(step * dt_s, 6),
                "sim_altitude_ft": read_property(fdm, "position/h-sl-ft"),
                "sim_speed_fps": read_property(fdm, "velocities/vt-fps"),
                "sim_roll_rad": read_property(fdm, "attitude/phi-rad"),
                "sim_pitch_rad": read_property(fdm, "attitude/theta-rad"),
                "sim_heading_rad": read_property(fdm, "attitude/psi-rad"),
                "health_proxy_observed": observed_proxy,
                "sensor_valid": int(valid),
            })

        coverage = sum(valid_flags) / len(valid_flags)
        max_gap_steps = 0
        gap = 0
        for valid in valid_flags:
            gap = 0 if valid else gap + 1
            max_gap_steps = max(max_gap_steps, gap)
        max_gap_s = max_gap_steps * dt_s
        observable = coverage >= float(config["minimum_coverage"]) and max_gap_s <= float(config["maximum_gap_s"])
        observability = "observable" if observable else "indeterminate"
        evaluation_label = truth_outcome if observable else "indeterminate"

        for row in episode_rows:
            row.update({
                "coverage_fraction": round(coverage, 6),
                "max_gap_s": round(max_gap_s, 6),
                "observability_status": observability,
                "evaluation_label": evaluation_label,
            })
            observations.append(row)
        oracle.append({
            "episode_id": episode_id,
            "ground_truth_outcome": truth_outcome,
            "functional_capacity_truth": round(capacity, 6),
            "synthetic_effect_family": "abstract_functional_state_only",
            "truth_definition": "capacity thresholds in config.json; not a physical damage model",
        })
        episode_summaries.append((truth_outcome, coverage, max_gap_s, observability, evaluation_label))

    with observations_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=observation_fields)
        writer.writeheader()
        writer.writerows(observations)
    with oracle_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=oracle_fields)
        writer.writeheader()
        writer.writerows(oracle)

    manifest = {
        "schema_version": "1.0",
        "simulator": "JSBSim",
        "model": model,
        "seed": config["seed"],
        "episodes": episodes,
        "samples_per_episode": steps + 1,
        "observation_file": observations_path.name,
        "oracle_file": oracle_path.name,
        "files_have_distinct_roles": True,
        "note": "Synthetic methodological data only; not a physical effect or tactical model.",
        "episode_counts": {
            label: sum(1 for s in episode_summaries if s[4] == label)
            for label in EVAL_LABELS
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2)
