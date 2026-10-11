"""Compare the existing Panda pose IK with and without bias compensation."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from smolvla_task.controllers.panda_ik_controller import PandaIKController
from smolvla_task.controllers.panda_ik_controller_no_gravity import (
    PandaIKControllerNoGravity,
)
from smolvla_task.envs.cube_tray_env import CubeTrayEnv
from smolvla_task.utils.episode_monitor import EpisodeEventMonitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS = (
    ("forward", (0.05, 0.0, 0.0)),
    ("backward", (-0.05, 0.0, 0.0)),
    ("left", (0.0, 0.05, 0.0)),
    ("right", (0.0, -0.05, 0.0)),
    ("up", (0.0, 0.0, 0.05)),
    ("down", (0.0, 0.0, -0.04)),
    ("diagonal", (0.03, 0.03, 0.03)),
    ("return", (0.0, 0.0, 0.0)),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    parser.add_argument("--settled-window-seconds", type=float, default=0.4)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.03)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs" / "evaluation" / "gravity_compensation",
    )
    args = parser.parse_args()
    values = (args.move_seconds, args.hold_seconds, args.settled_window_seconds,
              args.position_tolerance, args.orientation_tolerance)
    if not all(np.isfinite(value) and value > 0 for value in values):
        parser.error("Durations and tolerances must be finite and positive.")
    if args.settled_window_seconds > args.hold_seconds:
        parser.error("The settled window must not exceed the hold duration.")
    return args


def control_steps(seconds, period):
    steps = round(seconds / period)
    if steps < 1 or not np.isclose(steps * period, seconds, atol=1e-9, rtol=0):
        raise ValueError(f"Duration {seconds} must be a multiple of {period} s.")
    return steps


def rmse(values):
    return float(np.sqrt(np.mean(np.square(values))))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def command_pose(controller, target, rotation):
    """Check that the command differs from nominal IK only by compensation."""
    position, current_rotation = controller.get_ee_pose()
    position_error = target - position
    rotation_error = controller.compute_orientation_error(current_rotation, rotation)
    updated = (
        np.linalg.norm(position_error) >= controller.position_tolerance
        or np.linalg.norm(rotation_error) >= controller.orientation_tolerance
    )
    offset = np.zeros(7)
    bias = controller.data.qfrc_bias[controller.arm_dof_indices].copy()
    if updated:
        dq = controller.compute_joint_delta_pose(position_error, rotation_error)
        nominal = controller.data.qpos[controller.arm_qpos_indices].copy() + dq
        if controller.gravity_compensation:
            offset = controller.get_gravity_compensation_offset()
            expected_offset = np.clip(
                bias / controller.arm_kp * controller.gravity_compensation_scale,
                -controller.max_gravity_offset, controller.max_gravity_offset,
            )
            np.testing.assert_allclose(offset, expected_offset, atol=1e-12, rtol=0)
        expected = controller.clip_joint_limits(nominal + offset)
    controller.step_towards_pose(target, rotation)
    command = controller.data.ctrl[controller.arm_actuator_ids].copy()
    if updated:
        np.testing.assert_allclose(command, expected, atol=1e-12, rtol=0)
    return updated, offset, bias, command


def run_controller(args, label, controller_type, reference=None):
    env = CubeTrayEnv()
    rows = []
    waypoint_rows = []
    try:
        env.reset_policy_episode(seed=args.seed)
        controller = controller_type(
            env.model, env.data,
            position_tolerance=args.position_tolerance,
            orientation_tolerance=args.orientation_tolerance,
        )
        np.testing.assert_allclose(
            controller.arm_kp, env.model.actuator_gainprm[controller.arm_actuator_ids, 0],
            atol=1e-12, rtol=0,
        )
        initial_position, rotation = controller.get_ee_pose()
        initial = {
            "qpos": env.data.qpos.copy().tolist(),
            "qvel": env.data.qvel.copy().tolist(),
            "ctrl": env.data.ctrl.copy().tolist(),
            "ee_position": initial_position.tolist(),
            "ee_rotation": rotation.tolist(),
        }
        if reference is not None:
            for key in initial:
                np.testing.assert_array_equal(initial[key], reference[key])

        period = env.SIM_STEPS_PER_CONTROL * env.model.opt.timestep
        move_steps = control_steps(args.move_seconds, period)
        hold_steps = control_steps(args.hold_seconds, period)
        settled_steps = control_steps(args.settled_window_seconds, period)
        monitor = EpisodeEventMonitor(env, env.get_cube_position()[2])
        previous_target = initial_position.copy()
        global_step = 0

        for waypoint_index, (name, offset) in enumerate(WAYPOINTS):
            endpoint = initial_position + np.asarray(offset)
            waypoint_samples = []
            convergence_step = None
            for local_step in range(1, move_steps + hold_steps + 1):
                moving = local_step <= move_steps
                u = min(local_step / move_steps, 1.0)
                blend = 10 * u**3 - 15 * u**4 + 6 * u**5
                target = previous_target + blend * (endpoint - previous_target)
                updated, compensation, bias, command = command_pose(controller, target, rotation)
                for _ in range(env.SIM_STEPS_PER_CONTROL):
                    mujoco.mj_step(env.model, env.data)
                    monitor.observe()
                actual, actual_rotation = controller.get_ee_pose()
                position_error = float(np.linalg.norm(target - actual))
                orientation_error = float(np.linalg.norm(
                    controller.compute_orientation_error(actual_rotation, rotation)
                ))
                joint_actual = env.data.qpos[controller.arm_qpos_indices]
                servo_error = command - joint_actual
                within_tolerance = (
                    position_error < args.position_tolerance
                    and orientation_error < args.orientation_tolerance
                )
                if not moving and within_tolerance and convergence_step is None:
                    convergence_step = local_step
                global_step += 1
                row = {
                    "controller": label, "waypoint_index": waypoint_index,
                    "waypoint": name, "control_step": global_step,
                    "local_control_step": local_step,
                    "simulated_seconds": global_step * period,
                    "phase": "move" if moving else "hold",
                    "settled_window": local_step > move_steps + hold_steps - settled_steps,
                    "command_updated": updated, "within_tolerance": within_tolerance,
                    "position_error_mm": position_error * 1000,
                    "orientation_error_deg": float(np.degrees(orientation_error)),
                    "joint_tracking_rms_rad": rmse(servo_error),
                }
                for axis, target_value, actual_value in zip("xyz", target, actual):
                    row[f"target_{axis}_m"] = float(target_value)
                    row[f"actual_{axis}_m"] = float(actual_value)
                for i in range(3):
                    for j in range(3):
                        row[f"target_rotation_{i}{j}"] = float(rotation[i, j])
                        row[f"actual_rotation_{i}{j}"] = float(actual_rotation[i, j])
                for j in range(7):
                    row[f"joint{j + 1}_command_rad"] = float(command[j])
                    row[f"joint{j + 1}_actual_rad"] = float(joint_actual[j])
                    row[f"joint{j + 1}_compensation_rad"] = float(compensation[j])
                    row[f"joint{j + 1}_bias_nm"] = float(bias[j])
                if not all(np.isfinite(value) for value in (
                    position_error, orientation_error, row["joint_tracking_rms_rad"],
                )):
                    raise RuntimeError("Non-finite motion metrics.")
                rows.append(row)
                waypoint_samples.append(row)

            settled = waypoint_samples[-settled_steps:]
            converged = all(row["within_tolerance"] for row in settled)
            waypoint_rows.append({
                "controller": label, "waypoint_index": waypoint_index, "waypoint": name,
                "target_x_m": float(endpoint[0]), "target_y_m": float(endpoint[1]),
                "target_z_m": float(endpoint[2]), "converged": converged,
                "first_reach_control_step": convergence_step,
                "first_reach_seconds": None if convergence_step is None else convergence_step * period,
                "final_position_error_mm": waypoint_samples[-1]["position_error_mm"],
                "final_orientation_error_deg": waypoint_samples[-1]["orientation_error_deg"],
                "settled_position_error_mm": float(np.mean([r["position_error_mm"] for r in settled])),
                "settled_orientation_error_deg": float(np.mean([r["orientation_error_deg"] for r in settled])),
                "move_position_rmse_mm": rmse([r["position_error_mm"] for r in waypoint_samples[:move_steps]]),
            })
            print(f"[{label}] {name}: settled={waypoint_rows[-1]['settled_position_error_mm']:.3f} mm converged={converged}")
            previous_target = endpoint

        metrics = {
            "initial_state": initial, "control_period_seconds": period,
            "sample_count": len(rows), "waypoint_count": len(waypoint_rows),
            "mean_position_error_mm": float(np.mean([r["position_error_mm"] for r in rows])),
            "position_rmse_mm": rmse([r["position_error_mm"] for r in rows]),
            "move_position_rmse_mm": rmse([r["position_error_mm"] for r in rows if r["phase"] == "move"]),
            "max_position_error_mm": max(r["position_error_mm"] for r in rows),
            "mean_final_position_error_mm": float(np.mean([r["final_position_error_mm"] for r in waypoint_rows])),
            "mean_settled_position_error_mm": float(np.mean([r["settled_position_error_mm"] for r in waypoint_rows])),
            "orientation_rmse_deg": rmse([r["orientation_error_deg"] for r in rows]),
            "joint_tracking_rmse_rad": rmse([r["joint_tracking_rms_rad"] for r in rows]),
            "convergence_rate": sum(r["converged"] for r in waypoint_rows) / len(waypoint_rows),
            "max_command_compensation_rad": max(
                abs(r[f"joint{j}_compensation_rad"]) for r in rows for j in range(1, 8)
            ),
            "collision": monitor.collision, "collision_events": monitor.collision_events,
            "collision_sim_steps": monitor.collision_sim_steps,
        }
        write_json(args.output_dir / label / "metrics.json", metrics)
        return metrics, rows, waypoint_rows
    finally:
        env.close()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    trajectory = []
    waypoints = []
    reference = None
    for label, controller_type in (
        ("with_gravity", PandaIKController),
        ("without_gravity", PandaIKControllerNoGravity),
    ):
        metrics, rows, waypoint_rows = run_controller(args, label, controller_type, reference)
        reference = metrics["initial_state"]
        results[label] = metrics
        trajectory.extend(rows)
        waypoints.extend(waypoint_rows)

    numeric_metrics = [key for key, value in results["with_gravity"].items()
                       if isinstance(value, (float, int)) and not isinstance(value, bool)]
    summary = {
        "experiment": {
            "seed": args.seed, "move_seconds": args.move_seconds,
            "hold_seconds": args.hold_seconds,
            "settled_window_seconds": args.settled_window_seconds,
            "position_tolerance_m": args.position_tolerance,
            "orientation_tolerance_rad": args.orientation_tolerance,
            "target_orientation": "fixed initial end-effector rotation",
            "trajectory": "minimum-jerk interpolation between shared Cartesian endpoints",
            "waypoint_offsets_m": {name: list(offset) for name, offset in WAYPOINTS},
            "controller_parameters": {
                "damping": 1e-3, "step_size": 0.5, "max_joint_step": 0.05,
                "orientation_weight": 1.0,
                "gravity_compensation_scale": 1.0, "max_gravity_offset": 0.02,
            },
        },
        "metric_definitions": {
            "position_rmse_mm": "RMSE of Cartesian position-error norms, all move and hold samples.",
            "move_position_rmse_mm": "RMSE restricted to moving trajectory samples.",
            "settled_error": "Mean error during the final settled window of each waypoint hold.",
            "convergence_rate": "Fraction of waypoints within position AND orientation tolerance throughout the final window.",
            "first_reach_seconds": "Time from segment start to the first in-tolerance hold sample; not necessarily stable convergence.",
            "joint_tracking_rmse_rad": "RMS commanded actuator position minus actual joint position; compensated commands intentionally include an offset.",
            "compensation": "Existing controller uses qfrc_bias (gravity plus velocity-dependent bias), not pure gravity torque.",
            "collision": "New robot-obstacle contacts; reset baseline contacts are excluded.",
        },
        "initial_states_equal": True,
        "controllers": results,
        "with_minus_without": {
            key: results["with_gravity"][key] - results["without_gravity"][key]
            for key in numeric_metrics
        },
    }
    write_csv(args.output_dir / "trajectory.csv", trajectory)
    write_csv(args.output_dir / "waypoints.csv", waypoints)
    write_json(args.output_dir / "summary.json", summary)
    print("\ncontroller       RMSE(mm)  max(mm)  settled(mm)  orientation(deg)  convergence")
    for label, metrics in results.items():
        print(f"{label:<16} {metrics['position_rmse_mm']:8.3f} {metrics['max_position_error_mm']:8.3f} "
              f"{metrics['mean_settled_position_error_mm']:12.3f} "
              f"{metrics['orientation_rmse_deg']:17.3f} {metrics['convergence_rate']:11.1%}")
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
