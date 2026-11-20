#!/usr/bin/env python3
"""Run a matched real PX4 SITL 10 s circle with seeded horizontal gusts."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")

from wind_field_simulation import simulation as sim  # noqa: E402
from wind_field_simulation.sitl_process import request_managed_sitl_shutdown  # noqa: E402
from wind_field_simulation.training.dataset_v1 import file_sha256  # noqa: E402
from wind_field_simulation.training.study_v2 import write_json  # noqa: E402
from wind_field_simulation.wind_randomization import draw_drag_scales  # noqa: E402

WINDS = {42: (4., 4., 0.), 43: (-6., 4., 0.), 44: (0., 8.5, 0.)}
DURATION = 10.0
PX4_RUNTIME = ROOT / "runtime/px4"
DEFAULT_SEED_FILE = ROOT / "configs/pinn/x4_950_sitl_seed_parameters.bson"


class ObservedSimulation(sim.WindSimulationApp):
    """Record full wind-path timing and actual applied force without changing it."""

    maximum_abs_fz_n: float = 0.

    def _apply_wind_force(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track applied Z after every production force update."""
        result = super()._apply_wind_force()
        actual_z = self.data.xfrc_applied[self.bindings.quadrotor_body_id, 2]
        self.maximum_abs_fz_n = max(self.maximum_abs_fz_n, abs(float(actual_z)))
        return result

    def _run_log_sample(self) -> dict[str, Any] | None:
        """Attach force-path elapsed time and PX4 state to each sampled row."""
        sample = super()._run_log_sample()
        if sample is not None:
            sample.update(px4_offboard=bool(self.flight_control.offboard),
                          px4_armed=bool(self.flight_control.armed))
        return sample


def fly(
    app: ObservedSimulation, launcher: subprocess.Popen, wind: tuple[float, float, float],
    receipt: dict[str, Any],
) -> None:
    """Issue production commands at fixed flight phase offsets until disarmed."""
    app.bridge.connect()
    origin = time.perf_counter() - float(app.data.time)
    phase, phase_at = "warmup", float(app.data.time)
    control = app.flight_control
    offboard_lost_at = None

    def mark(event: str) -> None:
        item = {"event": event, "sim_time_s": float(app.data.time)}
        receipt["events"].append(item)
        print(json.dumps(item), flush=True)

    def advance(next_phase: str) -> None:
        nonlocal phase, phase_at
        phase, phase_at = next_phase, float(app.data.time)
        mark(phase)

    def command(kind: sim.UiCommandType, payload: Any = None) -> None:
        app._process_ui_command(sim.UiCommand(kind, payload))

    while launcher.poll() is None:
        now = float(app.data.time)
        elapsed = now - phase_at
        if app.last_error != "none":
            raise RuntimeError(app.last_error)
        if phase == "warmup":
            if elapsed >= 10. and app.bridge.heartbeat_is_recent():
                command(sim.UiCommandType.UNLOCK)
                advance("arm_requested")
            elif elapsed >= 40.:
                raise TimeoutError("PX4 heartbeat never became available")
        elif phase == "arm_requested":
            if control.armed:
                command(sim.UiCommandType.TAKEOFF)
                advance("takeoff_requested")
            elif elapsed >= 10.:
                raise TimeoutError("PX4 failed to arm")
        elif phase == "takeoff_requested":
            if control.flight_state is sim.FlightState.FLYING:
                command(sim.UiCommandType.SET_WIND, wind)
                advance("flying")
            elif elapsed >= 25.:
                raise TimeoutError("Takeoff did not reach FLYING")
        elif phase in ("flying", "wind_enabled", "circle_started"):
            if not control.armed or control.flight_state is not sim.FlightState.FLYING:
                raise RuntimeError("Armed FLYING state lost")
            if control.offboard:
                offboard_lost_at = None
            elif offboard_lost_at is None:
                offboard_lost_at = now
            elif now - offboard_lost_at >= 1.:
                raise RuntimeError("PX4 OFFBOARD lost for at least 1 s")
            if phase == "flying" and elapsed >= 8. - 1e-8:
                command(sim.UiCommandType.TOGGLE_WIND)
                if not control.wind_enabled or app.wind_field is None:
                    raise RuntimeError("Production wind command was not accepted")
                np.testing.assert_array_equal(app.wind_field.configured_velocity_world, wind)
                advance("wind_enabled")
            elif phase == "wind_enabled" and elapsed >= 10. - 1e-8:
                command(sim.UiCommandType.START_TRAJECTORY, "circle")
                trajectory = app.active_trajectory
                if trajectory is None or trajectory.duration != DURATION or trajectory.size_m != 1.:
                    raise RuntimeError("Production Circle command was not accepted")
                if app.trajectory_plot is not None:
                    app.trajectory_plot.close()
                    app.trajectory_plot = None
                advance("circle_started")
            elif phase == "circle_started" and elapsed >= DURATION + .5 - 1e-8:
                command(sim.UiCommandType.TOGGLE_WIND)
                mark("wind_disabled")
                command(sim.UiCommandType.LAND)
                advance("landing")
        elif phase == "landing":
            if not control.armed:
                command(sim.UiCommandType.EXIT)
                advance("exit_requested")
                return
            if elapsed >= 25.:
                raise TimeoutError("Landing/disarm not reached")

        if not app._run_iteration():
            raise RuntimeError("Unexpected Exit before landing")
        address = app.bindings.main_qpos_adr
        if np.max(np.abs(app.data.qpos[address:address + 3])) > 5.:
            raise RuntimeError("Position exceeded the diagnostic envelope")
        if (delay := origin + float(app.data.time) - time.perf_counter()) > 0:
            time.sleep(delay)
    raise RuntimeError(f"PX4 launcher stopped early, exit {launcher.returncode}")


def summarize(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Compare one complete phase-aligned circle using synchronized 50 Hz rows."""
    with path.open() as stream:
        records = [json.loads(line) for line in stream]
    start = next(event["sim_time_s"] for event in receipt["events"]
                 if event["event"] == "circle_started")
    selected = [row for row in records if row["type"] == "sample"
                and -1e-8 <= row["sim_time_s"] - start < DURATION - 1e-8]
    if (len(selected) != 500
            or not all(row["trajectory"] == "circle" and row["wind_enabled"]
                       and row["px4_offboard"] and row["px4_armed"] for row in selected)
            or not records[-1]["exit_requested"]
            or records[-1]["exception"] is not None):
        raise RuntimeError("Full clean armed/offboard circle window required")
    times = np.array([row["sim_time_s"] for row in selected])
    if not np.allclose(np.diff(times), .02, rtol=0, atol=1e-8):
        raise RuntimeError("Circle sample cadence was not 50 Hz")
    error = np.array([np.subtract(row["position_world_m"], row["target_world_m"])
                      for row in selected])
    wind = np.array([row["active_wind_world_m_s"] for row in selected])
    force = np.array([row["force_world_n"] for row in selected])
    force_timing = np.array([row["wind_compute_wall_s"] for row in selected]) * 1000
    norm = np.linalg.norm(error[:, :2], axis=-1)
    meta = records[0]
    return {
        "samples": len(selected), "horizontal_rmse_m": float(np.sqrt(np.mean(norm**2))),
        "horizontal_p95_m": float(np.quantile(norm, .95)),
        "horizontal_max_m": float(norm.max()),
        "vertical_rmse_m": float(np.sqrt(np.mean(error[:, 2]**2))),
        "force_xy_rms_n": float(np.sqrt(np.mean(np.sum(force[:, :2]**2, axis=1)))),
        "force_xy_peak_n": float(np.linalg.norm(force[:, :2], axis=1).max()),
        "active_wind_min_m_s": float(np.linalg.norm(wind[:, :2], axis=1).min()),
        "active_wind_max_m_s": float(np.linalg.norm(wind[:, :2], axis=1).max()),
        "force_path_p50_p95_p99_ms": np.quantile(force_timing, [.5, .95, .99]).tolist(),
        "outside_domain_samples": sum(row["outside_inspection_domain"] for row in selected),
        "limited_samples": sum(not np.array_equal(row["prelimit_acceleration_world_m_s2"],
                                                  row["filtered_acceleration_world_m_s2"])
                               for row in selected),
        "drag_group_scales": meta["drag_group_scales"],
        "experiment_drag_group_scales": meta["experiment_drag_group_scales"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "cda_m2": meta["wind_cda_m2"], "nominal_drag_gain": meta["wind_drag_scale"],
    }


def run(args: argparse.Namespace) -> None:
    """Run one isolated instance and preserve evidence, including failed attempts."""
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = out / "px4_state"
    state.mkdir()
    for name in ("parameters.bson", "parameters_backup.bson"):
        shutil.copyfile(args.seed_file, state / name)
    argv = [
        "--no-viewer", "--sim-tcp-port", str(4560 + args.instance), "--command-udp-port",
        str(14580 + args.instance), "--target-system", str(args.instance + 1),
        "--wind-backend", "analytic" if args.method == "analytic" else "pinn-x4",
        "--drag-variation", ".2", "--gust-amplitude", "1.5", "--gust-period", "4",
        "--wind-seed", str(args.wind_seed), "--run-log", str(out / "flight.jsonl"),
    ]
    if args.method == "v2":
        argv.extend(("--pinn-checkpoint", str(sim.bundled_x4_v2_checkpoint())))
    elif args.method == "v3" and args.checkpoint is not None:
        argv.extend(("--pinn-checkpoint", str(args.checkpoint)))
    launch = [sys.executable, str(ROOT / "scripts/run_px4_sitl.py"), "--runtime-dir",
              str(PX4_RUNTIME), "--state-dir", str(state), "-d", "-i", str(args.instance)]
    wind = WINDS[args.wind_seed]
    receipt: dict[str, Any] = {
        "method": args.method, "seed": args.wind_seed, "base_wind": wind, "events": [],
        "args": argv, "launcher_command": launch, "status": "failed",
        "driver_sha256": file_sha256(Path(__file__)),
        "px4_binary_sha256": file_sha256(PX4_RUNTIME / "bin/px4"),
        "px4_seed_sha256": file_sha256(args.seed_file),
    }
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PX4_")}
    try:
        app = ObservedSimulation(sim.parse_config(argv))
        if args.method == "v3" and app.pinn_predictor.drag_group_scales != draw_drag_scales(
            .2, args.wind_seed,
        ):
            app._close_runtime()
            raise ValueError("--method v3 requires a checkpoint that consumes the seeded drag draw")
        with (out / "px4_console.txt").open("x") as log:
            launcher = subprocess.Popen(
                launch, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                fly(app, launcher, wind, receipt)
            finally:
                signal.alarm(0)
                try:
                    app._close_runtime()
                finally:
                    receipt["managed_shutdown_request"] = request_managed_sitl_shutdown(state)
                    try:
                        code = launcher.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        launcher.send_signal(signal.SIGINT)
                        code = launcher.wait(timeout=23)
                    receipt["launcher_returncode"] = code
        receipt["max_abs_actual_fz_all_steps_n"] = app.maximum_abs_fz_n
        if not receipt["managed_shutdown_request"] or code != 0 or app.maximum_abs_fz_n != 0.:
            raise RuntimeError("Orderly PX4 shutdown or exact horizontal force was not verified")
        receipt["metrics"] = summarize(out / "flight.jsonl", receipt)
        receipt["status"] = "flown"
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        write_json(out / "result.json", receipt)


def timeout(_signum: int, _frame: Any) -> None:
    """Bound elapsed wall time of the owned flight loop."""
    raise TimeoutError("SITL flight did not complete within 125 wall seconds")


def main() -> int:
    """Select an explicit local SITL instance and matched wind condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("analytic", "v2", "v3"), required=True)
    parser.add_argument("--wind-seed", type=int, choices=WINDS, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--instance", type=int, choices=range(1, 10), default=7)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, timeout)
    with Path(f"/tmp/wind-field-x4-sitl-instance-{args.instance}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as ports:
            for kind, base in ((socket.SOCK_STREAM, 4560), (socket.SOCK_DGRAM, 14580),
                               (socket.SOCK_DGRAM, 14280), (socket.SOCK_DGRAM, 13030),
                               (socket.SOCK_DGRAM, 18570)):
                probe = ports.enter_context(socket.socket(socket.AF_INET, kind))
                if kind == socket.SOCK_STREAM:
                    # Match pymavlink's listener across orderly SITL restarts.
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    probe.bind(("127.0.0.1", base + args.instance))
                else:
                    probe.bind(("0.0.0.0", base + args.instance))
        signal.alarm(125)
        run(args)
    print(args.output_dir / "result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
