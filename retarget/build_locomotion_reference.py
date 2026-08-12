"""Build command-labelled, contact-consistent X2 locomotion references.

The input is an existing X2 retarget result (pickle or NPZ).  The output keeps
only the 29 policy-controlled joints; the two head joints are fixed to zero
only while running MuJoCo FK.  Root XY is recomputed from stance-foot anchors,
then body-frame ``[vx, vy, wz]`` labels are measured from that corrected root.

Example (run in the ``gmr`` environment)::

    python retarget/build_locomotion_reference.py \
      --src Any2Any/retargeted_dataset_v19_npz50 \
      --recipe-manifest Any2Any/x2_motion_tracking_recipe_v1/manifest.jsonl \
      --out-root Any2Any/locomotion_reference_x2_v1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle

import numpy as np

try:
    from ..kinematic_alignment import X2_MUJOCO_JOINTS
except ImportError:
    import sys

    ANY2ANY_ROOT = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ANY2ANY_ROOT))
    from kinematic_alignment import X2_MUJOCO_JOINTS


HERE = pathlib.Path(__file__).resolve().parent
X2_XML = HERE.parent.parent / "GMR/assets/agibot_x2/x2_mocap.xml"
CONTROLLED_JOINT_NAMES = tuple(X2_MUJOCO_JOINTS[:29])


def _as_scalar(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def load_motion(path: pathlib.Path) -> dict:
    """Load production pickle (wxyz) or NPZ50 (xyzw) into one wxyz schema."""
    if path.suffix == ".pkl":
        with path.open("rb") as stream:
            raw = pickle.load(stream)
        root_rot = np.asarray(raw["root_rot"], dtype=np.float64)
        source = str(raw.get("source", path))
        fps = float(raw.get("fps", 30.0))
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as raw:
            root_rot_xyzw = np.asarray(raw["root_rot"], dtype=np.float64)
            root_rot = root_rot_xyzw[:, [3, 0, 1, 2]]
            if "joint_names" in raw:
                joint_names = tuple(str(name) for name in raw["joint_names"].tolist())
                if joint_names not in (
                    CONTROLLED_JOINT_NAMES,
                    tuple(X2_MUJOCO_JOINTS),
                ):
                    raise ValueError(f"unexpected X2 joint order in {path}")
            source = str(_as_scalar(raw["source"])) if "source" in raw else str(path)
            fps = float(_as_scalar(raw["fps"]))
            return {
                "root_pos": np.asarray(raw["root_pos"], dtype=np.float64),
                "root_rot": root_rot,
                "dof_pos": np.asarray(raw["dof_pos"], dtype=np.float64),
                "fps": fps,
                "source": source,
            }
    else:
        raise ValueError(f"unsupported motion file: {path}")
    return {
        "root_pos": np.asarray(raw["root_pos"], dtype=np.float64),
        "root_rot": root_rot,
        "dof_pos": np.asarray(raw["dof_pos"], dtype=np.float64),
        "fps": fps,
        "source": source,
    }


def controlled_dof(dof_pos: np.ndarray) -> np.ndarray:
    """Return the 29 controlled X2 joints and reject ambiguous layouts."""
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    if dof_pos.ndim != 2 or dof_pos.shape[1] not in (29, 31):
        raise ValueError(f"expected (T,29) or (T,31) dof_pos, got {dof_pos.shape}")
    return dof_pos[:, :29].copy()


def expand_for_fk(dof_pos_29: np.ndarray) -> np.ndarray:
    """Append fixed neutral head joints for the 31-DoF MuJoCo articulation."""
    dof_pos_29 = controlled_dof(dof_pos_29)
    return np.pad(dof_pos_29, ((0, 0), (0, 2)), mode="constant")


class X2FootKinematics:
    def __init__(self, xml_path: pathlib.Path = X2_XML):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.foot_geoms = []
        for side in ("left", "right"):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{side}_ankle_roll_link",
            )
            geoms = [
                geom_id
                for geom_id in range(self.model.ngeom)
                if self.model.geom_bodyid[geom_id] == body_id
                and self.model.geom_contype[geom_id] != 0
                and self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
            ]
            if not geoms:
                raise RuntimeError(f"no sole collision geoms for {side} foot")
            self.foot_geoms.append(geoms)

    def trajectory(
        self,
        root_pos: np.ndarray,
        root_rot_wxyz: np.ndarray,
        dof_pos_29: np.ndarray,
    ) -> np.ndarray:
        dof_pos_31 = expand_for_fk(dof_pos_29)
        feet = np.empty((len(root_pos), 2, 3), dtype=np.float64)
        for frame in range(len(root_pos)):
            self.data.qpos[:3] = root_pos[frame]
            self.data.qpos[3:7] = root_rot_wxyz[frame]
            self.data.qpos[7:] = dof_pos_31[frame]
            self.mujoco.mj_forward(self.model, self.data)
            for foot, geoms in enumerate(self.foot_geoms):
                positions = self.data.geom_xpos[geoms]
                feet[frame, foot, :2] = positions[:, :2].mean(axis=0)
                feet[frame, foot, 2] = min(
                    self.data.geom_xpos[geom, 2] - self.model.geom_size[geom, 0]
                    for geom in geoms
                )
        return feet


def detect_contacts(
    foot_world: np.ndarray,
    fps: float,
    contact_height: float = 0.020,
    contact_vertical_speed: float = 0.25,
    double_support_speed: float = 0.08,
) -> np.ndarray:
    """Detect stance from sole height/vertical speed without trusting XY slip."""
    if len(foot_world) < 2:
        raise ValueError("a locomotion reference needs at least two frames")
    floor = float(np.percentile(foot_world[:, :, 2], 5))
    vertical_speed = np.gradient(foot_world[:, :, 2], 1.0 / fps, axis=0)
    horizontal_speed = np.linalg.norm(
        np.gradient(foot_world[:, :, :2], 1.0 / fps, axis=0),
        axis=2,
    )
    contact = (
        (foot_world[:, :, 2] <= floor + contact_height)
        & (np.abs(vertical_speed) <= contact_vertical_speed)
    )
    # Fill isolated one-frame detector gaps, and remove isolated blips.
    for foot in range(2):
        values = contact[:, foot].copy()
        values[1:-1] |= values[:-2] & values[2:]
        isolated = values[1:-1] & ~values[:-2] & ~values[2:]
        values[1:-1][isolated] = False
        contact[:, foot] = values
    # A low swing foot often enters the height band before heel strike.  Keep
    # true double support only while both feet move slowly; otherwise the
    # slower foot owns the stance anchor.
    for frame in np.flatnonzero(contact.all(axis=1)):
        if horizontal_speed[frame].max() > double_support_speed:
            stance = int(np.argmin(horizontal_speed[frame]))
            contact[frame] = False
            contact[frame, stance] = True
    # A low foot in an otherwise empty frame is stance, not flight caused by
    # noisy vertical differentiation.  True flight remains untouched.
    empty = ~contact.any(axis=1)
    lower = np.argmin(foot_world[:, :, 2], axis=1)
    for frame in np.flatnonzero(empty):
        foot = lower[frame]
        if foot_world[frame, foot, 2] <= floor + 0.06:
            contact[frame, foot] = True
    return contact


def correct_root_xy(
    root_pos: np.ndarray,
    foot_relative_xy: np.ndarray,
    contact: np.ndarray,
) -> np.ndarray:
    """Recompute root XY by pinning each stance foot to its landing anchor."""
    root_pos = np.asarray(root_pos, dtype=np.float64)
    foot_relative_xy = np.asarray(foot_relative_xy, dtype=np.float64)
    if foot_relative_xy.shape != (len(root_pos), 2, 2):
        raise ValueError(f"invalid foot_relative_xy shape {foot_relative_xy.shape}")
    if contact.shape != (len(root_pos), 2):
        raise ValueError(f"invalid contact shape {contact.shape}")

    corrected = root_pos.copy()
    anchors = [None, None]
    for frame in range(len(root_pos)):
        if frame == 0:
            provisional = root_pos[0, :2].copy()
        else:
            provisional = corrected[frame - 1, :2] + (
                root_pos[frame, :2] - root_pos[frame - 1, :2]
            )
        for foot in range(2):
            if not contact[frame, foot]:
                anchors[foot] = None
        existing = [foot for foot in range(2) if anchors[foot] is not None]
        if existing:
            provisional = np.mean(
                [anchors[foot] - foot_relative_xy[frame, foot] for foot in existing],
                axis=0,
            )
        for foot in range(2):
            if contact[frame, foot] and anchors[foot] is None:
                anchors[foot] = provisional + foot_relative_xy[frame, foot]
        active = [foot for foot in range(2) if anchors[foot] is not None]
        if active:
            corrected[frame, :2] = np.mean(
                [anchors[foot] - foot_relative_xy[frame, foot] for foot in active],
                axis=0,
            )
        else:
            corrected[frame, :2] = provisional
    return corrected


def _yaw_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion).T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _smooth(values: np.ndarray, fps: float, seconds: float) -> np.ndarray:
    window = max(1, int(round(seconds * fps)))
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack(
        [np.convolve(padded[:, column], kernel, mode="valid") for column in range(values.shape[1])],
        axis=1,
    )


def command_labels(
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    fps: float,
    smoothing_seconds: float = 0.2,
) -> np.ndarray:
    """Measure body-frame ``vx, vy, wz`` from the final corrected X2 root."""
    world_velocity = np.gradient(root_pos[:, :2], 1.0 / fps, axis=0)
    yaw = _yaw_from_wxyz(root_rot_wxyz)
    yaw_rate = np.gradient(yaw, 1.0 / fps)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    command = np.column_stack(
        [
            cosine * world_velocity[:, 0] + sine * world_velocity[:, 1],
            -sine * world_velocity[:, 0] + cosine * world_velocity[:, 1],
            yaw_rate,
        ]
    )
    return _smooth(command, fps, smoothing_seconds)


def stance_slip_speed(
    foot_world: np.ndarray,
    original_root: np.ndarray,
    corrected_root: np.ndarray,
    contact: np.ndarray,
    fps: float,
) -> np.ndarray:
    corrected_feet = foot_world.copy()
    corrected_feet[:, :, :2] += (
        corrected_root[:, None, :2] - original_root[:, None, :2]
    )
    speed = np.linalg.norm(np.diff(corrected_feet[:, :, :2], axis=0) * fps, axis=2)
    sustained_contact = contact[1:] & contact[:-1]
    return speed[sustained_contact]


def build_reference(
    motion: dict,
    kinematics: X2FootKinematics,
    min_mean_speed: float = 0.05,
    min_mean_turn_rate: float = 0.10,
    min_command_activity: float = 0.50,
    max_mean_stance_slip: float = 0.02,
    max_mean_root_correction: float = 0.15,
    max_root_correction: float = 0.50,
) -> tuple[dict, dict]:
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = controlled_dof(motion["dof_pos"])
    fps = float(motion["fps"])
    if root_pos.shape != (len(dof_pos), 3) or root_rot.shape != (len(dof_pos), 4):
        raise ValueError("root and joint trajectories have inconsistent shapes")
    if not np.isfinite(root_pos).all() or not np.isfinite(root_rot).all() or not np.isfinite(dof_pos).all():
        raise ValueError("motion contains non-finite values")

    foot_world = kinematics.trajectory(root_pos, root_rot, dof_pos)
    contact = detect_contacts(foot_world, fps)
    foot_relative_xy = foot_world[:, :, :2] - root_pos[:, None, :2]
    corrected_root = correct_root_xy(root_pos, foot_relative_xy, contact)
    command = command_labels(corrected_root, root_rot, fps)
    slip = stance_slip_speed(foot_world, root_pos, corrected_root, contact, fps)

    mean_speed = float(np.linalg.norm(command[:, :2], axis=1).mean())
    mean_turn_rate = float(np.abs(command[:, 2]).mean())
    command_activity = float(
        (
            (np.linalg.norm(command[:, :2], axis=1) >= 0.10)
            | (np.abs(command[:, 2]) >= 0.25)
        ).mean()
    )
    mean_slip = float(slip.mean()) if len(slip) else float("inf")
    p95_slip = float(np.percentile(slip, 95)) if len(slip) else float("inf")
    correction = np.linalg.norm(corrected_root[:, :2] - root_pos[:, :2], axis=1)
    mean_correction = float(correction.mean())
    peak_correction = float(correction.max())
    joint_velocity = np.abs(np.diff(dof_pos, axis=0) * fps)
    failure = None
    if (
        mean_correction > max_mean_root_correction
        or peak_correction > max_root_correction
    ):
        failure = "root_correction"
    elif (
        (mean_speed < min_mean_speed and mean_turn_rate < min_mean_turn_rate)
        or command_activity < min_command_activity
    ):
        failure = "not_locomotion"
    elif mean_slip > max_mean_stance_slip:
        failure = "stance_slip"

    report = {
        "fail": failure,
        "frames": len(dof_pos),
        "fps": fps,
        "mean_speed_mps": mean_speed,
        "mean_abs_turn_rate_rps": mean_turn_rate,
        "command_activity_fraction": command_activity,
        "mean_stance_slip_mps": mean_slip,
        "p95_stance_slip_mps": p95_slip,
        "contact_fraction": float(contact.mean()),
        "mean_root_correction_m": mean_correction,
        "max_root_correction_m": peak_correction,
        "max_joint_velocity_rps": float(joint_velocity.max(initial=0.0)),
    }
    reference = {
        "root_pos": corrected_root.astype(np.float32),
        "root_rot": root_rot.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "contact": contact.astype(np.uint8),
        "command": command.astype(np.float32),
        "fps": np.float32(fps),
        "joint_names": np.asarray(CONTROLLED_JOINT_NAMES),
        "command_names": np.asarray(["vx_body", "vy_body", "wz"]),
        "root_rot_format": np.asarray("wxyz"),
        "source": np.asarray(str(motion.get("source", ""))),
    }
    return reference, report


def collect_inputs(
    source_root: pathlib.Path,
    recipe_manifest: pathlib.Path | None,
    splits: set[str],
) -> list[pathlib.Path]:
    if source_root.is_file():
        return [source_root]
    if recipe_manifest is None:
        return sorted([*source_root.rglob("*.pkl"), *source_root.rglob("*.npz")])
    paths = []
    with recipe_manifest.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("category") != "locomotion" or entry.get("split") not in splits:
                continue
            source_path = source_root / entry["path"]
            pickle_path = source_path.with_suffix(".pkl")
            segmented = sorted(
                pickle_path.parent.glob(f"{pickle_path.stem}__part*.pkl")
            )
            if source_path.exists():
                paths.append(source_path)
            elif pickle_path.exists():
                paths.append(pickle_path)
            elif segmented:
                paths.extend(segmented)
            else:
                paths.append(source_path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=pathlib.Path, required=True)
    parser.add_argument("--out-root", type=pathlib.Path, required=True)
    parser.add_argument("--recipe-manifest", type=pathlib.Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-qc-fail", action="store_true")
    args = parser.parse_args()

    if not args.src.exists():
        raise SystemExit(f"retargeted source root does not exist: {args.src}")
    inputs = collect_inputs(args.src, args.recipe_manifest, set(args.splits))
    if args.max_clips is not None:
        inputs = inputs[: args.max_clips]
    if not inputs:
        raise SystemExit(f"no retargeted motion files found under {args.src}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    report_path = args.out_root / "manifest.jsonl"
    kinematics = X2FootKinematics()
    passed = failed = missing = 0
    with report_path.open("a", encoding="utf-8") as report_stream:
        for source in inputs:
            if not source.exists():
                result = {"source": str(source), "fail": "missing"}
                missing += 1
            else:
                relative = source.name if args.src.is_file() else str(source.relative_to(args.src))
                target = (args.out_root / relative).with_suffix(".npz")
                if target.exists() and not args.overwrite:
                    continue
                try:
                    motion = load_motion(source)
                    reference, result = build_reference(motion, kinematics)
                    result.update(source=str(source), output=str(target))
                    if result["fail"] is None or args.allow_qc_fail:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(target, **reference)
                    elif args.overwrite:
                        # Do not leave a previously accepted file in place
                        # after a stricter QC pass rejects it.
                        target.unlink(missing_ok=True)
                    if result["fail"] is None:
                        passed += 1
                    else:
                        failed += 1
                except Exception as error:  # noqa: BLE001
                    result = {
                        "source": str(source),
                        "fail": "error",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    failed += 1
            report_stream.write(json.dumps(result, sort_keys=True) + "\n")
            report_stream.flush()
            print(json.dumps(result, sort_keys=True))
    print(f"locomotion references: {passed} passed, {failed} failed, {missing} missing")


if __name__ == "__main__":
    main()
