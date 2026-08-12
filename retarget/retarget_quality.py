"""Shared kinematic quality checks for the X2 retargeting pipelines."""

from __future__ import annotations

import numpy as np


ROOT_LINEAR_VELOCITY_LIMIT = 10.0  # m/s; catches mocap discontinuities
ROOT_ANGULAR_VELOCITY_LIMIT = 15.0  # rad/s
JOINT_LIMIT_FRACTION_LIMIT = 0.25
JOINT_LIMIT_DWELL_SECONDS = 1.0
X2_ARM_COLLISION_MARGIN = 0.005
X2_JOINT_VELOCITY_LIMIT = 12.0  # rad/s; shared by offline and live references


def limit_joint_velocity(
    previous: np.ndarray | None,
    current: np.ndarray,
    fps: float,
    velocity_limit: float = X2_JOINT_VELOCITY_LIMIT,
) -> tuple[np.ndarray, bool]:
    """Apply the production per-frame joint safety clamp without filtering."""
    result = np.asarray(current, dtype=np.float64).copy()
    if previous is None:
        return result, False
    previous = np.asarray(previous, dtype=np.float64)
    if result.shape != previous.shape or result.size < 8:
        raise ValueError("X2 qpos samples must have matching floating-base shapes")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("reference fps must be finite and positive")
    step = float(velocity_limit) / float(fps)
    delta = result[7:] - previous[7:]
    clipped = bool(np.max(np.abs(delta), initial=0.0) > step)
    if clipped:
        result[7:] = previous[7:] + np.clip(delta, -step, step)
    return result, clipped

# Endpoint-aware rules are important for the asymmetric X2 arms. A straight
# elbow (upper endpoint) and shoulder-roll stops can both occur in valid poses
# (arms at the torso or overhead). Deep flexion and sustained shoulder pitch/yaw
# stops are the useful indicators of an unreachable target or bad IK branch.
ENFORCED_LIMIT_ENDPOINTS = {
    "waist_yaw_joint": ("lower", "upper"),
    "waist_pitch_joint": ("lower", "upper"),
    "waist_roll_joint": ("lower", "upper"),
    "left_shoulder_pitch_joint": ("lower", "upper"),
    "right_shoulder_pitch_joint": ("lower", "upper"),
    "left_shoulder_yaw_joint": ("lower", "upper"),
    "right_shoulder_yaw_joint": ("lower", "upper"),
    "left_elbow_joint": ("lower",),
    "right_elbow_joint": ("lower",),
}


def root_motion_metrics(qpos: np.ndarray, fps: float) -> tuple[float, float]:
    """Return maximum root linear and angular speeds."""
    if len(qpos) < 2:
        return 0.0, 0.0
    linear = np.linalg.norm(np.diff(qpos[:, :3], axis=0), axis=1) * fps
    quat = np.asarray(qpos[:, 3:7], dtype=np.float64)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    # q and -q encode the same orientation, hence abs(dot).
    dots = np.clip(np.abs(np.sum(quat[:-1] * quat[1:], axis=1)), 0.0, 1.0)
    angular = 2.0 * np.arccos(dots) * fps
    return float(linear.max()), float(angular.max())


def _longest_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run in a one-dimensional boolean mask."""
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2], initial=0))


def joint_limit_report(
    model,
    mujoco,
    dof: np.ndarray,
    margin: float = 0.02,
    fps: float | None = None,
):
    """Return per-joint saturation fractions and unsafe endpoint dwell.

    A violation is fatal when it occupies more than a quarter of a clip or,
    when ``fps`` is known, remains continuous for more than one second.
    """
    hinge_ids = [
        joint_id
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if dof.shape[1] != len(hinge_ids):
        raise ValueError(f"dof width {dof.shape[1]} != {len(hinge_ids)} hinge joints")
    lo = model.jnt_range[hinge_ids, 0]
    hi = model.jnt_range[hinge_ids, 1]
    lower_masks = dof < lo + margin
    upper_masks = dof > hi - margin
    fractions = (lower_masks | upper_masks).mean(axis=0)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in hinge_ids
    ]
    report = {
        name: round(float(fraction), 3)
        for name, fraction in zip(names, fractions)
        if fraction > 0.05
    }
    unexpected = {}
    for index, name in enumerate(names):
        endpoints = ENFORCED_LIMIT_ENDPOINTS.get(name, ())
        if not endpoints:
            continue
        unsafe = np.zeros(len(dof), dtype=bool)
        if "lower" in endpoints:
            unsafe |= lower_masks[:, index]
        if "upper" in endpoints:
            unsafe |= upper_masks[:, index]
        fraction = float(unsafe.mean())
        prolonged = (
            fps is not None
            and fps > 0
            and _longest_true_run(unsafe) / fps > JOINT_LIMIT_DWELL_SECONDS
        )
        if fraction > JOINT_LIMIT_FRACTION_LIMIT or prolonged:
            unexpected[name] = round(fraction, 3)
    return report, unexpected


def foot_collision_geoms(model, mujoco) -> list[int]:
    """Resolve the X2 sole collision spheres from the model."""
    body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    }
    geom_ids = [
        geom_id
        for geom_id in range(model.ngeom)
        if model.geom_bodyid[geom_id] in body_ids
        and model.geom_contype[geom_id] != 0
        and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    if not geom_ids:
        raise RuntimeError("X2 sole collision spheres were not found")
    return geom_ids


def sole_height(model, data, geom_ids: list[int]) -> float:
    """Return the actual lowest point of all X2 sole collision spheres."""
    return min(
        float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
        for geom_id in geom_ids
    )
