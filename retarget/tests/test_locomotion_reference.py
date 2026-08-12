from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np


ANY2ANY_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANY2ANY_ROOT))

from retarget.build_locomotion_reference import (  # noqa: E402
    CONTROLLED_JOINT_NAMES,
    command_labels,
    controlled_dof,
    correct_root_xy,
    detect_contacts,
    expand_for_fk,
    load_motion,
)


class _FixedKinematics:
    def __init__(self, feet):
        self.feet = feet

    def trajectory(self, root_pos, root_rot_wxyz, dof_pos_29):
        del root_pos, root_rot_wxyz, dof_pos_29
        return self.feet.copy()


def test_head_joints_are_not_part_of_the_reference_interface():
    dof31 = np.arange(4 * 31, dtype=np.float64).reshape(4, 31)
    dof29 = controlled_dof(dof31)
    assert dof29.shape == (4, 29)
    assert np.array_equal(dof29, dof31[:, :29])
    expanded = expand_for_fk(dof29)
    assert expanded.shape == (4, 31)
    assert np.array_equal(expanded[:, :29], dof29)
    assert np.all(expanded[:, 29:] == 0)


def test_npz_xyzw_root_is_loaded_as_explicit_wxyz():
    with tempfile.NamedTemporaryFile(suffix=".npz") as output:
        np.savez(
            output.name,
            root_pos=np.zeros((2, 3)),
            root_rot=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
            dof_pos=np.zeros((2, 31)),
            joint_names=np.asarray(
                [*CONTROLLED_JOINT_NAMES, "head_yaw_joint", "head_pitch_joint"]
            ),
            fps=np.int64(50),
        )
        motion = load_motion(pathlib.Path(output.name))
    assert np.allclose(motion["root_rot"], [1.0, 0.0, 0.0, 0.0])


def test_recipe_selects_only_requested_locomotion_split():
    from retarget.batch_retarget_x2 import recipe_locomotion_paths

    entries = [
        {"path": "A/walk_stageii.npz", "category": "locomotion", "split": "train"},
        {"path": "A/run_stageii.npz", "category": "locomotion", "split": "test"},
        {"path": "A/box_stageii.npz", "category": "dynamic", "split": "train"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as manifest:
        for entry in entries:
            manifest.write(json.dumps(entry) + "\n")
        manifest.flush()
        paths = recipe_locomotion_paths(manifest.name, splits=("train",))
    assert paths == [pathlib.Path("A/walk_stageii.npz")]


def test_stance_anchor_recomputes_root_instead_of_copying_source_translation():
    frames = 20
    source_root = np.zeros((frames, 3))
    source_root[:, 0] = np.arange(frames) * 0.02
    foot_relative = np.zeros((frames, 2, 2))
    foot_relative[:, 0, 0] = -np.arange(frames) * 0.01
    contact = np.zeros((frames, 2), dtype=bool)
    contact[:, 0] = True

    corrected = correct_root_xy(source_root, foot_relative, contact)
    corrected_foot = corrected[:, 0] + foot_relative[:, 0, 0]
    assert np.allclose(corrected_foot, corrected_foot[0])
    assert np.allclose(np.diff(corrected[:, 0]), 0.01)


def test_low_swing_foot_does_not_create_false_double_support():
    fps = 50.0
    feet = np.zeros((20, 2, 3))
    feet[:, :, 2] = 0.01
    feet[:, 1, 0] = np.arange(20) * 0.01
    contact = detect_contacts(feet, fps)
    assert np.all(contact[:, 0])
    assert not np.any(contact[:, 1])


def test_command_labels_recover_body_forward_speed_and_yaw_rate():
    fps = 50.0
    frames = 250
    dt = 1.0 / fps
    target_speed = 0.4
    target_yaw_rate = 0.3
    yaw = np.arange(frames) * dt * target_yaw_rate
    root = np.zeros((frames, 3))
    for frame in range(1, frames):
        root[frame, 0] = root[frame - 1, 0] + np.cos(yaw[frame]) * target_speed * dt
        root[frame, 1] = root[frame - 1, 1] + np.sin(yaw[frame]) * target_speed * dt
    quat = np.column_stack(
        [np.cos(yaw / 2), np.zeros(frames), np.zeros(frames), np.sin(yaw / 2)]
    )

    command = command_labels(root, quat, fps, smoothing_seconds=0.0)
    steady = slice(2, -2)
    assert np.allclose(command[steady, 0], target_speed, atol=2e-3)
    assert np.allclose(command[steady, 1], 0.0, atol=2e-3)
    assert np.allclose(command[steady, 2], target_yaw_rate, atol=1e-6)


def test_qc_rejects_large_root_reconstruction():
    from retarget.build_locomotion_reference import build_reference

    fps = 50.0
    frames = 100
    root = np.zeros((frames, 3))
    root[:, 0] = np.arange(frames) * 0.02
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    feet = np.zeros((frames, 2, 3))
    feet[:, :, 0] = root[:, None, 0]
    feet[:, :, 2] = 0.01
    feet[:, 1, 0] += 0.15
    motion = {
        "root_pos": root,
        "root_rot": quat,
        "dof_pos": np.zeros((frames, 29)),
        "fps": fps,
    }

    _, report = build_reference(motion, _FixedKinematics(feet))
    assert report["fail"] == "root_correction"
    assert report["mean_root_correction_m"] > 0.15


def test_qc_rejects_mostly_stationary_motion():
    from retarget.build_locomotion_reference import build_reference

    fps = 50.0
    frames = 200
    root = np.zeros((frames, 3))
    root[80:120, 0] = np.arange(40) * 0.01
    root[120:, 0] = root[119, 0]
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1))
    feet = np.zeros((frames, 2, 3))
    feet[:, 1, 0] = 0.15
    feet[:, :, 2] = 0.01
    motion = {
        "root_pos": root,
        "root_rot": quat,
        "dof_pos": np.zeros((frames, 29)),
        "fps": fps,
    }

    _, report = build_reference(motion, _FixedKinematics(feet))
    assert report["fail"] == "not_locomotion"
    assert report["command_activity_fraction"] < 0.50
