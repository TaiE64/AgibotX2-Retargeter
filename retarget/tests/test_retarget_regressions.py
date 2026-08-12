from __future__ import annotations

import pathlib
import sys

import numpy as np


RETARGET_ROOT = pathlib.Path(__file__).resolve().parents[1]
ANY2ANY_ROOT = RETARGET_ROOT.parent
GMR_ROOT = ANY2ANY_ROOT.parent / "GMR"
sys.path.insert(0, str(ANY2ANY_ROOT))
sys.path.insert(0, str(GMR_ROOT))


def test_target_sample_indices_are_exact_rate():
    from general_motion_retargeting.utils.smpl import target_sample_indices

    # 100 Hz no longer degrades to 33.33 Hz when 30 Hz was requested.
    indices = target_sample_indices(1001, 100.0, 30.0)
    assert len(indices) == 301
    assert np.allclose(np.diff(indices) / 100.0, 1.0 / 30.0)

    # Floating metadata just below 60 must not make int(src / target) == 1.
    indices = target_sample_indices(601, 59.9999885559082, 30.0)
    assert 300 <= len(indices) <= 301
    assert np.allclose(np.diff(indices) / 59.9999885559082, 1.0 / 30.0)

    # Low-rate inputs are upsampled and truthfully labelled at the target rate.
    indices = target_sample_indices(251, 25.0, 30.0)
    assert len(indices) == 301


def test_long_clips_are_balanced_into_bounded_non_overlapping_segments():
    from retarget.batch_retarget_x2 import segment_frame_ranges

    # A tiny tail must be redistributed, not rejected as a sub-2-second clip.
    ranges = segment_frame_ranges(2430, 30.0, 40.0)  # 81 seconds
    assert ranges[0][0] == 0
    assert ranges[-1][1] == 2430
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    durations = [(end - start) / 30.0 for start, end in ranges]
    assert len(ranges) == 3
    assert max(durations) <= 40.0
    assert min(durations) >= 2.0

    # Motions already within the bound retain their original single output.
    assert segment_frame_ranges(1200, 30.0, 40.0) == [(0, 1200)]


def test_root_motion_metrics_detect_discontinuity_and_quaternion_sign_flip():
    from retarget.retarget_quality import root_motion_metrics

    qpos = np.zeros((4, 8), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[2, 0] = 2.0
    linear, angular = root_motion_metrics(qpos, 30.0)
    assert linear == 60.0
    assert angular == 0.0

    # Quaternion sign changes do not represent physical angular motion.
    qpos[1, 3] = -1.0
    _, angular = root_motion_metrics(qpos, 30.0)
    assert angular == 0.0


def test_real_sole_and_unexpected_waist_limit_are_resolved_from_model():
    import mujoco

    from retarget.retarget_quality import (
        foot_collision_geoms,
        joint_limit_report,
        sole_height,
    )

    model = mujoco.MjModel.from_xml_path(
        str(GMR_ROOT / "assets/agibot_x2/x2_mocap.xml")
    )
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    geoms = foot_collision_geoms(model, mujoco)
    expected = min(
        data.geom_xpos[geom, 2] - model.geom_size[geom, 0] for geom in geoms
    )
    assert sole_height(model, data, geoms) == expected

    hinge_ids = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    names = [model.joint(joint_id).name for joint_id in hinge_ids]
    dof = np.zeros((20, len(hinge_ids)))
    waist = names.index("waist_pitch_joint")
    dof[:, waist] = model.jnt_range[hinge_ids[waist], 1]
    report, unexpected = joint_limit_report(model, mujoco, dof)
    assert report["waist_pitch_joint"] == 1.0
    assert unexpected["waist_pitch_joint"] == 1.0


def test_each_root_height_can_be_grounded_without_a_stale_offset():
    import mujoco

    from retarget.retarget_quality import foot_collision_geoms, sole_height

    model = mujoco.MjModel.from_xml_path(
        str(GMR_ROOT / "assets/agibot_x2/x2_mocap.xml")
    )
    data = mujoco.MjData(model)
    geoms = foot_collision_geoms(model, mujoco)
    data.qpos[3] = 1.0
    for root_z in (0.45, 0.80, 1.10):
        data.qpos[2] = root_z
        mujoco.mj_forward(model, data)
        data.qpos[2] -= sole_height(model, data, geoms)
        mujoco.mj_forward(model, data)
        assert abs(sole_height(model, data, geoms)) < 1e-12


def test_x2_arm_limit_filter_allows_rest_but_rejects_bad_branches():
    import mujoco

    from retarget.retarget_quality import joint_limit_report

    model = mujoco.MjModel.from_xml_path(
        str(GMR_ROOT / "assets/agibot_x2/x2_mocap.xml")
    )
    hinge_ids = [
        joint_id for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    names = [model.joint(joint_id).name for joint_id in hinge_ids]

    def at_limit(name, endpoint):
        dof = np.zeros((20, len(hinge_ids)))
        index = names.index(name)
        range_index = 0 if endpoint == "lower" else 1
        dof[:, index] = model.jnt_range[hinge_ids[index], range_index]
        return joint_limit_report(model, mujoco, dof)[1]

    # Straight elbows and shoulder-roll stops are valid poses. Shoulder roll
    # can reach either end for arms resting by the torso or moving overhead.
    assert "left_elbow_joint" not in at_limit("left_elbow_joint", "upper")
    assert "left_shoulder_roll_joint" not in at_limit(
        "left_shoulder_roll_joint", "lower"
    )
    assert "right_shoulder_roll_joint" not in at_limit(
        "right_shoulder_roll_joint", "upper"
    )
    assert "left_shoulder_roll_joint" not in at_limit(
        "left_shoulder_roll_joint", "upper"
    )
    assert "right_shoulder_roll_joint" not in at_limit(
        "right_shoulder_roll_joint", "lower"
    )

    # Deep elbow flexion and either yaw endpoint indicate an unreachable human
    # pose or an incorrect IK solution branch.
    assert at_limit("left_elbow_joint", "lower")["left_elbow_joint"] == 1.0
    assert at_limit("left_shoulder_yaw_joint", "upper")[
        "left_shoulder_yaw_joint"
    ] == 1.0

    # A continuous bad branch is rejected even when it occupies less than a
    # quarter of a long clip; isolated short contacts with a stop are allowed.
    yaw = names.index("left_shoulder_yaw_joint")
    dof = np.zeros((300, len(hinge_ids)))
    dof[100:155, yaw] = model.jnt_range[hinge_ids[yaw], 1]
    _, unexpected = joint_limit_report(model, mujoco, dof, fps=50.0)
    assert unexpected["left_shoulder_yaw_joint"] == round(55 / 300, 3)
    dof[:] = 0.0
    dof[100:125, yaw] = model.jnt_range[hinge_ids[yaw], 1]
    _, unexpected = joint_limit_report(model, mujoco, dof, fps=50.0)
    assert "left_shoulder_yaw_joint" not in unexpected


def test_x2_waist_posture_regularizer_is_active():
    import mujoco
    import mink

    from general_motion_retargeting import GeneralMotionRetargeting

    retargeter = GeneralMotionRetargeting(
        actual_human_height=1.8,
        src_human="smplx",
        tgt_robot="agibot_x2",
        verbose=False,
    )
    posture = next(task for task in retargeter.tasks1 if isinstance(task, mink.PostureTask))
    joint_id = mujoco.mj_name2id(
        retargeter.model, mujoco.mjtObj.mjOBJ_JOINT, "waist_pitch_joint"
    )
    assert posture.cost[retargeter.model.jnt_dofadr[joint_id]] == 20.0


def test_x2_arm_collision_projection_clears_hip_without_moving_body():
    import mujoco
    from mink.limits import CollisionAvoidanceLimit

    from general_motion_retargeting import GeneralMotionRetargeting

    retargeter = GeneralMotionRetargeting(
        actual_human_height=1.8,
        src_human="smplx",
        tgt_robot="agibot_x2",
        verbose=False,
    )
    model = retargeter.model

    def body_geoms(names):
        result = []
        for name in names:
            body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            result.extend(
                geom for geom in range(model.ngeom)
                if model.geom_bodyid[geom] == body
                and model.geom_contype[geom] != 0
            )
        return result

    left_arm = body_geoms([
        f"left_{part}_link"
        for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")
    ])
    right_arm = body_geoms([
        f"right_{part}_link"
        for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")
    ])
    lower = body_geoms(
        ["pelvis", "torso_link"]
        + [
            f"{side}_{part}_link"
            for side in ("left", "right")
            for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee")
        ]
    )
    retargeter.ik_limits.append(CollisionAvoidanceLimit(
        model,
        [(left_arm + right_arm, lower), (left_arm, right_arm)],
        minimum_distance_from_collisions=0.005,
    ))

    # A real take_01 hanging-arm configuration: the left wrist sits 33 mm
    # inside the left hip-roll shell before projection.
    qpos = model.qpos0.copy()
    joint_values = {
        "left_hip_pitch_joint": -0.296633,
        "left_hip_roll_joint": 0.031880,
        "left_hip_yaw_joint": 0.290689,
        "waist_yaw_joint": 0.091156,
        "waist_pitch_joint": 0.076548,
        "waist_roll_joint": 0.117686,
        "left_shoulder_pitch_joint": 0.481692,
        "left_shoulder_roll_joint": -0.056553,
        "left_shoulder_yaw_joint": -0.931580,
        "left_elbow_joint": -0.705026,
        "left_wrist_yaw_joint": -0.586200,
        "left_wrist_pitch_joint": 0.045378,
        "left_wrist_roll_joint": 0.718546,
    }
    for name, value in joint_values.items():
        qpos[model.joint(name).qposadr[0]] = value

    data = mujoco.MjData(model)
    fromto = np.empty(6)

    def clearance(sample):
        data.qpos[:] = sample
        mujoco.mj_forward(model, data)
        return min(
            mujoco.mj_geomDistance(
                model, data, arm_geom, lower_geom, 0.02, fromto
            )
            for arm_geom in left_arm
            for lower_geom in lower
        )

    assert clearance(qpos) < -0.015
    retargeter.configuration.update(qpos)
    retargeter.x2_arm_raw_quats = {}  # enable the SMPL-X analytic-arm path
    retargeter._x2_project_collisions()
    corrected = retargeter.configuration.data.qpos.copy()

    assert clearance(corrected) >= 0.0075
    roll_addr = model.joint("left_shoulder_roll_joint").qposadr[0]
    unchanged = np.ones(model.nq, dtype=bool)
    unchanged[roll_addr] = False
    assert np.array_equal(corrected[unchanged], qpos[unchanged])
