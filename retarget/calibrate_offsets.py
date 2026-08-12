"""Calibrate per-joint zero-pose offsets between G1 (source) and X2 (target).

The scattering matrix S_r maps joints by name, but the two robots define
different zero poses (G1: elbow bent 90 deg forward; X2: arms hanging, upper
arm flared outward). This script measures, at the zero pose, the world-frame
"bone" direction vectors (joint origin -> next joint origin) on both robots
and solves, joint by joint down each chain, the rotation about the (shared)
joint axis that aligns the X2 bone to the G1 bone. The result is an offset
vector b such that

    q_x2 = S_r^T q_g1 + b

Bone directions use body origins only, so the result is independent of how
each vendor oriented their link frames.

Run from Desktop/Byte inside .venv_sim; writes x2_offsets.json next to this file.
"""

import json
import os

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G1_XML = os.path.join(ROOT, "Sonic_Retarget/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml")
X2_XML = os.path.join(ROOT, "aimdk/X2_URDF-v1.3.0/x2_ultra.xml")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "x2_offsets.json")

# Chains to calibrate: (joint, bone start body, bone end body). Bone bodies are
# picked per robot because chain-tip link names differ (G1 wrist ends at
# wrist_yaw_link, X2 at wrist_roll_link). Joints whose bone cannot be probed
# (wrists, ankles, waist, head) keep offset 0 - their zero poses coincide or
# the residual is left to retargeting/LoRA, matching the port report's scope.
CHAINS = {
    "g1": [
        ("left_shoulder_roll_joint", "left_shoulder_pitch_link", "left_elbow_link"),
        ("left_elbow_joint", "left_elbow_link", "left_wrist_yaw_link"),
        ("right_shoulder_roll_joint", "right_shoulder_pitch_link", "right_elbow_link"),
        ("right_elbow_joint", "right_elbow_link", "right_wrist_yaw_link"),
        ("left_hip_pitch_joint", "left_hip_pitch_link", "left_knee_link"),
        ("left_knee_joint", "left_knee_link", "left_ankle_roll_link"),
        ("right_hip_pitch_joint", "right_hip_pitch_link", "right_knee_link"),
        ("right_knee_joint", "right_knee_link", "right_ankle_roll_link"),
    ],
    "x2": [
        ("left_shoulder_roll_joint", "left_shoulder_pitch_link", "left_elbow_link"),
        ("left_elbow_joint", "left_elbow_link", "left_wrist_roll_link"),
        ("right_shoulder_roll_joint", "right_shoulder_pitch_link", "right_elbow_link"),
        ("right_elbow_joint", "right_elbow_link", "right_wrist_roll_link"),
        ("left_hip_pitch_joint", "left_hip_pitch_link", "left_knee_link"),
        ("left_knee_joint", "left_knee_link", "left_ankle_roll_link"),
        ("right_hip_pitch_joint", "right_hip_pitch_link", "right_knee_link"),
        ("right_knee_joint", "right_knee_link", "right_ankle_roll_link"),
    ],
}


def body_pos(model, data, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert i >= 0, name
    return data.xpos[i].copy()


def bone_dir(model, data, start, end):
    v = body_pos(model, data, end) - body_pos(model, data, start)
    return v / np.linalg.norm(v)


def joint_qpos_addr(model, jname):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return model.jnt_qposadr[j], j


def solve_axis_rotation(axis, v_from, v_to):
    """Angle about `axis` rotating v_from as close as possible to v_to."""
    # project both vectors onto the plane orthogonal to axis
    a = axis / np.linalg.norm(axis)
    pf = v_from - a * (v_from @ a)
    pt = v_to - a * (v_to @ a)
    if np.linalg.norm(pf) < 1e-8 or np.linalg.norm(pt) < 1e-8:
        return 0.0
    pf /= np.linalg.norm(pf)
    pt /= np.linalg.norm(pt)
    return float(np.arctan2(np.cross(pf, pt) @ a, pf @ pt))


def main():
    g1 = mujoco.MjModel.from_xml_path(G1_XML)
    g1d = mujoco.MjData(g1)
    g1d.qpos[3] = 1
    mujoco.mj_kinematics(g1, g1d)

    x2 = mujoco.MjModel.from_xml_path(X2_XML)
    x2d = mujoco.MjData(x2)
    x2d.qpos[3] = 1
    mujoco.mj_kinematics(x2, x2d)

    offsets = {}
    print(f"{'joint':32s} {'bone angle G1-X2':>18s} {'offset (rad)':>13s}  X2 range")
    # calibrate root-outward: apply each solved offset before probing the next
    for (jname, s_g1, e_g1), (_, s_x2, e_x2) in zip(CHAINS["g1"], CHAINS["x2"]):
        vg = bone_dir(g1, g1d, s_g1, e_g1)
        vx = bone_dir(x2, x2d, s_x2, e_x2)
        angle = float(np.degrees(np.arccos(np.clip(vg @ vx, -1, 1))))
        adr, j = joint_qpos_addr(x2, jname)
        axis = x2d.xaxis[j].copy()
        off = solve_axis_rotation(axis, vx, vg)
        lo, hi = x2.jnt_range[j]
        off_clamped = float(np.clip(off, lo, hi))
        if abs(off) > np.radians(3):  # ignore sub-3deg noise
            offsets[jname] = off_clamped
            x2d.qpos[adr] = off_clamped
            mujoco.mj_kinematics(x2, x2d)  # re-pose before probing children
        print(f"{jname:32s} {angle:15.1f} deg {off_clamped:+13.3f}  [{lo:+.2f}, {hi:+.2f}]")

    # residual check: bone directions after calibration
    print("\nresidual bone misalignment after offsets:")
    for (jname, s_g1, e_g1), (_, s_x2, e_x2) in zip(CHAINS["g1"], CHAINS["x2"]):
        vg = bone_dir(g1, g1d, s_g1, e_g1)
        vx = bone_dir(x2, x2d, s_x2, e_x2)
        angle = float(np.degrees(np.arccos(np.clip(vg @ vx, -1, 1))))
        print(f"  {jname:30s} {angle:6.1f} deg")

    with open(OUT, "w") as f:
        json.dump(offsets, f, indent=2)
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
